import torch
from torch import nn
from torch.nn import functional as F

import extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)

class PSPNet(nn.Module):
    def __init__(self, n_classes=3, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax(dim = 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(p), self.classifier(auxiliary)
        
class PSPCircs(nn.Module):
    def __init__(self, n_classes=3, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True, w=64, h=54, extraend=False, combinecorners=False):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.conv1 = self._convlayer(1024, 256)
        self.conv2 = self._convlayer(256, 256)
        self.conv3 = self._convlayer(256, 256)
        if extraend:
            self.fin = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.Sigmoid(),
                nn.Conv2d(64, 256, kernel_size=(h,w)),
                nn.ReLU(),
                nn.Conv2d(256, 6,  kernel_size=1)
            )
        else:
            self.fin = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.Sigmoid(),
                nn.Conv2d(64, 6, kernel_size=(h,w))
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        
        self.combinecorners = combinecorners
        if(combinecorners):
            self.combiner = QuadCombine()
        
    def _convlayer(self, inchannels, outchannels):
        return nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, padding=1),
                                       nn.BatchNorm2d(outchannels),
                                       nn.PReLU(),
                                       nn.Dropout2d(p=0.15))
        
    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)
        p = self.conv1(p)
        p = self.conv2(p)
        p = self.conv3(p)
        p = self.fin(p)
        p = p.view(-1,6)
        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
        if(self.combinecorners):
            q = self.combiner(p, self.classifier(auxiliary))
            return (p, q), self.classifier(auxiliary)
        return p, self.classifier(auxiliary)
        
class PSPDual(nn.Module):
    def __init__(self, n_classes=3, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True, w=32, h=27):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.segout = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )
        
        conva = nn.Conv2d(64, 256, kernel_size=8, stride=8)
        convb = nn.Conv2d(256, 6, kernel_size=(h,w))
        conva.weight.data = torch.abs(conva.weight.data)/64
        convb.weight.data = torch.abs(convb.weight.data)/h
        self.numout = nn.Sequential(
            nn.Softmax(1),
            conva,
            nn.Dropout2d(p=0.3),
            convb
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.segout(p), self.numout(p).view(-1,6), self.classifier(auxiliary)
        
class QuadCombine(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = self._denselayer(36,64)
        self.dense2 = self._denselayer(64,64)
        self.fin = nn.Linear(64, 6)
        
    def forward(self, x, x_cls):
        y = torch.cat((x.view(-1,24), x_cls.view(-1,12)),1)
        y = self.dense1(y)
        y = self.dense2(y)
        y = self.fin(y)
        return y
        
    def _denselayer(self, infeats, outfeats):
        return nn.Sequential(nn.Linear(infeats, outfeats),
                             nn.BatchNorm1d(outfeats),
                             nn.ReLU(),
                             nn.Dropout(p=0.15))
                             
class DrawCirc(nn.Module):
    def __init__(self, w=64, h=54, phi=1, psi=1):
        yy, xx = np.mgrid[0:h][0:w]
        self.yy = torch.tensor(yy, dtype=torch.float, requires_grad=False)
        self.xx = torch.tensor(xx, dtype=torch.float, requires_grad=False)
        self.phi = phi
        self.psi = psi
        self.ls = nn.LogSoftmax(dim=0)
    def forward(self, xl, yl, rl, xi, yi, ri, inpres):
        lens = self.phi*(r1 - ((self.xx-x1)**2 + (self.yy-y1)**2)/r1)
        if inpres:
            inner = lens + self.psi*(r2 - ((self.xx-x2)**2 + (self.yy-y2)**2)/r2)*(lens>0).float() - self.psi*(lens<=0).float()
        else:
            inner = -torch.ones_like(lens)*self.psi-F.relu(-lens)
        return self.ls(torch.stack((torch.zeros_like(inner), lens, inner), dim=0))
