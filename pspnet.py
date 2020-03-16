import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import extractors


class SwitchConv(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=3, padding=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=kernel_size, padding=padding, **kwargs)
        self.extraweight = nn.Parameter((torch.rand(1,outchannels,1,1)*2-1)*outchannels**0.5)
        self.bn = nn.BatchNorm2d(outchannels)
        self.prelu = nn.PReLU()
        
    def forward(self, x, sw):
        x = self.conv(x)
        x = x + (2*sw-1).view(-1,1,1,1)*self.extraweight
        x = self.bn(x)
        x = self.prelu(x)
        return x

class AugConv(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=3, padding=1, stride=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs)
        self.yconv = nn.Conv2d(1, outchannels, kernel_size=1)
        self.xconv = nn.Conv2d(1, outchannels, kernel_size=1)
        self.bn = nn.BatchNorm2d(outchannels)
        self.prelu = nn.PReLU()
        
    def forward(self, x, yaug, xaug):
        x = self.conv(x)+self.yconv(yaug)+self.xconv(xaug)
        x = self.bn(x)
        x = self.prelu(x)
        return x
        
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
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
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
        p = F.upsample(input=x, size=(h, w), mode='bilinear', align_corners=False)
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
        #self.instnorm = nn.InstanceNorm2d(3)

    def forward(self, x):
        #x = self.instnorm(x)
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
    def __init__(self, n_classes=3, out_nums=6, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True, w=64, h=54, extraend=False, extraswitch=False, final_aug=False):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        if extraswitch:
            self.conv1 = SwitchConv(1024, 256)
        else:
            self.conv1 = self._convlayer(1024, 256)
        # Trying coord augmentation
        # Changes start here
        self.final_aug = final_aug
        if final_aug:
            self.conv2 = AugConv(256,256)
        else:
            self.conv2 = self._convlayer(256, 256)
        #self.conv2 = self._convlayer(256, 256)
        
        # Changes end here
        
        self.conv3 = self._convlayer(256, 256)
        
        if extraend:
            self.fin = nn.Sequential(
                self._downlayer(256,256), # ???
                self._downlayer(256,256),
                nn.Conv2d(256, 256, kernel_size=(h//4,w//4)),
                nn.Conv2d(256, out_nums, kernel_size=1)
            )
        else:
            self.fin = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.Sigmoid(),
                nn.Conv2d(64, out_nums, kernel_size=(h,w))
            )
            
        self.extraswitch = extraswitch
            
        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        
        #self.instnorm = nn.InstanceNorm2d(3)
        #self.classifierb = nn.Sequential(
        #    nn.Linear(256, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, n_classes)
        #)
        
    def _convlayer(self, inchannels, outchannels):
        return nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, padding=1),
                                       nn.BatchNorm2d(outchannels),
                                       nn.PReLU())
                                       
    def _downlayer(self, inchannels, outchannels):
        return nn.Sequential(nn.Conv2d(inchannels, outchannels, 2, stride=2),
                                       nn.BatchNorm2d(outchannels),
                                       nn.PReLU())
        
    def forward(self, x, switcher=None, yaug=None, xaug=None):
        #x = self.instnorm(x)
        f, class_f = self.feats(x)
        p = self.psp(f)
        if self.extraswitch:
            if switcher is None:
                raise TypeError("Didn't get a switcher")
                switcher = torch.ones((p.size(0),1,1,1),dtype=p.dtype,device=p.device)*0.5 # This is probably fine
            p = self.conv1(p, switcher)
        else:
            if switcher is not None:
                raise TypeError('Got a switcher with no switching')
            p = self.conv1(p)
            
        # Trying coordinate augmentation
        # Changes start here
        if self.final_aug:
            p = self.conv2(p,yaug,xaug)
        else:
            p = self.conv2(p)
        #p = self.conv2(p)
        # Changes end here
        
        p3 = self.conv3(p)
        p = self.fin(p3)
        p = p.view(-1,p.size(1))
        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
        #auxiliaryb = F.adaptive_max_pool2d(input=p3, output_size=(1, 1)).view(-1, p3.size(1))
        return p, self.classifier(auxiliary)#, self.classifierb(auxiliaryb)
        
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
        #self.dense2 = self._denselayer(64,64)
        self.fin = nn.Linear(64, 7)
        
    def forward(self, x):
        y = self.dense1(x)
        #y = self.dense2(y)
        y = self.fin(y)
        return y
        
    def _denselayer(self, infeats, outfeats):
        return nn.Sequential(nn.Linear(infeats, outfeats),
                             nn.BatchNorm1d(outfeats),
                             nn.ReLU(),
                             nn.Dropout(p=0.15))
                             
class PSPDraw(nn.Module):
    def __init__(self, n_classes=3, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True, w=32, h=27):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(66, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax(dim = 1)
        )

        self.conv1 = self._convlayer(1024, 256)
        self.conv2 = self._convlayer(256, 256)
        self.conv3 = self._convlayer(256, 256)
        
        self.nums = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(64, 6, kernel_size=(h,w))
        )
        
        self.draw = DrawCirc(w*4,h*4)
        
        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        q = self.up_1(p)
        q = self.drop_2(q)

        q = self.up_2(q)
        q = self.drop_2(q)

        p = self.conv1(p)
        p = self.conv2(p)
        p = self.conv3(p)
        p = self.nums(p)
        p = self.draw(*torch.split(p,1,1), 1)
        
        q = self.up_3(torch.cat((q,p),1))
        q = self.drop_2(q)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(q), self.classifier(auxiliary)
        
    def _convlayer(self, inchannels, outchannels):
        return nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, padding=1),
                                       nn.BatchNorm2d(outchannels),
                                       nn.PReLU(),
                                       nn.Dropout2d(p=0.15))
                             
class DrawCirc(nn.Module):
    def __init__(self, w=64, h=54, phi=1, psi=1):
        super().__init__()
        yy, xx = np.mgrid[0:h,0:w]
        self.yy = nn.Parameter(torch.tensor(yy, dtype=torch.float, requires_grad=False))
        self.xx = nn.Parameter(torch.tensor(xx, dtype=torch.float, requires_grad=False))
        self.phi = phi
        self.psi = psi
        self.th = nn.Tanh()
    def forward(self, xl, yl, rl, xi, yi, ri, inpres):
        lens = self.phi*(rl - ((self.xx-xl)**2 + (self.yy-yl)**2)/rl)
        if inpres:
            inner = lens + self.psi*(ri - ((self.xx-xi)**2 + (self.yy-yi)**2)/ri)*(lens>0).float()
        else:
            inner = -torch.ones_like(lens)*self.psi-F.relu(-lens)
        return self.th(torch.cat((lens, inner), dim=1))

class PSPRep(nn.Module):
    def __init__(self, n_classes=3, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True, w=32, h=27, reps=1):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 62)
        self.up_3 = CombUpsample(62, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)

        self.conv1 = self._convlayer(1024, 256)
        self.conv2 = self._convlayer(256, 128)
        self.conv3 = self._convlayer(128, 128)
        
        self.nums = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(32, 6, kernel_size=(h,w))
        )
        
        self.draw = DrawCirc(w*8,h*8)
        
        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

        self.pool = nn.AvgPool2d(8)
        
        self.reps = reps
        
    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        q = self.up_1(p)
        q = self.drop_2(q)

        q = self.up_2(q)
        q = self.drop_2(q)

        p = self.conv1(p)
        p = self.conv2(p)
        
        r = self.conv3(p)
        r = self.nums(r)
        
        for n in range(self.reps):
            r = self.draw(*torch.split(r,1,1), 1)
            s = self.up_3(q,r)
            s = self.pool(s)
            s = self.drop_2(s)
            r = self.conv3(torch.cat([p[:,0:64], s],1))
            r = self.nums(r)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return r.view(-1,6), self.classifier(auxiliary)
        
    def _convlayer(self, inchannels, outchannels):
        return nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, padding=1),
                                       nn.BatchNorm2d(outchannels),
                                       nn.PReLU(),
                                       nn.Dropout2d(p=0.15))
        
class CombUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, comb_channels=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels+comb_channels, out_channels, 3, padding=1)
        self.fin = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x, y):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear', align_corners=False)
        p = self.conv(torch.cat([p,y],1))
        return self.fin(p)