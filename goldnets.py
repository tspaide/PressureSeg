import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, feats=64, padding=2):
        super().__init__()
        self.size_decrease = 2*(2-padding)
        self.conv1 = nn.Conv2d(feats, feats, kernel_size=5, stride=1, padding=padding, dilation=1)
        self.conv2 = nn.Conv2d(feats, feats, kernel_size=5, stride=1, padding=padding, dilation=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(feats)
        self.bn2 = nn.BatchNorm2d(feats)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.size_decrease:
            x = x[:,:,self.size_decrease:-self.size_decrease,self.size_decrease:-self.size_decrease]
        return self.relu(x+y)
        
class ResNet(nn.Module):
    def __init__(self, infeats, outfeats, num_blocks=4, internal_feats=64, initpad=True, nonloc=None, nonloc_after=3):
        super().__init__()
        self.initpad = initpad
        if initpad:
            padding = 0
            #self.padlayer = nn.ReplicationPad2d(4*num_blocks+2)
        else:
            padding = 2
        self.start = nn.Sequential(nn.Conv2d(infeats, internal_feats, kernel_size=5, stride=1, padding=padding, dilation=1),
                                   nn.InstanceNorm2d(internal_feats),
                                   nn.ReLU())
                                   
        self.blocks = nn.ModuleList()
        for n in range(num_blocks):
            self.blocks.append(ResBlock(internal_feats, padding=padding))
        self.fin = nn.Conv2d(internal_feats, outfeats, kernel_size=1, stride=1, dilation=1)
        self.nonloc = nonloc
        if nonloc is not None:
            self.nonloc_after = nonloc_after
        if nonloc is 'dot':
            self.nonloc_layer = NonLocal(internal_feats)
        elif nonloc is 'exp':
            self.nonloc_layer = NonLocal(internal_feats, exp=True)
        elif nonloc is not None:
            raise ValueError(f'Unrecognized nonloc value {nonloc}')
        
        
    def forward(self, x):
        #if self.initpad:
        #    x = self.padlayer(x)
        x = self.start(x)
        if self.nonloc:
            for block in self.blocks[:self.nonloc_after]:
                x = block(x)
            x = self.nonloc_layer(x)
            for block in self.blocks[self.nonloc_after:]:
                x = block(x)
        else:
            for block in self.blocks:
                x = block(x)
        return self.fin(x)
        
class Matcher(nn.Module):
    def __init__(self, infeats, outfeats, internal_feats=16):
        super().__init__()
        self.start = nn.Conv2d(infeats, internal_feats, kernel_size=1)
        self.conva = nn.Conv2d(internal_feats, internal_feats, kernel_size=5)
        self.convb = nn.Conv2d(internal_feats, internal_feats, kernel_size=5)
        self.fin = nn.Conv2d(internal_feats, outfeats, kernel_size=1)
        self.rel = nn.ReLU()
    
    def forward(self, x):
        x = self.start(x)
        x = self.rel(x)
        x = self.conva(x)
        x = self.rel(x)
        x = self.convb(x)
        x = self.rel(x)
        return self.fin(x)
        
class Discrim(nn.Module):
    def __init__(self, infeats, h=256, w=256):
        super().__init__()
        #channelnums = [32,32,64,64,128,128]
        channelnums = [32,32,64,64]
        #kernel_sizes = [5,5,3,3,3,3]
        kernel_sizes = [5,5,3,3]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        inchans = infeats
        for chans, ker in zip(channelnums, kernel_sizes):
            self.convs.append(nn.Conv2d(inchans, chans, kernel_size=ker, stride=2, padding=ker//2, dilation=1))
            self.bns.append(nn.BatchNorm2d(chans))
            inchans = chans
        self.relu=nn.ReLU()
        self.patchout = nn.Conv2d(channelnums[-1], 1, kernel_size=1, stride=1, padding=0, dilation=1)
        #self.totalout = nn.Conv2d(128, 1, kernel_size=(h//64, w//64), stride=1, padding=0, dilation=1)
        
    def forward(self, x):
        for n in range(4):
            x = self.convs[n](x)
            x = self.bns[n](x)
            x = self.relu(x)
        x4 = x
        #for n in range(4,6):
        #    x = self.convs[n](x)
        #    x = self.bns[n](x)
        #    x = self.relu(x)
        return self.patchout(x4)#, self.totalout(x).reshape(-1)
        
def conv3x3(infeats, outfeats, stride=1, padding=1):
    return nn.Sequential(nn.ReplicationPad2d(padding),
                         nn.Conv2d(infeats, outfeats, kernel_size=3, stride=stride,
                                   padding=0, dilation=1),
                         nn.BatchNorm2d(outfeats),
                         nn.ReLU())
        
class NonLocal(nn.Module):
    def __init__(self, infeats, internal_feats=None, in_downsampling=16, internal_downsampling=16, exp=False):
        super().__init__()
        if internal_feats is None:
            internal_feats = infeats//2
        self.f = nn.Conv2d(infeats, internal_feats, kernel_size=1)
        self.g = nn.Conv2d(infeats, internal_feats, kernel_size=1)
        self.h = nn.Conv2d(infeats, infeats, kernel_size=1)
        self.in_pool = nn.MaxPool2d(in_downsampling)
        self.internal_pool = nn.MaxPool2d(internal_downsampling)
        self.exp = exp
        if exp:
            self.sm = nn.Softmax(4)
        
    def forward(self, x):
        selfvals = self.in_pool(self.f(x))
        othervals = self.internal_pool(self.g(x))
        selfvals = selfvals.unsqueeze(4).unsqueeze(4)
        othervals = othervals.unsqueeze(2).unsqueeze(2)
        dots = torch.sum(selfvals*othervals,1,keepdim=True)
        if self.exp:
            dotsize = dots.size()
            dots = dots.view(*dotsize[:4],-1)
            dots = self.sm(dots)
            dots = dots.view(dotsize)
        else:
            dots = dots/(selfvals.size(2)*selfvals.size(3))
        outvals = self.internal_pool(self.h(x))
        outvals = outvals.unsqueeze(2).unsqueeze(2)
        output = torch.sum(torch.sum(dots*outvals,4),4)
        return F.interpolate(output,size=(x.size(2),x.size(3)),mode='bilinear',align_corners=False)+x
        
class Upstage(nn.Module):
    def __init__(self, infeats, outfeats, bordercut=0, prevfeats=None):
        super().__init__()
        if prevfeats is None:
            prevfeats = outfeats
        self.convup = nn.ConvTranspose2d(infeats, outfeats, kernel_size=2, stride=2)
        self.conva = conv3x3(outfeats+prevfeats, outfeats, padding=1)
        self.convb = conv3x3(outfeats, outfeats, padding=1)
        self.bordercut = bordercut
        
    def forward(self, x, p):
        if self.bordercut:
            p = p[:,:,self.bordercut:-self.bordercut,self.bordercut:-self.bordercut]
        x = self.convup(x)
        x = self.conva(torch.cat((x,p), 1))
        x = self.convb(x)
        return x
        
class UNet(nn.Module):
    def __init__(self, features=1, out_features=2):
        super().__init__()
        self.start = nn.Sequential(conv3x3(features, 64),
                                   conv3x3(64, 64))
        self.down1 = self._downstage(64, 128)
        self.down2 = self._downstage(128, 256)
        self.down3 = self._downstage(256, 512)
        self.down4 = self._downstage(512, 1024)
        self.up4 = Upstage(1024,512)
        self.up3 = Upstage(512,256)
        self.up2 = Upstage(256,128)
        self.up1 = Upstage(128,64)
        self.fin = nn.Conv2d(64, out_features, kernel_size=1)
        
    def forward(self, x):
        x1 = self.start(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y5 = self.down4(x4)
        z4 = self.up4(y5, x4)
        z3 = self.up3(z4, x3)
        z2 = self.up2(z3, x2)
        z1 = self.up1(z2, x1)
        z0 = self.fin(z1)
        return z0
        
    def _downstage(self, infeats, outfeats):
        return nn.Sequential(nn.MaxPool2d(2),
                             conv3x3(infeats, outfeats),
                             conv3x3(outfeats, outfeats))