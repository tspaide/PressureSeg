import os
import math
import numpy as np
import json
import pickle
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import NoNorm
import skimage.transform
import skimage.exposure
from skimage import io
import PIL

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import circareas

import cv2
import time

plt.switch_backend('agg')


'''
Next five functions shamelessly stolen from Yue
'''

def calc_iop_tonomat(dia, tonometer=5):
    if dia>=4.3:
        iop = 25 - 10*(dia-4.3)
    elif dia>=3.7:
        iop = 37 - 20*(dia-3.7)
    elif dia>=3.5:
        iop = 43 - 30*(dia-3.5)
    else:   # not in chart
        iop = 43 - 30*(dia-3.5)

    # iop_dict = {}   # 5g tonometer  - joanne sent 2 formula, which are not the same
    # # for dia in np.arange(43, 61, 1):  # breaks because of floating point inaccuracy
    # for dia in range(43, 61):
    #     real_dia = dia/10.
    #     iop_dict[real_dia] = np.round(25 - (real_dia - 4.3) * 10, 0)
    # for dia in range(37, 44, 1):
    #     real_dia = dia/10.
    #     iop_dict[real_dia] = np.round(37 -20*(real_dia - 3.7), 0)
    # for dia in range(35, 37, 1):
    #     real_dia = dia/10.
    #     iop_dict[real_dia] = np.round(43 -30*(real_dia - 3.5), 0)

    return iop

def calc_iop_halberg(dia, tonometer=5):
    if dia>=4.3:
        iop = 26 - 10*(dia-4.3)
    elif dia>=3.8:
        iop = 36 - 20*(dia-3.8)
    else:   # not in chart
        iop = 36 - 20*(dia-3.8)
    return iop

def calc_iop_wrapper(dia, tonometer=5, do_halberg=True):
    if do_halberg:
        return calc_iop_halberg(dia, tonometer)
    else:
        return calc_iop_tonomat(dia, tonometer)

def calc_iop_from_circles(lens_circle, inner_circle):
    real_lens_dia = 9.1     # mm
    if(inner_circle == 0 or lens_circle == 0):
        return np.nan
    real_inner_dia = real_lens_dia * inner_circle/lens_circle
    iop = calc_iop_wrapper(real_inner_dia)
    return iop

def overlap(circa, circb, missloss = False):
    """Return the area of intersection of two circles and whether one contains another.
    Circles should be (3,) tensors of the form [x,y,r]
    If missloss, will return negative numbers if circles don't intersect at all
    Mostly taken from Yue.
    """
    r = torch.abs(circa[2])
    R = torch.abs(circb[2])
    
    d2 = (circa[0]-circb[0])**2 + (circa[1]-circb[1])**2
    d = torch.pow(d2, 0.5)
    if d <= torch.abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * torch.min(R, r)**2, True
    if d >= r + R:
        # The circles don't overlap at all.
        if(missloss):
            return r+R-d, False
        else:
            return 0, False
    
    r2, R2 = r**2, R**2
    alpha = torch.acos((d2 + r2 - R2) / (2*d*r))
    beta = torch.acos((d2 + R2 - r2) / (2*d*R))
    return ( r2 * alpha + R2 * beta - 0.5 * (r2 * torch.sin(2*alpha) + R2 * torch.sin(2*beta)) ), False
    
def dice_circle_loss(circa, circb, epsilon=0.001, size_average=True, reduce = True, negate = True,
                     missloss = True, addloss = None, cropbox = None):
    '''
        Smoothed dice for data of two (batches of) circles
        Circles should be (B,3) tensors of the form [[x1,y1,r1],[x2,y2,r2],...]
        If reduce, will take average or total Dice depending on size_average
        If negate, will use (1 - Dice)
        Gives additional penalty when circles don't overlap at all
    '''
    dices = torch.empty(circa.size()[0], device=circa.device)
    for n in range(circa.size()[0]):
        if(circb[n,2]==0):
            dices[n]=1
        else:
            if cropbox is None:
                over, cont = overlap(circa[n], circb[n], missloss)
                area = np.pi*(circa[n,2]**2+circb[n,2]**2)
            else:
                cropbox = torch.tensor(cropbox, dtype=torch.float, device=circa.device)
                xa, ya, ra = circa[n,0], circa[n,1], torch.abs(circa[n,2])
                xb, yb, rb = circb[n,0], circb[n,1], torch.abs(circb[n,2])
                over = circareas.cropcircoverlap(xa,ya,ra,xb,yb,rb, *cropbox)
                if(over==0 and missloss):
                    d = ((xa-xb)**2 + (ya-yb)**2)**0.5
                    over = torch.min(ra + rb - d, over)
                areaa = circareas.cropcircarea(xa,ya,ra, *cropbox)
                areab = circareas.cropcircarea(xb,yb,rb, *cropbox)
                area = areaa+areab
                if((areaa==circareas.boxarea(*cropbox) or areab==circareas.boxarea(*cropbox)) and missloss):
                    over = over-torch.abs(ra-rb)
            dices[n]=(2*over+epsilon)/(area+epsilon)
            if addloss is not None:
                dices[n] += addloss[n]
    if(negate):
        dices = 1-dices
    if(reduce):
        if(size_average):
            divisor = dices.size()[0]
        else:
            divisor = 1
        return torch.sum(dices)/divisor
    return dices
    
class Circles_Dice():
    '''
        Dice loss (sorta) for two classes (lens and inner)
        Note that lens dice is actually (lens+inner) dice
        Also calculation of inner stuff is wrong in situations where inner circle is cut off by the edge of lens
    '''
    def __init__(self, epsilon=0.001, size_average=True, reduce = True, missloss = True, cropbox=None):
        self.epsilon = epsilon
        self.size_average = size_average
        self.reduce = reduce
        self.missloss = missloss
        self.cropbox = cropbox
    def __call__(self, out, y, inprob=1, lsmall = None, insmall = None):
        outlens, outin = out.split([3,3],1)
        ylens, yin = y.split([3,3],1)
        lensdice = dice_circle_loss(outlens, ylens, epsilon=self.epsilon, size_average=self.size_average,
                                    reduce=self.reduce, missloss=self.missloss, addloss = lsmall, cropbox=self.cropbox)
        indice = dice_circle_loss(outin, yin, epsilon=self.epsilon, size_average=self.size_average,
                                  reduce=self.reduce, missloss=self.missloss, addloss = insmall, cropbox=self.cropbox)
        return lensdice+indice

def algproc(y):
    lenssmall = torch.min(torch.zeros_like(y[:,2]),y[:,2]-1)
    insmall = torch.min(torch.zeros_like(y[:,5]),y[:,5]-1)
    lrs = torch.max(y[:,2],torch.ones_like(y[:,2]))
    irs = torch.max(y[:,5],torch.ones_like(y[:,5]))
    processed = torch.t(torch.stack((y[:,0]/lrs, y[:,1]/lrs, torch.pow(lrs, 0.5),
                                     y[:,3]/irs, y[:,4]/irs, torch.pow(irs, 0.5))))
    return processed, lenssmall, insmall
        
def showcircs(entry):
    imgname = "../joanne/joanne_seg_manual/" + entry[0] + ".png"
    img = io.imread(imgname)
    plt.imshow(img)
    lens = entry[1]['lens_data']
    inner = entry[1]['inner_data']
    patches = []
    patches.append(Circle((lens[0],lens[1]),lens[2], color='r', fill=False))
    if inner[0] != 0.0:
        patches.append(Circle((inner[0],inner[1]),inner[2], color='y', fill=False))
    fig, ax = plt.subplots()
    ax.imshow(img)
    for p in patches:
        ax.add_artist(p)
    plt.savefig("circs.png")
     
def rotcheck(lens_data):
    x,y,r = lens_data
    ang = math.radians(15)
    left = math.cos(ang)*(y-540)-math.sin(ang)*(x-960)
    right = math.cos(ang)*(y-540)+math.sin(ang)*(x-960)
    toproom = left+540-r
    botroom = 540-right-r-1
    if botroom<toproom:
        return botroom, 'R'
    else:
        return toproom, 'L'
     
def dice2(ground, out):
    g = [0,0,0]
    o = [0,0,0]
    i = [0,0,0]
    h = len(ground)
    w = len(ground[0])
    for y in range(h):
        for x in range(w):
            g[ground[y][x]]+=1
            o[out[y][x]]+=1
            if(ground[y][x]==out[y][x]):
                i[ground[y][x]]+=1
    l1 = 1-(2*i[1]/(g[1]+o[1]))
    if(g[2]+o[2]==0):
        l2 = 0
    else:
        l2 = 1-(2*i[2]/(g[2]+o[2]))
    return l1+l2
                
def circfill(lens_data, inner_data, width=1920, height=1080, style=[0,1,2], classes = True):
    # style [x,y,z] indicates that outside of lens is x, inside of lens is y, inside of inner is z
    (lx, ly, lr) = lens_data
    (ix, iy, ir) = inner_data
    if(lr == 0):
        lx, ly = -1, -1
    if(ir == 0):
        ix, iy = -1, -1
    lines = []
    for y in range(height):
        leftlen = width if (y-ly)**2>lr**2 else math.floor(lx-(math.sqrt(lr**2-(y-ly)**2)))
        rightlen = width if (y-ly)**2>lr**2 else math.floor(lx+(math.sqrt(lr**2-(y-ly)**2)))
        leftin = rightlen if (y-iy)**2>ir**2 else math.floor(ix-(math.sqrt(ir**2-(y-iy)**2)))
        rightin = rightlen if (y-iy)**2>ir**2 else math.floor(ix+(math.sqrt(ir**2-(y-iy)**2)))
        leftlen = min(max(leftlen, 0), width)
        rightlen = max(min(rightlen, width), leftlen)
        leftin = min(max(leftin, leftlen), rightlen)
        rightin = max(leftin, min(rightin, rightlen))
        lines.append([style[0]]*leftlen + [style[1]]*(leftin-leftlen)
                        + [style[2]]*(rightin-leftin) + [style[1]]*(rightlen-rightin)
                        + [style[0]]*(width-rightlen))
        if(len(lines[-1])>width):
            print(y)
            print(len(lines[-1]))
            print(lens_data)
            print(inner_data)
            print(leftlen, leftin, rightin, rightlen, width)
    if classes:
        present = torch.zeros(max(style)+1, dtype=torch.float)
        present[style[0]]=1
        present[style[1]]=1
        if (ir>0):
            present[style[2]]=1
        return lines, present
    return lines

def circdraw(lens_data, inner_data, width=1920, height=1080, thickness=2, style=[0,1,2,0,0], classes = True):
    # style [a,b,c,d,e] indicates that base is a, lens circle is b, inner circle is c, inside of lens is d, inside of inner is e
    # doesn't work if thickness > one of the radii; this is fairly easy to avoid
    if(len(style)==2):
        style.append(style[1])
    if(len(style)==3):
        style.append(style[0])
    if(len(style)==4):
        style.append(style[0])
    (lx, ly, lr) = lens_data
    (ix, iy, ir) = inner_data
    lines = []

    for y in range(height):
        leftlenout = width if (y-ly)**2>(lr+thickness)**2 else math.floor(lx-(math.sqrt((lr+thickness)**2-(y-ly)**2)))
        rightlenout = width if (y-ly)**2>(lr+thickness)**2 else math.floor(lx+(math.sqrt((lr+thickness)**2-(y-ly)**2)))
        leftlenin = rightlenout if (y-ly)**2>(lr-thickness)**2 else math.floor(lx-(math.sqrt((lr-thickness)**2-(y-ly)**2)))
        rightlenin = rightlenout if (y-ly)**2>(lr-thickness)**2 else math.floor(lx+(math.sqrt((lr-thickness)**2-(y-ly)**2)))
        leftinout = rightlenout if (y-iy)**2>(ir+thickness)**2 else math.floor(ix-(math.sqrt((ir+thickness)**2-(y-iy)**2)))
        rightinout = rightlenout if (y-iy)**2>(ir+thickness)**2 else math.floor(ix+(math.sqrt((ir+thickness)**2-(y-iy)**2)))
        leftinin = rightinout if (y-iy)**2>(ir-thickness)**2 else math.floor(ix-(math.sqrt((ir-thickness)**2-(y-iy)**2)))
        rightinin = rightinout if (y-iy)**2>(ir-thickness)**2 else math.floor(ix+(math.sqrt((ir-thickness)**2-(y-iy)**2)))
        leftlenout = max(leftlenout, 0)
        rightlenout = min(rightlenout, width)
        leftlenin = max(leftlenin, leftlenout)
        rightlenin = min(rightlenin, rightlenout)
        leftinout = min(max(leftinout, leftlenin), rightlenin)
        rightinout = max(leftinout, min(rightinout, rightlenin))
        leftinin = min(max(leftinin, leftinout), rightinout)
        rightinin = max(leftinin, min(rightinin, rightinout))
        lines.append([style[0]]*leftlenout + [style[1]]*(leftlenin-leftlenout) + [style[3]]*(leftinout-leftlenin)
                        + [style[2]]*(leftinin-leftinout) + [style[4]]*(rightinin-leftinin)
                        + [style[2]]*(rightinout-rightinin) + [style[3]]*(rightlenin-rightinout)
                        + [style[1]]*(rightlenout-rightlenin) + [style[0]]*(width-rightlenout))
    if not classes:
        return lines
    present = torch.zeros(max(style)+1, dtype=torch.float)
    present[style[0]]=1
    present[style[1]]=1
    present[style[3]]=1
    if (ir>0):
        present[style[2]]=1
        present[style[4]]=1
    return lines, present
    
def findmancirc(imarr): # Find manually added red circle
    (h,w) = imarr.shape[:2]
    pixnum = 0
    xtot = 0
    ytot = 0
    for y in range(h):
        for x in range(w):
            if(imarr[y,x,0]==237):
                pixnum+=1
                xtot+=x
                ytot+=y
    if(pixnum==0):
        return (0,0,0)
    else:
        r = math.sqrt(pixnum/math.pi)
        return (xtot/pixnum, ytot/pixnum, r)

def checkbounds(data): # True iff lens circle hits top or bottom of image
    y = data['lens_data'][1]
    r = data['lens_data'][2]
    return ((y<r or y+r>=1080) and y !=0.0)

def getannotations(circle_file):
    infile = open(circle_file)
    annotations = [(k,v) for k,v in json.load(infile).items() if v['lens_data'][2]!=0.0] # Lens data of 0 seems to be incorrect
    infile.close()
    return annotations

class ToPil():
    def __call__(self,  image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        return F.to_pil_image(image), lens_data, inner_data

class ToTens():
    def __call__(self,  image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data, *a = image
        return (F.to_tensor(image), lens_data, inner_data, *a)

class Mapper():
    def __init__(self, op):
        self.op = op
    def __call__(self, input):
        return map(self.op, input)

class Unzipper():
    def __call__(self, input):
        return zip(*input)
        
class RandomResizedCropP():
    '''
        If tame='yes' it will try to make sure the eye is completely in frame
        If tame='sides' will only do this for x coordinate
        Will always at least make sure that the frame contains the center of the eye
    '''
    def __init__(self, width, height, scale, maxscale=None, tame = 'yes'):
        self.finwidth = width
        self.finheight = height
        self.minscale = scale
        self.maxscale = maxscale
        self.tame = tame
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        (width, height) = image.size
        (lx, ly, lr) = lens_data
        (ix, iy, ir) = inner_data
        if self.maxscale is None:
            scale = self.minscale
        else:
            scale = np.random.random_sample() * (self.maxscale-self.minscale)+self.minscale
        cropwidth = self.finwidth/scale
        cropheight  = self.finheight/scale
        if (self.tame == 'yes' or self.tame == 'sides'):
            minx = max(0, lx+lr-cropwidth)
            maxx = min(width-cropwidth, lx-lr) + 1
        else:
            minx = max(0, lx-cropwidth)
            maxx = min(width-cropwidth, lx) + 1
        if(self.tame == 'yes'):
            miny = max(0, ly+lr-cropheight)
            maxy = min(height-cropheight, ly-lr) + 1
        else:
            miny = max(0, ly-cropheight)
            maxy = min(height-cropheight, ly) + 1
        if(minx>maxx) or (miny>maxy):
            print(minx, maxx)
            print(miny, maxy)
            print(lens_data)
            print(inner_data)
            print(width, height)
            print(scale, cropwidth, cropheight)
        x = np.random.randint(minx, maxx)
        y = np.random.randint(miny, maxy)
        lx, ly, lr = scale*(lx-x),scale*(ly-y),scale*lr
        if(ir>0):
            ix, iy, ir = scale*(ix-x),scale*(iy-y),scale*ir
        return F.resized_crop(image,y,x,cropheight,cropwidth,(self.finheight,self.finwidth)), (lx,ly,lr), (ix,iy,ir)
        
class CropP():
    def __init__(self, minx, miny, maxx = None, maxy = None, width = None, height = None):
        if(maxx is None and width is None):
            raise ValueError("Need a maxx or width argument")
        if(maxy is None and height is None):
            raise ValueError("Need a maxy or height argument")
        self.minx = minx
        self.miny = miny
        if maxx is not None:
            self.width = maxx-minx
        else:
            self.width = width
        if maxx is not None:
            self.height = maxy-miny
        else:
            self.height = height
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        (lx, ly, lr) = lens_data
        (ix, iy, ir) = inner_data
        return F.crop(image,self.miny,self.minx,self.height,self.width), (lx-self.minx,ly-self.miny,lr), (ix-self.minx,iy-self.miny,ir)

class RandRotateP():
    def __init__(self, angle = 15, maxangle = None, resizing = False):
        '''
            The resizing attribute for this and Rotate controls whether the image will resize to keep the whole
            lens in the picture; it's only neccessary to ensure tame cropping
        '''
        if maxangle is None:
            maxangle = abs(angle)
            angle = -maxangle
        self.min = angle
        self.max = maxangle
        self.resizing = resizing
    def __call__(self, image, lens_data = None, inner_data = None):
        angle = np.random.random_sample() * (self.max-self.min)+self.min
        return rotateP(image, lens_data, inner_data, angle, self.resizing)
        
class ResizeP():
    def __init__(self, dim):
        self.dim=dim
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        (w,h) = image.size
        scale = self.dim/min(w,h)
        (lx, ly, lr) = lens_data
        (lx, ly, lr) = scale*lx, scale*ly, scale*lr
        (ix, iy, ir) = inner_data
        (ix, iy, ir) = scale*ix, scale*iy, scale*ir
        return F.resize(image, self.dim), (lx, ly, lr), (ix, iy, ir)
        
class FlipP():
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        (lx,ly,lr) = lens_data
        (ix,iy,ir) = inner_data
        w = image.size[0]
        if(ir!=0):
            ix = w-ix
        return F.hflip(image), (w-lx, ly, lr), (ix, iy, ir)
        
class PadP():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        (lx, ly, lr) = lens_data
        lens_data = (lx+self.x,ly+self.y,lr)
        (ix, iy, ir) = inner_data
        if(ir>0):
            inner_data=(ix+self.x,iy+self.y,ir)
        im_arr = np.asarray(image)
        im_arr = np.pad(im_arr, ((self.y, self.y), (self.x, self.x), (0,0)), 'edge')
        image = PIL.Image.fromarray(im_arr)
        return image, lens_data, inner_data
        
def rotateP(image, lens_data = None, inner_data = None, angle=0, resizing=False):
    if lens_data is None:
        image, lens_data, inner_data = image
    (lx, ly, lr) = lens_data
    (ix, iy, ir) = inner_data
    (w,h) = image.size
    arad = math.radians(angle)
    center = np.array([w/2, h/2])
    rotmat = np.array([[math.cos(arad), math.sin(arad)],[-math.sin(arad), math.cos(arad)]])
    [lx, ly] = np.matmul(rotmat, (np.array([lx, ly])-center))+center
    if (ir != 0):
        [ix, iy] = np.matmul(rotmat, (np.array([ix, iy])-center))+center
    if resizing: # reason one we need to expand the image: the circle gets rotated out of frame
        toppad = max(0,math.ceil(-ly+lr))
        botpad = max(0,math.ceil(ly+lr-h))
        leftpad = max(0,math.ceil(-lx+lr))
        rightpad = max(0,math.ceil(lx+lr-w))
        lx += leftpad
        ly += toppad
        if(ir != 0):
            ix += leftpad
            iy += toppad
    else:
        toppad, botpad, leftpad, rightpad = 0,0,0,0
    # reason two we need to expand the image: so we don't get a black background after rotating
    newcent = np.matmul(np.abs(rotmat), center)
    [px, py] = np.ceil(newcent - center)
    pl, pt, pr, pb = int(max(leftpad,px)),int(max(toppad,py)),int(max(rightpad,px)),int(max(botpad,py))
    im_arr = np.asarray(image)
    im_arr = np.pad(im_arr, ((pt, pb), (pl, pr), (0,0)), 'edge')
    padim = PIL.Image.fromarray(im_arr)
    padim = F.rotate(padim, angle)
    image = F.crop(padim, pt-toppad, pl-leftpad, h+toppad+botpad, w+leftpad+rightpad)
    return image, (lx, ly, lr), (ix, iy, ir)

class FourCuts():
    def __init__(self, xrange=0, yrange=0):
        self.xrange=xrange
        self.yrange=yrange
    def __call__(self, image, lens_data=None, inner_data=None):
        if lens_data is None:
            image, lens_data, inner_data = image
        (w,h) = image.size
        x = int(w/2 - self.xrange + 2*self.xrange*np.random.random_sample())
        y = int(h/2 - self.yrange + 2*self.yrange*np.random.random_sample())
        corners = [getquarter(np.asarray(image), n, lens_data, inner_data, x, y) for n in range(4)]
        return [(PIL.Image.fromarray(im), l, i) for (im, l, i) in corners]

ForkHuts = FourCuts

class CornerResize():
    def __init__(self, width, height, scale, maxscale=None, checkonly=False):
        self.finwidth = width
        self.finheight = height
        self.minscale = scale
        if maxscale is None:
            maxscale = scale
        self.maxscale = maxscale
        self.checkonly = checkonly
    def __call__(self, image, lens_data = None, inner_data = None):
        # Code for ignores[1] may leak some edge cases (ha);
        # I'll fix this if it actually comes up
        if lens_data is None:
            image, lens_data, inner_data = image
        (w, h) = image.size
        (lx, ly, lr) = lens_data
        (ix, iy, ir) = inner_data
        ignores = torch.zeros(3)
        if(ly<h):
            maxx = int(lx-lr)
        elif(ly<h+lr):
            maxx = int(lx-math.sqrt(lr**2-(ly-h)**2))
        else:
            maxx = w
        if maxx>=w:
            ignores[1]=1
            maxy = h
        elif(lx<w):
            maxy = int(ly-lr)
        else:
            maxy = int(ly-math.sqrt(lr**2-(lx-w)**2))
        if(ir>0):
            if(iy<h):
                maxix = int(ix-ir)
            elif(iy<h+ir):
                maxix = int(ix-math.sqrt(ir**2-(iy-h)**2))
            else:
                maxix = w
            if maxix>=w:
                ignores[2]=1
        if('checkonly' in vars(self) and self.checkonly):
            return image, lens_data, inner_data, ignores
        minh = h-maxy
        minw = w-maxx
        maxscale = min(self.maxscale, self.finwidth/minw, self.finheight/minh)
        minscale = min(maxscale, self.minscale)
        s = np.random.random_sample() * (maxscale-minscale)+minscale
        cropwidth = int(math.ceil(self.finwidth/s))
        cropheight = int(math.ceil(self.finheight/s))
        ph = max(cropheight-h,0)
        pw = max(cropwidth-w,0)
        if(pw>0 or ph>0):
            im_arr = np.asarray(image)
            im_arr = np.pad(im_arr, ((ph, 0), (pw, 0), (0,0)), 'edge')
            image = PIL.Image.fromarray(im_arr)
            lx += pw
            ly += ph
            if(ir>0):
                ix += pw
                iy += ph
        (w, h) = image.size
        x = w-cropwidth
        y = h-cropheight
        lx, ly, lr = s*(lx-x),s*(ly-y),s*lr
        if(ir>0):
            ix, iy, ir = s*(ix-x),s*(iy-y),s*ir
        return F.resized_crop(image,y,x,cropheight,cropwidth,(self.finheight,self.finwidth)), (lx,ly,lr), (ix,iy,ir), ignores
    
def getquarter(im_arr, n, lens_data=None, inner_data=None, x=None, y=None):
    if lens_data is None:
        im_arr, lens_data, inner_data = im_arr
    (lx, ly, lr) = lens_data
    (ix, iy, ir) = inner_data
    (h,w) = im_arr.shape[:2]
    if x is None:
        x = w//2
    if y is None:
        y = h//2
    if(n%2==1):
        lx = w-lx
        if(ir>0):
            ix = w-ix
        im_arr = np.flip(im_arr,1)
    if(n%4>1):
        ly = h-ly
        if(ir>0):
            iy = h-iy
        im_arr = np.flip(im_arr,0)
    return np.copy(im_arr[0:y,0:x]), (lx,ly,lr), (ix,iy,ir)

class RandomCrop():
    '''
        If tame='yes' it will try to make sure the eye is completely in frame
        If tame='sides' will only do this for x coordinate
        Will always at least make sure that the frame contains the center of the eye
    '''
    def __init__(self, width, height, tame = 'yes'):
        self.cropwidth = width
        self.cropheight = height
        self.tame = tame
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        (height, width) = image.shape[:2]
        (lx, ly, lr) = lens_data
        (ix, iy, ir) = inner_data
        if (self.tame == 'yes' or self.tame == 'sides'):
            minx = max(0, lx+lr-self.cropwidth)
            maxx = min(width-self.cropwidth, lx-lr) + 1
        else:
            minx = max(0, lx-self.cropwidth)
            maxx = min(width-self.cropwidth, lx) + 1
        if(self.tame == 'yes'):
            miny = max(0, ly+lr-self.cropheight)
            maxy = min(height-self.cropheight, ly-lr) + 1
        else:
            miny = max(0, ly-self.cropheight)
            maxy = min(height-self.cropheight, ly) + 1
        x = np.random.randint(minx, maxx)
        y = np.random.randint(miny, maxy)
        return image[y:y+self.cropheight,x:x+self.cropwidth], (lx-x,ly-y,lr), (ix-x,iy-y,ir)

class RandRescale():
    def __init__(self, scale=0.1, maxscale = None):
        if maxscale is None:
            self.maxscale = 1+scale
            self.minscale = 1-scale
        else:
            self.maxscale = maxscale
            self.minscale = scale
    def __call__(self, image, lens_data = None, inner_data = None):
        scale = np.random.random_sample() * (self.maxscale-self.minscale)+self.minscale
        return Rescale(scale)(image, lens_data, inner_data)

class RandRotate():
    def __init__(self, angle = 15, maxangle = None, resizing = False):
        '''
            The resizing attribute for this and Rotate controls whether the image will resize to keep the whole
            lens in the picture; it's only neccessary to ensure tame cropping
        '''
        if maxangle is None:
            maxangle = abs(angle)
            angle = -maxangle
        self.min = angle
        self.max = maxangle
        self.resizing = resizing
    def __call__(self, image, lens_data = None, inner_data = None):
        angle = np.random.random_sample() * (self.max-self.min)+self.min
        return Rotate(angle, self.resizing)(image, lens_data, inner_data)

class RandHorizFlip():
    def __init__(self, p =0.5):
        self.p = p
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        if(np.random.random_sample()<self.p):
            (lx,ly,lr) = lens_data
            (ix,iy,ir) = inner_data
            w = image.shape[1]
            return np.fliplr(image).copy(), (w-lx, ly, lr), (w-ix, iy, ir)
            # Torch will refuse to use it if you don't copy
        else:
            return image, lens_data, inner_data

class Crop():
    def __init__(self, minx, miny, maxx = None, maxy = None, width = None, height = None):
        if(maxx is None and width is None):
            raise ValueError("Need a maxx or width argument")
        if(maxy is None and height is None):
            raise ValueError("Need a maxy or height argument")
        self.minx = minx
        self.miny = miny
        if maxx is not None:
            self.maxx = maxx
        else:
            self.maxx = minx + width
        if maxx is not None:
            self.maxy = maxy
        else:
            self.maxy = miny + height
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        (lx, ly, lr) = lens_data
        (ix, iy, ir) = inner_data
        return image[self.miny:self.maxy,self.minx:self.maxx], (lx-self.minx,ly-self.miny,lr), (ix-self.minx,iy-self.miny,ir)

class Rescale():
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        newim = skimage.transform.rescale(image, self.scale)
        newlens = tuple(a*self.scale for a in lens_data)
        newin = tuple(a*self.scale for a in inner_data)
        return newim, newlens, newin

class Rotate():
    def __init__(self, angle, resizing=False):
        '''
            The resizing attribute controls whether the image will resize to keep the whole
            lens in the picture; it's only neccessary to ensure tame cropping
        '''
        self.angle = angle
        self.resizing = resizing
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        (lx, ly, lr) = lens_data
        (ix, iy, ir) = inner_data
        (h,w) = image.shape[:2]
        center = np.array([w/2, h/2])
        angle = math.radians(self.angle)
        rotmat = np.array([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]])
        [lx, ly] = np.matmul(rotmat, (np.array([lx, ly])-center))+center
        if (ir != 0):
            [ix, iy] = np.matmul(rotmat, (np.array([ix, iy])-center))+center
        if self.resizing:
            toppad = max(0,math.ceil(-ly+lr))
            botpad = max(0,math.ceil(ly+lr-h))
            leftpad = max(0,math.ceil(-lx+lr))
            rightpad = max(0,math.ceil(lx+lr-w))
            if(toppad > 0 or botpad > 0 or leftpad > 0 or rightpad > 0):
                lx += leftpad
                ly += toppad
                if(ir != 0):
                    ix += leftpad
                    iy += toppad
                image = np.pad(image, ((toppad, botpad), (leftpad, rightpad), (0,0)), mode='edge')
        newim = skimage.transform.rotate(image, self.angle, mode='edge')
        return newim, (lx, ly, lr), (ix, iy, ir)
   
class Pad():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __call__(self, image, lens_data = None, inner_data = None):
        if lens_data is None:
            image, lens_data, inner_data = image
        (lx, ly, lr) = lens_data
        (ix, iy, ir) = inner_data
        return np.pad(image, ((self.y, self.y), (self.x, self.x), (0,0)), 'edge'), (lx+self.x,ly+self.y,lr), (ix+self.x,iy+self.y,ir)
    
class HistEq():
    def __call__(self, im_arr, lens_data = None, inner_data = None):
        if lens_data is None:
            im_arr, lens_data, inner_data = im_arr
        for n in range(3):
            im_arr[:,:,n] = skimage.exposure.equalize_hist(im_arr[:,:,n])*256
        return im_arr, lens_data, inner_data
   
def histeqtens(x):
    x_arr = x[0].numpy().copy()
    for n in range(3):
        x_arr[n] = skimage.exposure.equalize_hist(x_arr[n])
    x = torch.tensor(x_arr, dtype=torch.float)
    x = torch.unsqueeze(x,0)
    return x
   
def splitset(img_dir, circle_file, train_prop = 0.8, shuffle = False, seed = None, validate_on = None,
             tame = 'yes', fill = True, coords = False, dual = False, corners = None):
    annotations = getannotations(circle_file)
    if(shuffle):
        np.random.seed(seed)
        np.random.shuffle(annotations)
    if(validate_on is not None):
        which_eye = ''
        if('OD' in validate_on):
            which_eye = 'OD'
        elif('OS' in validate_on):
            which_eye = 'OS'
        who = ''
        patnums = ['05','07','10','15','17','24','29','39','40','42','46','49','51'] # Magic list of patient numbers
        for n in patnums: 
            if n in validate_on:
                who = "iP0"+n
        valset = [(k,v) for (k,v) in annotations if which_eye in k and who in k]
        trainset = [(k,v) for (k,v) in annotations if which_eye not in k or who not in k]
        if(len(trainset) == 0 or len(valset) == 0):
            warnings.warn("validate_on didn't work; reverting to standard behavior")
            splitset(img_dir, circle_file, train_prop, shuffle, seed, tame = tame, fill = fill, coords = coords)
    else:
        train_len = int(train_prop*len(annotations))
        trainset = annotations[:train_len]
        valset = annotations[train_len:]
    if corners=='shuffle':
        train_trans = transforms.Compose([ToPil(), CornerResize(256, 216, 0.3, 0.5), ToTens()])
        val_trans = transforms.Compose([ToPil(), CornerResize(256, 216, 0.4), ToTens()])
    elif corners=='together':
        scalings = [transforms.Compose([PadP(320, 270), RandomResizedCropP(512, 432, 0.267, 0.4)]),
                    RandomResizedCropP(512, 432, 0.4)]
        train_trans = transforms.Compose([ToPil(), RandRotateP(45, resizing=True), transforms.RandomChoice(scalings), FourCuts(), Mapper(CornerResize(256,216,1,checkonly=True)), Mapper(ToTens()), Unzipper()])
        val_trans = transforms.Compose([ToPil(), CropP(320,0,1600,1080), ResizeP(432), FourCuts(), Mapper(CornerResize(256,216,1,checkonly=True)), Mapper(ToTens()), Unzipper()])
    elif tame=='yes':
        #train_trans = transforms.Compose([transforms.RandomApply([Pad(320, 270), RandRescale(0.67,1.)]), RandRotate(15, resizing=True), RandomCrop(1280, 1080), Rescale(0.2), RandHorizFlip()])
        #val_trans = transforms.Compose([Crop(320,0,1600,1080), Rescale(0.2)])
        scalings = [transforms.Compose([PadP(320, 270), RandomResizedCropP(256, 216, 0.134, 0.2)]),
                    RandomResizedCropP(256, 216, 0.2)]
        train_trans = transforms.Compose([HistEq(), ToPil(), RandRotateP(15, resizing=True),
                                          transforms.RandomChoice(scalings), ToTens()])
                                          #transforms.RandomChoice(scalings), transforms.RandomApply([FlipP()]), ToTens()])
        val_trans = transforms.Compose([HistEq(), ToPil(), CropP(320,0,1600,1080), ResizeP(216), ToTens()])
    else:
        train_trans = transforms.Compose([RandRescale(.25,.3), RandRotate(15), RandomCrop(256, 256, tame)])
        val_trans = transforms.Compose([Crop(448,28,1472,1052), Rescale(0.25)])
    
    if coords:
        train_dat = Coord_Dataset(img_dir, annotations = trainset, transform = train_trans, corners=corners)
        val_dat = Coord_Dataset(img_dir, annotations = valset, transform = val_trans, corners=corners)
    elif dual:
        train_dat = Dual_Dataset(img_dir, annotations = trainset, transform = train_trans)
        val_dat = Dual_Dataset(img_dir, annotations = valset, transform = val_trans)
    else:
        train_dat = Circ_Dataset(img_dir, annotations = trainset, transform = train_trans, fill = fill, corners=corners)
        val_dat = Circ_Dataset(img_dir, annotations = valset, transform = val_trans, fill = fill, corners=corners)

    return train_dat, val_dat
        
class Inner_Dataset(Dataset): # Only has inner data
    def __init__(self, img_dir, annotations, fill = True):
        self.annotations = list(annotations.items())
        self.img_dir = img_dir
        self.fill = fill
        self.transform = transforms.Compose([Crop(320,0,1600,1080), Rescale(0.2)])
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        k,v = self.annotations[idx]
        img_name = os.path.join(self.img_dir, k)
        im_array = io.imread(img_name)
        lens_data = v
        inner_data = v
        if self.transform is not None:
            im_array, lens_data, inner_data = self.transform((im_array, lens_data, inner_data))
        image = torch.tensor(im_array.transpose(2,0,1), dtype=torch.float)
        try:
            if self.fill:
                outarr, out_cls = circfill(lens_data, inner_data, im_array.shape[1], im_array.shape[0], style = [0,0,1])
            else:
                outarr, out_cls = circdraw(lens_data, inner_data, im_array.shape[1], im_array.shape[0])
        except AttributeError:
            self.fill = True
            outarr, out_cls = circfill(lens_data, inner_data, im_array.shape[1], im_array.shape[0], style = [0,0,1])
        out = torch.tensor(outarr, dtype=torch.long)
        return k[:-4], image, out, torch.cat((torch.ones(1), out_cls),0)
        
class Im_Dataset(Dataset): # Images only; no checking
    def __init__(self, img_dir, check_on, transform = None):
        self.img_dir = img_dir
        self.filelist = [x for x in os.listdir(img_dir) if ('iP0'+check_on) in x]
        self.filelist.sort(key=lambda x:int(x[14:-4]))
        if transform is None:
            transform = transforms.Compose([Crop(320,0,1600,1080), Rescale(0.2)])
        self.transform = transform
        
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        
        k = self.filelist[idx]
        img_name = os.path.join(self.img_dir, k)
        im_array = io.imread(img_name)
        if self.transform is not None:
            im_array, _, _ = self.transform((im_array, [960,540,0], [0,0,0]))
        image = torch.tensor(im_array.transpose(2,0,1), dtype=torch.float)

        #print(image.shape, out.shape)
        return k, image
        
class Circ_Dataset(Dataset):
    def __init__(self, img_dir, circle_file = None, annotations = None, transform = None, fill = True, corners = None):
        if(circle_file != None):
            self.annotations = getannotations(circle_file)
        elif(annotations != None):
            self.annotations = annotations
        else:
            raise ValueError('Circ_Dataset needs a circle_file or annotations argument')
        self.img_dir = img_dir
        self.transform = transform
        self.fill = fill
        self.corners = corners
        
    def __len__(self):
        if not self.corners or self.corners=='together':
            return len(self.annotations)
        else:
            return 4*len(self.annotations)

    def __getitem__(self, idx):
        if self.corners == 'shuffle':
            quadrant = idx % 4
            idx = idx // 4
        k,v = self.annotations[idx]
        img_name = os.path.join(self.img_dir, k + ".png")
        im_arr = io.imread(img_name)
        lens_data = v['lens_data']
        inner_data = v['inner_data']
        if self.corners == 'shuffle':
            im_arr, lens_data, inner_data = getquarter(im_arr, quadrant, lens_data, inner_data)
        if self.transform is not None:
            image, lens_data, inner_data, *a = self.transform((im_arr, lens_data, inner_data))
        else:
            image = torch.tensor(im_arr.transpose(2,0,1), dtype=torch.float)
            a = []
        out, out_cls = self._gettruth(image, lens_data, inner_data)
        if self.corners == 'shuffle':
            a.append(quadrant)
        return (k, image, out, out_cls, *a)
        
    def _gettruth(self, image, lens_data, inner_data):
        if isinstance(image, list):
            return zip(*[_gettruth(im, l, i) for (im, l, i) in zip(image, lens_data, inner_data)])
        (h,w) = image.size()[1:3]
        if self.fill:
            drawer = circfill
        else:
            drawer = circdraw
        outarr, out_cls = drawer(lens_data, inner_data, w, h)
        out = torch.tensor(outarr, dtype=torch.long)
        return out, out_cls
        
    def updateparams(self):
        if not 'fill' in vars(self):
            self.fill = True
        if not 'corners' in vars(self):
            self.corners = None
        if self.corners == True:
            self.corners = 'shuffle'
        
class Coord_Dataset(Dataset):
    def __init__(self, img_dir, circle_file = None, annotations = None, transform = None, corners = None):
        if(circle_file != None):
            self.annotations = getannotations(circle_file)
        elif(annotations != None):
            self.annotations = annotations
        else:
            raise ValueError('Coord_Dataset needs a circle_file or annotations argument')
        self.img_dir = img_dir
        self.transform = transform
        self.corners = corners
        self.starttime = time.time()
        
    def __len__(self):
        if self.corners=='shuffle':
            return 4*len(self.annotations)
        else:
            return len(self.annotations)

    def __getitem__(self, idx):
        if self.corners == 'shuffle':
            quadrant = idx % 4
            idx = idx // 4
        k,v = self.annotations[idx]
        img_name = os.path.join(self.img_dir, k + ".png")
        im_arr = io.imread(img_name)
        lens_data = v['lens_data']
        inner_data = v['inner_data']
        if self.corners == 'shuffle':
            im_arr, lens_data, inner_data = getquarter(im_arr, quadrant, lens_data, inner_data)
        if self.transform is not None:
            image, lens_data, inner_data, *a = self.transform((im_arr, lens_data, inner_data))
        else:
            image = im_arr
            a = []
        out, out_cls = self._gettruth(lens_data, inner_data)
        if self.corners == 'shuffle':
            a.append(quadrant)
        if self.corners == 'together':
            image, out, out_cls = torch.stack(image, 0), torch.stack(out, 0), torch.stack(out_cls, 0)
            a = [torch.stack(a[0],0)]
        if isinstance(image, np.ndarray):
            image = torch.tensor(image.transpose(2,0,1), dtype=torch.float)
        return (k, image, out, out_cls, *a)
        
    def _gettruth(self, lens_data, inner_data):
        if isinstance(lens_data[0], tuple):
            return zip(*[self._gettruth(l, i) for (l, i) in zip(lens_data, inner_data)])
        out = torch.tensor(lens_data + inner_data)
        out_cls = torch.zeros(3, dtype=torch.float)
        out_cls[0]=1
        out_cls[1]=1
        if (inner_data[2]>0):
            out_cls[2]=1
        return out, out_cls
        
    def updateparams(self):
        if not 'corners' in vars(self):
            self.corners = None

class Dual_Dataset(Dataset):
    def __init__(self, img_dir, circle_file = None, annotations = None, transform = None):
        if(circle_file != None):
            self.annotations = getannotations(circle_file)
        elif(annotations != None):
            self.annotations = annotations
        else:
            raise ValueError('Dual_Dataset needs a circle_file or annotations argument')
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        k,v = self.annotations[idx]
        img_name = os.path.join(self.img_dir, k + ".png")
        im_array = io.imread(img_name)
        lens_data = v['lens_data']
        inner_data = v['inner_data']
        if self.transform is not None:
            image, lens_data, inner_data = self.transform((im_array, lens_data, inner_data))
        else:
            image = torch.tensor(im_array.transpose(2,0,1), dtype=torch.float)
        (h,w) = image.size()[1:3]
        outarr, out_cls = circfill(lens_data, inner_data, w, h)
        outim = torch.tensor(outarr, dtype=torch.long)
        outdat = torch.tensor(lens_data + inner_data)
        return k, image, outim, outdat, out_cls

  
'''  
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30, (1920, 1080))
for x in range(160, 1761, 20):
    set1 = [[28,26,228], [184, 126, 55], [74,175,77]]
    ar = np.array(circfill([960, 540, 500], [x, 540, 300], style = set1, classes = False))
    out.write(np.array(ar).astype(np.dtype('uint8')))
    
out.release()
cv2.destroyAllWindows()
'''
    
      
'''
f = open('trainstuff.txt')
results = {}
for x in ['squeezenet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
    curves = {'train':[], 'validate':[]}
    for n in range(20):
        curves['train'].append(float(f.readline()[:7]))
        curves['validate'].append(float(f.readline()[:7]))
    results[x] = curves
f.close()

f = open('results')
res2 = json.load(f)
f.close()

for x in ['squeezenet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
    for y in ['train', 'validate']:
        results[x][y].extend(res2[x][y])


for x in ['squeezenet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
    fig, ax = plt.subplots()
    
    ax.plot(results[x]['train'], label='Training')
    ax.plot(results[x]['validate'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (categorical cross-entropy)')
    ax.legend()
    fig.savefig(x+'learn.png')
    ax.set_yscale('log')
    fig.savefig(x+'learnlog.png')
    plt.close(fig)
    
print(results['resnet50']['validate'])
    
fig, ax = plt.subplots()

for x in ['squeezenet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
    ax.plot(results[x]['validate'], label=x)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (categorical cross-entropy)')
ax.legend()
fig.savefig('compare.png')
ax.set_yscale('log')
fig.savefig('comparelog.png')
plt.close(fig)
'''

'''
f = open('numtrain.txt')
curves = {'train':[], 'validate':[]}
for n in range(40):
    curves['train'].append(float(f.readline()[:7]))
    curves['validate'].append(float(f.readline()[:7]))
f.close()

fig, ax = plt.subplots()
    
ax.plot(curves['train'], label='Training')
ax.plot(curves['validate'], label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (L1)')
ax.legend()
fig.savefig('learnnums.png')
ax.set_yscale('log')
fig.savefig('learnnumslog.png')
plt.close(fig)
'''