import os
import math
import numpy as np
import json
import pickle
import warnings
import logging
import random
import re
from functools import partial
from collections import namedtuple
import dill
import cv2
import time
import multiprocessing as mp
from scipy import stats
import itertools

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from skimage import io
import PIL
from PIL import Image

import torch
from torch import save, load
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader

import getdata
import goldsketch
import sys
sys.path.insert(0, "../pspnet-pytorch")
import pspnet
from pspnet import PSPNet, PSPCircs

#plt.switch_backend('agg')

CIRCLE_DATA_PATH = 'goldmann_measurements.json'
TWO_CIRCLE_DATA_PATH = 'goldmann_measurements_2.json'
RYAN_OMAR_PATH = 'ryan_omar_segs.json'
IMAGE_BASE_PATH = '../../yue/joanne/GAT SL videos'
HOLDOUT_SET = ['40-J34-OS','10-J49-OD','21-J48-OS','11-F35-OS','55-F18-OD','19-F55-OD']
GOLDMANN_OUTER_DIAM = 7
with open('clinical_iops.json') as f:
    CLINICAL_IOP_DICT = json.load(f)

models = {
    'squeezenet': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet', w=w, h=h),
    'densenet': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet', w=w, h=h),
    'resnet18': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', w=w, h=h),
    'resnet34': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34', w=w, h=h),
    'resnet50': lambda h, w, **kwargs: PSPCircs(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50', w=w, h=h, extraend=True,out_nums=9,**kwargs),
    'resnet101': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101', w=w, h=h),
    'resnet152': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152', w=w, h=h)
}

def arrToTens(arr):
    return torch.tensor(arr.transpose(2,0,1).copy(), dtype=torch.float)
    
def arrToTens3d(arr, chans):
    return torch.tensor(arr.transpose(2,0,1).copy(), dtype=torch.float).view(chans,-1,arr.shape[0],arr.shape[1])
    
def tensToArr(tens):
    return tens.detach().cpu().numpy()
    
def arrsToTens(info, in_im, out_im):
    return arrToTens(in_im), arrToTens(out_im)
    
def arrsToTens3d(info, in_im, out_im, chans):
    return arrToTens3d(in_im, chans[0]), arrToTens3d(out_im, chans[1])
    
def im_filereader(path='', nameedit=(lambda x: x)):
    def lookup(key):
        im_name = os.path.join(path, nameedit(key))
        logging.info(f'Opening {im_name}')
        return np.atleast_3d(io.imread(im_name))
    return lookup
    
def stdfilename(key):
    return '.'.join([str(key[0]),'png'])
    
def maybefake(p=0.5, fake_key = ('fake','fake','fake')):
    def keyadj(key):
        if np.random.sample()<p:
            return fake_key
        return key
    return keyadj
    
class MaybeFaker():
    def __init__(self, p=0.5, fake_key = ('fake','fake','fake')):
        self.p = p
        self.fake_key = fake_key
    def __call__(self, key):
        if np.random.sample()<self.p:
            return self.fake_key
        return key
    
def constim(val):
    def getfile(key):
        return val
    return getfile
    
def get_golddat(coord_dict):
    def lookup(key):
        group, vid, file = key
        coords = coord_dict[group][vid][file]
        if 'j' in group:
            groupfold = 'goldmann_new'
        else:
            groupfold = 'goldmann_shu'
        imloc = os.path.join(IMAGE_BASE_PATH,groupfold,'raw',vid[:-4],file)
        im = io.imread(imloc)
        return im,coords
    return lookup
    
def ro_lookup():
    def lookup(key):
        name = key[0]
        im = io.imread(key[1])
        return name, im, key[2:]
    return lookup
    
def im_only(num_entries=7):
    def lookup(key):
        return io.imread(key), [1]*num_entries
    return lookup

def fakesonly(both_mires=True, color_mode='rand', extrarate=0.5, with_name=False):
    def lookup(key):
        if with_name:
            return ('fake', *goldsketch.goldmann_fake(bothmires=both_mires,color_mode=color_mode,extra=(np.random.rand()<extrarate)))
        return goldsketch.goldmann_fake(bothmires=both_mires,color_mode=color_mode,extra=(np.random.rand()<extrarate))
    return lookup
    
def lookup_switch(test, lookup_a, lookup_b):
    def lookup(key):
        if test(key):
            return lookup_a(key)
        return lookup_b(key)
    return lookup
    
def consty(width, height, max=None, depth=1):
    yy,xx,_ = np.mgrid[0:height,0:width,0:depth]
    if max:
        yy = yy*max/(height-1)
    return constim(yy)
    
def constx(width, height, div=1):
    yy,xx,_ = np.mgrid[0:height,0:width,0:1]/div
    return constim(xx)
    
def pframe(width, height, val=1):
    frame = np.full((height,width,1),val)
    frame[1:-1,1:-1] = np.full((height-2,width-2,1),0)
    return constim(frame)

def edges(width, height, val=1):
    frame = np.full((height,width,4),0)
    frame[:,0,0] = np.full(height,val)
    frame[:,-1,1] = np.full(height,val)
    frame[0,:,2] = np.full(width,val)
    frame[-1,:,3] = np.full(width,val)
    return constim(frame)
    
def cat_ims(readers):
    def finread(key):
        return np.concatenate([r(key) for r in readers], axis=2)
    return finread
    
def saveorig(info, dat, name='orig'):
    im, *a = dat
    info[name] = im
    return dat
    
def fixedcrop(info, in_arr, out_arr, l=None, r=None, t=None, b=None):
    logging.info(f'Cropping with dimensions [{t}:{b},{l}:{r}]')
    return in_arr[t:b,l:r], out_arr[t:b,l:r]
    
def addavgs2d(info, in_arr, out_arr):
    avgs = in_arr.mean((0,1))
    broadcast_avgs = np.zeros_like(in_arr)+avgs
    return np.concatenate((in_arr, broadcast_avgs), 2), out_arr
    
def addy(info, in_arr, out_arr, div=1):
    yy,_,_ = np.mgrid[0:in_arr.shape[0],0:in_arr.shape[1],0:1]/div
    return np.concatenate((in_arr,yy),2), out_arr
    
def constinfo(transform):
    def fintrans(info, dat):
        return transform(dat)
    return fintrans
    
def randResizedCrop(width, height, scale, maxscale=None, tame = 'yes', circle_fullness=1):
    return constinfo(getdata.RandomResizedCropP(width, height, scale, maxscale, tame, circle_fullness))
    
def constCrop(minx, miny, maxx = None, maxy = None, width = None, height = None):
    return constinfo(getdata.CropP(minx, miny, maxx, maxy, width, height))
    
def brightness_focus(width, height=None, mem=0):
    if height == None:
        height = width
    if mem:
        prevs = []
    def t(info, dat):
        image, lens_data, inner_data, *i2 = dat
        (lx, ly, lr) = lens_data
        (ix, iy, ir) = inner_data
        if type(image) is Image.Image:
            image = np.asarray(image)
            rePil = True
        else:
            rePil = False
        mass = np.sum(image)
        yy,xx,_ = np.mgrid[0:image.shape[0],0:image.shape[1],0:1]
        ymom = np.sum(image*yy)
        xmom = np.sum(image*xx)
        yc = ymom/mass
        xc = xmom/mass
        top = yc-height/2
        left = xc-height/2
        top = min(max(top,0),image.shape[0]-height)
        left = min(max(left,0),image.shape[1]-width)
        if mem:
            nonlocal prevs
            prevs.append((top,left))
            top,left = np.mean(prevs,0)
            prevs = prevs[-mem:]
        top = int(top)
        left = int(left)
        extrarets = []
        if i2:
            (ix2, iy2, ir2) = i2[0]
            extrarets.append((ix2-left,iy2-top,ir2))
        image = image[top:top+height,left:left+width]
        if rePil:
            image = Image.fromarray(image)
        return (image, (lx-left,ly-top,lr), (ix-left,iy-top,ir), *extrarets)
    return t
    
def location_cropper(default_width=None, default_height=None, mem=0):
    if default_height is None:
        default_height = default_width
    if mem:
        prevs = []
    def t(info, dat):
        image, lens_data, inner_data, *i2 = dat
        (lx, ly, lr) = lens_data
        (ix, iy, ir) = inner_data
        if type(image) is Image.Image:
            image = np.asarray(image)
            rePil = True
        else:
            rePil = False
        if 'box' in info:
            top, left, bot, right = info['box']
        else:
            if default_width is None:
                raise ValueError("location_cropper is out of locations and has no default_width value given")
            xsum = np.sum(im,(1,2))
            xcsum = np.cumsum(xsum)
            xwsum = xcsum[default_width:]-xcsum[:-default_width]
            left = np.argmax(xwsum)
            right = left+default_width
            ysum = np.sum(im,(0,2))
            ycsum = np.cumsum(ysum)
            ywsum = ycsum[default_height:]-ycsum[:-default_height]
            top = np.argmax(ywsum)
            bot = top+default_height
        if mem:
            nonlocal prevs
            prevs.append(((top+bot-default_height)/2,(left+right-default_width)/2))
            if 'box' not in info:
                top,left = np.mean(prevs,0)
                top = min(max(0,top),image.shape[0]-default_height)
                left = min(max(0,left),image.shape[1]-default_width)
            prevs = prevs[-mem:]
        top = int(top)
        left = int(left)
        extrarets = []
        if i2:
            (ix2, iy2, ir2) = i2[0]
            extrarets.append((ix2-left,iy2-top,ir2))
        image = image[top:bot,left:right]
        if rePil:
            image = Image.fromarray(image)
        return (image, (lx-left,ly-top,lr), (ix-left,iy-top,ir), *extrarets)
    return t
    
class Tracker():
    def __init__(self, width, height = None, startpos=None, maxchange=None, center_rad=0, image_shape = None, calibration = 'window'):
        self.width = width
        if height is None:
            height = width
        self.height = height
        if startpos is None:
            self.prevpos = None
        else:
            self.prevpos = np.asarray(startpos)
        self.maxchange = maxchange
        self.image_shape = image_shape
        if calibration == 'window':
            self.focus = self._window_focus
        elif calibration == 'com':
            self.focus = self._com_focus
        else:
            raise ValueError("'calibration' argument for Tracker class should be 'window' or 'com'")
        self.center_rad = center_rad
        self.latest_box = None
            
    def __call__(self, info, dat):
        image, lens_data, inner_data, *i2 = dat
        (lx, ly, lr) = lens_data
        (ix, iy, ir) = inner_data
        if type(image) is Image.Image:
            image = np.asarray(image)
            rePil = True
        else:
            rePil = False
        self.image_shape = image.shape[:2]
        if self.prevpos is None:
            yc, xc = self.focus(image)
        else:
            yc, xc = self.prevpos
        top = yc-self.height//2
        left = xc-self.width//2
        top = min(max(top,0),image.shape[0]-self.height)
        left = min(max(left,0),image.shape[1]-self.width)
        self.latest_box = (int(top), int(left), int(top+self.height), int(left+self.width))
        extrarets = []
        if i2:
            (ix2, iy2, ir2) = i2[0]
            extrarets.append((ix2-left,iy2-top,ir2))
        image = image[top:top+self.height,left:left+self.width]
        if rePil:
            image = Image.fromarray(image)
        return (image, (lx-left,ly-top,lr), (ix-left,iy-top,ir), *extrarets)
        
    def _com_focus(self, image, reposition=True):
        mass = np.sum(image)
        yy,xx,_ = np.mgrid[0:image.shape[0],0:image.shape[1],0:1]
        ymom = np.sum(image*yy)
        xmom = np.sum(image*xx)
        yc = ymom/mass
        xc = xmom/mass
        yc = int(round(yc))
        xc = int(round(xc))
        if reposition:
            self.prevpos = np.array([yc,xc])
        return yc, xc
    
    def _window_focus(self, image, reposition=True):
        xsum = np.sum(image,(1,2))
        xcsum = np.cumsum(xsum)
        xwsum = xcsum[self.height:]-xcsum[:-self.height]
        ysum = np.sum(image,(0,2))
        ycsum = np.cumsum(ysum)
        ywsum = ycsum[self.width:]-ycsum[:-self.width]
        yc = np.argmax(xwsum)+self.height/2
        xc = np.argmax(ywsum)+self.width/2
        yc = int(round(yc))
        xc = int(round(xc))
        if reposition:
            self.prevpos = np.array([yc,xc])
        return yc, xc
    
    def update(self, pos):
        pos = np.asarray(pos,dtype=np.int64)
        if self.prevpos is None:
            self.prevpos = pos
        else:
            ymov = round(pos[0]-self.height/2-self.center_rad)
            xmov = round(pos[1]-self.width/2-self.center_rad)
            if ymov>self.center_rad:
                ymov -= self.center_rad
            elif ymov<-self.center_rad:
                ymov += self.center_rad
            else:
                ymov = 0
            if xmov>self.center_rad:
                xmov -= self.center_rad
            elif xmov<-self.center_rad:
                xmov += self.center_rad
            else:
                xmov = 0
            if self.maxchange:
                xmov = max(-self.maxchange,min(xmov, self.maxchange))
                ymov = max(-self.maxchange,min(ymov, self.maxchange))
            newy = max(self.prevpos[0]+int(ymov),self.height//2)
            newx = max(self.prevpos[1]+int(xmov),self.width//2)
            if self.image_shape is not None:
                newy = min(newy,self.image_shape[0]-self.height//2)
                newx = min(newx,self.image_shape[1]-self.width//2)
            self.prevpos = np.array([newy,newx])
    
def randRotate(angle = 15, maxangle = None, resizing = False):
    return constinfo(getdata.RandRotateP(angle, maxangle, resizing))
    
def resize(dim):
    return constinfo(getdata.ResizeP(dim))
    
def flip():
    return constinfo(getdata.FlipP())
    
def pad(x,y):
    return constinfo(getdata.PadP(x,y))
    
def normalize(targetmin=0, targetmax=1):
    def trans(info, dat):
        logging.info('Normalizing image')
        im, *a = dat
        cmins,_ = torch.min(torch.min(im,2,keepdim=True)[0],1,keepdim=True)
        cmaxs,_ = torch.max(torch.max(im,2,keepdim=True)[0],1,keepdim=True)
        im = (targetmax-targetmin)*(im-cmins)/(cmaxs-cmins)+targetmin
        return (im, *a)
    return trans
    
def package_goldcoords(transform_line=False,expected_coord_len=10,extraname=None):
    def t(info, dat):
        im, coords = dat
        if transform_line:
            if len(coords)>expected_coord_len:
                line_coords = coords[-3:]
                coords = coords[:-3]
            else:
                line_coords = [1.,1.,1.]
            extraret = [line_coords]
        else:
            extraret = []
        if len(coords)==7:
            if extraname is None:
                info['isbot']=int(coords[6])
            else:
                info[extraname]=int(coords[6])
            return (im,coords[:3],coords[3:6],*extraret)
        else:
            if extraname is None:
                info['origseg']=int(coords[9])
            else:
                info[extraname]=int(coords[9])
            return (im,coords[:3],coords[3:6],coords[6:9],*extraret)
    return t
    
def package_ro(extraname=None):
    def t(info, dat):
        name, im, coords = dat
        info['filename'] = name
        if extraname is None:
            info['is_ryan']=int(coords[9])
        else:
            info[extraname]=int(coords[9])
        return (im,coords[:3],coords[3:6],coords[6:9],)
    return t
    
def add_classes():
    def t(info, dat):
        im, lens_data, inner_data, *i2 = dat
        if 'origseg' in info:
            innerrad = i2[0][2] if info['origseg'] else inner_data[2]
        else:
            innerrad = inner_data[2]
        classes = np.array((1,1,int(innerrad!=0)),dtype=np.float32)
        return im, np.concatenate([lens_data, inner_data,*i2]).astype(np.float32), classes
        #return im, np.concatenate([lens_data, inner_data, line_dat]).astype(np.float32), classes # Trying midline segmentation
    return t
    
def composetransforms(transforms):
    def fintrans(info, dat):
        for t in transforms:
            dat = t(info, dat)
        return dat
    return fintrans
    
def unfold(tens, inaxis=0, stackaxis=1):
    return torch.cat([torch.squeeze(layer,inaxis) for layer in torch.split(tens, 1, inaxis)], stackaxis)
    
class ImtoNumsDataset(Dataset):
    def __init__(self, keys, datlookup, transformer = None, dispstring=None, infobase={}, keyadj=None):
        super().__init__()
        self.keys = keys
        self.datlookup = datlookup
        if transformer is None:
            transformer = composetransforms([getdata.ToPil(),getdata.ToTens()])
        self.transformer = transformer
        self.dispstring = dispstring
        self.infobase = infobase
        self.keyadj = keyadj
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        logging.info(f'Looking up key {key}')
        if self.keyadj is not None:
            key = self.keyadj(key)
            logging.info(f'Adjusting key to {key}')
        dat = self.datlookup(key)
        info = self.infobase.copy()
        info['key']=key
        dat = self.transformer(info, dat)
        im, coords, classes = dat
        return info, im, coords, classes
    
def maken(cl, num_to_make=2, multinumbering=True, **kwargs):
    if multinumbering and num_to_make>10:
        raise ValueError('multinumbering not possible if num_to_make>10')
    if multinumbering:
        regexes = ['(.*)_[0-9]*'+str(n)+'[0-9]*' for n in range(num_to_make)]
    else:
        regexes = ['(.*)_'+str(n) for n in range(num_to_make)]
    basekwargs = {arg:val for arg,val in kwargs.items() if all([not re.fullmatch(reg,arg) for reg in regexes])}
    kwargses = []
    for n in range(num_to_make):
        kwargses.append({m.group(1):val for m,val in 
                             ((re.fullmatch(regexes[n],arg),val) for arg,val in kwargs.items())
                         if m})
    return [cl(**{**basekwargs, **k}) for k in kwargses]
    
def loadsets(models_path):
    if 'tests' not in models_path:
        models_path = os.path.join('tests',models_path)
    fname = next(name for name in os.listdir(models_path) if '_dat' in name)
    with open(os.path.join(models_path, fname), 'rb') as f:
        sets = dill.load(f)
    train_dat = sets['train_dat']
    val_dat = sets['val_dat']
    if 'out_dat' in sets:
        out_dat = sets['out_dat']
    else:
        out_dat = val_dat
    for dat in [train_dat, val_dat, out_dat]:
        if not 'infobase' in vars(dat):
            dat.infobase = {}
    return train_dat, val_dat, out_dat
    
def infiniloader(dataset, **kwargs):
    loader = DataLoader(dataset, **kwargs)
    stop = None
    while stop is None:
        for data in loader:
            stop = yield data
            if stop is not None:
                break
    del loader

def dividefolds(pic_shuffle=True, seed=None, valprop=0.2, num_out=3, dataloc=CIRCLE_DATA_PATH, imitate_single=True, readd_extra=False, ro_mode=False):
    if ro_mode:
        return divide_ro_vids(pic_shuffle, seed, valprop, num_out, dataloc)
    if seed:
        pic_shuffle=True
    if dataloc==CIRCLE_DATA_PATH:
        imitate_single = False
    if imitate_single:
        trainkeys,valkeys,outkeys = dividefolds(pic_shuffle, seed, valprop, num_out)
        trainfolds = set(b for (a,b,c) in trainkeys)
        valfolds = set(b for (a,b,c) in valkeys)
        outfolds = set(b for (a,b,c) in outkeys)
        added_video=[]
    vids = []
    with open(dataloc) as f:
        circle_data = json.load(f)
    for foldname in circle_data:
        for subfold in circle_data[foldname]:
            if subfold[:-4] in HOLDOUT_SET:
                continue
            if imitate_single and subfold == '27-J33-OD.csv': # Magic string, sorta
                added_video = [(foldname,subfold,x) for x in circle_data[foldname][subfold]]
                continue
            vids.append([(foldname,subfold,x) for x in circle_data[foldname][subfold]])
    
    if imitate_single:
        trainfolds = [vid for vid in vids if vid[0][1] in trainfolds]
        valfolds = [vid for vid in vids if vid[0][1] in valfolds]
        outfolds = [vid for vid in vids if vid[0][1] in outfolds]
        if readd_extra:
            trainfolds.append(added_video)
    else:
        r = np.random.RandomState(seed)
        validxs = r.choice(len(vids), size=int(len(vids)*valprop), replace=False)
        isval = np.zeros(len(vids), dtype=np.bool)
        for n in validxs:
            isval[n]=1
            
        valfolds = [v for n,v in enumerate(vids) if isval[n]]
        trainfolds = [v for n,v in enumerate(vids) if not isval[n]]
        r.shuffle(valfolds)
        if type(num_out) is slice:
            outfolds = valfolds[num_out]
        else:
            outfolds = valfolds[:num_out]

    trainkeys = [k for f in trainfolds for k in f]
    valkeys = [k for f in valfolds for k in f]
    outkeys = [k for f in outfolds for k in f]
    
    return trainkeys,valkeys,outkeys
   
def divide_ro_vids(pic_shuffle=True, seed=None, valprop=0.2, num_out=3, dataloc=RYAN_OMAR_PATH, patsplit=True):
    if seed:
        pic_shuffle=True
    with open(dataloc) as f:
        circle_data = json.load(f)
    vids = sorted(set(s[0].split('_')[0] for s in circle_data))
    r = np.random.RandomState(seed)
    if patsplit:
        patlist = []
        pateyes = {}
        for k,g in itertools.groupby(vids, key=(lambda x: x.split('-')[0])):
            patlist.append(k)
            pateyes[k] = list(g)
        r.shuffle(patlist)
        valvids = []
        trainvids = []
        for pat in patlist:
            if len(valvids)<len(vids)*valprop:
                valvids += pateyes[pat]
            else:
                trainvids += pateyes[pat]
        
    else:
        r.shuffle(vids)
        valvids = vids[:int(len(vids)*valprop)]
        trainvids = vids[int(len(vids)*valprop):]
        print(len(trainvids))
        print(len(valvids))
    if type(num_out) is slice:
        outvids = valvids[num_out]
    else:
        outvids = valvids[:num_out]
    print(outvids)
    trainkeys = [s for s in circle_data if s[0].split('_')[0] in trainvids]
    valkeys = [s for s in circle_data if s[0].split('_')[0] in valvids]
    outkeys = [s for s in circle_data if s[0].split('_')[0] in outvids]
    return trainkeys, valkeys, outkeys
   
'''
def dividefolds(pic_shuffle=True, seed=None, valprop=0.2, num_out=3, dataloc=CIRCLE_DATA_PATH, imitate_single=True, readd_extra=False):
    if seed:
        pic_shuffle=True
    if dataloc==CIRCLE_DATA_PATH:
        imitate_single = False # redundant
    if imitate_single:
        trainkeys,valkeys,testkeys = dividefolds(pic_shuffle, seed, valprop, num_out)
        trainfolds = set(b for (a,b,c) in trainkeys)
        valfolds = set(b for (a,b,c) in valkeys)
        testfolds = set(b for (a,b,c) in testkeys)
        added_video=[]
    vids = []
    with open(dataloc) as f:
        circle_data = json.load(f)
    for foldname in circle_data:
        for subfold in circle_data[foldname]:
            if subfold[:-4] in HOLDOUT_SET:
                continue
            if imitate_single and subfold == '27-J33-OD.csv': # Magic string, sorta
                added_video = [(foldname,subfold,x) for x in circle_data[foldname][subfold]]
                continue
            vids.append([(foldname,subfold,x) for x in circle_data[foldname][subfold]])
    
    if imitate_single:
        trainfolds = [vid for vid in vids if vid[0][1] in trainfolds]
        valfolds = [vid for vid in vids if vid[0][1] in valfolds]
        testfolds = [vid for vid in vids if vid[0][1] in testfolds]
        outfolds = [vid for vid in vids if vid[0][1] in outfolds]
        if readd_extra:
            trainfolds.append(added_video)
    else:
        r = np.random.RandomState(seed)
        perm = r.permutation(len(vids))
        validxs = r.choice(len(vids), size=int(len(vids)*valprop), replace=False)
        isval = np.zeros(len(vids), dtype=np.bool)
        for n in validxs:
            isval[n]=1
            
        valfolds = [v for n,v in enumerate(vids) if isval[n]]
        trainfolds = [v for n,v in enumerate(vids) if not isval[n]]
        r.shuffle(valfolds)
        if type(num_out) is slice:
            outfolds = testfolds[num_out]
        else:
            outfolds = testfolds[:num_out]

    trainkeys = [k for f in trainfolds for k in f]
    valkeys = [k for f in valfolds for k in f]
    testkeys = [k for f in testfolds for k in f]
    outkeys = [k for f in outfolds for k in f]
    
    return trainkeys,valkeys,testkeys,outkeys
'''
    
def csvout(val_path, training_curve, epoch_losses, datsize):
    with open(os.path.join(val_path,'Training curve.csv'), 'w') as f:
        for n in range(len(training_curve['train'])):
            f.write(f"{training_curve['train'][n]},{training_curve['validate'][n]}")
            if (n % datsize == 0) and (n/datsize < len(epoch_losses['validate'])):
                f.write(f",{epoch_losses['train'][n//datsize]},{epoch_losses['validate'][n//datsize]}")
            f.write('\n')
            
def smoothed_tc(outloc, training_curve, alpha=0.999):
    tclen = min(len(training_curve['train']),len(training_curve['validate']))
    if 'two_mire' in training_curve:
        tclen = min(len(training_curve['two_mire']), tclen)
    jointtc = np.array([training_curve['train'][:tclen],training_curve['validate'][:tclen]])
    if 'two_mire' in training_curve:
        jointtc = np.concatenate((jointtc, np.array([training_curve['two_mire'][:tclen]])))
    tcmeans = np.empty_like(jointtc)
    curr = np.zeros_like(jointtc[...,0])
    ct = 0
    for m in range(tclen):
        curr, ct = curr*alpha+jointtc[...,m], ct*alpha+1
        tcmeans[...,m] = curr/ct
    fig, ax = plt.subplots()
    ax.plot(tcmeans[0,tclen//2:])
    ax.plot(tcmeans[1,tclen//2:])
    if 'two_mire' in training_curve:
        ax.plot(tcmeans[2,tclen//2:])
    ymin, ymax = ax.get_ylim()
    ax.clear()
    ax.plot(tcmeans[0], linewidth=0.5, label='Training')
    if 'two_mire' in training_curve:
        ax.plot(tcmeans[1], linewidth=0.5, label='One-Mire Validation')
        ax.plot(tcmeans[2], linewidth=0.5, label='Two-Mire Validation')
    else:
        ax.plot(tcmeans[1], linewidth=0.5, label='Validation')
    ax.set_ylim(ymin,ymax)
    ax.legend(loc='lower left')
        
    plt.savefig(os.path.join(outloc, 'Training_curve.png'))
    plt.close()
    
def window_tc(outloc, training_curve, window_size=1000):
    tclen = min(len(training_curve['train']),len(training_curve['validate']))
    if 'two_mire' in training_curve:
        tclen = min(len(training_curve['two_mire']), tclen)
    jointtc = np.array([training_curve['train'][:tclen],training_curve['validate'][:tclen]])
    if 'two_mire' in training_curve:
        jointtc = np.concatenate((jointtc, np.array([training_curve['two_mire'][:tclen]])))
    cs = np.cumsum(jointtc, 1)
    cts = np.concatenate((np.arange(1,window_size+1),np.ones(cs.shape[1]-window_size)*window_size))[np.newaxis]
    padcs = np.concatenate((np.zeros((*cs.shape[:-1],window_size)),cs[...,:-window_size]),1)
    tcmeans = (cs-padcs)/cts
    fig, ax = plt.subplots()
    ax.plot(tcmeans[0], linewidth=0.5, label='Training')
    if 'two_mire' in training_curve:
        ax.plot(tcmeans[1], linewidth=0.5, label='One-Mire Validation')
        ax.plot(tcmeans[2], linewidth=0.5, label='Two-Mire Validation')
    else:
        ax.plot(tcmeans[1], linewidth=0.5, label='Validation')
    ax.set_ylim(0,1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    plt.savefig(os.path.join(outloc, 'Training_curve.pdf'))
    plt.close()
    
def warmup():
    testnet = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
    testnet = testnet.cuda()
    testtens = torch.randn(4,128,128,128)
    testtens = testtens.cuda()
    testtens = testnet(testtens)
    testtens = testnet(testtens)
    global WARMED_UP
    WARMED_UP = True
    
def cudarun(outsuffix, q, setloc = None, nettype='UNet'):
    pos,cuda = q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    testcycle(outsuffix, setloc=setloc, nettype=nettype, position=pos)
    q.put((pos,os.environ["CUDA_VISIBLE_DEVICES"]))
    
def testbatch(suffixes, setlocs=None, nettypes='UNet', cudae=range(4)):
    q = mp.Manager().Queue()
    for n,cuda in enumerate(cudae):
        q.put((n,cuda))
    if type(nettypes) is str:
        nettypes = [nettypes for _ in suffixes]
    if setlocs is None or type(setlocs) is str:
        setlocs = [setlocs for _ in suffixes]
    argses = [(suff,q,nettype,setloc) for (suff,nettype,setloc) in zip(suffixes,nettypes,setlocs)]
    with mp.pool(processes=len(cudae)) as pool:
        pool.starmap(cudarun, argses)
  
def setseed(worker_id):
    np.random.seed(torch.initial_seed()&((1<<32)-1))

def get_iops(outc, out_clsc, coords, classes, fallback = None):
    iopss = []
    for n in range(outc.shape[0]):
        ldiam = abs(outc[n,5]/outc[n,2])*GOLDMANN_OUTER_DIAM
        liop = 168.694/ldiam**2 if out_clsc[n,2]>0 else 0
        rdiam = abs(outc[n,8]/outc[n,2])*GOLDMANN_OUTER_DIAM
        riop = 168.694/rdiam**2 if out_clsc[n,2]>0 else 0
        if coords.size(1)<9 or coords[n,8]==0:
            inner_seg_rad = coords[n,5]
        elif coords[n,5]==0:
            inner_seg_rad = coords[n,8]
        else:
            ldist = (coords[n,0]-coords[n,3])**2+(coords[n,1]-coords[n,4])**2
            rdist = (coords[n,0]-coords[n,6])**2+(coords[n,1]-coords[n,7])**2
            inner_seg_rad = coords[n,5] if ldist < rdist else coords[n,8]
        if fallback is not None and inner_seg_rad==0:
            inner_seg_rad = fallback[n]
        segdiam = (inner_seg_rad/coords[n,2]).detach().item()*GOLDMANN_OUTER_DIAM
        segiop = 168.694/segdiam**2 if classes[n,2]>0 else 0
        iops = [liop if (outc[n,0]-outc[n,3])**2+(outc[n,1]-outc[n,4])**2
                         <(outc[n,0]-outc[n,6])**2+(outc[n,1]-outc[n,7])**2 else riop,
                segiop]
        iopss.append(iops)
    return iopss
    
def save_ex_pic(im, outc, has_inner, picname):
    yy, xx = np.mgrid[0:im.shape[0],0:im.shape[1]]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
    ax1.imshow(im)
    ax1.axis('off')
    outr = ((xx-outc[0])**2+(yy-outc[1])**2)**0.5
    if has_inner:
        inra = ((xx-outc[3])**2+(yy-outc[4])**2)**0.5
        inrb = ((xx-outc[6])**2+(yy-outc[7])**2)**0.5
    else:
        inra = np.ones_like(outr)*65535
        inrb = inra
    outfill = (outr<abs(outc[2])).astype(np.uint8)
    infill = ((inra<abs(outc[5])).astype(np.uint8)+(inrb<abs(outc[8])).astype(np.uint8))*outfill
    outbord = (abs(outr-abs(outc[2]))<1.5).astype(np.uint8)
    inbord = ((abs(inra-abs(outc[5]))<1.5).astype(np.uint8)+(abs(inrb-abs(outc[8]))<1.5).astype(np.uint8))*outfill
    ax2.imshow(outfill+infill, cmap='Set1', norm = NoNorm())
    ax2.axis('off')
    ax3.imshow((im*255).astype(np.uint8)|((outbord|inbord)*255)[:,:,np.newaxis])
    ax3.axis('off')
    fig.savefig(picname)
    plt.close(fig)
    
def save_pressure_graphs(pressures, outloc):
    vid_dict = {}
    for key in pressures:
        vid = key.split('_')[0]
        if vid not in vid_dict:
            vid_dict[vid] = []
        vid_dict[vid].append(key)
    
    for vid, frames in vid_dict.items():
        fnums = []
        iops = [[],[]]
        frames.sort()
        for f in frames:
            fnums.append(int(f.split('_')[1].split('.')[0]))
            for pres, preslist in zip(pressures[f], iops):
                preslist.append(pres)
        fig, ax = plt.subplots()
        ax.plot(fnums,iops[0],'bo',label='Inference')
        ax.plot(fnums,iops[1],'go',label='Ground truth')
        try:
            measured_pressure = int(vid.split('-')[0])
        except ValueError:
            try:
                measured_pressure = CLINICAL_IOP_DICT[vid]
            except KeyError:
                warnings.warn(f"Couldn't get measured pressure from {vid}")
        else:
            ax.plot([min(fnums),max(fnums)],[measured_pressure,measured_pressure],'--',label='Measured pressure')
        ax.legend()
        bot, top = ax.get_ylim()
        ax.set_ylim(top=min(top,50))
        fig.savefig(os.path.join(outloc,vid+' pressures.png'))
        plt.close(fig)
        with open(os.path.join(outloc,vid+' pressures.csv'),'w') as f:
            for frame, pres in zip(frames,iops[0]):
                f.write(f'{frame},{pres}\n')
    
def save_bland_altman(pressures, outloc, modified=True):
    vid_dict = {}
    for key in pressures:
        vid = key.split('_')[0]
        if vid not in vid_dict:
            vid_dict[vid] = []
        vid_dict[vid].append(key)
    iopss = []
    all_results = []
    for vid, frames in vid_dict.items():
        iops = []
        frames.sort()
        for f in frames:
            iops.append(pressures[f][0])
        try:
            measured_pressure = int(vid.split('-')[0])
        except ValueError:
            try:
                measured_pressure = CLINICAL_IOP_DICT[vid]
            except KeyError:
                warnings.warn(f"Couldn't get measured pressure from {vid}")
                measured_pressure = None
        nonzeros = [iop for iop in iops if iop!=0]
        if nonzeros:
            med_iop = np.median(nonzeros)
            low_iop = np.percentile(nonzeros, 25)
        else:
            warnings.warn(f'No mires found for {vid}')
            med_iop = None
        all_results.append([vid, measured_pressure, med_iop, any(iop for iop in (pressures[f][1] for f in frames))])
        if measured_pressure is None or med_iop is None:
            continue
        iopss.append([measured_pressure, med_iop, low_iop])
        
    with open(os.path.join(outloc,'video_iops.csv'),'w') as f:
        f.write('video, clinical, automated, mires in video\n')
        for [vid, measured, auto, exists] in all_results:
            f.write(f'{vid}, {measured}, {auto}, {exists}\n')
       
    '''
    fig, ax = plt.subplots()
    diffs = [iops[0]-iops[1] for iops in iopss]
    ax.scatter([iops[0] for iops in iopss],diffs,label='Difference')
    diff_mean = np.mean(diffs)
    sd_mean = np.std(diffs)
    left, right = ax.get_xlim()
    ax.set_xlabel('Clinical iop')
    ax.set_ylabel('Clinical-predicted')
    ax.plot([left,right],[diff_mean,diff_mean],'--',label='Mean difference')
    ax.plot([left,right],[diff_mean+1.96*sd_mean,diff_mean+1.96*sd_mean],'--',label='Mean+1.96*sd')
    ax.plot([left,right],[diff_mean-1.96*sd_mean,diff_mean-1.96*sd_mean],'--',label='Mean-1.96*sd')
    ax.set_xlim(left,right)
    fig.savefig(os.path.join(outloc,'bland-altman.png'))
    plt.close(fig)
    
    fig, ax = plt.subplots()
    low_diffs = [iops[0]-iops[2] for iops in iopss]
    ax.scatter([iops[0] for iops in iopss],diffs,label='Difference')
    low_diff_mean = np.mean(low_diffs)
    low_sd_mean = np.std(low_diffs)
    left, right = ax.get_xlim()
    ax.set_xlabel('Clinical iop')
    ax.set_ylabel('Clinical-predicted')
    ax.plot([left,right],[low_diff_mean,low_diff_mean],'--',label='Mean difference')
    ax.plot([left,right],[low_diff_mean+1.96*low_sd_mean,low_diff_mean+1.96*low_sd_mean],'--',label='Mean+1.96*sd')
    ax.plot([left,right],[low_diff_mean-1.96*low_sd_mean,low_diff_mean-1.96*low_sd_mean],'--',label='Mean-1.96*sd')
    ax.set_xlim(left,right)
    fig.savefig(os.path.join(outloc,'bland-altman-low.png'))
    plt.close(fig)
    '''

    iopss = [iops for iops in iopss if iops[0]]

    diff_mean = np.mean(diffs)
    sd_mean = np.std(diffs)

    fig, ax = plt.subplots()
    diffs = [iops[0]-iops[1] for iops in iopss]
    if modified:
        xs = [iops[0] for iops in iopss]
        xlab = 'Clinical IOP'
    else:
        xs = [(iops[0]+iops[1])/2 for iops in iopss]
        xlab = '(Clinical+Automated)/2'
    ax.scatter(xs,diffs,label='Difference')
    diff_mean = np.mean(diffs)
    sd_mean = np.std(diffs)
    left, right = ax.get_xlim()
    ax.set_xlabel(xlab)
    ax.set_ylabel('Clinical-Automated')
    ax.plot([left,right],[diff_mean,diff_mean],'--',label='Mean difference')
    ax.plot([left,right],[diff_mean+1.96*sd_mean,diff_mean+1.96*sd_mean],'--',label='Mean+1.96*sd',color='orange')
    ax.plot([left,right],[diff_mean-1.96*sd_mean,diff_mean-1.96*sd_mean],'--',label='Mean-1.96*sd',color='orange')
    ax.set_xlim(left,right)
    fig.savefig(os.path.join(outloc,'bland-altman-nz.png'))
    plt.close(fig)
    
    fig, ax = plt.subplots()
    ax.scatter([iops[0] for iops in iopss],[iops[1] for iops in iopss],label='Difference')
    diff_mean = np.mean(diffs)
    sd_mean = np.std(diffs)
    left, right = ax.get_xlim()
    ax.set_xlabel('Clinical iop')
    ax.set_ylabel('Predicted')
    ax.set_xlim(left,right)
    fig.savefig(os.path.join(outloc,'scatter-nz.png'))
    plt.close(fig)
    
    '''
    fig, ax = plt.subplots()
    low_diffs = [iops[0]-iops[2] for iops in iopss]
    ax.scatter([iops[0] for iops in iopss],diffs,label='Difference')
    low_diff_mean = np.mean(low_diffs)
    low_sd_mean = np.std(low_diffs)
    left, right = ax.get_xlim()
    ax.set_xlabel('Clinical iop')
    ax.set_ylabel('Clinical-predicted')
    ax.plot([left,right],[low_diff_mean,low_diff_mean],'--',label='Mean difference')
    ax.plot([left,right],[low_diff_mean+1.96*low_sd_mean,low_diff_mean+1.96*low_sd_mean],'--',label='Mean+1.96*sd')
    ax.plot([left,right],[low_diff_mean-1.96*low_sd_mean,low_diff_mean-1.96*low_sd_mean],'--',label='Mean-1.96*sd')
    ax.set_xlim(left,right)
    fig.savefig(os.path.join(outloc,'bland-altman-low-nz.png'))
    plt.close(fig)
    '''
    
def load_nearest_model(outloc, start_epoch, max_epoch):
    nettype = 'UNet' # Change this
    if start_epoch=='curr' or start_epoch=='prev':
        start_epoch_name = start_epoch
        model_state = load(os.path.join(outloc, nettype+f"_{start_epoch}"))
        start_epoch = model_state['epoch']+1
        print(f'Loading {nettype}_{start_epoch_name} (epoch {start_epoch-1})')
        return model_state, start_epoch
    else:
        prevs = [int(match.group(1)) for match in (re.fullmatch(nettype+'_(\d*)',s) for s in os.listdir(outloc)) if match]
        if start_epoch-1 in prevs:
            print(f'Loading {nettype}_{start_epoch-1}')
            model_state = load(os.path.join(outloc, nettype+f"_{start_epoch-1}"))
            return model_state, start_epoch
        else:
            test_model_state = load(os.path.join(outloc, nettype+"_curr"))
            last_epoch = test_model_state['epoch']
            if last_epoch==start_epoch-1:
                print(f'Loading {nettype}_curr (epoch {start_epoch-1})')
                return test_model_state, start_epoch
            elif last_epoch==start_epoch:
                prev_model_state = load(os.path.join(outloc, nettype+"_prev"))
                if prev_model_state['epoch']==start_epoch-1:
                    print(f'Loading {nettype}_prev (epoch {start_epoch-1})')
                    return prev_model_state, start_epoch
                else:
                    print(f'Epoch mismatch in {nettype}_prev')
            prev_candidates = [p for p in prevs if p<start_epoch]
            bestprev = max(prev_candidates) if prev_candidates else -1
            curr_usable = (last_epoch<start_epoch)
            if not prev_candidates and not curr_usable:
                print('No loadable point found; starting from scratch')
                return None, 0
            elif not curr_usable or bestprev>last_epoch:
                print(f'Loading {nettype}_{bestprev}')
                model_state = load(os.path.join(outloc, nettype+f"_{bestprev}"))
                return model_state, bestprev+1
                start_epoch = bestprev+1
            else:
                print('Loading {nettype}_curr (epoch {last_epoch})')
                return test_model_state, last_epoch+1
    
def framereader(video_path, dispname, requested_info=[], locations=[]):
    transform = composetransforms([constinfo(getdata.ToPil()), location_cropper(768, mem=4),
                                   resize(256), constinfo(getdata.ToTens())])
    cap = cv2.VideoCapture(video_path)
    for n in requested_info:
        yield cap.get(n)
    ret, inframe = cap.read()
    framenum = 0
    while(ret):
        name = ' '.join([dispname, 'frame', str(framenum)])
        inframe = inframe.transpose(1,0,2)
        inframe = np.flip(np.flip(inframe,2),1).copy()
        info = {}
        if locations:
            info['box'] = locations.pop(0)
        x, _, _ = transform(info,(inframe, [960,540,0], [0,0,0]))
        x = torch.unsqueeze(x,0)
        yield name, x
        ret, inframe = cap.read()
        framenum +=1
    
def tracking_framereader(video_path, dispname, tracker, requested_info=[]):
    transform = composetransforms([constinfo(getdata.ToPil()), tracker,
                                   resize(256), constinfo(getdata.ToTens())])
    cap = cv2.VideoCapture(video_path)
    for n in requested_info:
        yield cap.get(n)
    ret, inframe = cap.read()
        
    framenum = 0
    while(ret):
        name = ' '.join([dispname, 'frame', str(framenum)])
        inframe = inframe.transpose(1,0,2)
        inframe = np.flip(np.flip(inframe,2),1).copy()
        x, _, _ = transform({},(inframe, [960,540,0], [0,0,0]))
        x = torch.unsqueeze(x,0)
        yield name, x
        ret, inframe = cap.read()
        framenum +=1
    
class Circle_Find_Framereader():
    def __init__(self, video_path, dispname, requested_info=[], tracker=None,
                 **trargs):
        self.val = 0
        if tracker is not None:
            self.tracker = tracker
        elif 'width' in trargs:
            self.tracker = Tracker(**trargs)
        else:
            raise ValueError('Circle_Find_Framereader needs a tracker or width argument')
        self.dispname = dispname
        self.transform = composetransforms([constinfo(getdata.ToPil()), tracker,
                         resize(256), constinfo(getdata.ToTens())])
        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.inframe = self.cap.read()
        self.requested_info = requested_info
        self.framenum = 0
        
    def __next__(self):
        if self.requested_info:
            return self.cap.get(self.requested_info.pop(0))
        if not self.ret:
            raise StopIteration
        name = ' '.join([self.dispname, 'frame', str(self.framenum)])
        outframe = self.inframe.transpose(1,0,2)
        outframe = np.flip(np.flip(outframe,2),1).copy()
        x, _, _ = self.transform({},(outframe, [960,540,0], [0,0,0]))
        x = torch.unsqueeze(x,0)
        self.ret, self.inframe = self.cap.read()
        self.framenum +=1
        yield name, x
        
    def update(self, pos):
        self.tracker.update(pos)
        
    def __iter__(self):
        return self
    
def readjust(net, tracker, im, iters=2, from_scratch=True, perimeter_check=True, size_check=True,
             loc_check=True, net_im_size=256, shift_amount=20, name=None):
    if name:
        error_spec = ' for '+name
    else:
        error_spec = ''
    net.eval()
    transform = composetransforms([constinfo(getdata.ToPil()), tracker,
                                   resize(net_im_size), constinfo(getdata.ToTens())])
    try:
        net_im_size[0]
    except TypeError:
        net_im_size = (net_im_size, net_im_size)
    if from_scratch:
        tracker.focus(im)
    maxchange = tracker.maxchange
    tracker.maxchange = None
    center_rad = tracker.center_rad
    tracker.center_rad = 0
    for _ in range(iters):
        inp, _, _ = transform({}, (im, [960,540,0], [0,0,0]))
        xa = inp.cuda().unsqueeze(0)
        out, out_cls = net(xa)
        outc = out.detach().cpu()[0]
        tracker.update([outc[1]*tracker.height/net_im_size[0],outc[0]*tracker.width/net_im_size[1]])
        if size_check:
            if outc[2]>0.45*min(net_im_size):
                tracker.width = int(1.1*tracker.width)
                tracker.height = int(1.1*tracker.height)
            if outc[2]<0.1*min(net_im_size):
                tracker.width = int(tracker.width/1.1)
                tracker.height = int(tracker.height/1.1)
    tracker.maxchange = maxchange
    tracker.center_rad = center_rad
    if loc_check or perimeter_check:
        inp, _, _ = transform({}, (im, [960,540,0], [0,0,0]))
        xa = inp.cuda().unsqueeze(0)
        out, out_cls = net(xa)
        outc = out.detach().cpu()[0]
        basepos = tracker.prevpos
        padded_x = min(max(outc[0], (tracker.width-basepos[1])*net_im_size[1]/tracker.width),
                       (im.shape[1]-basepos[1])*net_im_size[1]/tracker.width)
        padded_y = min(max(outc[1], (tracker.height-basepos[0])*net_im_size[0]/tracker.height),
                       (im.shape[0]-basepos[0])*net_im_size[0]/tracker.height)
        if (abs(padded_x-net_im_size[1]/2)*tracker.width/net_im_size[1]>shift_amount/2
            or abs(padded_y-net_im_size[0]/2)*tracker.height/net_im_size[0]>shift_amount/2):
            #fi
            warnings.warn(f"Readjusting{error_spec} didn't get clean lock on circle")
        if size_check and (outc[2]>0.45*min(net_im_size) or outc[2]<0.1*min(net_im_size)):
            warnings.warn(f"Couldn't appropriately resize window{error_spec}")
            
    if perimeter_check:
        current_measured = (outc[0], outc[1], outc[2])
        offs = [(y,x) for x in [-1,0,1] for y in [-1,0,1] if (x,y)!=(0,0)]
        misses = 0
        miss_dirs = []
        total_checks = 0
        if size_check:
            resize_errors = 0
        for y,x in offs:
            ydiff = -y*basepos[0]-tracker.height/2+(1+y)*im.shape[0]//2
            if y and ydiff<shift_amount/2:
                continue
            xdiff = -x*basepos[1]-tracker.width/2+(1+x)*im.shape[1]//2
            if x and xdiff<shift_amount/2:
                continue
            total_checks += 1
            ydiff = int(min(ydiff, shift_amount))*y
            xdiff = int(min(xdiff, shift_amount))*x
            tracker.prevpos = basepos + np.array([ydiff, xdiff])
            inp, _, _ = transform({}, (im, [960,540,0], [0,0,0]))
            xa = inp.cuda().unsqueeze(0)
            out, out_cls = net(xa)
            outc = out.detach().cpu()[0]
            if (abs((current_measured[0]-outc[0])*tracker.width/net_im_size[1]-xdiff)>shift_amount/2
                and abs((current_measured[1]-outc[1])*tracker.height/net_im_size[0]-ydiff)>shift_amount/2):
                #fi
                misses += 0
                miss_dirs.append((x,y))
            if size_check and (outc[2]>current_measured[2]*1.1 or outc[2]<current_measured[2]*0.9):
                resize_errors += 1
        tracker.prevpos = basepos
        if misses:
            warnings.warn(f'Readjusting sanity check{error_spec} failed in {miss_dirs} directions out of {total_checks} possible')
        if size_check and resize_errors:
            warnings.warn(f'Readjusting sanity check{error_spec} failed {resize_errors} size checks out of {total_checks} possible')
            
def _sqtol(r2, rad, tolerance):
    return (r2<(rad+tolerance)**2)&(r2>max(rad-tolerance,0)**2)
    
def movwrite(net, val_on=None, data_path=None, save_path=None, video_path=None, dispname=None, thickness=1.5):
    set1 = np.array([[28,26,228], [184,126,55], [74,175,77], [163,78,152]],dtype=np.uint8)
    if (val_on is not None) and (data_path is not None):
        frames = [os.path.join(data_path,fname) for fname in os.listdir(data_path) if val_on in fname and '.png' in fname]
        frames.sort()
        transform = composetransforms([package_goldcoords, constinfo(getdata.ToPil()), constCrop(177,673,945,1441),
                                       resize(256), constinfo(getdata.ToTens())])
        dat = ImtoNumsDataset(frames,im_only,transformer=transform)
        if(len(dat)==0):
            return None
        dat_loader = DataLoader(dat)
        dat_iterator = tqdm(dat_loader)
        framerate = 25
    elif video_path is not None:
        if dispname is None:
            _,dispname = os.path.split(video_path)
        tracker = Tracker(768, maxchange=5, center_rad=2)
        dat_iterator = tracking_framereader(video_path, dispname, tracker, [5,7]) # Magic numbers; blame the cv2 people
        framerate = next(dat_iterator)
        num_frames = next(dat_iterator)
        dat_iterator = tqdm(dat_iterator, total=int(num_frames))
    else:
        raise ValueError('movwrite needs either a val_on and data_path argument (for frame input) or '
                         'video_path argument (for video input)')
    if(save_path is not None):
        dat_iterator.set_description(save_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter('.'.join([save_path,'avi']), fourcc, framerate, (808, 296))
    pressures = []
    results = []
    net.eval()
    torch.set_grad_enabled(False)
    yy,xx = np.mgrid[0:256,0:256]
    yy2,xx2 = yy**2,xx**2
    for name, x in dat_iterator:
        xa = x.cuda()
        out, out_cls = net(xa)
        outc = out.detach().cpu()[0]
        outc = torch.abs(outc).numpy()
        im_arr = np.clip((x[0]*256).numpy(),0,255).astype('uint8')
        im_arr = np.flip(im_arr,0).transpose(1,2,0)
        lr = outc[2]
        tracker.update([outc[1]*tracker.height/256,outc[0]*tracker.width/256])
        outr2 = xx2-2*outc[0]*xx+outc[0]**2+yy2-2*outc[1]*yy+outc[1]**2
        if(out_cls[0,2]>0):
            rightr2 = xx2-2*outc[3]*xx+outc[3]**2+yy2-2*outc[4]*yy+outc[4]**2
            leftr2 = xx2-2*outc[6]*xx+outc[6]**2+yy2-2*outc[7]*yy+outc[7]**2
            ir_r = outc[5]
            ir_l = outc[8]
            ir = ir_r if (outc[0]-outc[3])**2+(outc[1]-outc[4])**2<(outc[0]-outc[6])**2+(outc[1]-outc[7])**2 else ir_l
        else:
            rightr2 = np.full_like(outr2,255)
            leftr2 = rightr2
            ir_r = 0
            ir_l = ir_r
            ir = ir_r
        circmask = (outr2<outc[2]**2)*(1+(rightr2<ir_r**2)+(leftr2<ir_l**2))
        out_arr = set1[circmask]
        overlay = _sqtol(outr2,outc[2],thickness)|_sqtol(rightr2,ir_r,thickness)|_sqtol(leftr2,ir_l,thickness)
        circover = overlay[:,:,np.newaxis]*np.array([255,255,255],dtype=np.uint8)
        frame = np.full((296, 808, 3), 255, np.dtype('uint8'))
        frame[10:266, 10:266] = im_arr
        frame[10:266, 276:532] = out_arr
        frame[10:266, 542:798] = cv2.max(im_arr, circover)
        cv2.putText(frame, name, (246, 288), 0, 0.8, (0,0,0))
        vid.write(frame)
        
        diam = abs(ir/lr)*GOLDMANN_OUTER_DIAM
        iop = 168.694/diam**2 if diam>0 else 0
        pressures.append(iop)
        results.append(outc.tolist())
    torch.set_grad_enabled(True)
    
    ndpressures = np.array(pressures)
    fnums = np.mgrid[0:ndpressures.size][ndpressures>0]
    fig, ax = plt.subplots()
    tlim = np.percentile(ndpressures[ndpressures>0],75)*1.5
    ax.plot(fnums,ndpressures[ndpressures>0],'bo')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Pressure (mm Hg)')
    ax.set_ylim(0,tlim)
    fig.savefig('.'.join([save_path,'png']))
    plt.close(fig)
    vid.release()
    cv2.destroyAllWindows()
    return pressures, results
    
def process_videos(netloc,vidlist,savelocs=None,dispnames=None):
    if not 'tests' in netloc:
        netloc = os.path.join('tests',netloc)
    if not 'UNet' in netloc: # TODO: fix net name
        netloc = os.path.join(netloc,'UNet_best') # Ditto
    model_state = load(netloc)
    net = models['resnet50'](32,32)
    net = nn.DataParallel(net)
    net = net.cuda()
    net.load_state_dict(model_state['model'])
    
    if savelocs is None:
        netdir,_ = os.path.split(netloc)
        savelocs = (os.path.join(netdir,vidname.split('.')[0]) for _,vidname in (os.path.split(vidloc) for vidloc in vidlist))
    if dispnames is None:
        dispnames = (None for _ in vidlist)
    pressureses = {}
    resultses = {}
    for video_path, save_path, dispname in zip(vidlist,savelocs,dispnames):
        p,r = movwrite(net, video_path=video_path, save_path=save_path, dispname=dispname)
        pressureses[video_path] = p
        resultses[video_path] = r
    return pressureses, resultses
    
def pressures_from_vid(net, save_path=None, val_on=None, data_path=None, video_path=None, measured_pressure=None,
                       return_framerate=False, return_locations=False, skip_measured=False, graph_legend=True):
    if (val_on is not None) and (data_path is not None):
        frames = [os.path.join(data_path,fname) for fname in os.listdir(data_path) if val_on in fname and '.png' in fname]
        frames.sort()
        transform = composetransforms([package_goldcoords, constinfo(getdata.ToPil()), constCrop(177,673,945,1441),
                                       resize(256), constinfo(getdata.ToTens())])
        dat = ImtoNumsDataset(frames,im_only,transformer=transform)
        if(len(dat)==0):
            return None
        dat_loader = DataLoader(dat)
        dat_iterator = tqdm(dat_loader)
        framerate = 25
        dat_iterator.set_description(data_path)
    elif video_path is not None:
        tracker = Tracker(768, maxchange=5, center_rad=2)
        if return_locations:
            cap = cv2.VideoCapture(video_path)
            ret, inframe = cap.read()
            inframe = inframe.transpose(1,0,2)
            inframe = np.flip(np.flip(inframe,2),1).copy()
            readjust(net, tracker, inframe, iters=3, name=os.path.split(video_path)[1])
        dat_iterator = tracking_framereader(video_path, '',tracker,[5,7])
        framerate = next(dat_iterator)
        num_frames = next(dat_iterator)
        dat_iterator = tqdm(dat_iterator, total=int(num_frames))
        dat_iterator.set_description(video_path)
        if not skip_measured and measured_pressure is None:
            try:
                _,fname = os.path.split(video_path)
                n = fname.index('O')
                clean_vidname = fname[:n-1]+'-'+fname[n:n+2]
                measured_pressure = CLINICAL_IOP_DICT[clean_vidname]
            except:
                warnings.warn(f"Couldn't get iop reading from {video_path}")
    else:
        raise ValueError('pressures_from_vid needs either a val_on and data_path argument (for frame input) or '
                         'video_path argument (for video input)')
    pressures = []
    results = []
    if return_locations:
        locations = []
    net.eval()
    torch.set_grad_enabled(False)
    for name, x in dat_iterator:
        xa = x.cuda()
        out, out_cls = net(xa)
        outc = out.detach().cpu()[0]
        outc = torch.abs(outc).numpy()
        lr = outc[2]
        if(out_cls[0,2]>0):
            ir_r = outc[5]
            ir_l = outc[8]
            ir = ir_r if (outc[0]-outc[3])**2+(outc[1]-outc[4])**2<(outc[0]-outc[6])**2+(outc[1]-outc[7])**2 else ir_l
        else:
            ir_r = 0
            ir_l = ir_r
            ir = ir_r
        if return_locations:
            locations.append(tracker.latest_box)
        tracker.update([outc[1]*tracker.height/256,outc[0]*tracker.width/256])
        diam = abs(ir/lr)*GOLDMANN_OUTER_DIAM
        iop = 168.694/diam**2 if diam>0 else 0
        pressures.append(iop)
        results.append(outc.astype(float).tolist()+[out_cls.detach().cpu().numpy().astype(float)[0,2]])
    torch.set_grad_enabled(True)
    
    if save_path:
        ndpressures = np.array(pressures)
        fnums = np.mgrid[0:ndpressures.size][ndpressures>0]
        fig, ax = plt.subplots()
        if fnums.size:
            tlim = np.percentile(ndpressures[ndpressures>0],75)*1.5
            xmin, xmax = min(fnums/framerate), max(fnums/framerate)
        else:
            tlim = 40
            xmin, xmax = 0, len(pressures)/framerate
        ax.plot(fnums/framerate,ndpressures[ndpressures>0],'bo', label='Machine-detected pressure')
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Pressure (mm Hg)')
        ax.set_ylim(0,tlim)
        if not skip_measured and measured_pressure:
            ax.plot([xmin, xmax],[measured_pressure,measured_pressure],'--',label='Clinically-measured pressure')
        if graph_legend:
            ax.legend()
        if not os.path.splitext(save_path)[1]:
            save_path = save_path+'.png'
        fig.savefig(save_path)
        plt.close(fig)
    
    extrarets = []
    
    if return_locations:
        extrarets.append(locations)
    if return_framerate:
        extrarets.append(framerate)
        
    return (pressures, results, *extrarets)
    
def graph_vid(net=None, save_path=None, val_on=None, data_path=None, video_path=None, dispname=None, measured_pressure=None,
              results=None, pressures=None, framerate=None, locations=None, skip_measured=False, pregraph_data=True):
    if (val_on is not None) and (data_path is not None):
        frames = [os.path.join(data_path,fname) for fname in os.listdir(data_path) if val_on in fname and '.png' in fname]
        frames.sort()
        transform = composetransforms([package_goldcoords, constinfo(getdata.ToPil()), constCrop(177,673,945,1441),
                                       resize(256), constinfo(getdata.ToTens())])
        dat = ImtoNumsDataset(frames,im_only,transformer=transform)
        if(len(dat)==0):
            return None
        dat_loader = DataLoader(dat)
        dat_iterator = tqdm(dat_loader)
        if framerate is None:
            framerate = 25
        dat_iterator.set_description(data_path)
        update_tracker = False
    elif video_path is not None:
        if dispname is None:
            _,dispname = os.path.split(video_path)
        if locations:
            dat_iterator = framereader(video_path, dispname, [5,7], locations)
            update_tracker = False
        else:
            tracker = Tracker(768, maxchange=5, center_rad=2)
            dat_iterator = tracking_framereader(video_path, dispname, tracker, [5,7])
            update_tracker = True
        if framerate is None:
            framerate = next(dat_iterator)
        else:
            next(dat_iterator)
        num_frames = next(dat_iterator)
        dat_iterator = tqdm(dat_iterator, total=int(num_frames))
        dat_iterator.set_description(video_path)
        if measured_pressure is None and not skip_measured:
            try:
                _,fname = os.path.split(video_path)
                n = fname.index('O')
                clean_vidname = fname[:n-1]+'-'+fname[n:n+2]
                measured_pressure = CLINICAL_IOP_DICT[clean_vidname]
            except:
                warnings.warn(f"Couldn't get iop reading from {video_path}")
    else:
        raise ValueError('graph_vid needs either a val_on and data_path argument (for frame input) or '
                         'video_path argument (for video input)')
    if results is None:
        p, results = pressures_from_vid(net, save_path=None, val_on=val_on, data_path=data_path,
                                        video_path=video_path, measured_pressure=measured_pressure, 
                                        return_framerate=False, skip_measured=skip_measured)
        if pressures is None:
            pressures = p
    if pressures is None:
        pressures = []
        for r in results:
            if(r[9]>0):
                lr = abs(r[2])
                ir_r = abs(r[5])
                ir_l = abs(r[8])
                ir = ir_r if (r[0]-r[3])**2+(r[1]-r[4])**2<(r[0]-r[6])**2+(r[1]-r[7])**2 else ir_l
                diam = abs(ir/lr)*GOLDMANN_OUTER_DIAM
                iop = 168.694/diam**2 if diam>0 else 0
            else:
                iop = 0
            pressures.append(iop)
    fig, ax = plt.subplots(figsize = (5.2,3.9))
    ndpressures = np.array(pressures)
    fnums = np.mgrid[0:ndpressures.size][ndpressures>0]
    if fnums.size:
        tlim = np.percentile(ndpressures[ndpressures>0],75)*1.5
    else:
        tlim = 40
    xmax = len(pressures)/framerate
    if pregraph_data:
        ax.plot(fnums/framerate,ndpressures[ndpressures>0],'bo', label='Machine-detected pressure')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('IOP (mm Hg)')
    tlim = 25
    ax.set_ylim(0,tlim)
    axis_left, axis_right, axis_bot, axis_top = 0.1,0.98,0.11,0.98
    ax.set_position((axis_left, axis_bot, axis_right-axis_left, axis_top-axis_bot))
    
    xmin, xmax = 0, (len(pressures)-1)/framerate
    xmax = 19
    if not skip_measured and measured_pressure:
        ax.plot([xmin, xmax],[measured_pressure,measured_pressure],'--',label='Clinically-measured pressure')
    ax.set_xlim(xmin, xmax)
    #ax.legend()
    fig.savefig(save_path+'_temp.png')
    if pregraph_data:
        plt.close(fig)
    
    graph = np.asarray(Image.open(save_path+'_temp.png'))
    graph = np.atleast_3d(graph)
    if graph.shape[2] == 4:
        graph = graph[:,:,:3]
    if graph.shape[2] == 1:
        graph = graph*np.array([1,1,1],dtype=np.uint8)
    graph = np.flip(graph,2)
    os.remove(save_path+'_temp.png')
        
    yy, xx = np.mgrid[0:11,0:11] # Magic number zone
    add_circ_rad = (yy.shape[0]-1)//2
    ydist, xdist = abs(yy-add_circ_rad), abs(xx-add_circ_rad)
    ydist, xdist = ydist + (ydist>0)/2, xdist + (xdist>0)/2
    add_circle = (xdist**2+ydist**2<34)
        
    yy,xx = np.mgrid[0:256,0:256]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if not os.path.splitext(save_path)[1]:
        save_path = save_path+'.avi'
    vid = cv2.VideoWriter(save_path, fourcc, framerate, (536, 670))
    
    for n,((name, x),r,p) in enumerate(zip(dat_iterator,results, pressures)):
        im_arr = np.clip((x[0]*256).numpy(),0,255).astype('uint8')
        im_arr = np.flip(im_arr,0).transpose(1,2,0)
        lr = abs(r[2])
        outr2 = (xx-r[0])**2+(yy-r[1])**2
        if update_tracker:
            tracker.update([r[1]*tracker.height/256,r[0]*tracker.width/256])
        if(r[9]>0):
            rightr2 = (xx-r[3])**2+(yy-r[4])**2
            leftr2 = (xx-r[6])**2+(yy-r[7])**2
            ir_r = abs(r[5])
            ir_l = abs(r[8])
        else:
            rightr2 = np.full_like(outr2,255)
            leftr2 = rightr2
            ir_r = 0
            ir_l = ir_r
        thickness = 1.5
        overlay = _sqtol(outr2,lr,thickness)|_sqtol(rightr2,ir_r,thickness)|_sqtol(leftr2,ir_l,thickness)
        circover = overlay[:,:,np.newaxis]*np.array([255,255,255],dtype=np.uint8)
        frame = np.full((670, 536, 3), 255, np.dtype('uint8'))
        frame[8:264, 8:264] = im_arr
        frame[8:264, 272:528] = cv2.max(im_arr, circover)
        graph_left, graph_right, graph_top, graph_bot = 8, 528, 272, 662
        frame[graph_top:graph_bot, graph_left:graph_right] = graph
        #relx, rely = n/(len(pressures)-1), p/tlim
        relx, rely = n/(framerate*xmax), p/tlim
        absx = int(round(graph_left + (axis_left + (axis_right-axis_left)*relx)*(graph_right-graph_left)))
        absy = int(round(graph_bot + (axis_bot + (axis_top-axis_bot)*rely)*(graph_top-graph_bot)))
        linetop = int(round(graph_bot + axis_top*(graph_top-graph_bot)))
        linebot = int(round(graph_bot + axis_bot*(graph_top-graph_bot)))
        frame[linetop:linebot,absx] &= 0x7f
        if r[9]>0:
            absy = max(absy,266)
            if not pregraph_data:
                ax.plot([n/framerate],[p],'bo', label='Machine-detected pressure')
                fig.savefig(save_path+'_temp.png')
                graph = np.asarray(Image.open(save_path+'_temp.png'))
                graph = np.atleast_3d(graph)
                if graph.shape[2] == 4:
                    graph = graph[:,:,:3]
                if graph.shape[2] == 1:
                    graph = graph*np.array([1,1,1],dtype=np.uint8)
                graph = np.flip(graph,2)
                os.remove(save_path+'_temp.png')
            try:
                (frame[absy-add_circ_rad:absy+add_circ_rad+1,absx-add_circ_rad:absx+add_circ_rad+1])[add_circle] = [0,255,0]
            except IndexError:
                print(absy, absx)
                print(add_circ_rad)
                print(frame.shape)
                raise IndexError
        
        #cv2.putText(frame, name, (246, 288), 0, 0.8, (0,0,0))
        vid.write(frame)
        
    if not pregraph_data:
        plt.close(fig)
    vid.release()
    cv2.destroyAllWindows()
   
def annotated_graph(datloc, saveloc=None, legend=True):
    with open(datloc) as f:
        dat = json.load(f)
    pres = dat['pressures']
    fr = dat['framerate']
    pres.append(0)
    segs = []
    segstarts = []
    curr_seg = []
    minframe = None
    maxframe = 0
    for n,p in enumerate(pres):
        if minframe is None and p>0:
            minframe = n
        if p==0:
            if curr_seg:
                segs.append(curr_seg)
                maxframe = n-1
                segstarts.append(n-len(curr_seg))
                curr_seg = []
        else:
            curr_seg.append(p)
            
    if minframe is None:
        minframe = 0
        maxframe = len(pres)-1
            
    longsegs = []
    shortsegs = []
    for seg, ss in zip(segs, segstarts):
        if len(seg)<20:
            shortsegs.append(np.array([seg, range(ss,ss+len(seg))]))
        else:
            longsegs.append(np.array([seg, range(ss,ss+len(seg))]))
    if shortsegs:
        shortsegs = np.concatenate(shortsegs, 1)
    else:
        shortsegs = np.zeros((2,0))
    ends = []
    cleaned = []
    changey = []
    for seg in longsegs:
        ends.append(seg[:,:2])
        ends.append(seg[:,-2:])
        midseg = seg[:,2:-2]
        pseg = np.pad(midseg,((0,0),(10,10)),mode='edge')
        maxs = np.amax(np.array([pseg[0,n-10:n+11] for n in range(10,pseg.shape[1]-10)]),1)
        mins = np.amin(np.array([pseg[0,n-10:n+11] for n in range(10,pseg.shape[1]-10)]),1)
        changey.append(midseg[:,maxs/mins>=1.2])
        cleaned.append(midseg[:,maxs/mins<1.2])
        #changey.append((seg[:,2:-2])[:,abs(seg[0,2:-2]-seg[0,1:-3])+abs(seg[0,2:-2]-seg[0,3:-1])>=0.1*seg[0,2:-2]])
        #cleaned.append((seg[:,2:-2])[:,abs(seg[0,2:-2]-seg[0,1:-3])+abs(seg[0,2:-2]-seg[0,3:-1])<0.1*seg[0,2:-2]])
        
    if longsegs:
        ends = np.concatenate(ends, 1)
        changey = np.concatenate(changey, 1)
        cleaned = np.concatenate(cleaned, 1)
    else:
        ends = np.zeros((2,0))
        changey = ends
        cleaned = ends
    
    if saveloc:
        fig, ax = plt.subplots(figsize=(4,3))
        ndpressures = np.array(pres)
        fnums = np.mgrid[0:ndpressures.size][ndpressures>0]
        starttime = np.min(fnums)
        ax.scatter((cleaned[1]-starttime)/fr,cleaned[0],label='Readings used')
        ax.scatter((shortsegs[1]-starttime)/fr,shortsegs[0],label='Mire not present long enough')
        ax.scatter((ends[1]-starttime)/fr,ends[0],label='Right after/before mire not present')
        ax.scatter((changey[1]-starttime)/fr,changey[0],label='Pressure changing too fast')
        #tlim = np.percentile(ndpressures[ndpressures>0],75)*1.5
        tlim = 25
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Pressure (mm Hg)')
        if legend:
            ax.legend()
            
        ax.set_xlim(-0.5409250908677874, 11.146523682417083)
        ax.set_ylim(0,tlim)
        fig.savefig(saveloc, bbox_inches='tight')
        plt.close(fig)
    return cleaned
    
def testcycle(outloc, setloc = None, nettype='UNet', position=None, outmode=False, loadloc=None, saveevery=None, start_epoch=0):
    synth_decrease = 'decsynth' in outloc
    if 'verysynth' in outloc:
        synthrate = 0.5
    elif 'nosynth' in outloc:
        synthrate = 0
    else:
        m = re.search('(\d*\.?\d*)synth',outloc) # Walrus in 3.8
        if m and m.group(1):
            synthrate = float(m.group(1))
            if synthrate > 1:
                raise ValueError(f'based on {m.group(0)} got synthrate {m.group(1)}>1')
        elif 'synth' in outloc:
            synthrate = 0.2
        else:
            synthrate = 0
    if 'extra' in outloc:
        extrarate = 0.5
    else:
        extrarate = 0
    if 'colconv' in outloc:
        color_mode = 'randcomb'
    else:
        color_mode = 'rand'
    both_mires = True
    segment_lines = False
    one_mire_compatibility = False
    ro_mode = True
    
    extraswitch = False
    
    rand_inner = False
    orig_inner = False
    track_orig_inner = False
    one_mire_compatibility ^= rand_inner|orig_inner
    
    if ro_mode:
        dataloc = RYAN_OMAR_PATH
        trainkeys,valkeys,outkeys = divide_ro_vids(seed=int(outloc.split('.')[1]), num_out=5,dataloc=dataloc)
        
        traintransform = composetransforms([package_ro(), constinfo(getdata.ToPil()), randRotate(resizing=True),
                                           randResizedCrop(256,256,0.25,0.5,circle_fullness=0.75), constinfo(getdata.ToTens()), add_classes()])
                                           
        outtransform = composetransforms([package_ro(), constinfo(getdata.ToPil()), constCrop(177,673,945,1441),
                                          resize(256), constinfo(getdata.ToTens()), add_classes()])
                                          
        switchname = 'is_ryan'
        
        lookup = lookup_switch((lambda x: x[0]=='fake'), fakesonly(extrarate=extrarate,both_mires=both_mires, with_name=True), ro_lookup())
        
        fake_key = ('fake','fake',0.,0.,0.,0.,0.,0.5)
    
    else:
        dataloc = TWO_CIRCLE_DATA_PATH if both_mires else CIRCLE_DATA_PATH
        
        trainkeys,valkeys,outkeys = dividefolds(seed=int(outloc.split('.')[1]), num_out=5,dataloc=dataloc)
        
        traintransform = composetransforms([package_goldcoords(), constinfo(getdata.ToPil()), randRotate(resizing=True),
                                           randResizedCrop(256,256,0.25,0.5,circle_fullness=0.75), constinfo(getdata.ToTens()), add_classes()])
                                           
        outtransform = composetransforms([package_goldcoords(), constinfo(getdata.ToPil()), constCrop(177,673,945,1441),
                                          resize(256), constinfo(getdata.ToTens()), add_classes()])
                                          
        lookup = lookup_switch((lambda x: x[0]=='fake'), fakesonly(extrarate=extrarate,both_mires=both_mires), get_golddat(coordindex)) # Trying adding synthetic data
        
        switchname = 'origseg'
        
        fake_key = ('fake','fake','fake')
        
        #testkeys = testkeys + [('fake','fake','fake')]*(len(testkeys)//5)  #Trying adding synthetic data
        outkeys = outkeys + [('fake','fake','fake')]*5
    
    testlocs = ['../../yue/joanne/GAT SL videos/other_techs/raw/I01-OS',
                '../../yue/joanne/GAT SL videos/other_techs/raw/I03-OD',
                '../../yue/joanne/GAT SL videos/other_techs/raw/I06-OS']
    testkeys = [os.path.join(testloc,f) for testloc in testlocs for f in os.listdir(testloc)]
    
    with open(dataloc) as f:
        coordindex = json.load(f)
    
    
    train_dat, val_dat, out_dat, inf_dat = maken(ImtoNumsDataset, 4,
                                                 keys_0=trainkeys,
                                                 keys_1=valkeys,
                                                 keys_2=outkeys,
                                                 keys_3=testkeys, 
                                                 keyadj_0=MaybeFaker(synthrate, fake_key=fake_key),
                                                 datlookup=lookup,
                                                 #datlookup_01=get_golddat(coordindex),
                                                 datlookup_3=im_only(7+3*both_mires),
                                                 transformer_01=traintransform,
                                                 transformer_23=outtransform)
    
    if outmode and setloc is None:
        setloc = outloc
    
    #out_dat.keys = outkeys
    
    train_loader = DataLoader(train_dat, shuffle=True, batch_size = 4, num_workers = 6, worker_init_fn=setseed, pin_memory=True)
    val_loader = DataLoader(val_dat, shuffle=True, batch_size = 2, num_workers = 4, worker_init_fn=setseed, pin_memory=True)
    out_loader = DataLoader(out_dat, batch_size=2, num_workers = 2, worker_init_fn=setseed, pin_memory=True)
    
    max_epoch = 200
    if outmode:
        start_epoch = max_epoch
        
    net = models['resnet50'](32,32,extraswitch=extraswitch)
    
    net = nn.DataParallel(net)
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    scheduler = ExponentialLR(optimizer, 0.99)
    net = net.cuda()
    
    # Trying coord augmentation
    # Changes start here
    #yaug = torch.arange(0,1.001,1/(32-1)).view(1,1,-1,1).cuda()
    #xaug = torch.arange(0,1.001,1/(32-1)).view(1,1,1,-1).cuda()
    # Changes end here
    
    if start_epoch and (max_epoch!=start_epoch):
        model_state, start_epoch = load_nearest_model(outloc, start_epoch, max_epoch)
        if model_state is not None:
            net.load_state_dict(model_state['model'])
            optimizer.load_state_dict(model_state['optimizer'])
            if 'scheduler' in model_state:
                scheduler.load_state_dict(model_state['scheduler'])
            else:
                for _ in range(start_epoch):
                    scheduler.step()
    
    save_every = max(int(np.round(max_epoch/50)),1)*5
    
    autobest = True
    best_loaded = False
    if 'tests' not in outloc:
        outloc = 'tests/' + outloc
        
    print('Validations going to', outloc)
    
    outputonly = False
    if(max_epoch==start_epoch):
        outputonly = True
        max_epoch+=1
    elif start_epoch>0:
        with open(os.path.join(outloc, 'Epoch_losses.json')) as f:
            epoch_losses = json.load(f)
        with open(os.path.join(outloc, 'Training_curve.json')) as f:
            training_curve = json.load(f)
        if not outputonly:
            prevbest = min(epoch_losses['validate'])
            if len(epoch_losses['train'])<start_epoch:
                warnings.warn(f"Training data only goes to epoch {len(epoch_losses['train'])-1}")
                epoch_losses['train'] += ['No data']*(start_epoch-len(epoch_losses['train']))
                training_curve['train'] += [float('nan')]*int(len(training_curve['train'])*(start_epoch/len(epoch_losses['train'])-1))
            if len(epoch_losses['validate'])<start_epoch:
                warnings.warn(f"Validation data only goes to epoch {len(epoch_losses['validate'])-1}")
                epoch_losses['validate'] += ['No data']*(start_epoch-len(epoch_losses['validate']))
                training_curve['validate'] += [float('nan')]*int(len(training_curve['validate'])*(start_epoch/len(epoch_losses['validate'])-1))
            if len(epoch_losses['train'])>start_epoch:
                warnings.warn("Previous data had data beyond the current starting epoch.  Truncating; please see 'Epoch_losses.json.old' "
                              "and 'Training_curve.json.old' for previous data")
                with open(os.path.join(outloc, 'Epoch_losses.json.old'),'w') as f:
                    json.dump(epoch_losses, f)
                with open(os.path.join(outloc, 'Training_curve.json.old'),'w') as f:
                    json.dump(training_curve, f)
                training_curve['train'] = training_curve['train'][0:start_epoch*len(training_curve['train'])//len(epoch_losses['train'])]
                training_curve['validate'] = training_curve['train'][0:start_epoch*len(training_curve['validate'])//len(epoch_losses['validate'])]
                epoch_losses['train'] = epoch_losses['train'][0:start_epoch]
                epoch_losses['validate'] = epoch_losses['validate'][0:start_epoch]
                prevbest = min(epoch_losses['validate'])
    else:
        epoch_losses={'train':[],'validate':[]}
        os.makedirs(outloc, exist_ok=True)
        with open(os.path.join(outloc, '_'.join([nettype, 'dat'])), 'wb') as f:
            dill.dump({'train_dat':train_dat, 'val_dat':val_dat, 'out_dat':out_dat}, f)
        prevbest = float('inf')
        training_curve = {'train':[],'validate':[]}
        if rand_inner or orig_inner or track_orig_inner:
            training_curve['two_mire'] = []
        
    
    seg_criterion = getdata.Circles_Dice()
    #seg_criterion = nn.L1Loss()
    class_weights = None
    cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    line_loss = nn.L1Loss()
    
    if not outputonly: # Set up validationsource
        validationsource = infiniloader(val_dat, shuffle=True, batch_size=2, num_workers=1, worker_init_fn=setseed, pin_memory=True)
        next(validationsource)
        
    if loadloc is not None:
        model_state = load(os.path.join(loadloc, nettype+"_best"))
        net.load_state_dict(model_state['model'])
        print(f"Loading epoch {model_state['epoch']} from {loadloc}")
    
    if position is None:
        outiter = range(start_epoch, max_epoch)
    else:
        outiter = trange(start_epoch, max_epoch, position=position)
        outiter.set_description(outloc)
    
    for epoch in outiter:
        scheduler.step()
        modes = ['train', 'validate']
        if epoch == max_epoch-1:
            pass
            modes.append('output')
            #modes.append('inference')
            if outputonly:
                modes = ['output']
                #modes = ['compare iops']
        for mode in modes:
            if mode == 'train':
                if synth_decrease:
                    train_dat.keyadj.p = 0.1+0.8*(99-epoch)/99
                loader = train_loader
                noun = 'Training'
                best_loaded = False
                net.train()
                torch.autograd.set_grad_enabled(True)
            elif mode == 'validate':
                loader = val_loader
                noun = 'Validation'
                net.eval()
                torch.autograd.set_grad_enabled(False)
            elif mode == 'output':
                loader = out_loader
                noun = 'Output'
                if autobest and not best_loaded:
                    model_state = load(os.path.join(outloc, nettype+"_best"))
                    net.load_state_dict(model_state['model'])
                    tqdm.write(f"Outputting using best model: epoch {model_state['epoch']}")
                    best_loaded=True
                net.eval()
                torch.autograd.set_grad_enabled(False)
                results = []
                pressures = {}
                numfakes = 0
            elif mode == 'inference':
                loader = DataLoader(inf_dat, batch_size=2, num_workers = 2, worker_init_fn=setseed, pin_memory=True)
                os.makedirs(os.path.join(outloc,'inferences'), exist_ok=True)
                noun = 'Inference'
                if autobest and not best_loaded:
                    model_state = load(os.path.join(outloc, nettype+"_best"))
                    net.load_state_dict(model_state['model'])
                    tqdm.write(f"Inference using best model: epoch {model_state['epoch']}")
                    best_loaded=True
                net.eval()
                torch.autograd.set_grad_enabled(False)
            elif mode == 'compare iops':
                loader = val_loader
                noun = 'Testing'
                if autobest and not best_loaded:
                    model_state = load(os.path.join(outloc, nettype+"_best"))
                    net.load_state_dict(model_state['model'])
                    tqdm.write(f"Outputting using best model: epoch {model_state['epoch']}")
                    best_loaded=True
                net.eval()
                torch.autograd.set_grad_enabled(False)
                pressures = {}
            
            if position is None:
                data_iterator = tqdm(loader)
            else:
                data_iterator = loader
            cur_losses = []
            num = 0
            for info, ims, coords, classes in data_iterator:
                if mode == 'train':
                    optimizer.zero_grad()
                ims = ims.cuda()
                coords = coords.cuda()
                classes = classes.cuda()
                if extraswitch:
                    switch = info[switchname].float().cuda()
                    if mode=='output' or mode=='inference':
                        switch = torch.ones_like(switch)*0.5
                    out, out_cls = net(ims,switch)
                else:
                    out, out_cls = net(ims)
                
                if one_mire_compatibility:
                    out = torch.cat([out[:,:3],out[:,6:9],out[:,3:6]],1)
                
                if rand_inner:
                    fakeswitch = torch.randint(high=2,size=(ims.size(0),1),device=ims.device)
                    info['isbot'] = fakeswitch[:,0]
                    fakeswitch = fakeswitch.cuda()
                    radsum = coords[:,5]+coords[:,8]
                    coords = torch.cat((coords[:,:3],coords[:,3:6]*(1-fakeswitch)+coords[:,6:9]*fakeswitch),1)
                        
                if orig_inner:
                    fakeswitch = info['origseg']
                    for a in range(fakeswitch.size(0)):
                        if info['key'][0][a] == 'fake':
                            fakeswitch[a] = np.random.randint(2)
                    fakeswitch = fakeswitch.float()
                    info['isbot'] = fakeswitch
                    fakeswitch = torch.unsqueeze(fakeswitch,1).cuda()
                    coords = torch.cat((coords[:,:3],coords[:,3:6]*(1-fakeswitch)+coords[:,6:9]*fakeswitch),1)
                
                if 'isbot' in info:
                    selected_out = getdata.circle_select(out, info['isbot'])
                    seg_loss = seg_criterion(selected_out,coords[:,:6])
                else:
                    seg_loss = seg_criterion(out[:,:9],coords[:,:9])
                cls_loss = cls_criterion(out_cls, classes)
                loss = seg_loss + cls_loss
                if torch.isnan(loss):
                    print('nan loss')
                    print(seg_loss)
                    print(cls_loss)
                    quit()
                cur_losses.append(loss.data.item())
                if position is None:
                    if mode=='train':
                        status = f'[{epoch}] - Training - loss = {loss.data.item():8.5f} - avg = {np.mean(cur_losses):0.5f} - LR = {scheduler.get_lr()[0]:0.6f}'
                    elif mode=='validate':
                        status = f'[{epoch}] - Validation - loss = {loss.data.item():8.5f} - avg = {np.mean(cur_losses):0.5f}'
                    else:
                        status = f'[{max_epoch-1}] - Output - loss = {loss.data.item():8.5f} - avg = {np.mean(cur_losses):0.5f}'
                    data_iterator.set_description(status)
                if mode == 'train':
                    training_curve['train'].append(loss.data.item())
                    
                    if segment_lines:
                        ms = torch.tan(coords[:,-3:-2]) #Trying midline segmentation
                        line_loc = torch.cat((ms,coords[:,-1:]-ms*coords[:,-2:-1]),1)
                        for n in range(out.size(0)):
                            if info['key'][2][n]=='fake':
                                loss = loss + line_loss(out[n,9:],line_loc[n])/(3*out.size(0))
                    
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        net.eval()
                        testinfo, testims, testcoords, testclasses = next(validationsource)
                        testims, testcoords, testclasses = testims.cuda(), testcoords.cuda(), testclasses.cuda()
                        if extraswitch:
                            testswitch = testinfo[switchname].float().cuda()
                            testout, testout_cls = net(testims,testswitch)
                        else:
                            testout, testout_cls = net(testims)
                        
                        if rand_inner or orig_inner or track_orig_inner:
                            test_seg_loss = seg_criterion(testout[:,:9],testcoords[:,:9])
                            test_cls_loss = cls_criterion(testout_cls, testclasses)
                            testloss = test_seg_loss + test_cls_loss
                            training_curve['two_mire'].append(testloss.data.item())
                            if rand_inner:
                                fakeswitch = torch.randint(high=2,size=(testims.size(0),1),device=ims.device)
                            if orig_inner or track_orig_inner:
                                fakeswitch = testinfo['origseg']
                                for a in range(fakeswitch.size(0)):
                                    if testinfo['key'][0][a] == 'fake':
                                        fakeswitch[a] = np.random.randint(2)
                                fakeswitch = torch.unsqueeze(fakeswitch,1).float()
                            testinfo['isbot'] = fakeswitch[:,0]
                            fakeswitch = fakeswitch.cuda()
                            #testcoords = torch.cat((testcoords[:,:3],testcoords[:,3:6]*(1-fakeswitch)+testcoords[:,6:9]*fakeswitch),1)
                            testcoords = getdata.circle_select(testcoords, testinfo['isbot'])
                        
                        if one_mire_compatibility^track_orig_inner:
                            testout = torch.cat([testout[:,:3],testout[:,6:9],testout[:,3:6]],1)
                        
                        if 'isbot' in testinfo:
                            test_selected_out = getdata.circle_select(testout, testinfo['isbot'])
                            test_seg_loss = seg_criterion(test_selected_out,testcoords[:,:6])
                        else:
                            test_seg_loss = seg_criterion(testout[:,:9],testcoords[:,:9])
                        test_cls_loss = cls_criterion(testout_cls, testclasses)
                        testloss = test_seg_loss + test_cls_loss
                        training_curve['validate'].append(testloss.data.item())
                        net.train()
                if mode == 'output' or mode=='inference':
                    if 'orig' in info:
                        imsc = info['orig'].numpy().transpose(0,2,3,1)
                    else:
                        imsc = ims.detach().cpu().numpy().transpose(0,2,3,1)
                    outc = out.detach().cpu().numpy()
                    out_clsc = out_cls.detach().cpu().numpy()
                    yy, xx = np.mgrid[0:imsc.shape[1],0:imsc.shape[2]]
                    for n in range(ims.size(0)):
                        if mode == 'inference':
                            picname = os.path.join(outloc,'inferences',info['key'][n].split('/')[-1])
                        else:
                            if 'filename' in info:
                                picname = os.path.join(outloc, info['filename'][n]+'.png')
                            else:
                                picname = os.path.join(outloc, info['key'][2][n])
                            if picname[:4]=='fake':
                                picname = os.path.join(outloc,f'synthetic{numfakes}.png')
                                numfakes += 1
                        save_ex_pic(imsc[n], outc[n], out_clsc[n,2]>0, picname)
                if mode == 'output' or mode == 'compare iops':
                    outc = out.detach().cpu().numpy()
                    out_clsc = out_cls.detach().cpu().numpy()
                    iopss = get_iops(outc, out_clsc, coords, classes, fallback=(radsum if rand_inner else None))
                    for n,iops in enumerate(iopss):
                        if 'filename' in info:
                            if info['filename'][n]!='fake':
                                pressures[info['filename'][n]+'.png'] = iops
                        elif info['key'][2][n]!='fake':
                            pressures[info['key'][2][n]] = iops
                        
            if mode == 'train' or mode == 'validate':
                epoch_losses[mode].append(np.mean(cur_losses))
                with open(os.path.join(outloc, 'Epoch_losses.json'), 'w') as f:
                    json.dump(epoch_losses, f)
            if mode == 'train':
                with open(os.path.join(outloc, 'Training_curve.json'), 'w') as f:
                    json.dump(training_curve, f)
            if mode == 'validate':
                if epoch>0:
                    os.rename(os.path.join(outloc, nettype+"_curr"), os.path.join(outloc, nettype+"_prev"))
                model_state = {'model': net.state_dict(),
                               'optimizer': optimizer.state_dict(),
                               'epoch': epoch}
                if epoch%save_every==0:
                    save(model_state, os.path.join(outloc, nettype+f"_{epoch}"))
                save(model_state, os.path.join(outloc, nettype+"_curr"))
                if prevbest>np.mean(cur_losses):
                    save(model_state, os.path.join(outloc, nettype+"_best"))
                    prevbest = np.mean(cur_losses)
                csvout(outloc, training_curve, epoch_losses, math.ceil(len(train_dat)/train_loader.batch_size))
                smoothed_tc(outloc, training_curve)
            if mode == 'output':
                #with open(os.path.join(outloc,'results'), 'w') as f:
                    #f.write('\n'.join([str(x) for x in results]))
                save_pressure_graphs(pressures, outloc)
                            
            if mode == 'compare iops':
                save_bland_altman(pressures, outloc)
                
    if not outputonly:
        try:
            validationsource.send(1)
        except StopIteration:
            pass
    
if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)
    
    net = models['resnet50'](32,32)
    
    net = nn.DataParallel(net)
    net = net.cuda()
    model_state = load('tests/99dec-ro-verysynth-colswitchextra.0/UNet_best')
    net.load_state_dict(model_state['model'])
    
    vids_to_proc = ['J006_OD', 'J021_OD', 'Tech013_OD', 'Tech018_OS']        
    video_dir = '/data/yue/joanne/GAT SL videos/reproduceability_videos'
    assert all(v+'.MOV' in os.listdir(video_dir) for v in vids_to_proc)
    
    times = []
    framelens = []
    vidtimes = []
    
    for vname in vids_to_proc:
        vidloc = os.path.join(video_dir,vname+'.MOV')
        stime = time.time()
        p, r, fr = pressures_from_vid(net, video_path=vidloc, skip_measured=True, return_framerate=True)
        tot_time = time.time()-stime
        times.append(tot_time)
        framelens.append(len(p))
        vidtimes.append(len(p)/fr)
    
    print(times)
    print(framelens)
    print(vidtimes)
    
    quit()
    '''
    basedir = 'tests/99dec-ro-verysynth-colswitchextra.0/intervideos'
    to_proc = (os.path.join(basedir,x) for x in os.listdir(basedir) if '_res.json' in x)
    vids_to_proc = ['J006_OD', 'J021_OD', 'Tech013_OD', 'Tech018_OS']
    for floc in to_proc:
        if not any(v in floc for v in vids_to_proc):
            continue
        with open(floc) as f:
            dat = json.load(f)
        ndpressures = np.array(dat['pressures'])
        framerate = dat['framerate']
        fnums = np.mgrid[0:ndpressures.size][ndpressures>0]
        fnums = fnums - np.min(fnums)
        print(np.max(fnums))
        fig, ax = plt.subplots(figsize=(8,6))
        if fnums.size:
            tlim = 25
            xmin, xmax = min(fnums/framerate), max(fnums/framerate)
        else:
            tlim = 40
            xmin, xmax = 0, len(pressures)/framerate
        ax.scatter(fnums/framerate,ndpressures[ndpressures>0], label='Machine-detected pressure')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('IOP (mm Hg)')
        ax.set_ylim(0,tlim)
        ax.set_xlim(-0.5409250908677874, 11.146523682417083)
        save_path = floc.replace('_res.json','.svg')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        saveloc = floc.replace('_res.json','_label.svg')
        annotated_graph(floc, saveloc, False)
    
    quit()
    
    '''
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print(f'Using CUDA {os.environ["CUDA_VISIBLE_DEVICES"]}:')
    
    testcycle('99dec-ro-verysynth-75circ-vlong-patsplit.0')
    
    quit()
    '''
    
    shus = []
    with open('shu_iops.txt') as f:
        for l in f:
            shus.append(l.rstrip().split(','))

    joannes = []
    with open('joanne_iops.txt') as f:
        for l in f:
            joannes.append(l.rstrip().split(','))
    
    dlreads = []
    with open('dl_pressures_fit.csv') as f:
        for l in f:
            dlreads.append(l.rstrip().split(','))
    
    udlreads = []
    with open('dl_pressures_prefit.csv') as f:
        for l in f:
            udlreads.append(l.rstrip().split(','))
            
    dataloc = RYAN_OMAR_PATH
    trainkeys,valkeys,outkeys = divide_ro_vids(seed=0, num_out=5,dataloc=dataloc,patsplit=False)
    
    trainsets = sorted(set(a[0].split('_')[0] for a in trainkeys))
    print(len(trainsets))
    #print(sorted(trainsets))
    
    valsets = sorted(set(a[0].split('_')[0] for a in valkeys))
    print(len(valsets))
    #print(sorted(valsets))
    
    readings = []
    infers = []
    for t in trainsets:
        locs = [a for a in dlreads if a[0]==t[0] and int(a[1])==int(t.split('-')[0][1:])]
        assert len(locs)==1
        if 'OD' in t:
            which = 0
        else:
            which = 1
        if t[0] == 'F':
            inferlist = shus
        else:
            inferlist = joannes
        infer = locs[0][2+which]
        read = inferlist[int(locs[0][1])-1][which]
        if infer == '--':
            continue
        if read == '--' or read=='0':
            continue
        readings.append(float(read))
        infers.append(float(infer))
    readings = np.array(readings)
    infers = np.array(infers)
    print(np.mean(infers-readings))
    
    fig, ax = plt.subplots(figsize=(4,3))
    diffs = infers-readings
    ax.scatter((readings+infers)/2,diffs,label='Difference')
    diff_mean = np.mean(diffs)
    sd_mean = np.std(diffs)
    left, right = 0,30
    ax.set_xlabel('Mean of Automated and GAT (mmHg)')
    ax.set_ylabel('Automated - GAT (mmHg)')
    print(diff_mean, diff_mean-1.96*sd_mean, diff_mean+1.96*sd_mean)
    ax.plot([left,right],[diff_mean,diff_mean],'--',label='Mean difference')
    ax.plot([left,right],[diff_mean+1.96*sd_mean,diff_mean+1.96*sd_mean],'--',label='Mean+1.96*sd',color='orange')
    ax.plot([left,right],[diff_mean-1.96*sd_mean,diff_mean-1.96*sd_mean],'--',label='Mean-1.96*sd',color='orange')
    ax.set_xlim(left,right)
    ax.set_ylim(-10,10)
    fig.savefig('training_b-a_fit.svg', bbox_inches='tight')
    plt.close(fig)
    
    quit()
    
    '''
    testcycle('99dec-ro-verysynth-colswitchextra-75circ.0', outmode=True)
    
    quit()
    '''
    '''
    vidlist = ['../../yue/joanne/GAT SL videos/videos/I01-OS.MOV',
               '../../yue/joanne/GAT SL videos/videos/I03-OS.MOV',
               '../../yue/joanne/GAT SL videos/videos/I06-OS.MOV']
    types = [amt+'synthdata-col'+randtype+extra+'.0' for amt in ['very','dec'] for randtype in ['switch','conv'] for extra in ['extra','']]
    vidlist = vidlist + [os.path.join('videos',f) for f in os.listdir('videos')]
    
    for type in types:
        pss, rss = process_videos(type,vidlist)
        for k, v in pss.items():
            _, vidname = os.path.split(k)
            with open(os.path.join('tests',type,vidname.split('.')[0]+'.json'),'w') as f:
                json.dump(v,f)
    quit()
    '''
    
    #s = [a.split('_')[0] for a in os.listdir('tests/99dec-ro-verysynth-colswitchextra.0/videos/') if 'res' in a]
    #trainkeys,valkeys,outkeys = divide_ro_vids(seed=0, num_out=5,dataloc=RYAN_OMAR_PATH)
    #valsets = sorted(set(ent[0].split('_')[0] for ent in valkeys))
    '''
    valsets = [a[:-4] for a in os.listdir('tests/99dec-ro-verysynth-colswitchextra.0/intervideos/') if '.avi' in a]
    r = np.random.RandomState(0)
    r.shuffle(valsets)
    meds = []
    clinicals = []
    used_names = []
    
    vid_pressures = {}
    fit_vid_pressures = {}
    
    for vid in valsets:
        datloc = 'tests/99dec-ro-verysynth-colswitchextra.0/intervideos/'+vid+'_res.json'
        saveloc = os.path.join('tests/99dec-ro-verysynth-colswitchextra.0/intervideos',vid+'_label.png')
        cleaned_vals = annotated_graph(datloc)
        if cleaned_vals.size:
            med = np.median(cleaned_vals[0])
            meds.append(med)
            vid_pressures[vid] = med
            fit_vid_pressures[vid] = med*1.26128949-3.62980616
            used_names.append(vid)
        else:
            print(vid)
    quit()
    '''
    
    with open('tests/99dec-ro-verysynth-colswitchextra.0/intervideos/vid_pressures.json') as f:
        vid_pressures = json.load(f)
    with open('tests/99dec-ro-verysynth-colswitchextra.0/intervideos/fit_vid_pressures.json') as f:
        fit_vid_pressures = json.load(f)
        
    joanne_readings = []
    tech_readings = []
    with open('../../yue/joanne/GAT SL videos/reproducibility.csv') as f:
        started_reading = False
        for line in f:
            pieces = line.rstrip().split(',')
            if not started_reading:
                if pieces[0]!='5':
                    continue
                started_reading = True
            if not pieces:
                continue
            joanne_readings.append(float(pieces[2]))
            joanne_readings.append(float(pieces[3]))
            tech_readings.append(float(pieces[4]))
            tech_readings.append(float(pieces[5]))
        
    joanne_readings = np.array(joanne_readings, dtype=np.float64)
    tech_readings = np.array(tech_readings, dtype=np.float64)
        
    joanne_infers = np.zeros(50)
    tech_infers = np.zeros(50)
    for k,v in fit_vid_pressures.items():
        if k[0]=='J':
            arr = joanne_infers
            numstart = 1
        elif k[:4]=='Tech':
            arr = tech_infers
            numstart = 4
        else:
            raise KeyError(k)
        if k[-1]=='D':
            offs = 0
        elif k[-1]=='S':
            offs = 1
        else:
            raise KeyError(k)
        a = int(k[numstart:-3])
        assert a>=5 and a<30
        ind = 2*(a-5)+offs
        assert arr[ind]==0
        arr[ind] = v
    assert np.all(joanne_infers>0) and np.all(tech_infers>0)
    
    all_readings = np.concatenate([joanne_readings,tech_readings])
    all_infers = np.concatenate([joanne_infers,tech_infers])
    
    
    fig, ax = plt.subplots(figsize=(4,3))
    diffs = all_infers-all_readings
    ax.scatter((all_readings+all_infers)/2,diffs,label='Difference')
    diff_mean = np.mean(diffs)
    sd_mean = np.std(diffs)
    left, right = 0,30
    ax.set_xlabel('Mean of Automated and GAT (mmHg)')
    ax.set_ylabel('Automated - GAT (mmHg)')
    print(diff_mean, diff_mean-1.96*sd_mean, diff_mean+1.96*sd_mean)
    ax.plot([left,right],[diff_mean,diff_mean],'--',label='Mean difference')
    ax.plot([left,right],[diff_mean+1.96*sd_mean,diff_mean+1.96*sd_mean],'--',label='Mean+1.96*sd',color='orange')
    ax.plot([left,right],[diff_mean-1.96*sd_mean,diff_mean-1.96*sd_mean],'--',label='Mean-1.96*sd',color='orange')
    ax.set_xlim(left,right)
    ax.set_ylim(-10,10)
    fig.savefig('inter_b-a_fit.svg', bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(4,3))
    diffs = tech_infers-joanne_infers
    ax.scatter((tech_infers+joanne_infers)/2,diffs,label='Difference')
    diff_mean = np.mean(diffs)
    sd_mean = np.std(diffs)
    left, right = 0,30
    ax.set_xlabel('Mean of Video 1 and 2 (mmHg)')
    ax.set_ylabel('Video 1 - Video 2 (mmHg)')
    print(diff_mean, diff_mean-1.96*sd_mean, diff_mean+1.96*sd_mean)
    ax.plot([left,right],[diff_mean,diff_mean],'--',label='Mean difference')
    ax.plot([left,right],[diff_mean+1.96*sd_mean,diff_mean+1.96*sd_mean],'--',label='Mean+1.96*sd',color='orange')
    ax.plot([left,right],[diff_mean-1.96*sd_mean,diff_mean-1.96*sd_mean],'--',label='Mean-1.96*sd',color='orange')
    ax.set_xlim(left,right)
    ax.set_ylim(-10,10)
    fig.savefig('retest_auto_b-a.svg', bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(4,3))
    diffs = tech_readings-joanne_readings
    r = np.random.RandomState(777)
    tech_readingsr = tech_readings + r.randn(50)*0.1
    joanne_readingsr = joanne_readings + r.randn(50)*0.1
    ax.scatter((tech_readingsr+joanne_readingsr)/2,tech_readingsr-joanne_readingsr,label='Difference')
    diff_mean = np.mean(diffs)
    sd_mean = np.std(diffs)
    left, right = 0,30
    ax.set_xlabel('Mean of Observer 1 and 2 (mmHg)')
    ax.set_ylabel('Observer 1 - Observer 2 (mmHg)')
    print(diff_mean, diff_mean-1.96*sd_mean, diff_mean+1.96*sd_mean)
    ax.plot([left,right],[diff_mean,diff_mean],'--',label='Mean difference')
    ax.plot([left,right],[diff_mean+1.96*sd_mean,diff_mean+1.96*sd_mean],'--',label='Mean+1.96*sd',color='orange')
    ax.plot([left,right],[diff_mean-1.96*sd_mean,diff_mean-1.96*sd_mean],'--',label='Mean-1.96*sd',color='orange')
    ax.set_xlim(left,right)
    ax.set_ylim(-10,10)
    fig.savefig('retest_gat_b-a.svg', bbox_inches='tight')
    plt.close(fig)
    