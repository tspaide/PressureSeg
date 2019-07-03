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

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from skimage import io
import PIL

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

plt.switch_backend('agg')

CIRCLE_DATA_PATH = 'goldmann_measurements.json'
IMAGE_BASE_PATH = '../../yue/joanne/GAT SL videos'
HOLDOUT_SET = ['40-J34-OS','10-J49-OD','21-J48-OS','11-F35-OS','55-F18-OD','19-F55-OD']
GOLDMANN_OUTER_DIAM = 7

models = {
    'squeezenet': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet', w=w, h=h),
    'densenet': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet', w=w, h=h),
    'resnet18': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', w=w, h=h),
    'resnet34': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34', w=w, h=h),
    'resnet50': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50', w=w, h=h, extraend=True,out_nums=9),
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
    
def maybefake(p=0.5):
    def keyadj(key):
        if np.random.sample()<p:
            return ('fake','fake','fake')
        return key
    return keyadj
    
class MaybeFaker():
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, key):
        if np.random.sample()<self.p:
            return ('fake','fake','fake')
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
    
def im_only(key):
    return io.imread(key), [1,1,1,1,1,1,1]
    #return io.imread(key), [1,1,1,1,1,1,1,1,1,1] # Trying midline segmentation
        
def fakesonly(key):
    return goldsketch.goldmann_fake()
    
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
    
def saveorig(info, in_arr, out_arr, name='orig'):
    info[name] = arrToTens(in_arr)
    return in_arr, out_arr
    
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
    
def randRotate(angle = 15, maxangle = None, resizing = False):
    return constinfo(getdata.RandRotateP(angle, maxangle, resizing))
    
def resize(dim):
    return constinfo(getdata.ResizeP(dim))
    
def flip():
    return constinfo(getdata.FlipP())
    
def pad(x,y):
    return constinfo(getdata.PadP(x,y))
    
def package_goldcoords(info, dat):
    im, coords = dat
    info['isbot']=int(coords[6])
    return im,coords[:3],coords[3:6]
    #return im,coords[:3],coords[3:6],coords[7:10] if len(coords)>7 else [1.,1.,1.] # Trying midline segmentation
    
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
        im, coords = dat
        dat = self.transformer(info, dat)
        im, lens_data, inner_data = dat
        #im, lens_data, inner_data, line_dat = dat # Trying midline segmentation
        classes = np.array((1,1,int(inner_data[2]!=0)),dtype=np.float32)
        return info, im, np.concatenate([lens_data, inner_data]).astype(np.float32), classes
        #return info, im, np.concatenate([lens_data, inner_data, line_dat]).astype(np.float32), classes # Trying midline segmentation
    
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

def dividefolds(pic_shuffle=True, seed=None, valprop=0.2, num_out=3):
    if seed:
        pic_shuffle=True
    vids = []
    with open(CIRCLE_DATA_PATH) as f:
        circle_data = json.load(f)
    for foldname in circle_data:
        for subfold in circle_data[foldname]:
            if subfold[:-4] in HOLDOUT_SET:
                continue
            vids.append([(foldname,subfold,x) for x in circle_data[foldname][subfold]])
    
    r = np.random.RandomState(seed)
    validxs = r.choice(len(vids), size=int(len(vids)*valprop), replace=False)
    isval = np.zeros(len(vids), dtype=np.bool)
    for n in validxs:
        isval[n]=1
        
    valfolds = [v for n,v in enumerate(vids) if isval[n]]
    testfolds = [v for n,v in enumerate(vids) if not isval[n]]
    r.shuffle(valfolds)
    if type(num_out) is slice:
        outfolds = valfolds[num_out]
    else:
        outfolds = valfolds[:num_out]

    testkeys = [k for f in testfolds for k in f]
    valkeys = [k for f in valfolds for k in f]
    outkeys = [k for f in outfolds for k in f]
    
    return testkeys,valkeys,outkeys
    
def csvout(val_path, training_curve, epoch_losses, datsize):
    with open(os.path.join(val_path,'Training curve.csv'), 'w') as f:
        for n in range(len(training_curve['train'])):
            f.write(f"{training_curve['train'][n]},{training_curve['validate'][n]}")
            if (n % datsize == 0) and (n/datsize < len(epoch_losses['validate'])):
                f.write(f",{epoch_losses['train'][n//datsize]},{epoch_losses['validate'][n//datsize]}")
            f.write('\n')
            
def smoothed_tc(outloc, training_curve, alpha=0.999):
    tclen = min(len(training_curve['train']),len(training_curve['validate']))
    jointtc = np.array([training_curve['train'][:tclen],training_curve['validate'][:tclen]])
    tcmeans = np.empty_like(jointtc)
    curr = np.zeros_like(jointtc[...,0])
    ct = 0
    for m in range(tclen):
        curr, ct = curr*alpha+jointtc[...,m], ct*alpha+1
        tcmeans[...,m] = curr/ct
    fig, ax = plt.subplots()
    ax.plot(tcmeans[0,tclen//2:])
    ax.plot(tcmeans[1,tclen//2:])
    ymin, ymax = ax.get_ylim()
    ax.clear()
    ax.plot(tcmeans[0], linewidth=0.5, label='Training')
    ax.plot(tcmeans[1], linewidth=0.5, label='Validation')
    ax.set_ylim(ymin,ymax)
    ax.legend()
        
    plt.savefig(os.path.join(outloc, 'Training_curve.png'))
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
  
def testcycle(outloc, setloc = None, nettype='UNet', position=None, outmode=False, loadloc=None, saveevery=None, start_epoch=0):
    trainkeys,valkeys,outkeys = dividefolds(seed=int(outloc.split('.')[1]), num_out=5)
    
    testtransform = composetransforms([package_goldcoords, constinfo(getdata.ToPil()), randRotate(resizing=True),
                                       randResizedCrop(256,256,0.25,0.5,circle_fullness=0.95), constinfo(getdata.ToTens())])
                                       
    outtransform = composetransforms([package_goldcoords, constinfo(getdata.ToPil()), constCrop(177,673,945,1441),
                                      resize(256), constinfo(getdata.ToTens())])
    
    testlocs = ['../../yue/joanne/GAT SL videos/other_techs/raw/F07-OD','../../yue/joanne/GAT SL videos/other_techs/raw/I01-OS',
                '../../yue/joanne/GAT SL videos/other_techs/raw/I03-OD','../../yue/joanne/GAT SL videos/other_techs/raw/I06-OS']
    testkeys = [os.path.join(testloc,f) for testloc in testlocs for f in os.listdir(testloc)]
    
    with open(CIRCLE_DATA_PATH) as f:
        coordindex = json.load(f)
    
    #testkeys = testkeys + [('fake','fake','fake')]*(len(testkeys)//5)  #Trying adding synthetic data
    outkeys = outkeys + [('fake','fake','fake')]*5
    
    train_dat, val_dat, out_dat, inf_dat = maken(ImtoNumsDataset, 4,
                                                 keys_0=trainkeys,
                                                 keys_1=valkeys,
                                                 keys_2=outkeys,
                                                 keys_3=testkeys, 
                                                 keyadj_0=MaybeFaker(0.5),
                                                 datlookup=lookup_switch((lambda x: x[0]=='fake'), fakesonly, get_golddat(coordindex)), # Trying adding synthetic data
                                                 #datlookup_01=get_golddat(coordindex),
                                                 datlookup_3=im_only,
                                                 transformer_01=testtransform,
                                                 transformer_23=outtransform)
    
    if outmode and setloc is None:
        setloc = outloc
    
    #out_dat.keys = outkeys
    
    train_loader = DataLoader(train_dat, shuffle=True, batch_size = 4, num_workers = 6, worker_init_fn=setseed)
    val_loader = DataLoader(val_dat, shuffle=True, batch_size = 2, num_workers = 4, worker_init_fn=setseed)
    out_loader = DataLoader(out_dat, batch_size=2, num_workers = 2, worker_init_fn=setseed)
    
    max_epoch = 100
    if outmode:
        start_epoch = max_epoch
        
    net = models['resnet50'](32,32)
    
    net = nn.DataParallel(net)
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    scheduler = ExponentialLR(optimizer, 0.95)
    net = net.cuda()
    start_epoch='curr'
    if start_epoch and (max_epoch!=start_epoch):
        if start_epoch=='curr' or start_epoch=='prev':
            start_epoch_name = start_epoch
            model_state = load(os.path.join(outloc, nettype+f"_{start_epoch}"))
            start_epoch = model_state['epoch']+1
            print(f'Loading {nettype}_{start_epoch_name} (epoch {start_epoch-1})')
        else:
            prevs = [int(match.group(1)) for match in (re.fullmatch(nettype+'_(\d*)',s) for s in os.listdir(outloc)) if match]
            if start_epoch-1 in prevs:
                print(f'Loading {nettype}_{start_epoch-1}')
                model_state = load(os.path.join(outloc, nettype+f"_{start_epoch-1}"))
            else:
                test_model_state = load(os.path.join(outloc, nettype+"_curr"))
                last_epoch = test_model_state['epoch']
                got_model = False
                if last_epoch==start_epoch-1:
                    print(f'Loading {nettype}_curr')
                    model_state = test_model_state
                    got_model = True
                elif last_epoch==start_epoch:
                    prev_model_state = load(os.path.join(outloc, nettype+"_prev"))
                    if prev_model_state['epoch']==start_epoch-1:
                        print(f'Loading {nettype}_prev')
                        model_state = prev_model_state
                        got_model = True
                    else:
                        print(f'Epoch mismatch in {nettype}_prev')
                if not got_model:
                    prev_candidates = [p for p in prevs if p<start_epoch]
                    bestprev = max(prev_candidates) if prev_candidates else -1
                    curr_usable = (last_epoch<start_epoch)
                    if not prev_candidates and not curr_usable:
                        print('No loadable point found; starting from scratch')
                        start_epoch = 0
                    elif not curr_usable or bestprev>last_epoch:
                        print(f'Loading {nettype}_{bestprev}')
                        model_state = load(os.path.join(outloc, nettype+f"_{bestprev}"))
                        start_epoch = bestprev+1
                    else:
                        print('Loading {nettype}_curr (epoch {last_epoch})')
                        model_state = test_model_state
                        start_epoch = last_epoch+1
    
    save_every = max(int(np.round(max_epoch/50)),1)*5
    
    autobest = True
    if 'tests' not in outloc:
        outloc = 'tests/' + outloc
        
    print('Validations going to', outloc)
    
    outputonly = False
    if(max_epoch==start_epoch):
        outputonly = True
        max_epoch+=1
    if(start_epoch>0):
        with open(os.path.join(outloc, 'Epoch_losses.json')) as f:
            epoch_losses = json.load(f)
        if not outputonly:
            prevbest = min(epoch_losses['validate'])
            if len(epoch_losses['train'])<start_epoch:
                warnings.warn(f"Training data only goes to epoch {len(epoch_losses['train'])-1}")
                epoch_losses['train'] += ['No data']*(start_epoch-len(epoch_losses['train']))
            if len(epoch_losses['validate'])<start_epoch:
                warnings.warn(f"Validation data only goes to epoch {len(epoch_losses['validate'])-1}")
                epoch_losses['validate'] += ['No data']*(start_epoch-len(epoch_losses['validate']))
            if len(epoch_losses['train'])>start_epoch:
                warnings.warn("Previous data had data beyond the current starting epoch.  Truncating; please see 'Epoch_losses.json.old' "
                              "for previous data")
                with open(os.path.join(outloc, 'Epoch_losses.json.old'),'w') as f:
                    json.dump(epoch_losses, f)
                epoch_losses['train'] = epoch_losses['train'][0:start_epoch]
                epoch_losses['validate'] = epoch_losses['validate'][0:start_epoch]
    else:
        epoch_losses={'train':[],'validate':[]}
        os.makedirs(outloc, exist_ok=True)
        with open(os.path.join(outloc, '_'.join([nettype, 'dat'])), 'wb') as f:
            dill.dump({'train_dat':train_dat, 'val_dat':val_dat, 'out_dat':out_dat}, f)
        prevbest = float('inf')
    
    seg_criterion = getdata.Circles_Dice()
    class_weights = None
    cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    line_loss = nn.L1Loss()
    
    if not outputonly:
        validationsource = infiniloader(val_dat, shuffle=True, batch_size=2, num_workers=2, worker_init_fn=setseed)
        next(validationsource)
        training_curve = {'train':[],'validate':[]}
        
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
            modes.append('inference')
            if outputonly:
                modes = ['inference']
        for mode in modes:
            if mode == 'train':
                train_dat.keyadj.p = 0.1+0.8*(99-epoch)/99
                loader = train_loader
                noun = 'Training'
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
                if autobest:
                    model_state = load(os.path.join(outloc, nettype+"_best"))
                    net.load_state_dict(model_state['model'])
                    tqdm.write(f"Outputting using best model: epoch {model_state['epoch']}")
                    autobest=False
                net.eval()
                torch.autograd.set_grad_enabled(False)
                results = []
                pressures = {}
                numfakes = 0
            elif mode == 'inference':
                loader = DataLoader(inf_dat, batch_size=2, num_workers = 2, worker_init_fn=setseed)
                os.makedirs(os.path.join(outloc,'inferences'), exist_ok=True)
                noun = 'Inference'
                if autobest:
                    model_state = load(os.path.join(outloc, nettype+"_best"))
                    net.load_state_dict(model_state['model'])
                    tqdm.write(f"Inference using best model: epoch {model_state['epoch']}")
                    autobest=False
                net.eval()
                torch.autograd.set_grad_enabled(False)
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
                out, out_cls = net(ims)
                seg_loss = seg_criterion(out[:,:9],coords[:,:6],switch=info['isbot']) # Trying midline segmentation
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
                    
                    '''
                    ms = torch.tan(coords[:,6:7]) #Trying midline segmentation
                    line_loc = torch.cat((ms,coords[:,8:9]-ms*coords[:,7:8]),1)
                    for n in range(out.size(0)):
                        if info['key'][2][n]=='fake':
                            loss = loss + line_loss(out[n,9:],line_loc[n])/(3*out.size(0))
                    '''
                    
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        net.eval()
                        testinfo, testims, testcoords, testclasses = next(validationsource)
                        testims, testcoords, testclasses = testims.cuda(), testcoords.cuda(), testclasses.cuda()
                        testout, testout_cls = net(testims)
                        test_seg_loss = seg_criterion(testout[:,:9],testcoords[:,:6],switch=testinfo['isbot'])
                        test_cls_loss = cls_criterion(testout_cls, testclasses)
                        testloss = test_seg_loss + test_cls_loss
                        training_curve['validate'].append(testloss.data.item())
                        net.train()
                if mode == 'output' or mode=='inference':
                    imsc = ims.detach().cpu().numpy().transpose(0,2,3,1)
                    outc = out.detach().cpu().numpy()
                    out_clsc = out_cls.detach().cpu().numpy()
                    yy, xx = np.mgrid[0:imsc.shape[1],0:imsc.shape[2]]
                    for n in range(ims.size(0)):
                        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
                        ax1.imshow(imsc[n])
                        ax1.axis('off')
                        outr = ((xx-outc[n,0])**2+(yy-outc[n,1])**2)**0.5
                        if out_clsc[n,2]>0:
                            inra = ((xx-outc[n,3])**2+(yy-outc[n,4])**2)**0.5
                            inrb = ((xx-outc[n,6])**2+(yy-outc[n,7])**2)**0.5
                        else:
                            inra = np.ones_like(outr)*65535
                            inrb = inra
                        #tophalf = (yy<outc[n,9]*xx+outc[n,10]) #Trying midline segmentation
                        #bothalf = (yy>outc[n,9]*xx+outc[n,10]) #Trying midline segmentation
                        outfill = (outr<abs(outc[n,2])).astype(np.uint8)
                        #infill = ((inra<abs(outc[n,5])).astype(np.uint8)*bothalf+(inrb<abs(outc[n,8])).astype(np.uint8)*tophalf)*outfill #Trying midline segmentation
                        infill = ((inra<abs(outc[n,5])).astype(np.uint8)+(inrb<abs(outc[n,8])).astype(np.uint8))*outfill
                        #midline = (abs(yy-outc[n,9]*xx-outc[n,10])<1.5).astype(np.uint8)&outfill #Trying midline segmentation
                        outbord = (abs(outr-abs(outc[n,2]))<1.5).astype(np.uint8)
                        #inbord = ((abs(inra-abs(outc[n,5]))<1.5).astype(np.uint8)*tophalf+(abs(inrb-abs(outc[n,8]))<1.5).astype(np.uint8)*bothalf)*outfill #Trying midline segmentation
                        inbord = ((abs(inra-abs(outc[n,5]))<1.5).astype(np.uint8)+(abs(inrb-abs(outc[n,8]))<1.5).astype(np.uint8))*outfill
                        #ax2.imshow(outfill+infill+midline, cmap='Set1', norm = NoNorm()) #Trying midline segmentation
                        ax2.imshow(outfill+infill, cmap='Set1', norm = NoNorm())
                        ax2.axis('off')
                        #ax3.imshow((imsc[n]*255).astype(np.uint8)|((outbord|inbord|midline)*255)[:,:,np.newaxis]) #Trying midline segmentation
                        ax3.imshow((imsc[n]*255).astype(np.uint8)|((outbord|inbord)*255)[:,:,np.newaxis])
                        ax3.axis('off')
                        if mode == 'inference':
                            fig.savefig(os.path.join(outloc,'inferences',info['key'][n].split('/')[-1]))
                            plt.close(fig)
                            continue
                        if info['key'][2][n]=='fake':
                            fig.savefig(os.path.join(outloc,f'synthetic{numfakes}.png'))
                            plt.close(fig)
                            numfakes += 1
                            continue
                        fig.savefig(os.path.join(outloc, info['key'][2][n]))
                        plt.close(fig)
                        ldiam = abs(outc[n,5]/outc[n,2])*GOLDMANN_OUTER_DIAM
                        liop = 168.694/ldiam**2 if out_clsc[n,2]>0 else 0
                        rdiam = abs(outc[n,8]/outc[n,2])*GOLDMANN_OUTER_DIAM
                        riop = 168.694/rdiam**2 if out_clsc[n,2]>0 else 0
                        segdiam = (coords[n,5]/coords[n,2]).detach().item()*GOLDMANN_OUTER_DIAM
                        segiop = 168.694/segdiam**2 if classes[n,2]>0 else 0
                        iops = [liop if (outc[n,0]-outc[n,3])**2+(outc[n,1]-outc[n,4])**2
                                         <(outc[n,0]-outc[n,6])**2+(outc[n,1]-outc[n,7])**2 else riop,
                                segiop]
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
        with open(os.path.join(outloc,'results'), 'w') as f:
            f.write('\n'.join([str(x) for x in results]))
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
            ax.legend()
            bot, top = ax.get_ylim()
            ax.set_ylim(top=min(top,50))
            fig.savefig(os.path.join(outloc,vid+' pressures.png'))
            plt.close(fig)
            with open(os.path.join(outloc,vid+' pressures.csv'),'w') as f:
                for frame, pres in zip(frames,iops[0]):
                    f.write(f'{frame},{pres}\n')
            
            
    if not outputonly:
        try:
            validationsource.send(1)
        except StopIteration:
            pass
  
if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    print(f'Using CUDA {os.environ["CUDA_VISIBLE_DEVICES"]}:')
    
    testcats = ['tests/decsynthdata-colswitch.0']
    
    for testcat in testcats:
        testcycle(testcat,outmode=False)