import os

import torch
import torch.autograd
from torch import save, load
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from skimage import io
import skimage
import cv2
from PIL import Image

import pspnet
from pspnet import PSPNet, PSPCircs

import pickle
import logging
import warnings
import math
import json

import sys
sys.path.insert(0, "../pressure-seg")
import getdata

models = {
    'squeezenet': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet', w=w, h=h),
    'densenet': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet', w=w, h=h),
    'resnet18': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', w=w, h=h),
    'resnet34': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34', w=w, h=h),
    'resnet50': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50', w=w, h=h),
    'resnet101': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101', w=w, h=h),
    'resnet152': lambda h, w: PSPCircs(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152', w=w, h=h)
}
                
videolist = ['58', '61', '62', '65', '66', '69', '71']
num_classes = 3
    
def accumulate_values(dict1, dict2):
    for (k,v) in dict2.items():
        if k in dict1:
            dict1[k].append(v)
        else:
            dict1[k] = [v]
    return dict1
    
def makenames(models_path, validation_path, snapshot, validate_on, backend, name_suffix, prev_epoch):
    if models_path is None and name_suffix is None:
        raise ValueError('Please supply a models-path or name-suffix value')
    if name_suffix is not None:
        if backend == 'squeezenet':
            type = 'squeeze'
        else:
            type = backend[6:]
        if models_path is None:
            models_path='_'.join(['snaps',validate_on,type]) + name_suffix
        if validation_path is None:
            validation_path='_'.join(['../joanne/validations',validate_on,type]) + name_suffix
    if prev_epoch is not None and snapshot is None:
        snapshot = os.path.join(models_path, '_'.join(['PSPNet', str(prev_epoch)]))
    return models_path, validation_path, snapshot
    
def build_network(snapshot, backend, start_lr, milestones, gamma = 0.1, oldmode = False):
    epoch = 0
    backend = backend.lower()
    net = models[backend](27, 32)
    net = nn.DataParallel(net)
    optimizer = optim.Adam(net.parameters(), lr=start_lr)
    milestones=[int(x) for x in milestones.split(',')]
    if snapshot is not None:
        old = load(snapshot)
        try:
            epoch = old['epoch']
            net.load_state_dict(old['model'])
            optimizer.load_state_dict(old['optimizer'])
            scheduler = MultiStepLR(optimizer, milestones = milestones, gamma = gamma, last_epoch = epoch-1)
        except KeyError:
            net = models[backend](27, 32)
            _, epoch = os.path.basename(snapshot).split('_')
            epoch = int(epoch)
            net.load_state_dict(load(snapshot))
            scheduler = MultiStepLR(optimizer, milestones = [x - epoch for x in milestones], gamma = gamma)
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    else:
        scheduler = MultiStepLR(optimizer, milestones = milestones, gamma = gamma)
        #scheduler = ExponentialLR(optimizer, 0.8)
    net = net.cuda()
    return net, optimizer, epoch, scheduler

def framereader(video_path, dispname):
    dummyset = getdata.Im_Dataset("./", "asdwef")
    transform = dummyset.transform # This is bad and I should feel bad
    cap = cv2.VideoCapture(video_path)
    ret, inframe = cap.read()
    framenum = 0
    while(ret):
        name = ' '.join([dispname, 'frame', str(framenum)])
        inframe = inframe.astype(float)/256
        inframe, _, _ = transform((inframe, [960,540,0], [0,0,0]))
        inframe = np.flip(inframe,2).copy()
        x = torch.tensor(inframe.transpose(2,0,1), dtype=torch.float)
        x = torch.unsqueeze(x,0)
        yield name, x
        ret, inframe = cap.read()
        framenum +=1
    
def movwrite(net, val_on=None, data_path=None, save_path=None, video_path=None, dispname=None):
    set1 = [[28,26,228], [184, 126, 55], [74,175,77]]
    if (val_on is not None) and (data_path is not None):
        dat = getdata.Im_Dataset(data_path, val_on)
        if(len(dat)==0):
            return None
        dat_loader = DataLoader(dat)
        dat_iterator = tqdm(dat_loader)
        if(save_path is not None):
            dat_iterator.set_description(save_path)
    elif video_path is not None:
        if dispname is None:
            dispname = video_path
        dat_iterator = framereader(video_path, dispname)
    else:
        raise ValueError('movwrite needs either a val_on and data_path argument (for frame input) or '
                         'video_path argument (for video input)')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter('.'.join([save_path,'avi']), fourcc, 25, (808, 256))
    pressures = []
    results = []
    net.eval()
    torch.set_grad_enabled(False)
    for name, x in dat_iterator:
        xv = Variable(getdata.histeqtens(x)).cuda()
        out, out_cls = net(xv)
        outc = out.detach().cpu()[0]
        outc = torch.abs(outc)
        im_arr = (x[0]*256).numpy().astype('uint8')
        im_arr = np.flip(im_arr,0).transpose(1,2,0)
        lr = outc[2]
        if(out_cls[0,2]>0.5):
            out_arr = np.array(getdata.circfill(outc[:3], outc[3:], width=256, height=216, style=set1, classes=False)).astype('uint8')
            circover = np.array(getdata.circdraw(outc[:3], outc[3:], width=256, height=216, thickness=1, style=[[0,0,0],[255,255,255]], classes=False)).astype('uint8')
            ir = outc[5]
        else:
            out_arr = np.array(getdata.circfill(outc[:3], [0,0,0], width=256, height=216, style=set1, classes=False)).astype('uint8')
            circover = np.array(getdata.circdraw(outc[:3], [0,0,0], width=256, height=216, thickness=1, style=[[0,0,0],[255,255,255]], classes=False)).astype('uint8')
            ir = torch.tensor(0)
        frame = np.full((256, 808, 3), 255, np.dtype('uint8'))
        frame[10:226, 10:266] = im_arr
        frame[10:226, 276:532] = out_arr
        frame[10:226, 542:798] = cv2.max(im_arr, circover)
        cv2.putText(frame, f'{name[0]}', (246, 248), 0, 0.8, (0,0,0))
        vid.write(frame)
        
        if(out_cls[0,2]>0.5):
            ir = outc[5]
        else:
            ir = torch.tensor(0)
        pressures.append(getdata.calc_iop_from_circles(lr.item(),ir.item()))
        results.append(outc.numpy().tolist())
    torch.set_grad_enabled(True)
    fig, ax = plt.subplots()
    ax.plot(pressures)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Pressure (mm Hg)')
    ax.set_ylim(0,30)
    fig.savefig('.'.join([save_path,'png']))
    vid.release()
    cv2.destroyAllWindows()
    return pressures, results
    
def writeresults(net, dat, epoch, alpha, beta, class_weights = None, corners = None, oldres = None):
    newres = validate(net, dat, epoch, alpha, beta, class_weights = class_weights, corners=corners, writeout=True)
    if oldres is not None:
        newres = accumulate_values(oldres, newres)
    return newres
    
def validate(net, val_dat, epoch, alpha, beta, validation_path = None, class_weights = None, corners = None, writeout = None):
    net.eval()
    val_loader = DataLoader(val_dat)
    seg_criterion = getdata.Circles_Dice()
    cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    val_iterator = tqdm(val_loader)
    results = []
    val_losses = []
    if(validation_path != None):
        os.makedirs(validation_path, exist_ok=True)
    if writeout:
        calcres = {}
    for name, x, y, y_cls, *a in val_iterator:
        if(validation_path is not None):
            im = x[0].numpy().transpose(1,2,0)
            yc = y[0]
        x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()
        out, out_cls = net(x)
        outc, out_clsc=out.detach().cpu()[0], out_cls.detach().cpu()[0]
        if corners == 'shuffle':
            mults = (1-a[0]).cuda()
            out_cls = out_cls*mults
            y_cls = y_cls*mults
            y = torch.cat((y[:,0:2], y[:,2:3]*mults[:,1:2], y[:,3:5], y[:,5:6]*mults[:,2:3]), 1)
            if class_weights is not None:
                corr = torch.mean(torch.log(2-mults)*class_weights)
            else:
                corr = torch.mean(torch.log(2-mults))
        seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
        if corners == 'shuffle':
            cls_loss = cls_loss-corr
        loss = seg_loss + alpha * cls_loss
        val_losses.append(loss.data.item())
        if(validation_path == None):
            status = '[{0}Val] loss = {1:0.5f} avg = {2:0.5f}'.format(
                epoch + 1, loss.data.item(), np.mean(val_losses))
        else:
            status = '[{0}Output-Val] loss = {1:0.5f} avg = {2:0.5f}'.format(
                epoch + 1, loss.data.item(), np.mean(val_losses))
        val_iterator.set_description(status)
        if(validation_path != None):
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
            ax1.set_title(name[0])
            ax1.imshow(im)
            h, w = im.shape[0:2]
            if(out_clsc[2]<0):
                outs = getdata.circfill(outc[:3], [0,0,0], w, h)[0]
            else:
                outs = getdata.circfill(outc[:3], outc[3:], w, h)[0]
            trues = getdata.circfill(yc[:3], yc[3:], w, h)[0]
            ax2.set_title(' '.join([name[0], 'output']))
            ax2.imshow(outs, cmap='Set1', norm = NoNorm())
            #temp = torch.zeros_like(out[0])
            #temp.index_copy_(0, torch.tensor([0,2,1]).cuda(), out[0])
            #ax1.imshow(torch.exp(temp).detach().transpose(0,1).transpose(1,2))
            ax3.set_title(' '.join([name[0], 'truth']))
            ax3.imshow(trues, cmap='Set1', norm = NoNorm())
            fig.text(0.5, 0.05, f'Loss={loss}', horizontalalignment='center', verticalalignment='bottom')
            if corners=='shuffle':
                outname = '_'.join([name[0], str(a[1].item())])
            else:
                outname = name[0]
            fig.savefig(os.path.join(validation_path, '.'.join([outname,'png'])))
            plt.close(fig)
            results.append((outname, yc, outc, loss.data.item()))
        if writeout:
            if corners=='shuffle':
                outname = '_'.join([name[0], str(a[1].item())])
            else:
                outname = name[0]
            calcres[outname] = (outc.numpy().tolist(), out_clsc.numpy().tolist())
    if(validation_path != None):
        f = open(os.path.join(validation_path,'results'), 'w')
        f.write('\n'.join([f'({name}, {y}, {out}, {loss})' for (name, y, out, loss) in results]))
        f.close()
        val_loss = np.mean(val_losses)
        print(f'Validation loss: {val_loss}')
    if writeout:
        return calcres
    else:
        return np.mean(val_losses)
    

@click.command()
@click.option('--data-path', type=str, default=None, help='Path to dataset folder')
@click.option('--models-path', type=str, default=None, help='Path for storing model snapshots')
@click.option('--backend', type=str, default='resnet50', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=256, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=256, help='Vertical random crop size')
@click.option('--corners', type=str, default=None, help='Mode for splitting into quadrants')
@click.option('--batch-size', type=int, default=4)
@click.option('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
@click.option('--beta', type=float, default=1.0, help='Coefficient for cornerwise loss term')
@click.option('--epochs', type=int, default=20, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.001)
@click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
@click.option('--train-prop', type=float, default=0.8, help='Proportion of dataset used for training')
@click.option('--validation-path', type=str, default=None, help='If not None, path to store validation outputs')
@click.option('--shuffle', type=bool, default=True, help='Shuffle dataset before splitting')
@click.option('--validate-freq', type=int, default=1, help='Validation frequency')
@click.option('--validate-on', type=str, default=None, help='Validate on a particular subject, e.g. \'51\' or \'46OS\'; overrides other split modifiers')
@click.option('--name-suffix', type=str, default=None, help='Autogen path names with the given suffix')
@click.option('--prev-epoch', type=int, default=None, help='Previous epoch to load')

def ctrain(data_path='', models_path=None, backend='resnet50', snapshot=None, crop_x=256, crop_y=256, corners=None,
           batch_size=4, alpha=1.0, beta=1.0, epochs=20, start_lr=0.001, milestones='10,20,30', gpu='0', train_prop=0.8,
           validation_path=None, shuffle=True, validate_freq=1, validate_on=None, name_suffix=None, prev_epoch=None):
    models_path, validation_path, snapshot = makenames(models_path, validation_path, snapshot,
                                                       validate_on, backend, name_suffix, prev_epoch)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    models_path = os.path.abspath(os.path.expanduser(models_path))
    #validation_path = os.path.abspath(os.path.expanduser(validation_path))
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    if(corners == 's'):
        corners = 'shuffle'
    if(corners == 't'):
        corners = 'together'
    
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN FloatTensor y_cls) where (EDIT: y_cls needs to be a FloatTensor rather than a LongTensor)
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    train_loader, class_weights, n_images = None, None, None
    
    # optimizer = optim.Adam(net.parameters(), lr=start_lr)
    if snapshot is None:
        (train_dat, val_dat) = getdata.splitset("../joanne/joanne_seg_manual/", "../joanne/true_avg_circles.json",
                                                train_prop, shuffle = shuffle, validate_on = validate_on, tame = 'yes',
                                                corners = corners, coords = True)
    else:
        with open('_'.join([snapshot, 'set']), 'rb') as f:
            (train_dat, val_dat) = pickle.load(f)
        train_dat.updateparams()
        val_dat.updateparams()
        corners = train_dat.corners
    net, optimizer, starting_epoch, scheduler = build_network(snapshot, backend, start_lr, milestones)
    #scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])
    #scheduler = ExponentialLR(optimizer, 0.8)
    
    
    
    train_loader = DataLoader(train_dat, shuffle=True, batch_size = batch_size, drop_last = True)
    
    if(starting_epoch>0):
        try:
            with open(os.path.join(validation_path, 'Training_curve.json')) as f:
                prevres = json.load(f)
            train_losses = prevres['train'][:starting_epoch]
            val_losses = prevres['validate'][:starting_epoch]
        except FileNotFoundError:
            train_losses = [f'Results from epochs up to {starting_epoch} not found']
            val_losses = [f'Results from epochs up to {starting_epoch} not found']
    else:
        train_losses = []
        val_losses = []
    
    for epoch in range(starting_epoch, starting_epoch + epochs):
        seg_criterion = getdata.Circles_Dice()
        cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        reg = nn.L1Loss()
        epoch_losses = []
        train_iterator = tqdm(train_loader)
        scheduler.step()
        net.train()
        for name, x, y, y_cls, *a in train_iterator:
            optimizer.zero_grad()
            x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()
            out, out_cls = net(x)
            if corners == 'shuffle':
                mults = (1-a[0]).cuda()
                out_cls = out_cls*mults
                y_cls = y_cls*mults
                y = torch.cat((y[:,0:2], y[:,2:3]*mults[:,1:2], y[:,3:5], y[:,5:6]*mults[:,2:3]), 1)
                if class_weights is not None:
                    corr = torch.mean(torch.log(2-mults)*class_weights)
                else:
                    corr = torch.mean(torch.log(2-mults))
                
            seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
            #if corners == 'together':
                #seg_loss = beta*seg_loss + seg_criterion(out_tot, y[::4])
            if corners == 'shuffle' or corners == 'together':
                cls_loss = cls_loss - corr
            loss = seg_loss + alpha * cls_loss
            epoch_losses.append(loss.data.item())
            status = '[{0}] loss = {1:0.5f} avg = {2:0.5f}, LR = {3:0.7f}'.format(
                epoch + 1, loss.data.item(), np.mean(epoch_losses), scheduler.get_lr()[0])
            train_iterator.set_description(status)
            loss.backward()
            optimizer.step()
        
        model_state = {'model': net.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'epoch': epoch+1}
        save(model_state, os.path.join(models_path, '_'.join(["PSPNet", str(epoch + 1)])))
        # save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch + 1)])))
        f = open(os.path.join(models_path, '_'.join(["PSPNet", str(epoch + 1), 'set'])), 'wb')
        pickle.dump ((train_dat, val_dat), f)
        f.close()
        train_losses.append(np.mean(epoch_losses))
        if((epoch+1) % validate_freq == 0 and epoch != starting_epoch + epochs - 1):
            val_losses.append(validate(net, val_dat, epoch, alpha, beta, class_weights = class_weights, corners=corners))
        with open(os.path.join(validation_path, 'Training_curve.json'), 'w') as f:
            json.dump({'train':train_losses, 'validate':val_losses}, f)
        
    val_losses.append(validate(net, val_dat, starting_epoch + epochs - 1, alpha, beta,
                               validation_path, class_weights, corners))
    
    if epochs>0:
        with open(os.path.join(validation_path, 'Training_curve.json'), 'w') as f:
            json.dump({'train':train_losses, 'validate':val_losses}, f)
    return {'train':train_losses, 'validate':val_losses}
    
if __name__ == '__main__':
    #ctrain()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net, optimizer, starting_epoch, scheduler = build_network('snaps_29_50nums_histeq/PSPNet_13', 'resnet50', 0.001, '10,20,30')
    filepath = '../../yue/joanne/Ocular Manometry 31Oct2018/Eye 2 Ascending'
    files = os.listdir(filepath)
    for filename in files:
        if filename[3] == 'P':
            continue
        dispname = ' '.join(filename.split()[:3])
        pressures, results = movwrite(net, save_path = os.path.join('manometry', dispname),
                                      video_path = os.path.join(filepath, filename), dispname = dispname)
        #with open(f'manometry/{dispname}_results.json','w') as f:
        #    json.dump(results, f)
        #with open(f'manometry/{dispname}_pressures.json','w') as f:
        #    json.dump(pressures, f)