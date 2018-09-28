import os

import torch
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
import skimage.transform
import cv2

import pspnet
from pspnet import PSPNet

import pickle
import logging
import warnings
import math

import sys
sys.path.insert(0, "../pressure-seg")
import getdata

import json

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

videolist = ['58', '61', '62', '65', '66', '69', '71']
corners = 'shuffle'
num_classes = 3

class Diceloss():
    '''
    Dice loss for this specific problem
    To ensure apples to apples comparison to the regression loss, measures (1+2) dice loss
    and (2) dice loss
    '''
    def __init__(self, epsilon=0.001, size_average=True, reduce = True):
        self.epsilon = epsilon
        self.size_average = size_average
        self.reduce = reduce
    def __call__(self, out, y, logprobs = True, inpres = None, inprobs = None):
        if logprobs:
            out = torch.exp(out)
        if inprobs is not None:
            inprobs = inprobs.view(-1,1,1)
            out = torch.stack((out[:,0],out[:,1] + out[:,2]*(1-inprobs),out[:,2]*inprobs),1)
        ylens = (y>=1).float()
        outlens = torch.sum(out[:,1:3],1)
        lensdice = (2*bisum(ylens*outlens)+self.epsilon)/(bisum(ylens)+bisum(outlens)+self.epsilon)
        yin = (y==2).float()
        outin = out[:,2]
        indice = (2*bisum(yin*outin)+self.epsilon)/(bisum(yin)+bisum(outin)+self.epsilon)
        if inpres is not None:
            indice = indice*inpres + torch.ones_like(indice)*(1-inpres)
        dicetot = 2-(lensdice + indice)
        if not self.reduce:
            return dicetot
        if self.size_average:
            divisor = dicetot.size()[0]
        else:
            divisor = 1
        return torch.sum(dicetot)/divisor

def bisum(x):
    return torch.sum(torch.sum(x,1),1)

def makenames(models_path, validations_path, name_suffix)
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
            validation_path='_'.join(['../validations',validate_on,type]) + name_suffix
    return models_path, validation_path
    
def build_network(snapshot, backend, start_lr, milestones, gamma = 0.1):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
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
            _, epoch = os.path.basename(snapshot).split('_')
            epoch = int(epoch)
            net.load_state_dict(load(snapshot))
            scheduler = MultiStepLR(optimizer, milestones = [x - epoch for x in milestones], gamma = gamma)
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    else:
        scheduler = MultiStepLR(optimizer, milestones = milestones, gamma = gamma)
    net = net.cuda()
    return net, optimizer, epoch, scheduler

def movwrite(net, val_on, data_path, save_path):
    set1 = [[28,26,228], [184, 126, 55], [74,175,77]]
    colors = torch.ones(216,256,1,1, dtype=torch.long)*torch.tensor(set1)
    dat = getdata.Im_Dataset(data_path, val_on)
    if(len(dat)==0):
        return
    dat_loader = DataLoader(dat)
    dat_iterator = tqdm(dat_loader)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter(save_path, fourcc, 25, (808, 256))
    for name, x in dat_iterator:
        xv = Variable(x).cuda()
        out, out_cls = net(xv)
        out = torch.exp(out)
        inprob = torch.sigmoid(out_cls[0,2])
        out[:,1] += out[:,2]*(1-inprob)
        out[:,2] *= inprob
        outc = out.detach().cpu()[0]
        segs = torch.unsqueeze(torch.unsqueeze(torch.argmax(outc, 0),2),3)*torch.ones(3, dtype=torch.long)
        im_arr = (x[0]*256).numpy().astype('uint8')
        im_arr = np.flip(im_arr,0).transpose(1,2,0)
        out_arr = torch.squeeze(torch.gather(colors, 2, segs),2).numpy().astype('uint8')
        frame = np.full((256, 808, 3), 255, np.dtype('uint8'))
        frame[10:226, 10:266] = im_arr
        frame[10:226, 276:532] = out_arr
        frame[10:226, 542:798] = cv2.addWeighted(im_arr, 0.6, out_arr, 0.4, 0)
        cv2.putText(frame, f'{name[0]}', (246, 248), 0, 0.8, (0,0,0))
        vid.write(frame)
    vid.release()
    cv2.destroyAllWindows()
    
def validate(net, val_dat, epoch, alpha, validation_path = None, class_weights = None): 
    net.eval()
    val_loader = DataLoader(val_dat)
    #seg_criterion = nn.NLLLoss(weight=class_weights)
    seg_criterion = Diceloss()
    cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    val_iterator = tqdm(val_loader)
    results = []
    val_losses = []
    if(validation_path != None):
        os.makedirs(validation_path, exist_ok=True)
    for name, x, y, y_cls, *a in val_iterator:
        if corners=='together':
            x, y = torch.cat(*x, 0), torch.cat(*y, 0)
            y_cls = torch.max(torch.stack(*y_cls), 0)[0]
        if(validation_path is not None):
            im = x[0].numpy().transpose(1,2,0)
        x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()
        out, out_cls = net(x)
        mults = torch.ones_like(y_cls)
        if corners == 'shuffle':
            mults = (1-a[0]).cuda()
            if class_weights is not None:
                corr = torch.mean(torch.log(2-mults)*class_weights)
            else:
                corr = torch.mean(torch.log(2-mults))
        if corners == 'together':
            out_cls = torch.max(out_cls.view(-1, 4, num_classes), 1)[0]
        #seg_loss, cls_loss = seg_criterion(out, y, inprobs=torch.sigmoid(out_cls[:,2])), cls_criterion(out_cls, y_cls)
        seg_loss, cls_loss = seg_criterion(out, y, inpres=y_cls[:,2]), cls_criterion(out_cls*mults, y_cls*mults)
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
        if(validation_path is not None):
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
            #imarr = io.imread(os.path.join('../joanne/joanne_seg_manual/', name[0] +'.png'))
            #transim = skimage.transform.rescale(imarr[:,320:1600], 0.2)
            ax1.set_title(name[0])
            ax1.imshow(im)
            out = torch.exp(out)
            inprob = torch.sigmoid(out_cls[0,2])
            out[:,1] += out[:,2]*(1-inprob)
            out[:,2] *= inprob
            ax2.set_title(' '.join([name[0], 'output']))
            ax2.imshow(torch.argmax(out[0], 0), cmap='Set1', norm = NoNorm())
            #temp = torch.zeros_like(out[0])
            #temp.index_copy_(0, torch.tensor([0,2,1]).cuda(), out[0])
            #ax1.imshow(torch.exp(temp).detach().transpose(0,1).transpose(1,2))
            ax3.set_title(' '.join([name[0], 'truth']))
            ax3.imshow(y[0], cmap='Set1', norm = NoNorm())
            fig.text(0.5, 0.05, f'Loss={loss}', horizontalalignment='center', verticalalignment='bottom')
            if corners=='shuffle':
                outname = '_'.join([name[0], str(a[1].item())])
            else:
                outname = name[0]
            fig.savefig(os.path.join(validation_path, '.'.join([outname,'png'])))
            plt.close(fig)
            results.append((name[0], loss.item()))
    if(validation_path != None):
        f = open(os.path.join(validation_path,'results'), 'w')
        f.write('\n'.join([f'{name}, {loss}' for (name, loss) in results]))
        f.close()
        val_loss = np.mean(val_losses)
        print(f'Validation loss: {val_loss}')
    return np.mean(val_losses)


@click.command()
@click.option('--data-path', type=str, default="", help='Path to dataset folder')
@click.option('--models-path', type=str, default=None, help='Path for storing model snapshots')
@click.option('--backend', type=str, default='resnet34', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=256, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=256, help='Vertical random crop size')
@click.option('--batch-size', type=int, default=4)
@click.option('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
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
def train(data_path, models_path, backend, snapshot, crop_x, crop_y, batch_size,
          alpha, epochs, start_lr, milestones, gpu, train_prop, validation_path,
          shuffle, validate_freq, validate_on, name_suffix):
    models_path, validations_path = makenames(models_path, validations_path, name_suffix)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    models_path = os.path.abspath(os.path.expanduser(models_path))
    validation_path = os.path.abspath(os.path.expanduser(validation_path))
    os.makedirs(models_path, exist_ok=True)
    
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
    net, optimizer, starting_epoch, scheduler = build_network(snapshot, backend, start_lr, milestones)
    #scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])
    #scheduler = ExponentialLR(optimizer, 0.8)
    
    
    if snapshot is None:
        (train_dat, val_dat) = getdata.splitset("../joanne/joanne_seg_manual/", "../joanne/true_avg_circles.json",
                                                train_prop, shuffle = shuffle, validate_on = validate_on, tame = 'yes',
                                                corners = corners)
    else:
        f = open('_'.join([snapshot, 'set']), 'rb')
        (train_dat, val_dat) = pickle.load(f)
        train_dat.updateparams()
        val_dat.updateparams()
        f.close()
    train_loader = DataLoader(train_dat, shuffle=True, batch_size = batch_size)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(starting_epoch, starting_epoch + epochs):
        #seg_criterion = nn.NLLLoss(weight=class_weights)
        seg_criterion = Diceloss()
        cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        epoch_losses = []
        train_iterator = tqdm(train_loader)
        scheduler.step()
        net.train()
        for name, x, y, y_cls, *a in train_iterator:
            optimizer.zero_grad()
            if corners=='together':
                x, y = torch.cat(*x, 0), torch.cat(*y, 0)
                y_cls = torch.max(torch.stack(*y_cls), 0)[0]
            x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()
            out, out_cls = net(x)
            mults = torch.ones_like(y_cls)
            if corners == 'shuffle':
                mults = (1-a[0]).cuda()
                if class_weights is not None:
                    corr = torch.mean(torch.log(2-mults)*class_weights)
                else:
                    corr = torch.mean(torch.log(2-mults))
            if corners == 'together':
                out_cls = torch.max(out_cls.view(-1, 4, num_classes), 1)[0]
            #seg_loss, cls_loss = seg_criterion(out, y, inprobs=torch.sigmoid(out_cls[:,2])), cls_criterion(out_cls, y_cls)
            seg_loss, cls_loss = seg_criterion(out, y, inpres=y_cls[:,2]), cls_criterion(out_cls*mults, y_cls*mults)
            if corners == 'shuffle':
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
            val_losses.append(validate(net, val_dat, epoch, alpha, class_weights = class_weights))
        
    val_losses.append(validate(net, val_dat, starting_epoch + epochs - 1, alpha, validation_path, class_weights))    
    return {'train':train_losses, 'validate':val_losses}

if __name__ == '__main__':
    train()
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #net, optimizer, starting_epoch, scheduler = build_network('snaps_29_50sdz/PSPNet_22', 'resnet50', 0.001, '10,20,30')
    #for x in videolist:
    #    for y in ['OD','OS']:
    #        movwrite(net, '_'.join([x,y]), '../../yue/joanne/video_frames_test/', '_'.join(['segs',x,y])+'.avi')