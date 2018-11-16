import os
import json
import logging

import torch
from torch import save, load
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm, trange
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

from pspnet import QuadCombine

import sys
sys.path.insert(0, "../pressure-seg")
import getdata

def _allcornersin(k, dict):
    return all([('_'.join([k,str(n)]) in dict) for n in range(4)])

def getkeys(corners, truths):
    return [k for k in truths.keys() if _allcornersin(k,corners)]
    
def maketwo(cl, **kwargs):
    kwargs0 = {**{arg:val for arg,val in kwargs.items() if arg[-2:]!='_0' and arg[-2:]!='_1'},
               **{arg[:-2]:val for arg,val in kwargs.items() if arg[-2:]=='_0'}}
    kwargs1 = {**{arg:val for arg,val in kwargs.items() if arg[-2:]!='_0' and arg[-2:]!='_1'},
               **{arg[:-2]:val for arg,val in kwargs.items() if arg[-2:]=='_1'}}
    return cl(**kwargs0), cl(**kwargs1)
    
class Combiner_Dataset(Dataset):
    def __init__(self, cornerdir, truthdir, keys=None, goodinds = None):
        with open(cornerdir) as f:
            self.corners = json.load(f)
        with open(truthdir) as f:
            self.truths = json.load(f)
        if keys is None:
            self.keys = getkeys(self.corners, self.truths)
        else:
            self.keys = keys
        if goodinds==None:
            goodinds = range(len(list(corners.values())[0]))
        self.goodinds = goodinds
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        key = self.keys[idx]
        inval_lists = [self.corners['_'.join([key,str(n)])] for n in range(4)]
        if self.goodinds:
            ind = np.random.choice(self.goodinds)
        else:
            ind = np.random.randint(len(inval_lists[0]))
        invals = [l[ind] for l in inval_lists]
        intens = torch.tensor([out+outc for (out, outc) in invals]).view(36)
        outval = self.truths[key]
        outtens = torch.tensor(outval['lens_data']+outval['inner_data'])
        return key, standardizequarters(intens), outtens

def standardizequarters(intens):
    subtens = torch.tensor([320.,0.,0.,320.,0.,0.])
    wtens = torch.tensor([512.,0.,0.,512.,0.,0.])
    htens = torch.tensor([0.,432.,0.,0.,432.,0.])
    pcs = torch.split(intens, [6,3,6,3,6,3,6,3])
    return torch.cat([torch.abs(pcs[0])/0.4+subtens,pcs[1],
                      torch.abs(wtens-pcs[2])/0.4+subtens,pcs[3],
                      torch.abs(htens-pcs[4])/0.4+subtens,pcs[5],
                      torch.abs(wtens+htens-pcs[6])/0.4+subtens,pcs[7]])
        
def averager(intens):
    pcs = torch.split(intens, [9,9,9,9], 1)
    return (pcs[0]+pcs[1]+pcs[2]+pcs[3])/4
    
        
truthdir = '../joanne/true_avg_circles.json'
im_dir = '../joanne/joanne_seg_manual'
traincornerdir = '../joanne/validations_29_50numcorners/Training_results_long.json'
valcornerdir = '../joanne/validations_29_50numcorners/Validation_results_long.json'

train_dat = Combiner_Dataset(traincornerdir, truthdir, goodinds = range(5,15))
val_dat = Combiner_Dataset(valcornerdir, truthdir, goodinds = [4])

num_epochs = 40
models_path = 'snaps_29_50numcorners_combo_short'
val_path = '../joanne/validations_29_50numcorners_combo_short'
print('Validations going to', val_path)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print(f'Using CUDA {os.environ["CUDA_VISIBLE_DEVICES"]}:')

net = QuadCombine()
net = nn.DataParallel(net)
optimizer = optim.Adam(net.parameters())
scheduler = ExponentialLR(optimizer, 0.9)
os.makedirs(models_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
loss_fn = getdata.Circles_Dice()
cls_loss_fn = nn.BCEWithLogitsLoss()
results = []

for epoch in trange(num_epochs):
    scheduler.step()
    for mode in ['train', 'validate']:
        if mode == 'train':
            loader = DataLoader(train_dat, shuffle=True, batch_size = 4, drop_last=True)
            noun = 'Training'
            displr = f'{scheduler.get_lr()[0]:0.6f}'
            net.train()
        else:
            loader = DataLoader(val_dat, batch_size = 1)
            noun = 'Validation'
            displr = 'n/a'
            net.eval()
        data_iterator = tqdm(loader)
        epoch_losses = []
        cls_improvs = []
        seg_improvs = []
        for key, x, y in data_iterator:
            if mode == 'train':
                optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            out = net(x)
            
            # Postprocess
            av = averager(x)
            out = out+torch.cat([av[:,0:6], av[:,8:9]], 1)
            
            cls_base = cls_loss_fn(av[:,8], (y[:,5]>0).float())
            seg_base = loss_fn(av[:,:6], y)
            
            cls_loss = cls_loss_fn(out[:,6], (y[:,5]>0).float())
            seg_loss = loss_fn(out[:,:6], y)
            loss = seg_loss + cls_loss
            cls_improv = cls_base-cls_loss
            seg_improv = seg_base-seg_loss
            epoch_losses.append(loss.data.item())
            cls_improvs.append(cls_improv.data.item())
            seg_improvs.append(seg_improv.data.item())
            status = f'[{epoch}] - {noun} - loss = {loss.data.item():8.5f} - avg = {np.mean(epoch_losses):0.5f} - vs base = {np.mean(cls_improvs):0.5f} {np.mean(seg_improvs):0.5f} - LR = {displr}'
            data_iterator.set_description(status)
            if mode == 'train':
                loss.backward()
                optimizer.step()
            if mode == 'validate' and epoch==num_epochs-1:
                # Draw stuff
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7,7))
                imgname = os.path.join("../joanne/joanne_seg_manual/", '.'.join([key[0],"png"]))
                im = io.imread(imgname)
                im = im[:,320:1600]
                ax1.set_title(key[0])
                ax1.imshow(im)
                h, w = im.shape[0:2]
                adjust = torch.tensor([320.,0.,0.])
                yc = y.detach().cpu()[0]
                trues = getdata.circfill(yc[:3]-adjust, yc[3:6]-adjust, w, h)[0]
                ax2.set_title(' '.join([key[0], 'truth']))
                ax2.imshow(trues, cmap='Set1', norm = NoNorm())
                outc = out.detach().cpu()[0]
                if(outc[6]<0):
                    outs = getdata.circfill(outc[:3]-adjust, [0,0,0], w, h)[0]
                else:
                    outs = getdata.circfill(outc[:3]-adjust, outc[3:6]-adjust, w, h)[0]
                ax3.set_title(' '.join([key[0], 'output']))
                ax3.imshow(outs, cmap='Set1', norm = NoNorm())
                avc = av.detach().cpu()[0]
                if(avc[8]<0):
                    avs = getdata.circfill(avc[:3]-adjust, [0,0,0], w, h)[0]
                else:
                    avs = getdata.circfill(avc[:3]-adjust, avc[3:6]-adjust, w, h)[0]
                ax4.set_title(' '.join([key[0], 'averaging']))
                ax4.imshow(avs, cmap='Set1', norm = NoNorm())
                fig.text(0.5, 0.05, f'Loss={loss}', horizontalalignment='center', verticalalignment='bottom')
                fig.savefig(os.path.join(val_path, '.'.join([key[0],'png'])))
                plt.close(fig)
                results.append(loss.data.item())
        tqdm.write(f'Epoch {epoch} {noun.lower()} loss {np.mean(epoch_losses):0.5f} improvements class {np.mean(cls_improvs):0.5f} seg {np.mean(seg_improvs):0.5f}')
    model_state = {'model': net.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch+1}
    save(model_state, os.path.join(models_path, '_'.join(["Combinations", str(epoch + 1)])))
with open(os.path.join(val_path,'results'), 'w') as f:
    f.write('\n'.join([str(x) for x in results]))