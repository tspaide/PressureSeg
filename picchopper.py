import math, os, cv2, re
import numpy as np
import json
from PIL import Image

goldpath = '../goldmann'
linepath = '../goldmann/lineims'
savepath = '../goldmann/cutims'

def findline(filename):
    linecolor = (36,28,237)
    im = cv2.imread(os.path.join(linepath, filename))
    xs = []
    ys = []
    lineinds = np.nonzero((im == linecolor).all(2))
    if len(lineinds[0])==0:
        raise ValueError(f'Cannot find line in {filename}')
    xs = [[x,1] for x in lineinds[1]]
    m,b = np.linalg.lstsq(xs,lineinds[0], rcond=None)[0]
    xcenter, ycenter = np.mean(lineinds[1]), np.mean(lineinds[0])
    xmin,xmax = min(lineinds[1]),max(lineinds[1])
    return m,b,xcenter,ycenter,int(xmin),int(xmax)
    
def getline(filename, linelist = None)
    if linelist is None or filename not in linelist:
        return findline(filename)
    return linelist[filename]
    
def cut(filename):
    m,b,xcenter,ycenter,xmin,xmax = getline(filename)
    im_arr = cv2.imread(os.path.join(goldpath, filename))
    angle = np.arctan(m)
    h,w = im_arr.shape[:2]
    image = Image.fromarray(np.flip(im_arr,2))
    image = image.rotate(math.degrees(angle), center=(xcenter, ycenter))
    image.crop((0,0,w,ycenter)).save(os.path.join(savepath, filename[:-4]+'_top.png'))
    image.crop((0,ycenter,w,h)).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(savepath, filename[:-4]+'_bot.png'))
    
def savelines():
    cutinfo = []
    for filename in os.listdir(linepath):
        cutinfo.append((filename, findline(filename)))
    with open(os.path.join(goldpath, 'linedata.json'),'w') as f:
        json.dump(cutinfo, f)

