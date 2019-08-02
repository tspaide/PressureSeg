import os
import json
import numpy as np
from skimage import io
from tqdm import tqdm

def circfit(ptlist, ycheck=None, tol=0.01):
    x1, y1, x2, y2, x3, y3 = [float(s) for s in ptlist]
    # Can throw np.linalg.LinAlgError
    a,b = np.linalg.solve([[2*(x1-x2),2*(y1-y2)],[2*(x1-x3),2*(y1-y3)]],[x1**2-x2**2+y1**2-y2**2,x1**2-x3**2+y1**2-y3**2])
    r = ((x1-a)**2+(y1-b)**2)**0.5
    assert abs(r**2-(x2-a)**2-(y2-b)**2)<tol
    assert abs(r**2-(x3-a)**2-(y3-b)**2)<tol
    if ycheck is not None:
        sw = (np.mean((y1,y2,y3))>ycheck)
        return a,b,r,sw
    return a,b,r
    
def _sqtol(r2, rad, tolerance=2):
    return ((r2<(rad+tolerance)**2)&(r2>max(rad-tolerance,0)**2)).astype(np.uint8)
    
bpath = '2miresegs'

with open('goldmann_measurements.json') as f:
    prevdat = json.load(f)

with open('switchmatch_adjusts.json') as f:
    adjusts = json.load(f)
    
sings = []
keymisses = []
dat = {}
outsings = 0
in1sings = 0
in2sings = 0
switchmatches = 0
    
yy,xx,_ = np.mgrid[0:1920,0:1080,0:1]
   
for spath in os.listdir(bpath):
    section_dat = {}
    dat[spath] = section_dat
    path = os.path.join(bpath,spath)
    if 'j' in spath:
        imsection = 'goldmann_new'
    else:
        imsection = 'goldmann_shu'
    for fname in os.listdir(path):
        if '04-J02-OD-color' in fname and 'v2' not in fname:
            continue
        print(fname)
        shortname = fname.replace('_v2','').replace(' (1)','')
        folder_dat = {}
        section_dat[shortname] = folder_dat
        if shortname not in prevdat[spath] or 'v2' in fname:
            print(f"Starting over with {shortname}")
            overwrite = True
        else:
            overwrite = False
            foldrec = prevdat[spath][shortname]
        fpath = os.path.join(path,fname)
        lines = []
        with open(fpath,encoding='utf-8-sig') as f:
            for line in f:
                lines.append(line.rstrip().split(','))
        newkeys = set()
        for l in lines:
            if len(l)<2 or not l[1]:
                continue
            if l[0][0]=='\'':
                l[0] = l[0][1:]
            if l[0][-1]=='\'':
                l[0] = l[0][:-1]
            if '04-J02-OD-color' in fname:
                if l[0][0]=='2':
                    l[0] = '04-J0' + l[0]
            newkeys.add(l[0])
            if overwrite or l[0] not in foldrec:
                if len(l)<13 or not l[12] or any(float(c)<0 for c in l[1:13]):
                    continue
                try:
                    a,b,r = circfit(l[1:7])
                except (np.linalg.LinAlgError, AssertionError):
                    sings.append(' - '.join([spath,fname,l[0],'outer']))
                    outsings += 1
                    continue
                entry = [a,b,r]
                if all(s=='0' for s in l[7:13]):
                    a,b,r = 0,0,0
                    sw=False
                else:
                    try:
                        a,b,r,sw = circfit(l[7:13],b)
                    except (np.linalg.LinAlgError, AssertionError):
                        sings.append(' - '.join([spath,fname,l[0],'inner 1']))
                        in1sings += 1
                        a,b,r,sw = 0,0,0,False
                entry = entry + [a,b,r,int(sw)]
            else:
                entry = foldrec[l[0]]
            outcirc = entry[0:3]
            firstincirc = entry[3:6]
            firstswitch = int(entry[6])
            
            if len(l)>13 and l[13] and any(float(c) for c in l[13:19]):
                try:
                    a,b,r,sw = circfit(l[13:19],entry[1])
                except (np.linalg.LinAlgError, AssertionError):
                    sings.append(' - '.join([spath,fname,l[0],'inner 2']))
                    in2sings += 1
                    a,b,r,sw = 0,0,0,False
            else:
                a,b,r,sw = 0,0,0,False
            secondincirc = [a,b,r]
            secondswitch = int(sw)
            if firstswitch==secondswitch and firstincirc[2] and secondincirc[2]:
                if l[0] in adjusts:
                    try:
                        a,b,r,sw = circfit(adjusts[l[0]][1:],entry[1])
                        if int(adjusts[l[0]][0]):
                            secondincirc = [a,b,r]
                            secondswitch = int(sw)
                        else:
                            firstincirc = [a,b,r]
                            firstswitch = int(sw)
                    except (np.linalg.LinAlgError, AssertionError):
                        sings.append(' - '.join([spath,fname,l[0],'adjusted inner']))
                        continue
                sings.append(' - '.join([spath,fname,l[0],'switchmatch']))
                switchmatches += 1
                firstswitch = int(firstincirc[0]>secondincirc[0])
                secondswitch = 1-firstswitch
            if firstswitch:
                firstincirc, secondincirc = secondincirc, firstincirc
            folder_dat[l[0]] = outcirc + firstincirc + secondincirc + [firstswitch]
        if not overwrite:
            oldkeys = set(foldrec.keys())
            holdouts = oldkeys-newkeys
            if holdouts:
                print(f'copying {len(holdouts)} from previous data')
            for k in holdouts:
                entry = foldrec[k]
                outcirc = entry[0:3]
                firstincirc = entry[3:6]
                switch = int(entry[6])
                secondincirc = [0,0,0]
                if switch:
                    firstincirc, secondincirc = secondincirc, firstincirc
                folder_dat[l[0]] = outcirc + firstincirc + secondincirc + [switch]


with open('goldmann_measurements_2.json','w') as f:
    json.dump(dat,f)

    