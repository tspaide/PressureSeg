from pathlib import Path
import cv2
import re
import numpy as np
from skimage import io
from tqdm import tqdm
from PIL import Image
import json

shopping_list = {}
frame_path = Path('/data')/'yue'/'joanne'/'all_videos_seg'/'manual_seg'/'ryan'

with open('ryan_smartphone_segs.json') as f:
    dat = json.load(f)
    
for l in dat:
    vname = ' '.join(l.split('_')[:-1])
    framenum = int(l.split('_')[-1][5:-4])
    if vname not in shopping_list:
        shopping_list[vname] = []
    shopping_list[vname].append(framenum)
    
for v in shopping_list.values():
    v.sort()
    
vidpath = Path('/data')/'yue'/'joanne'/'videos'

vidlist = []
for folder in vidpath.iterdir():
    if 'Goldman' in folder.name:
        continue
    vidlist = vidlist + list(folder.iterdir())
    
for vidname, framelist in tqdm(shopping_list.items()):
    possible_videos = [v for v in vidlist if vidname in v.name]
    if len(possible_videos) == 2:
        possible_videos = [v for v in possible_videos if 'low flou' not in v.name and 'Francy' not in v.name]
    assert len(possible_videos)==1
    vid = possible_videos[0]
    cap = cv2.VideoCapture(str(vid))
    prev_frame = 0
    cap.grab()
    qbar = tqdm(framelist)
    for frame_num in qbar:
        qbar.set_description(f'frame {frame_num}')
        for _ in range(frame_num-prev_frame):
            cap.grab()
        ret, inframe = cap.retrieve()
        inframe = inframe.transpose(1,0,2)
        #inframe = np.flip(inframe,2)
        cv2.imwrite(str(Path('smartphone_frames')/f'{vidname.replace(" ","_")}_frame{frame_num}.png'),inframe)
        prev_frame = frame_num
        
print('')