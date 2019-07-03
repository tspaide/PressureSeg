import numpy as np
import json
import warnings
from skimage import io
import json

import logging
logging.getLogger('imageio').setLevel(logging.ERROR)

try:
    with open('colors.json') as f:
        COLOR_SETS = json.load(f)
except:
    COLOR_SETS = [[153,255,255,3,20,255,12,19,222,30,143,255,5,182,170,10,107,169]]
COLOR_SETS = np.array(COLOR_SETS)

try:
    with open('extracolors.json') as f:
        EXTRA_COLOR_SETS = json.load(f)
except:
    EXTRA_COLOR_SETS = [[21,0,241,17,72,182,17,39,228,34,0,247,21,255,169,23,255,195]]
EXTRA_COLOR_SETS = np.array(EXTRA_COLOR_SETS)

# Recorded values: x 598-621, y 1017-1055, m .003-.007, outer r ~ 259, inner r 75-140?, innter thickness 20-30?

def initstyle_sketch(innerprob=0.3, resprob=0.4, size=256, noise=0.05, padding=0):
    p = np.random.random_sample()
    if p<innerprob:
        drawinnercirc = True
        residue = False
    elif p<innerprob+resprob:
        drawinnercirc = False
        residue = True
    else:
        drawinnercirc = False
        residue = False

    # Plausible-looking parameters
    basesize = 800
    base_outer_r = 259
    outer_x = (300+np.random.random_sample()*200)*size/basesize + padding
    outer_y = (300+np.random.random_sample()*200)*size/basesize + padding
    outer_r = (200+np.random.random_sample()*100)*size/basesize
    outer_r_mult = outer_r/base_outer_r
    outer_thickness_top = (20+np.random.random_sample()*3)*outer_r_mult
    top_displacement = 10*(np.random.random_sample()**0.6)*outer_r_mult
    top_disp_angle = 2*np.pi*np.random.random_sample()
    top_x = outer_x + top_displacement*np.cos(top_disp_angle)
    top_y = outer_y + top_displacement*np.sin(top_disp_angle)
    outer_thickness_bot = (20+np.random.random_sample()*3)*outer_r_mult
    bot_displacement = 10*(np.random.random_sample()**0.6)*outer_r_mult
    bot_disp_angle = 2*np.pi*np.random.random_sample()
    bot_x = outer_x + bot_displacement*np.cos(bot_disp_angle)
    bot_y = outer_y + bot_displacement*np.sin(bot_disp_angle)
    shift = (120+np.random.random_sample()*10)*outer_r_mult
    if drawinnercirc or residue:
        base_inner_r = (75+np.random.random_sample()*65)
        inner_r = base_inner_r*outer_r_mult
        inner_thickness = (base_inner_r/10+12.5+np.random.random_sample()*4)*outer_r_mult
        inner_displacement = 80*(np.random.random_sample()**0.6)*outer_r_mult
        inner_disp_angle = 2*np.pi*np.random.random_sample()
        inner_x = outer_x + inner_displacement*np.cos(inner_disp_angle)
        inner_y = outer_y + inner_displacement*np.sin(inner_disp_angle)
    else:
        inner_r, inner_x, inner_y, inner_thickness = 1, None, None, None
    line_angle = 0.4*np.random.random_sample()-0.2
    return sketch(size+2*padding,size+2*padding,line_angle,outer_x,outer_y,outer_r,
                  top_x,top_y,outer_thickness_top,bot_x,bot_y,outer_thickness_bot,shift,
                  inner_x,inner_y,inner_r,inner_thickness,drawinnercirc,residue,noise,circshiftmult=1.4)
    
def sketch(width,height,line_angle,outer_x,outer_y,outer_r,
           top_x,top_y,outer_thickness_top,bot_x,bot_y,outer_thickness_bot,shift,
           inner_x,inner_y,inner_r,inner_thickness,drawinnercirc,residue,noise=0.05,circshiftmult=1,color_mode=0,coltypes=COLOR_SETS):
    # Draw circles and lines
    yy,xx,_ = np.mgrid[0:height,0:width,0:1]
    tophalf = (yy-outer_y<(np.tan(line_angle)*(xx-outer_x)))
    bothalf = (yy-outer_y>(np.tan(line_angle)*(xx-outer_x)))
    outr2 = (xx-outer_x)**2+(yy-outer_y)**2
    outcirc = (outr2<outer_r**2)
    midline = (np.abs((yy-outer_y)-(np.tan(line_angle)*(xx-outer_x)))<1)*outcirc
    topcirc = ((xx-top_x)**2+(yy-top_y)**2<(outer_r-outer_thickness_top)**2)
    botcirc = ((xx-bot_x)**2+(yy-bot_y)**2<(outer_r-outer_thickness_bot)**2)
    xshift = shift*np.cos(line_angle)
    yshift = shift*np.sin(line_angle)
    topr2 = (xx-outer_x+xshift*circshiftmult)**2+(yy-outer_y+yshift*circshiftmult)**2
    botr2 = (xx-outer_x-xshift*circshiftmult)**2+(yy-outer_y-yshift*circshiftmult)**2
    topshiftcirc = topr2<outer_r**2
    botshiftcirc = botr2<outer_r**2
    if drawinnercirc or residue:
        topinr2 = (xx-inner_x+xshift)**2+(yy-inner_y+yshift)**2
        botinr2 = (xx-inner_x-xshift)**2+(yy-inner_y-yshift)**2
    else:
        topinr2 = 0
        botinr2 = 0
    if drawinnercirc:
        topinner = (topinr2>inner_r**2)*(topinr2<(inner_r+inner_thickness)**2)*topcirc*topshiftcirc
        botinner = (botinr2>inner_r**2)*(botinr2<(inner_r+inner_thickness)**2)*botcirc*botshiftcirc
        topinfill = (topinr2<(inner_r-inner_thickness)**2)*topcirc*topshiftcirc
        botinfill = (botinr2<(inner_r-inner_thickness)**2)*botcirc*botshiftcirc
    else:
        topinner, botinner, topinfill, botinfill = None, None, None, None
    if residue:
        topres = (topinr2<(inner_r/2)**2)*topcirc*topshiftcirc
        botres = (botinr2<(inner_r/2)**2)*botcirc*botshiftcirc
    else:
        topres, botres = None, None
        
      
    # Colors
    if color_mode=='rand':
        color_mode = np.random.randint(coltypes.shape[0])
    if color_mode=='randcomb':
        wts = np.random.dirichlet(np.ones(coltypes.shape[0]))*1.1-0.1/coltypes.shape[0]
        colors = np.matmul(wts,coltypes)
    else:
        colors = coltypes[color_mode]
        colors = colors+np.random.normal(size=18)*10
    outbordercol = np.array(colors[:3]).reshape(1,1,3)
    outcol = np.array(colors[3:6]).reshape(1,1,3)
    midcol = np.array(colors[6:9]).reshape(1,1,3)
    sidecol = np.array(colors[9:12]).reshape(1,1,3)
    inringcol = np.array(colors[12:15]).reshape(1,1,3)
    infillcol = np.array(colors[15:18]).reshape(1,1,3)
    rescol = infillcol
      
    # Synthesize
    outcircborder = outcirc*((1-topcirc)*tophalf+(1-botcirc)*bothalf)
    sides = tophalf*topcirc*(1-topshiftcirc)+bothalf*botcirc*(1-botshiftcirc)
    outdist = (outr2**0.5)
    indist = (topr2**0.5)*tophalf+(botr2**0.5)*bothalf
    featdist = (topinr2**0.5)*tophalf+(botinr2**0.5)*bothalf
    
    top = (1-outcirc)*outcol+topcirc*(1-topshiftcirc)*sidecol+outcirc*(1-topcirc)*outbordercol+topcirc*topshiftcirc*midcol
    bot = (1-outcirc)*outcol+botcirc*(1-botshiftcirc)*sidecol+outcirc*(1-botcirc)*outbordercol+botcirc*botshiftcirc*midcol
    if drawinnercirc:
        top = (1-topinner-topinfill)*top+topinner*inringcol+topinfill*infillcol
        bot = (1-botinner-botinfill)*bot+botinner*inringcol+botinfill*infillcol
        central_feature = bothalf*(botinner+botinfill)+tophalf*(topinner+topinfill)
    elif residue:
        top = (1-topres)*top+topres*rescol
        bot = (1-botres)*bot+botres*rescol
        central_feature = bothalf*botres+tophalf*topres
    else:
        central_feature = np.zeros_like(sides)
        
    out = (top*tophalf + bot*bothalf)
    if noise>0:
        out = np.clip(out + noise*255*(np.random.normal(size=out.shape)),0,255)
    gen_info = {'lx':outer_x, 'ly':outer_y, 'lr':outer_r, 'shift':shift, 'ix':inner_x, 'iy':inner_y, 'ir':inner_r,
                'inner':drawinnercirc, 'residue':residue}
    masks = {'outcirc':outcircborder, 'tophalf':tophalf, 'bothalf':bothalf, 'outdist':outdist, 'indist':indist, 'midline':midline,
             'central_feature':central_feature, 'sides':sides, 'featdist':featdist}
    return out, gen_info, masks
    
def goldmann_fake(innerprob=3885/5099, extra=False):
    top_x, top_y = 552+np.random.normal()*67, 1072+np.random.normal()*55
    line_angle = 0.05*np.random.normal()
    top_r = 246+np.random.normal()*10
    outer_thickness_top = 20+np.random.normal()*5
    outer_r = top_r+outer_thickness_top
    outer_thickness_bot = 10+np.random.normal()*2
    top_displacement = np.random.sample()*outer_thickness_top
    top_disp_angle = line_angle+0.05*np.random.normal()
    bot_displacement = np.random.sample()*outer_thickness_bot
    bot_disp_angle = -line_angle+0.05*np.random.normal()
    outer_x = top_x-top_displacement*np.cos(top_disp_angle)
    outer_y = top_y-top_displacement*np.sin(top_disp_angle)
    bot_x = outer_x+bot_displacement*np.cos(bot_disp_angle)
    bot_y = outer_y+bot_displacement*np.sin(bot_disp_angle)
    shift = (0.45+0.01*np.random.normal())*top_r
    if np.random.random_sample()<innerprob:
        drawinnercirc = True
        inner_x_disp = np.random.normal()*0.22*top_r
        inner_y_disp = np.random.normal()*0.15*top_r
        inner_x = top_x + inner_x_disp
        inner_y = top_y + inner_y_disp
        inner_r = top_r*np.random.gamma(15,0.03)
        inner_thickness = 20+np.random.sample()*2
    else:
        drawinnercirc = False
        inner_r, inner_x, inner_y, inner_thickness = 0, 0, 0, 0
    if extra:
        coltypes = EXTRA_COLOR_SETS
    else:
        coltypes = COLOR_SETS
    im,gen_info,masks = sketch(1080,1920,line_angle,outer_x,outer_y,outer_r,
                               top_x,top_y,outer_thickness_top,bot_x,bot_y,outer_thickness_bot,shift,
                               inner_x,inner_y,inner_r,inner_thickness,drawinnercirc,False,noise=0.01,circshiftmult=1.4,
                               color_mode='rand',coltypes=coltypes)
    shadow_range = 0.1+np.random.sample()*3
    shadow_mask = np.clip((masks['outdist']/outer_r-1.2)/shadow_range,0,1)
    if np.random.random_sample()<0.5: # Other ways to do this?
        switch = 1
    else:
        switch = 0
    if not drawinnercirc:
        ix,iy,ir = 0,0,0
    elif switch:
        ix,iy,ir = inner_x+shift*np.cos(line_angle),inner_y+shift*np.sin(line_angle),inner_r
    else:
        ix,iy,ir = inner_x-shift*np.cos(line_angle),inner_y-shift*np.sin(line_angle),inner_r
    return (im*(1-shadow_mask)).astype(np.uint8), (top_x,top_y,top_r,ix,iy,ir,switch)
    #return (im*(1-shadow_mask)).astype(np.uint8), (top_x,top_y,top_r,ix,iy,ir,switch,line_angle,outer_x,outer_y) # Trying midline segmentation
    
if __name__=='__main__':
    quit()
    for n in range(5):
        im, coords = goldmann_fake(extra=n%2)
        yy,xx,_ = np.mgrid[0:im.shape[0],0:im.shape[1],0:1]
        outx,outy,outr,inx,iny,inr,s,la,ox,oy = coords
        m = np.tan(la)
        b = oy-m*ox
        outcirc = (abs(((xx-outx)**2+(yy-outy)**2)**0.5-outr)<2).astype(np.uint8)
        if s:
            valreg = (yy>m*xx+b)
        else:
            valreg = (yy<m*xx+b)
        midline = (abs(yy-m*xx-b)<2).astype(np.uint8)
        if inr:
            incirc = (abs(((xx-inx)**2+(yy-iny)**2)**0.5-inr)<2).astype(np.uint8)*valreg
        else:
            incirc = np.zeros_like(outcirc)
        im = im.astype(np.uint8)|(outcirc*np.array([255,0,0],dtype=np.uint8))|(incirc*np.array([0,255,255],dtype=np.uint8))|(midline*np.array([255,255,255],dtype=np.uint8))
        print(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(f'fakegoldex{n}.png', im, check_contrast=False)
        