import torch
import math

'''
    Calculate (analytically) areas of (intersections of) circles, possibly contained in some rectangle.
    This system needs O(n^2) rules to handle n different kinds of shapes, so it's best not to extend it too much.
'''

def hclip(curve, left=None, right=None):
    if left is not None:
        curve = [a for a in curve if a['right']>left]
        if(len(curve)>0):
            curve[0]['left'] = left
    if right is not None:
        curve = [a for a in curve if a['left']<right]
        if(len(curve)>0):
            curve[-1]['right'] = right
    return curve

def circints(xa, ya, ra, xb, yb, rb):
    d2 = (xa-xb)**2 + (ya-yb)**2
    d = d2**0.5
    if(d>ra+rb):
        return []
    if(d<abs(ra-rb)):
        return []
    alpha = torch.acos((d2 + ra**2 - rb**2) / (2*d*ra))
    #gamma = torch.atan(torch.abs((yb-ya)/(xb-xa)))
    gamma = torch.atan((yb-ya)/(xb-xa))
    if(xb<xa):
        gamma = gamma + math.pi
    ans1 = (xa+ra*torch.cos(gamma+alpha), ya+ra*torch.sin(gamma+alpha))
    ans2 = (xa+ra*torch.cos(gamma-alpha), ya+ra*torch.sin(gamma-alpha))
    return [ans1,ans2]
    
def circfn(xc, y, r, dir, x):
    return y+dir*(r**2-(xc-x)**2)**0.5
    
def arcmax(arca, arcb):
    left = max(arca['left'], arcb['left'])
    right = min(arca['right'], arcb['right'])
    if left >= right:
        return []
    if(arca['type'] == 'hline'):
        if(arcb['type'] == 'hline'):
            y = max(arca['y'], arcb['y'])
            ansarc = {'type':'hline', 'left':left, 'right':right, 'y':y}
            return [ansarc]
        if(arcb['type']=='circ'):
            ya = arca['y']
            xb, yb, rb = arcb['x'], arcb['y'], arcb['r']
            if(arcb['dir']==-1):
                toplim = yb
                botlim = yb-rb
            else:
                toplim = yb+rb
                botlim = yb
            if(ya>=toplim):
                ansarc = arca.copy()
                ansarc['left'] = left
                ansarc['right'] = right
                return [ansarc]
            elif(ya<=botlim):
                ansarc = arcb.copy()
                ansarc['left'] = left
                ansarc['right'] = right
                return [ansarc]
            else:
                xl = xb - (rb**2 - (yb-ya)**2)**0.5
                xr = xb + (rb**2 - (yb-ya)**2)**0.5
                if(arcb['dir']==-1):
                    cl, cr = arcb.copy(), arcb.copy()
                    cc = arca.copy()
                else:
                    cl, cr = arca.copy(), arca.copy()
                    cc = arcb.copy()
                cl['right'] = xl
                cr['left'] = xr
                cc['left'], cc['right'] = xl, xr
                return hclip([cl, cc, cr], left, right)
                
    if(arca['type']=='circ'):
        if(arcb['type']=='hline'):
            return arcmax(arcb, arca)
        if(arcb['type'] == 'circ'):
            xa, ya, ra, dira = arca['x'], arca['y'], arca['r'], arca['dir']
            xb, yb, rb, dirb = arcb['x'], arcb['y'], arcb['r'], arcb['dir']
            if(xa == xb and ya == yb and ra == rb):
                if(dira==1):
                    ansarc = arca.copy()
                else:
                    ansarc = arcb.copy()
                ansarc['left']=left
                ansarc['right']=right
                return [ansarc]
            ints = circints(xa, ya, ra, xb, yb, rb)
            ints = [x for (x,y) in ints if dira*y>dira*ya and dirb*y>dirb*yb and x>left and x<right]
            ints.append(left)
            ints.sort()
            ints.append(right)
            tval = (left+ints[1])/2
            if(circfn(xa,ya,ra,dira,tval)>circfn(xb,yb,rb,dirb,tval)):
                minarc,maxarc = arcb,arca
            else:
                minarc,maxarc = arca,arcb
            ans=[]
            for n in range(len(ints)-1):
                ansarc = maxarc.copy()
                ansarc['left']=ints[n]
                ansarc['right']=ints[n+1]
                ans.append(ansarc)
                minarc, maxarc = maxarc, minarc
            return ans
    raise ValueError(f"Intersection for arcs {arca} and {arcb} not written")
            
def arcneg(arc):
    if(arc['type']=='hline'):
        ans = arc.copy()
        ans['y']=-arc['y']
        return ans
    if(arc['type']=='circ'):
        ans = arc.copy()
        ans['y']=-arc['y']
        ans['dir']=-arc['dir']
        return ans
            
def curveneg(curve):
    return [arcneg(arc) for arc in curve]
            
def curvemax(curvea, curveb):
    i,j = 0,0
    ans=[]
    while(i<len(curvea) and j<len(curveb)):
        if(curvea[i]['right']<=curveb[j]['left']):
            i += 1
            continue
        if(curvea[i]['left']>=curveb[j]['right']):
            j += 1
            continue
        left = max(curvea[i]['left'], curveb[j]['left'])
        right = min(curvea[i]['right'], curveb[j]['right'])
        ans.extend(hclip(arcmax(curvea[i], curveb[j]), left, right))
        if(curvea[i]['right']>curveb[j]['right']):
            j+=1
        else:
            i+=1
    return ans
    
def curvemin(curvea, curveb):
    return curveneg(curvemax(curveneg(curvea), curveneg(curveb)))
    
def arcmin(arca, arcb):
    return curvemin([arca],[arcb])
    
def circaderiv(x, r):
    if(x<=-r):
        return -math.pi*(r**2)/4
    elif(x>=r):
        return math.pi*(r**2)/4
    else:
        return (r**2 * torch.asin(x/r) + x*(r**2 - x**2)**0.5)/2
    
def areaunderarc(arc):
    left = arc['left']
    right = arc['right']
    if(arc['type']=='hline'):
        return arc['y']*(right-left)
    if(arc['type']=='circ'):
        rect =  arc['y']*(right-left)
        x, r = arc['x'], arc['r']
        circ = circaderiv(right-x, r) - circaderiv(left-x, r)
        return rect+arc['dir']*circ
    
def areaundercurve(curve, proto=None):
    #print([areaunderarc(arc) for arc in curve])
    if proto is None:
        return sum([areaunderarc(arc) for arc in curve])
    else:
        return sum([areaunderarc(arc) for arc in curve], torch.zeros_like(proto))
    
def vclip(curve, bot=None, top=None):
    if(len(curve)==0):
        return curve
    left = curve[0]['left']
    right = curve[-1]['right']
    if bot is not None:
        botline = {'type':'hline', 'y':bot, 'left':left, 'right':right}
        curve = curvemax(curve, [botline])
    if top is not None:
        topline = {'type':'hline', 'y':top, 'left':left, 'right':right}
        curve = curvemin(curve, [topline])
    return curve

def halfcirc(x,y,r,dir):
    return {'type':'circ', 'left':x-r, 'right':x+r, 'x':x, 'y':y, 'r':r, 'dir':dir}
    
def circarcs(x,y,r):
    return halfcirc(x,y,r,torch.ones_like(x)), halfcirc(x,y,r,-torch.ones_like(x))
    
def circarea(x,y,r):
    toparc, botarc = circarcs(x,y,r)
    return areaunderarc(toparc,x)-areaunderarc(botarc,x)

def cropcircarea(x,y,r,left,right,bot,top):
    toparc, botarc = circarcs(x,y,r)
    topcurve = vclip([toparc], bot, top)
    topcurve = hclip(topcurve, left, right)
    botcurve = vclip([botarc], bot, top)
    botcurve = hclip(botcurve, left, right)
    return areaundercurve(topcurve,x)-areaundercurve(botcurve,x)
    
def circoverlap(xa,ya,ra,xb,yb,rb):
    topa, bota = circarcs(xa,ya,ra)
    topb, botb = circarcs(xb,yb,rb)
    top = arcmin(topa, topb)
    bot = arcmax(bota, botb)
    bot = curvemin(bot, top)
    return areaundercurve(top,xa)-areaundercurve(bot,xa)
    
def cropcircoverlap(xa,ya,ra,xb,yb,rb,left,right,bot,top):
    topa, bota = circarcs(xa,ya,ra)
    topb, botb = circarcs(xb,yb,rb)
    topcurve = arcmin(topa, topb)
    topcurve = vclip(topcurve, bot, top)
    topcurve = hclip(topcurve, left, right)
    botcurve = arcmax(bota, botb)
    botcurve = vclip(botcurve, bot, top)
    botcurve = hclip(botcurve, left, right)
    botcurve = curvemin(botcurve, topcurve)
    return areaundercurve(topcurve,xa)-areaundercurve(botcurve,xa)
    
def boxarea(left, right, bot, top):
    return (right-left)*(top-bot)
    