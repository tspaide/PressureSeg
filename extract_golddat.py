import numpy as np
import os
import json

folders = ['jcw30-csv','sf30-csv']
basepath = '../../yue/joanne/GAT SL videos'

sings = []

dats = {}
for foldname in folders:
    folddict = {}
    csvfiles = [f for f in os.listdir(os.path.join(basepath,foldname)) if '(1)' not in f]
    print(foldname)
    for fname in csvfiles:
        lines = []
        print('  '+fname)
        filepath = os.path.join(basepath,foldname,fname)
        if fname in os.listdir('corrected_circles'):
            print('    reading from corrected')
            filepath = os.path.join('corrected_circles',fname)
        with open(filepath,encoding="ISO-8859-1") as f:
            for line in f:
                if line[-1]=='\n':
                    line = line[:-1]
                if line[:2]=='2-':
                    line = '04-J0'+line
                if line[0]=="'":
                    continue
                    line = line.replace('\'','')
                if line[0]=="\\":
                    line = line[1:]
                if line[6] != '-':
                    line=line[:6]+chr(8211)+line[7:]
                lines.append(line.split(','))
        folddict[fname] = {}
        nameexprinted = False
        for l in lines:
            if len(l)<13 or not l[1]:
                continue
            if not nameexprinted:
                print('    '+l[0])
                nameexprinted=True
            x1, y1, x2, y2, x3, y3 = [float(s) for s in l[1:7]]
            try:
                a,b = np.linalg.solve([[2*(x1-x2),2*(y1-y2)],[2*(x1-x3),2*(y1-y3)]],[x1**2-x2**2+y1**2-y2**2,x1**2-x3**2+y1**2-y3**2])
            except np.linalg.linalg.LinAlgError:
                sings.append(' - '.join([foldname,fname,l[0],'outer']))
                continue
            r = ((x1-a)**2+(y1-b)**2)**0.5
            assert abs(r**2-(x2-a)**2-(y2-b)**2)<0.01
            assert abs(r**2-(x3-a)**2-(y3-b)**2)<0.01
            dat = [a,b,r]
            if all(s=='0' for s in l[7:13]):
                dat = dat + [0,0,0,0]
            else:
                x1, y1, x2, y2, x3, y3 = [float(s) for s in l[7:13]]
                try:
                    a,b = np.linalg.solve([[2*(x1-x2),2*(y1-y2)],[2*(x1-x3),2*(y1-y3)]],[x1**2-x2**2+y1**2-y2**2,x1**2-x3**2+y1**2-y3**2])
                except np.linalg.linalg.LinAlgError:
                    sings.append(' - '.join([foldname,fname,l[0],'inner']))
                    continue
                r = ((x1-a)**2+(y1-b)**2)**0.5
                assert abs(r**2-(x2-a)**2-(y2-b)**2)<0.01
                assert abs(r**2-(x3-a)**2-(y3-b)**2)<0.01
                sw = (np.mean((y1,y2,y3))>dat[1])
                #if sw!=(dat[0]<a):
                #    print(l[0],sw)
                dat = dat + [a,b,r,int(sw)]
            folddict[fname][l[0]] = dat
        if fname[6] != '-':
            fname = fname[:6]+chr(8211)+fname[7:]
    dats[foldname] = folddict
    
print(sings)
    
with open('goldmann_measurements.json','w') as f:
    json.dump(dats,f)