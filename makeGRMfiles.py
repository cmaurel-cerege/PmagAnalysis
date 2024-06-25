import numpy as np
import sys

file = sys.argv[1]
path = ''
for k in np.arange(len(file.split('/'))-1):
    path += str(file.split('/')[k])+'/'
name = file.split('/')[-1].split('.')[0]

fp = open(file, 'r')

Mx, My, Mz, step = [], [], [], []
for j, line in enumerate(fp):
    if j > 0:
        cols = line.split()
        if 'NA' not in cols[1:4]:
            Mx.append(float(cols[1])*1e-3)
            My.append(float(cols[2])*1e-3)
            Mz.append(float(cols[3])*1e-3)
            step.append(int(cols[-1])*0.1)
fp.close()

## AF XYZ -> AF X -> AF Y
Mxgrm = Mx[:1]+[(Mx[k]+Mx[k+1]+Mx[k+2])/3 for k in np.arange(1,len(Mx)-2,3)]
Mygrm = My[:1]+[(My[k]+My[k+1]+My[k+2])/3 for k in np.arange(1,len(My)-2,3)]
Mzgrm = Mz[:1]+[(Mz[k]+Mz[k+1]+Mz[k+2])/3 for k in np.arange(1,len(Mz)-2,3)]
stepgrm = step[:1]+[(step[k]+step[k+1]+step[k+2])/3 for k in np.arange(1,len(step)-2,3)]

## AF XYZ
Mxnogrm = Mx[:1]+[Mx[k] for k in np.arange(1,len(Mx)-2,3)]
Mynogrm = My[:1]+[My[k] for k in np.arange(1,len(My)-2,3)]
Mznogrm = Mz[:1]+[Mz[k] for k in np.arange(1,len(Mz)-2,3)]
stepnogrm = step[:1]+[step[k] for k in np.arange(1,len(step)-2,3)]

fpwgrm = open(path+name+'_GRMavg.txt', 'w')
fpwnogrm = open(path+name+'_noGRM.txt', 'w')
for j in np.arange(len(Mxgrm)):
    fpwgrm.write(str(int(stepgrm[j]))+' '+str(Mxgrm[j])+' '+str(Mygrm[j])+' '+str(Mzgrm[j])+' '+'\n')
fpwgrm.close()
for j in np.arange(len(Mxnogrm)):
    fpwnogrm.write(str(int(stepnogrm[j]))+' '+str(Mxnogrm[j])+' '+str(Mynogrm[j])+' '+str(Mznogrm[j])+' '+'\n')
fpwnogrm.close()