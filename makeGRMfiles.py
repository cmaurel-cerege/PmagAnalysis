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

## If GRM protocol was started after some demag steps
id_start = 1
for k in np.arange(len(step)-1):
    if step[k] == step[k+1] and step[k] == step[k+2]:
        id_start = k
        break

## AF XYZ -> AF X -> AF Y
Mxgrm = Mx[:id_start]+[(Mx[k]+Mx[k+1]+Mx[k+2])/3 for k in np.arange(id_start,len(Mx)-2,3)]
Mygrm = My[:id_start]+[(My[k]+My[k+1]+My[k+2])/3 for k in np.arange(id_start,len(My)-2,3)]
Mzgrm = Mz[:id_start]+[(Mz[k]+Mz[k+1]+Mz[k+2])/3 for k in np.arange(id_start,len(Mz)-2,3)]
stepgrm = step[:id_start]+[int((step[k]+step[k+1]+step[k+2])/3) for k in np.arange(id_start,len(step)-2,3)]

## AF XYZ
Mxnogrm = Mx[:id_start]+[Mx[k] for k in np.arange(id_start,len(Mx)-2,3)]
Mynogrm = My[:id_start]+[My[k] for k in np.arange(id_start,len(My)-2,3)]
Mznogrm = Mz[:id_start]+[Mz[k] for k in np.arange(id_start,len(Mz)-2,3)]
stepnogrm = step[:id_start]+[step[k] for k in np.arange(id_start,len(step)-2,3)]

fpwgrm = open(path+name+'_GRMavg.txt', 'w')
#fpwnogrm = open(path+name+'_noGRM.txt', 'w')
for j in np.arange(len(Mxgrm)):
    if j != len(Mxgrm)-1:
        fpwgrm.write(str(int(stepgrm[j]))+', '+f'{Mxgrm[j]:.4e}'+', '+f'{Mygrm[j]:.4e}'+', '+f'{Mzgrm[j]:.4e}'+'\n')
    else:
        fpwgrm.write(str(int(stepgrm[j]))+', '+f'{Mxgrm[j]:.4e}'+', '+f'{Mygrm[j]:.4e}'+', '+f'{Mzgrm[j]:.4e}')
fpwgrm.close()
# for j in np.arange(len(Mxnogrm)):
#     fpwnogrm.write(str(int(stepnogrm[j]))+' '+str(Mxnogrm[j])+' '+str(Mynogrm[j])+' '+str(Mznogrm[j])+' '+'\n')
# fpwnogrm.close()