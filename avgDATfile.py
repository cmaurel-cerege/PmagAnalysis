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

Mxavg, Myavg, Mzavg, stepavg = [], [], [], []
Mxfirst, Myfirst, Mzfirst, stepfirst = [], [], [], []
id = 0
for k in np.arange(len(step)-1):
    if k == id:
        stepavg.append(step[k])
        stepfirst.append(step[k])
        Mxfirst.append(Mx[k])
        Myfirst.append(My[k])
        Mzfirst.append(Mz[k])
        if step[k] == step[k+1]:
            if step[k] == step[k+2]:
                Mxavg.append((Mx[k]+Mx[k+1]+Mx[k+2])/3)
                Myavg.append((My[k]+My[k+1]+My[k+2])/3)
                Mzavg.append((Mz[k]+Mz[k+1]+Mz[k+2])/3)
                id += 3
            else:
                Mxavg.append((Mx[k]+Mx[k+1])/2)
                Myavg.append((My[k]+My[k+1])/2)
                Mzavg.append((Mz[k]+Mz[k+1])/2)
                id += 2
        else:
            Mxavg.append(Mx[k])
            Myavg.append(My[k])
            Mzavg.append(Mz[k])
            id += 1

fpwavg = open(path+name+'_avg.txt', 'w')
fpwfirst = open(path+name+'_first.txt', 'w')
for j in np.arange(len(Mxavg)):
    fpwavg.write(str(int(stepavg[j]))+', '+f'{Mxavg[j]:.4e}'+', '+f'{Myavg[j]:.4e}'+', '+f'{Mzavg[j]:.4e}')
    if j != len(Mxavg)-1: fpwavg.write('\n')
    else: continue
fpwavg.close()

for j in np.arange(len(Mxfirst)):
    fpwfirst.write(str(int(stepfirst[j]))+', '+f'{Mxfirst[j]:.4e}'+', '+f'{Myfirst[j]:.4e}'+', '+f'{Mzfirst[j]:.4e}')
    if j != len(Mxfirst)-1: fpwfirst.write('\n')
    else: continue
fpwfirst.close()