import numpy as np
import sys
import os

files = sys.argv[1:]
path = ''
for k in np.arange(len(files[0].split('/'))-1):
    path += str(files[0].split('/')[k])+'/'

for file in files:
    filename = (file.split('/')[-1]).split('.')[0]
    fp = open(str(file),'r')
    Mx, My, Mz, T, name = [], [], [], [], []
    ## This assumes moment in A m2, field in mT
    for j, line in enumerate(fp):
        if j > 0:
            cols = line.split('\t')
            if cols[15] == 'NA':
                T.append(20.)
            else:
                T.append(float(cols[15]))
            Mx.append(float(cols[1]))
            My.append(float(cols[2]))
            Mz.append(float(cols[3]))
            name.append(cols[0])
    fp.close()

    fpw = open(path+filename+'.txt','w')
    fpw.write(name[0]+'\n')
    fpw.write('step\tMx\tMy\tMz\n')
    for k in np.arange(len(T)):
        fpw.write(str(T[k])+'\t'+str(Mx[k])+'\t'+str(My[k])+'\t'+str(Mz[k])+'\n')
    fpw.close()