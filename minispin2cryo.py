import numpy as np
import sys

files = sys.argv[1:]
if len(files) > 2:
    print('\nInput only one Minispin file and one 2G file')
    sys.exit()

format = input('\nDid you remove the "," separator and replace "," with "." for numbers in minispin file? (Y/n)  ')
if format == 'n':
    print(' -> Do it :)')
    sys.exit()

path = ''
for k in np.arange(len(files[0].split('/'))-1):
    path += str(files[0].split('/')[k])+'/'

if len(files) == 1:
    name = files[0].split('/')[-1].split('.')[0]
    fp = open(str(files[0]), 'r')
    if files[0][len(files[0])-3:] == 'txt':
        MxMS, MyMS, MzMS, stepMS = [], [], [], []
        for j, line in enumerate(fp):
            cols = line.split()
            MxMS.append(float(cols[5])*1e-8)
            MyMS.append(float(cols[6])*1e-8)
            MzMS.append(float(cols[7])*1e-8)
            stepMS.append(int(cols[1])*0.1)
        fp.close()

elif len(files) == 2:
    for file in files:
        fp = open(str(file), 'r')
        if file[len(file) - 3:] == 'txt':
            Mx, My, Mz, step = [], [], [], []
            for j, line in enumerate(fp):
                cols = line.split()
                Mx.append(float(cols[5])*1e-8)
                My.append(float(cols[6])*1e-8)
                Mz.append(float(cols[7])*1e-8)
                step.append(int(cols[1])*0.1)
            Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)

            MxMS, MyMS, MzMS, stepMS = Mx, My, Mz, step
            xMS2cryo = input('Mx2G = MxMS? (Y/n)  ')
            if xMS2cryo == 'n':
                xMS2cryo = input('Mx2G = ? (-x, y, -y, z, -z)  ')
                if xMS2cryo == '-x':
                    MxMS = -Mx
                elif xMS2cryo == 'y':
                    MxMS = My
                elif xMS2cryo == '-y':
                    MxMS = -My
                elif xMS2cryo == 'z':
                    MxMS = Mz
                elif xMS2cryo == '-z':
                    MxMS = -Mz
            yMS2cryo = input('My2G = MyMS? (Y/n)  ')
            if yMS2cryo == 'n':
                yMS2cryo = input('My2G = ? (x, -x, -y, z, -z)  ')
                if yMS2cryo == 'x':
                    MyMS = Mx
                elif yMS2cryo == '-x':
                    MyMS = -Mx
                elif yMS2cryo == '-y':
                    MyMS = -My
                elif yMS2cryo == 'z':
                    MyMS = Mz
                elif yMS2cryo == '-z':
                    MyMS = -Mz
            zMS2cryo = input('Mz2G = MzMS? (Y/n)  ')
            if zMS2cryo == 'n':
                zMS2cryo = input('Mz2G = ? (x, -x, y, -y, -z)  ')
                if zMS2cryo == 'x':
                    MzMS = Mx
                elif zMS2cryo == '-x':
                    MzMS = -Mx
                elif zMS2cryo == 'y':
                    MzMS = My
                elif zMS2cryo == '-y':
                    MzMS = -My
                elif zMS2cryo == '-z':
                    MzMS = -Mz

        elif file[len(file)-3:] == 'DAT':
            name = file.split('/')[-1].split('.')[0]
            Mx2G, My2G, Mz2G, step2G = [], [], [], []
            for j, line in enumerate(fp):
                if j > 0:
                    cols = line.split()
                    Mx2G.append(float(cols[1])*1e-3)
                    My2G.append(float(cols[2])*1e-3)
                    Mz2G.append(float(cols[3])*1e-3)
                    step2G.append(int(cols[-1])*0.1)
        fp.close()

fpcomb = open(path+name+'_cmb.txt','w')
for j in np.arange(len(MxMS)):
    fpcomb.write(str(int(stepMS[j]))+'  '+f'{MxMS[j]:.3e}'+'  '+f'{MyMS[j]:.3e}'+'  '+f'{MzMS[j]:.3e}'+'\n')
if len(files) == 2:
    for j in np.arange(len(Mx2G)):
        if step2G[j] not in stepMS:
            fpcomb.write(str(int(step2G[j]))+'  '+f'{Mx2G[j]:.3e}'+'  '+f'{My2G[j]:.3e}'+'  '+f'{Mz2G[j]:.3e}'+'\n')

fpcomb.close()
