import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from plotHyst import *

save = input('Save the figure? (y/N)')

path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'

files = sys.argv[1:]
name_hyst, name_bcr, Mrs, Ms, Bc, Bcr = [], [], [], [], [], []
for file in files:
    if 'Hy' in file:
        fp_hyst = open(file,'r',encoding="utf8", errors='ignore')
        name_hyst.append(file.split('/')[-1].split('_')[0])
        Bhyst, Mhyst = [], []
        for j, line in enumerate(fp_hyst):
            cols = line.split(',')
            if len(cols) > 1:
                if cols[1] == '0':
                    Bhyst.append(float(cols[3]))
                    Mhyst.append(float(cols[4]))
        fp_hyst.close()
        Ms_tmp, Mrs_tmp, Bc_tmp, sHF, kHF, alpha = plotHyst(Bhyst, Mhyst)
        plt.close()
        Mrs.append(Mrs_tmp)
        Ms.append(Ms_tmp)
        Bc.append(Bc_tmp)

    elif 'Bcr' in file:
        fp_bcr = open(file, 'r', encoding="utf8", errors='ignore')
        name_bcr.append(file.split('/')[-1].split('_')[0])
        Bbcr, Mbcr = [], []
        for j, line in enumerate(fp_bcr):
            cols = line.split(',')
            if len(cols) > 1:
                if cols[1] == '0':
                    Bbcr.append(float(cols[3]))
                    Mbcr.append(float(cols[4]))
        fp_bcr.close()
        Bcr_tmp = plotBcr(Bbcr, Mbcr)
        plt.close()
        Bcr.append(Bcr_tmp)

name, MrsMs, BcrBc = [], [], []
for k in np.arange(len(Mrs)):
    name.append(name_bcr[k][:3])
    id = name_hyst.index(name_bcr[k])
    MrsMs.append(Mrs[id]/Ms[id])
    BcrBc.append(Bcr[k]/Bc[id])

fig, ax = plt.subplots(figsize=(5,4))
ax.set_xscale("log")
ax.set_yscale("log")
plt.xlim(1,20)
plt.ylim(0.01,1.)
plt.xlabel('Bcr/Bc')
plt.ylabel('Mrs/Ms')
plt.plot([2,2],[0,1],'k-',lw=0.5)
plt.plot([5,5],[0,1],'k-',lw=0.5)
plt.plot([0,20],[0.02,0.02],'k-',lw=0.5)
plt.plot([0,20],[0.5,0.5],'k-',lw=0.5)
for k in np.arange(len(BcrBc)):
    plt.plot(BcrBc[k], MrsMs[k], marker='o', ms=6, mew=0.5, mec='k', lw=0, label=name[k])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()

plt.show()