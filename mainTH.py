import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from plotDemag import *
from plotZijderveld import *
from plotEqualArea import *
import PCAZijderveld as pca
from calcPaleointensities import *

save = input('Save the figures? (y/N)')
thellier = input('Thellier-Thellier? (y/N)')

if len(sys.argv) == 1:
    input('No files...')
    sys.exit

path = ''
fp = open(sys.argv[1],'r')
for k in np.arange(len(sys.argv[1].split('/'))-1):
    path += str(sys.argv[1].split('/')[k])+'/'
if not os.path.exists(path+'Plots'):
    os.makedirs(path+'Plots')

sample = sys.argv[1].split('/')[-1].split('.')[0]

Mx, My, Mz, Thstep, Thtype = [], [], [], [], []
for j, line in enumerate(fp):
    if j > 0:
        cols = line.split()
        Mx.append(float(cols[1]) * 1e-3)
        My.append(float(cols[2]) * 1e-3)
        Mz.append(float(cols[3]) * 1e-3)
        ## Name code to know which step (Z, I, C, T) you're looking at:
        if thellier == 'y':
            step = str(cols[17]).split('.')
            if step[-1] == '00': Thtype.append('Z')  ## Z step
            elif step[-1] == '01': Thtype.append('I')  ## I step
            elif step[-1] == '02': Thtype.append('C')  ## pTRM check
            elif step[-1] == '03': Thtype.append('T')  ## pTRM tail check
            else: print('Temperature notation problem...'); sys.exit()
            Thstep.append(int(cols[-8].split('.')[0]))
        else:
            Thstep.append(float(cols[17]))
fp.close()
Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)

if thellier == 'y':
    Plot_Thellier(Mx, My, Mz, Thstep, Thtype)
    if save == 'y':
        plt.savefig(path + sample + '-Thellier.pdf', format='pdf', dpi=400, bbox_inches="tight")

    NRMx, NRMy, NRMz, NRM, pTRMgained, cHgained, zStep, cStep = Plot_Aray(Mx, My, Mz, Thstep, Thtype)
    if save == 'y':
        plt.savefig(path + sample + '-Aray.pdf', format='pdf', dpi=400, bbox_inches="tight")

    #### TO DO
    Stat_Thellier(NRM, pTRMgained, cHgained, zStep, cStep, field=30)
    if save == 'y':
        plt.savefig(path + sample + '-Aray-fit.pdf', format='pdf', dpi=400, bbox_inches="tight")

    Plot_Zijderveld(NRMx, NRMy, NRMz, zStep, unit='A m2', title='NRM@TH', color='TH')
    if save == 'y':
        plt.savefig(path + sample + '-Thellier-zijd.pdf', format='pdf', dpi=400, bbox_inches="tight")

    plt.show(block=False)

else:
    Plot_thermal_demag(Mx, My, Mz, Thstep, norm=True)
    if save == 'y':
        plt.savefig(path + sample + '-Therm-int.pdf', format='pdf', dpi=400, bbox_inches="tight")

    fig = plt.figure(figsize=(5,5))
    Plot_Zijderveld(Mx, My, Mz, Thstep, unit='A m2', title='NRM@TH', color='TH')
    if save == 'y':
        plt.savefig(path + sample + '-Therm-zijd.pdf', format='pdf', dpi=400, bbox_inches="tight")

    fig = plt.figure(figsize=(5, 5))
    plot_equal_area_sequence(Mx, My, Mz, fig, 'NRM@TH', color='k')
    if save == 'y':
        plt.savefig(path + sample + '-Therm-eqarea.pdf', format='pdf', dpi=400, bbox_inches="tight")

    dopca = str(input('Run PCA analysis? (y/N)  '))
    if dopca == 'y':
        id1 = input('Index first datapoint?  ')
        if id1 != '':
            id1 = int(eval(id1))
            id2 = input('Index last datapoint? (None for last)  ')
            if id2 != '':
                id2 = int(eval(id2))
            else:
                id2 = len(Mx)
        MAD, DANG, vec, MAD95, Mcmax = pca.Calc_MAD_and_DANG(Mx,My,Mz,Thstep,id1,id2)

plt.show()