import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
from plotDemag import *
import PCAZijderveld as pca

save = input('Save the figures? (y/N)')

fp = open(sys.argv[1],'r')
thellier = input('Thellier-Thellier? (y/N)')

path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'

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
            if step[-1] == '00': Thtype.append('Z')
            elif step[-1] == '01': Thtype.append('I')
            elif step[-1] == '02': Thtype.append('C')
            elif step[-1] == '03': Thtype.append('T')
            else: print('Temperature notation problem...'); sys.exit()
            Thstep.append(int(cols[-8].split('.')[0]))
        else:
            Thstep.append(float(cols[17]))
fp.close()
Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)

if thellier == 'y':
    arm = input('ARM sample? (y/N)')
    if arm == 'y':
        Plot_thellier_arm(Mx, My, Mz, Thstep, Thtype)
        if save == 'y':
            plt.savefig(path + sample + '-ARMvsT.pdf', format='pdf', dpi=400, bbox_inches="tight")
    else:
        Plot_Thellier(Mx, My, Mz, Thstep, Thtype)
        if save == 'y':
            plt.savefig(path + sample + '-Thellier_int.pdf', format='pdf', dpi=400, bbox_inches="tight")

        NRMx, NRMy, NRMz, NRM, pTRMgained, cHgained, zStep, cStep = Plot_aray(Mx, My, Mz, Thstep, Thtype)
        if save == 'y':
            plt.savefig(path + sample + '-Aray.pdf', format='pdf', dpi=400, bbox_inches="tight")

        Stat_Thellier(NRM, pTRMgained, cHgained, zStep, cStep, field=30)
        if save == 'y':
            plt.savefig(path + sample + '-Aray_fit.pdf', format='pdf', dpi=400, bbox_inches="tight")

        Plot_Zijderveld(NRMx, NRMy, NRMz, zStep, unit='mom', newfig='y', gui='nogui', color='k')
        if save == 'y':
            plt.savefig(path + sample + '-Thellier_zijd.pdf', format='pdf', dpi=400, bbox_inches="tight")

        plt.show()
        dopca = str(input('Run PCA analysis? (Y/n)'))
        if dopca != 'n':
            MAD, DANG, vec, MAD95, Mcmax, id1, id2 = pca.PCA_analysis(NRMx, NRMy, NRMz,,
else:
    Plot_thermal_demag(Mx, My, Mz, Thstep, line=0)
    if save == 'y':
        plt.savefig(path + sample + '-ThDemag.pdf', format='pdf', dpi=400, bbox_inches="tight")
    M0 = np.sqrt(Mx[0] ** 2 + My[0] ** 2 + Mz[0] ** 2)
    Plot_thermal_demag(Mx/M0, My/M0, Mz/M0, Thstep, norm=True, line=0)
    if save == 'y':
        plt.savefig(path + sample + '-ThDemagNormalized.pdf', format='pdf', dpi=400, bbox_inches="tight")
    Plot_Zijderveld(Mx, My, Mz, Thstep, unit='mom',newfig='y',color='k')
    if save == 'y':
        plt.savefig(path + sample + '-ThDemagZijd.pdf', format='pdf', dpi=400, bbox_inches="tight")

    # Plot_Zijderveld(Mx, My, Mz, Thstep, unit='mom',newfig='y',color='k')
    # plt.xlim(-3e-9,0.4e-9)
    # plt.ylim(-1e-9,1e-9)
    # if save == 'y':
    #     plt.savefig(path + sample + '-ThDemagZijdzoom.pdf', format='pdf', dpi=400, bbox_inches="tight")

    plot_equal_area_sequence(Mx, My, Mz, Thstep, newfig='y', color='k')
    if save == 'y':
        plt.savefig(path + sample + '-ThDemagEqArea.pdf', format='pdf', dpi=400, bbox_inches="tight")

plt.show()