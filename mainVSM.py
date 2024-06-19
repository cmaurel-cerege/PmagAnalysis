import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
from scipy.optimize import curve_fit
from plotHyst import *

def Get_closest_id(L,value):
    return list(L).index(min(L, key=lambda x:abs(x-value)))

save = input('Save the figures? (y/N)')

path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'
sample = sys.argv[1].split('/')[-1].split('-')[0]
# if len(sys.argv[1].split('/')[-1].split('_')) == 2:
#     sample = sys.argv[1].split('/')[-1].split('_')[0]
# elif len(sys.argv[1].split('/')[-1].split('_')) >= 3:
#     sample = sys.argv[1].split('/')[-1].split('_')[0]+'_'+sys.argv[1].split('/')[-1].split('_')[1]


fp = open(sys.argv[1],'r',encoding="utf8", errors='ignore')
if 'IRMacq' in sys.argv[1]:
    type = 'IRM'
elif 'Hy' in sys.argv[1]:
    type = 'Hy'
elif 'Bcr' in sys.argv[1]:
    type = 'Bcr'
elif 'LT' in sys.argv[1]:
    type = 'LT'

B, M, T = [], [], []
if 'txt' in sys.argv[1]:
    for j,line in enumerate(fp):
        cols = line.split(',')
        B.append(float(cols[0])*1e-3) ## field in T for plotting
        M.append(float(cols[1]))
else:
    for j,line in enumerate(fp):
        cols = line.split(',')
        if len(cols) > 1:
            if cols[1] == '0':
                B.append(float(cols[3]))
                M.append(float(cols[4]))

    if type == 'IRM':
        simp = str(input('Simplify file? (y/N)'))
        if simp == 'y':
            fpw = open(sys.argv[1][:-3] + 'txt', 'w')
            for k in np.arange(1, len(B)):
                fpw.write(f'{B[k] * 1e3:.3e}' + ',' + f'{M[k]:.3e}' + '\n')
            fpw.close()
fp.close()

if type == 'IRM':
    der = input('Plot derivative ? (y/N)')
    if der == 'y':
        plotVSMdata(B,M,type=type,der='y',ylim=0,newfig='y')
    else:
        plotVSMdata(B, M, type=type, ylim=0, newfig='y')
    if save == 'y':
        if der == 'y':
            plt.savefig(path + sample + '_IRMacqder.pdf', format='pdf', dpi=400, bbox_inches="tight")
        else:
            plt.savefig(path + sample + '_IRMacq.pdf', format='pdf', dpi=400, bbox_inches="tight")
    IRM150 = getIRMacq(B,M,0.150)
    print("IRMacq. 150 mT = "+f'{IRM150:.3e}'+" A m2")

elif type == 'Bcr':
    plotVSMdata(B,M,type=type,ylim=0,newfig='y')
    if save == 'y':
        plt.savefig(path + sample + '_Bcr.pdf', format='pdf', dpi=400, bbox_inches="tight")
    Bcr = getBcr(B,M)
    print("Bcr = "+f'{Bcr*1000:.2f}'+" mT")

elif type == 'Hy':

    normalize = input("Mass normalize ? (y/N)")
    if normalize == 'y':
        mass = float(eval(input("Mass (g)? ")))*1e-3
    else:
        mass = 1

    Bcut = [B[k] for k in np.arange(len(B)) if np.absolute(B[k])<1.6]
    Mcut = [M[k] for k in np.arange(len(M)) if np.absolute(B[k])<1.6]
    corr = input('Linear correction? (Y/n)')
    if corr == 'n':
        plotCorrHysteresis(Bcut, Mcut, type=type, mass=mass, linear=False, interval=0.2, ylim=0, shownocorr=True)
        if save == 'y':
            plt.savefig(path + sample + '_HyNC.pdf', format='pdf', dpi=400, bbox_inches="tight")
    else:
        plotCorrHysteresis(Bcut, Mcut, type=type, mass=mass, linear=True, interval=0.1, ylim=0, shownocorr=True)
        if save == 'y':
            plt.savefig(path + sample + '_HyC.pdf', format='pdf', dpi=400, bbox_inches="tight")


if type == 'LT':
    plotVSMLTdata(T, M, norm=True, newfig='y')
    if save == 'y':
        plt.savefig(path + sample + '_LT.pdf', format='pdf', dpi=400, bbox_inches="tight")

plt.show()

