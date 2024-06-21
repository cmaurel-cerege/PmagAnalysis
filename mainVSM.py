import numpy as np
import matplotlib.pyplot as plt
import sys
from plotHyst import *

save = input('Save the figures? (y/N)')

path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'
sample = sys.argv[1].split('/')[-1].split('-')[0]

files = sys.argv[1:]
fp_irm, fp_bcr, fp_hyst = None, None, None
for fp in files:
    if 'IRMacq' in fp:
        fp_irm = open(fp,'r',encoding="utf8", errors='ignore')
    elif 'Hy' in fp:
        fp_hyst = open(fp,'r',encoding="utf8", errors='ignore')
    elif 'Bcr' in fp:
        fp_bcr = open(fp,'r',encoding="utf8", errors='ignore')
    elif 'LT' in fp:
        fp_LT = open(fp,'r',encoding="utf8", errors='ignore')

if fp_irm != None:
    Birm, Mirm = [], []
    if 'txt' in fp_irm.name:
        for j,line in enumerate(fp_irm):
            cols = line.split(',')
            Birm.append(float(cols[0])*1e-3) ## field in T for plotting
            Mirm.append(float(cols[1]))
    else:
        for j,line in enumerate(fp_irm):
            cols = line.split(',')
            if len(cols) > 1:
                if cols[1] == '0':
                    Birm.append(float(cols[3]))
                    Mirm.append(float(cols[4]))

    simp = input('Simplify file? (y/N)  ')
    if simp == 'y':
        fpw = open(fp_irm[:-4]+'.txt', 'w')
        for k in np.arange(1, len(Birm)):
            fpw.write(f'{Birm[k] * 1e3:.3e}' + ',' + f'{Mirm[k]:.3e}' + '\n')
        fpw.close()
    fp_irm.close()

    der = input('Plot derivative ? (y/N)  ')
    IRMacq, Bvalue = plotIRMacq(Birm, Mirm, Bvalue=150, der=der, ylim=())
    plt.show(block=False)

    if save == 'y':
        plt.savefig(path + sample + '_IRMacq.pdf', format='pdf', dpi=200, bbox_inches="tight")

if fp_bcr != None:
    Bbcr, Mbcr = [], []
    for j, line in enumerate(fp_bcr):
        cols = line.split(',')
        if len(cols) > 1:
            if cols[1] == '0':
                Bbcr.append(float(cols[3]))
                Mbcr.append(float(cols[4]))
    fp_bcr.close()
    Bcr = plotBcr(Bbcr, Mbcr, ylim=())
    plt.show(block=False)

    if save == 'y':
        plt.savefig(path + sample + '_Bcr.pdf', format='pdf', dpi=200, bbox_inches="tight")

if fp_hyst != None:
    Bhyst, Mhyst = [], []
    for j, line in enumerate(fp_hyst):
        cols = line.split(',')
        if len(cols) > 1:
            if cols[1] == '0':
                Bhyst.append(float(cols[3]))
                Mhyst.append(float(cols[4]))
    fp_hyst.close()

    mass = input("Mass of the sample ? ")
    if mass != '':
        mass = float(eval(mass))
    else:
        mass = 1

    nonlincorr = input('Non-linear correction? (y/N)  ')
    nocorr = input('Show data not corrected? (y/N)'  )
    interval = 0.2

    if nonlincorr == 'y':
        if nocorr != 'y':
            Ms, Mrs, Bc, sHF, kHF, alpha = plotHyst(Bhyst, Mhyst, mass=mass, linear=False, interval=interval, ylim=(), shownocorr=False)
        else:
            Ms, Mrs, Bc, sHF, kHF, alpha = plotHyst(Bhyst, Mhyst, mass=mass, linear=False, interval=interval, ylim=(), shownocorr=True)
    else:
        if nocorr != 'y':
            Ms, Mrs, Bc, sHF, kHF, alpha = plotHyst(Bhyst, Mhyst, mass=mass, linear=True, interval=interval, ylim=(), shownocorr=False)
        else:
            Ms, Mrs, Bc, sHF, kHF, alpha = plotHyst(Bhyst, Mhyst, mass=mass, linear=True, interval=interval, ylim=(), shownocorr=True)
    if save == 'y':
        plt.savefig(path + sample + '_Hyst.pdf', format='pdf', dpi=200, bbox_inches="tight")

print('\n')
if fp_irm != None:
    if Bvalue != 0:
        print(' * IRM acq. at '+str(int(Bvalue))+' mT = '+f'{IRMacq:.3e}'+' A m2\n')

if fp_bcr != None:
    print(' * Bcr = '+f'{Bcr*1000:.0f}'+' mT\n')

if fp_hyst != None:
    if nocorr != 'y':
        if nonlincorr == 'y':
            if mass != 1:
                unit = 'A m2 kg-1'
            else:
                unit = 'A m2'
            print(' * Ms = '+f'{Ms/mass:.3e}'+' '+unit)
            print(' * Mrs = '+f'{Mrs/mass:.3e}'+' '+unit)
            print(' * Bc = '+f'{Bc*1000:.2f}'+' mT')
            print(' * HF slope = ' + f'{kHF:.3e}' + ' A m2 T-1')
            print(' * alpha = ' + f'{alpha:.3e}' + ' A m2 T-2'+'\n')
        else:
            if mass != 1:
                unit = 'A m2 kg-1'
                unitHF = 'm3 kg-1'
            else:
                unit = 'A m2'
                unitHF = 'm3'
            print(' * Ms = ' + f'{Ms/mass:.3e}'+' '+unit)
            print(' * Mrs = ' + f'{Mrs/mass:.3e}'+' '+unit)
            print(' * Bc = ' + f'{Bc*1000:.1f}'+' mT')
            print(' * HF slope = ' + f'{sHF:.2e}'+' A m2 T-1')
            print(' * HF slope = ' + f'{kHF/mass:.2e}'+' '+unitHF+'\n')

if fp_LT != None:
    TLT, MLT = [], []
    for j, line in enumerate(fp_hyst):
        cols = line.split(',')
        if len(cols) > 1:
            if cols[1] == '0':
                TLT.append(float(cols[3])) ## Not sure about the column
                MLT.append(float(cols[4]))
    fp_LT.close()

    plotVSMLT(Bbcr, Mbcr, norm=False, ylim=())
    if save == 'y':
        plt.savefig(path + sample + '_hyLT.pdf', format='pdf', dpi=200, bbox_inches="tight")


plt.show()

