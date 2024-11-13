
#### /!\ IMPORTANT NOTE /!\ ####

# For the code to work flawlessly:
#   IRM acquisition CSV or TXT files must contain "IRMacq"
#   Hysteresis CSV files must contain "Hy"
#   Bcr CSV files must contain "Bcr"

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from plotHyst import *
from builtins import *

save = input('Save the figures? (y/N)')

files = sys.argv[1:]
for fp in files:
    path = ''
    for k in np.arange(len(fp.split('/')) - 1):
        path += str(fp.split('/')[k]) + '/'
    sample = fp.split('/')[-1].split('-')[0]
    type = fp.split('/')[-1].split('-')[-1]

    if 'IRMacq' in type:
        fp_irm = open(fp,'r',encoding="utf8", errors='ignore')
        fp_bcr, fp_hyst, fp_LT = None, None, None
    elif 'Hy' in type:
        fp_hyst = open(fp,'r',encoding="utf8", errors='ignore')
        fp_irm, fp_bcr, fp_LT = None, None, None
    elif 'Bcr' in type:
        fp_bcr = open(fp,'r',encoding="utf8", errors='ignore')
        fp_irm, fp_hyst, fp_LT = None, None, None
    elif 'LT' in type:
        fp_LT = open(fp,'r',encoding="utf8", errors='ignore')
        fp_irm, fp_bcr, fp_hyst = None, None, None

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
            down = input('Downsample data? (y/N)  ')
            if down == 'y':
                x = float(eval(input('Fraction of sIRM between two steps (%)? (e.g., 0.5)  ')))/100
                dsBirm, dsMirm = Birm[:1],Mirm[:1]
                f = 1
                for k in np.arange(len(Birm)-1):
                    if f > 1:
                        f -= 1
                        continue
                    while (Mirm[k+f]-Mirm[k]) <= x*Mirm[-1]:
                        if k+f == len(Mirm)-1:
                            break
                        else:
                            f += 1
                    dsBirm.append(Birm[k+f])
                    dsMirm.append(Mirm[k+f])
                print('Number of datapoints kept: '+str(len(dsMirm))+'/'+str(len(Mirm)))
                fpw = open(fp_irm.name[:-4]+'_ds.txt', 'w')
                for k in np.arange(1,len(dsBirm)):
                    fpw.write(f'{dsBirm[k]*1e3:.3e}' + ',' + f'{dsMirm[k]:.3e}' + '\n')
                fpw.close()
            else:
                fpw = open(fp_irm.name[:-4]+'.txt', 'w')
                for k in np.arange(1,len(Birm)):
                    fpw.write(f'{Birm[k]*1e3:.3e}' + ',' + f'{Mirm[k]:.3e}' + '\n')
                fpw.close()

        IRMacq, Bvalue = plotIRMacq(Birm, Mirm, Bvalue=150, ylim=())
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

        mass = input("Mass of the sample (kg)? (default = 1)  ")
        if mass != '':
            mass = float(eval(mass))
        else:
            mass = 1

        nonlincorr = input('Non-linear correction? (y/N)  ')
        nocorr = input('Show data not corrected? (y/N)  ')
        print('\n')
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

    if fp_irm != None:
        if Bvalue != 0:
            print(' * IRM acq. at '+str(int(Bvalue))+' mT = '+f'{IRMacq:.3e}'+' A m2')
        fp_irm.close()

    if fp_bcr != None:
        print(' * Bcr = '+f'{Bcr*1000:.1f}'+' mT')
        fp_bcr.close()

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
                print(' * alpha = ' + f'{alpha:.3e}' + ' A m2 T-2')
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
                print(' * HF slope = ' + f'{kHF/mass:.2e}'+' '+unitHF)
        fp_hyst.close()

    if fp_LT != None:
        TLT, MLT = [], []
        for j, line in enumerate(fp_LT):
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

