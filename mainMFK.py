import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import interpolate

type = input('RT, LT, HT measurements? (r/l/h)  ')
mass = input('Mass of the sample (g)?  ')
if mass != '':
    mass = float(eval(mass))
else:
    mass = 1

fp = open(sys.argv[1],'r',encoding="utf8",errors='ignore')

save = input('Save the figures? (y/N)')
path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'
sample = sys.argv[1].split('/')[-1].split('.')[0]

## Measurements conducted at room temperature but sample heated in the oven:
if type == 'r':

    Tstep, K = [], []
    for j, line in enumerate(fp):
        cc = line.split(' ')
        cols = [c for c in cc if c != '']
        if j > 0:
            Tstep.append(float(cols[0]))
            K.append(float(cols[6])*1e-5/(mass*1e-3))
    fp.close()

    normsus = input('Normalize susceptibility? (y/N)')
    fig = plt.figure()
    plt.xlabel('Temperature (C)')
    plt.xlim(0, np.max(Tstep)+20)
    if normsus == 'N':
        plt.ylabel('Susceptibility (m3 kg-1)')
        plt.ylim(0, 1.1*np.max(K))
    else:
        plt.ylabel('Normalized susceptibility')
        plt.ylim(0, 1.1*np.max(K/K[0]))
    plt.plot(Tstep,K/K[0],'k-',marker='o',ms='3',lw=0.5,mfc='k')
    fig.tight_layout()

    if save == 'y':
        if normsus == 'N':
            plt.savefig(path + sample + '_susc.pdf', format='pdf', dpi=200, bbox_inches="tight")
        else:
            plt.savefig(path + sample + '_normsusc.pdf', format='pdf', dpi=200, bbox_inches="tight")

## Measurements conducted in the MFK at low or high temperatures:
if type == 'l' or type == 'h':

    T, K = [], []
    for j, line in enumerate(fp):
        cc = line.split(' ')
        cols = [c for c in cc if c != '']
        if j > 0:
            T.append(float(cols[0]))
            K.append(float(cols[1]))
    fp.close()

    ## If sample holder correction:
    if len(sys.argv) > 2:
        fppe = open(sys.argv[2],'r',encoding="utf8",errors='ignore')
        Tpe, Kpe = [], []
        for j, line in enumerate(fppe):
            cc = line.split(' ')
            cols = [c for c in cc if c != '']
            if j > 0:
                Tpe.append(float(cols[0]))
                Kpe.append(float(cols[1]))
        fppe.close()
    else:
        input('No sample holder file...')
        sys.exit()

    if type == 'l':

        T_LT, K_LT = [], []
        for k in np.arange(1, len(T)):
            if T[k] > T[k-1]:
                T_LT.append(T[k])
                K_LT.append(K[k])
        T_LTpe, K_LTpe = [], []
        for k in np.arange(1, len(Tpe)):
            if Tpe[k] != Tpe[k-1]:
                T_LTpe.append(Tpe[k])
                K_LTpe.append(Kpe[k])
        K_LT, T_LT, K_LTpe, T_LTpe = np.array(K_LT), np.array(T_LT), np.array(K_LTpe), np.array(T_LTpe)

        T_LT_interp = np.linspace(-194, 0, 80)
        tck = interpolate.splrep(T_LT, K_LT, s=0)
        tckpe = interpolate.splrep(T_LTpe, K_LTpe, s=0)
        K_LT_interp = interpolate.splev(T_LT_interp, tck, der=0)
        K_LTpe_interp = interpolate.splev(T_LT_interp, tckpe, der=0)
        K_LT_corr = (K_LT_interp-K_LTpe_interp)*1e-5/(mass*1e-3)

        tckprime = interpolate.splrep(T_LT_interp, K_LT_corr, s=0)
        K_LT_corr_prime = interpolate.splev(T_LT_interp, tckprime, der=1)

        plotderivative = input('Plot derivative? (Y/n)')
        if plotderivative != 'n':
            plotderivativeonfig = input('On the same figure? (Y/n)')
        else:
            plotderivativeonfig = 'n'

        fig, ax1 = plt.subplots(1,1,figsize=(6,4))
        plt.xlabel('Temperature (K)')
        ax1.set_ylabel('Susceptibility (m3 kg-1)')
        ax1.set_ylim(0, np.max(K_LT_corr)+0.1*np.max(K_LT_corr))
        ax1.plot(T_LT_interp+273.15, K_LT_corr, 'k-', marker='.', ms='0', lw=1.5)
        if plotderivative != 'n' and plotderivativeonfig != 'n':
            ax2 = ax1.twinx()
            ax2.set_ylabel('Derivative of susceptibility (m3 kg-1 K-1)')
            ax2.set_ylim(np.min(K_LT_corr_prime)-0.1*np.min(K_LT_corr_prime), np.max(K_LT_corr_prime)+0.1*np.max(K_LT_corr_prime))
            ax2.plot(T_LT_interp+273.15, K_LT_corr_prime, 'r-', marker='.', ms='0', lw=1.5)
            fig.tight_layout()
            if save == 'y':
                plt.savefig(path + sample + '_suscLT.pdf', format='pdf', dpi=200, bbox_inches="tight")
        if plotderivative != 'n' and plotderivativeonfig == 'n':
            fig = plt.figure()
            plt.xlabel('Temperature (K)')
            plt.ylabel('Derivative of susceptibility (m3 kg-1 K-1)')
            plt.plot(T_LT_interp+273.15, K_LT_corr_prime, 'k-', marker='o', ms='4', lw=0.5)
            fig.tight_layout()
            if save == 'y':
                plt.savefig(path + sample + '_suscLT.pdf', format='pdf', dpi=200, bbox_inches="tight")


    if type == 'h':

        ## Distinguishing the heating from the cooling curve:
        T_HTh = np.array(T[1:T.index(np.max(T)) + 1])
        K_HTh = np.array(K[1:T.index(np.max(T)) + 1])
        T_HTc = np.array(T[T.index(np.max(T)):])[::-1]
        K_HTc = np.array(K[T.index(np.max(T)):])[::-1]

        T_HThpe = np.array(Tpe[:Tpe.index(np.max(Tpe)) + 1])
        K_HThpe = np.array(Kpe[:Tpe.index(np.max(Tpe)) + 1])
        T_HTcpe = np.array(Tpe[Tpe.index(np.max(Tpe)):])[::-1]
        K_HTcpe = np.array(Kpe[Tpe.index(np.max(Tpe)):])[::-1]

        tckh = interpolate.splrep(T_HTh, K_HTh, s=0)
        tckc = interpolate.splrep(T_HTc, K_HTc, s=0)
        tckhpe = interpolate.splrep(T_HThpe, K_HThpe, s=0)
        tckcpe = interpolate.splrep(T_HTcpe, K_HTcpe, s=0)

        T_LT_interp = np.linspace(40, np.min([T_HTh[-1],T_HThpe[-1]]),80)
        K_HTh_interp = interpolate.splev(T_LT_interp, tckh, der=0)
        K_HThpe_interp = interpolate.splev(T_LT_interp, tckhpe, der=0)
        K_HTc_interp = interpolate.splev(T_LT_interp, tckc, der=0)
        K_HTcpe_interp = interpolate.splev(T_LT_interp, tckcpe, der=0)

        K_HTh_corr = (K_HTh_interp-K_HThpe_interp)*1e-5/(mass*1e-3)
        K_HTc_corr = (K_HTc_interp-K_HTcpe_interp)*1e-5/(mass*1e-3)

        fig = plt.figure(figsize=(6,4))
        plt.xlabel('Temperature (C)')
        plt.ylabel('Susceptibility (m3 kg-1)')
        plt.xlim(0, 710)
        plt.ylim(0, 2.4)
        plt.plot(T_LT_interp, K_HTh_corr, 'r-', marker='o', ms='0', lw=1.5)
        plt.plot(T_LT_interp, K_HTc_corr, 'b-', marker='o', ms='0', lw=1.5)
        fig.tight_layout()

        if save == 'y':
            plt.savefig(path + sample + 'suscHT.pdf', format='pdf', dpi=400, bbox_inches="tight")

plt.show()

