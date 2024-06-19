import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import interpolate
from plotMFKLT import *

type = input('RT? LT? HT? (r/l/h)')
mass = float(eval(input('Mass of the sample (g)?  ')))

fp = open(sys.argv[1],'r',encoding="utf8", errors='ignore')

save = input('Save the figures? (y/N)')
path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'
sample = sys.argv[1].split('/')[-1].split('.')[0]

if type == 'r':

    T, K = [], []
    for j, line in enumerate(fp):
        cc = line.split(' ')
        cols = [c for c in cc if c != '']
        if j > 0:
            T.append(float(cols[0]))
            K.append(float(cols[6]))
    fp.close()

    T, K = np.array(T), np.array(K)
    K = K * 10e-6 / (mass * 1e-3)

    fig = plt.figure()

    plt.xlabel('Temperature (C)')
    #plt.ylabel('Susceptibility (10-6 m3 kg-1)')
    plt.ylabel('Normalized susceptibility')
    plt.xlim(0,710)
    #plt.ylim(0,1.1*np.max(K*1e6))
    #plt.ylim(0, 1.1 * np.max(K / K[0]))

    #plt.ylim(0,6)

    plt.plot(T,K*1e6/(K[0]*1e6),'k-',marker='o',ms='6',lw=0.5,mfc='k')
    fig.tight_layout()
    if save == 'y':
        plt.savefig(path + sample + '_norm.pdf', format='pdf', dpi=400, bbox_inches="tight")

if type == 'l' or type == 'h':

    T, K = [], []
    for j, line in enumerate(fp):
        cc = line.split(' ')
        cols = [c for c in cc if c != '']
        if j > 0:
            T.append(float(cols[0]))
            K.append(float(cols[1]))
    fp.close()

    fppe = open(sys.argv[2], 'r', encoding="utf8", errors='ignore')
    Tpe, Kpe = [], []
    for j, line in enumerate(fppe):
        cc = line.split(' ')
        cols = [c for c in cc if c != '']
        if j > 0:
            Tpe.append(float(cols[0]))
            Kpe.append(float(cols[1]))
    fppe.close()

    if type == 'l':

        TT, KK = [], []
        for k in np.arange(1, len(T)):
            if T[k] > T[k - 1]:
                TT.append(T[k])
                KK.append(K[k])
        TTpe, KKpe = [], []
        for k in np.arange(1, len(Tpe)):
            if Tpe[k] != Tpe[k - 1]:
                TTpe.append(Tpe[k])
                KKpe.append(Kpe[k])
        K, T, Kpe, Tpe = np.array(KK), np.array(TT), np.array(KKpe), np.array(TTpe)
        T_i = np.linspace(-194, 0, 80)
        tck = interpolate.splrep(T, K, s=0)
        tckpe = interpolate.splrep(Tpe, Kpe, s=0)
        K_i = interpolate.splev(T_i, tck, der=0)
        Kpe_i = interpolate.splev(T_i, tckpe, der=0)
        Khi = (K_i - Kpe_i) * 10e-6 / (mass * 1e-3)

        tckprime = interpolate.splrep(T_i, Khi, s=0)
        Khi_prime = interpolate.splev(T_i, tckprime, der=1)

        fig = plt.figure(figsize=(5,3))
        plt.xlabel('Temperature (K)')
        plt.ylabel('Susceptibility (10-6 m3 kg-1)')
        plt.ylim(0, np.max(Khi) + 0.1 * np.max(Khi))
        plt.plot(T_i+273.15, Khi, 'k-', marker='.', ms='4', lw=0.5)
        fig.tight_layout()
        if save == 'y':
            plt.savefig(path + sample + '_sus.pdf', format='pdf', dpi=400, bbox_inches="tight")

        # fig = plt.figure()
        # plt.title(sys.argv[1].split('/')[-1].split('.')[0])
        # plt.xlabel('Temperature (K)')
        # plt.ylabel('Derivative of susceptibility (10-6 m3 kg-1 K-1)')
        # plt.plot(T_i + 273.15, Khi_prime, 'k-', marker='.', ms='4', lw=0.5)
        # fig.tight_layout()
        #if save == 'y':
        #    plt.savefig(path + sample + '_sus.pdf', format='pdf', dpi=200, bbox_inches="tight")

    if type == 'h':
        Theat = np.array(T[1:T.index(np.max(T)) + 1])
        Kheat = np.array(K[1:T.index(np.max(T)) + 1])
        Tcool = np.array(T[T.index(np.max(T)):])[::-1]
        Kcool = np.array(K[T.index(np.max(T)):])[::-1]

        Tpeheat = np.array(Tpe[:Tpe.index(np.max(Tpe)) + 1])
        Kpeheat = np.array(Kpe[:Tpe.index(np.max(Tpe)) + 1])
        Tpecool = np.array(Tpe[Tpe.index(np.max(Tpe)):])[::-1]
        Kpecool = np.array(Kpe[Tpe.index(np.max(Tpe)):])[::-1]

        tck_heat = interpolate.splrep(Theat, Kheat, s=0)
        tck_cool = interpolate.splrep(Tcool, Kcool, s=0)
        tckpe_heat = interpolate.splrep(Tpeheat, Kpeheat, s=0)
        tckpe_cool = interpolate.splrep(Tpecool, Kpecool, s=0)

        T_i = np.linspace(40, np.min([Theat[-1], Tpeheat[-1]]), 80)

        Kheat_i = interpolate.splev(T_i, tck_heat, der=0)
        Kpeheat_i = interpolate.splev(T_i, tckpe_heat, der=0)
        Kcool_i = interpolate.splev(T_i, tck_cool, der=0)
        Kpecool_i = interpolate.splev(T_i, tckpe_cool, der=0)

        Khi_heat = (Kheat_i - Kpeheat_i) * 10e-6 / (mass * 1e-3)
        Khi_cool = (Kcool_i - Kpecool_i) * 10e-6 / (mass * 1e-3)

        fig = plt.figure()
        plt.title(sys.argv[1].split('/')[-1].split('.')[0])
        plt.xlabel('Temperature (C)')
        plt.ylabel('Susceptibility (10-6 m3 kg-1)')
        plt.xlim(0, 710)
        plt.ylim(0, 2.4)
        plt.plot(T_i, Khi_heat, 'r-', marker='.', ms='4', lw=0.5)
        plt.plot(T_i, Khi_cool, 'b-', marker='.', ms='4', lw=0.5)
        fig.tight_layout()

        if save == 'y':
            plt.savefig(path + sample + '.pdf', format='pdf', dpi=400, bbox_inches="tight")

plt.show()

