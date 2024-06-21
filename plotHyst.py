import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib import ticker
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline

def Get_closest_id(L,value):
    return list(L).index(min(L, key=lambda x:abs(x-value)))

def line(x, a, b):
    return a*x+b

def poly(x,a,b,c):
    return a + b*x + c*x**(-2)

def plotIRMacq(B,M,Bvalue=0,der='y',ylim=()):

    if Bvalue != 0:
        id1 = Get_closest_id(B, Bvalue*1e-3)
        if B[id1] < Bvalue*1e-3:
            B1, B2, M1, M2 = B[id1], B[id1+1], M[id1], M[id1+1]
        else:
            B1, B2, M1, M2 = B[id1-1], B[id1], M[id1-1], M[id1]
        slope, intercept = np.polyfit([B1,B2], [M1,M2], 1)
        IRMacq = slope*Bvalue*1e-3+intercept

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(visible=1, which='major', axis='both')

    B = np.array([B[k] for k in np.arange(len(B)) if B[k] > 0])
    M = np.array([M[k] for k in np.arange(len(B)) if B[k] > 0])
    ax.set_xscale('log')
    ax.set_xlabel('Field (T)')
    ax.set_ylabel('M (A m2)')
    ax.plot(B, M, 'k-', lw=1.5, marker='.', ms=0)
    ax.set_xlim(1e-3, np.max(B))
    if ylim != ():
        ax.set_ylim(ylim[0], ylim[1])

    if der == 'y':
        cs = CubicSpline(B, M)
        xs = np.logspace(np.min(np.log10(B[3:])), np.max(np.log10(B[3:] * 1000)), 30)
        ax2 = ax.twinx()
        ax2.plot(xs, cs(xs, 1), ls='-', color='darkred', lw=1.5)
        ax2.tick_params(axis="y")
        ax2.set_ylabel('dM/dB (A m2 T-1)')
    else:
        ax.set_ylabel('IRM (A m2)')

    return IRMacq, Bvalue

def plotBcr(B,M,ylim=()):

    id1 = Get_closest_id(M, 0)
    if M[id1] > 0:
        B1, B2, M1, M2 = B[id1], B[id1+1], M[id1], M[id1+1]
    else:
        B1, B2, M1, M2 = B[id1-1], B[id1], M[id1-1], M[id1]
    slope, intercept = np.polyfit([B1,B2], [M1,M2], 1)
    Bcr = -intercept/slope

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(visible=1, which='major', axis='both')

    ax.set_xlabel('Field (T)')
    ax.set_ylabel('M (A m2)')
    if ylim != ():
        ax.set_ylim(ylim[0],ylim[1])
    ax.plot(B, M, 'k-', lw=1.5, marker='.', ms=0)

    return Bcr

def plotHyst(B,M,mass=1,linear=True,interval=0.3,ylim=(),shownocorr=False):

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(visible=1, which='major', axis='both')
    ax.set_xlabel('Field (T)')
    ax.set_ylabel('Moment (A m2)')
    if shownocorr == True:
        ax.plot(B, M, 'k-', lw=1.5)

    B, M = np.array(B), np.array(M)
    Bhalf = np.array([B[k] for k in np.arange(int(len(B)/2))])
    Mhalf = np.array([M[k] for k in np.arange(int(len(M)/2))])
    Bfit = np.array([Bhalf[k] for k in np.arange(len(Bhalf)) if Bhalf[k] > np.max(B) - interval])
    Mfit = np.array([Mhalf[k] for k in np.arange(len(Bhalf)) if Bhalf[k] > np.max(B) - interval])

    Ms, Mrs, Bc, sHF, kHF, alpha = 0, 0, 0, 0, 0, 0

    if linear == True:
        popt, pcov = curve_fit(line, Bfit, Mfit)
        Mcorr = M-popt[0]*B
        Mcorrhalf = Mhalf-popt[0]*Bhalf
        idb = Get_closest_id(Bhalf,0)
        Ms = Mcorrhalf[0]
        sHF = popt[0]
        kHF = sHF*4*np.pi*1e-7
        if Bhalf[idb] > 0:
            B1, B2, M1, M2 = Bhalf[idb], Bhalf[idb + 1], Mcorrhalf[idb], Mcorrhalf[idb + 1]
        else:
            B1, B2, M1, M2 = Bhalf[idb], Bhalf[idb - 1], Mcorrhalf[idb], Mcorrhalf[idb - 1]
        slope, intercept = np.polyfit([B1, B2], [M1, M2], 1)
        Mrs = intercept
        idm = Get_closest_id(Mcorrhalf, 0)
        if Mcorrhalf[idm] > 0:
            B1, B2, M1, M2 = Bhalf[idm], Bhalf[idm + 1], Mcorrhalf[idm], Mcorrhalf[idm + 1]
        else:
            B1, B2, M1, M2 = Bhalf[idm], Bhalf[idm - 1], Mcorrhalf[idm], Mcorrhalf[idm - 1]
        slope, intercept = np.polyfit([B1, B2], [M1, M2], 1)
        Bc = -intercept/slope

        ax.plot(B, Mcorr, '-', color='darkred', lw=1.5)

    elif linear == False:
        idb = Get_closest_id(Bhalf, 0)
        if Bhalf[idb] > 0:
            B1, B2, M1, M2 = Bhalf[idb], Bhalf[idb + 1], Mhalf[idb], Mhalf[idb + 1]
        else:
            B1, B2, M1, M2 = Bhalf[idb], Bhalf[idb - 1], Mhalf[idb], Mhalf[idb - 1]
        slope, intercept = np.polyfit([B1, B2], [M1, M2], 1)
        Mrs = intercept
        idm = Get_closest_id(Mhalf, 0)
        if Mhalf[idm] > 0:
            B1, B2, M1, M2 = Bhalf[idm], Bhalf[idm + 1], Mhalf[idm], Mhalf[idm + 1]
        else:
            B1, B2, M1, M2 = Bhalf[idm], Bhalf[idm - 1], Mhalf[idm], Mhalf[idm - 1]
        slope, intercept = np.polyfit([B1, B2], [M1, M2], 1)
        Bc = -intercept/slope

        popt, pcov = curve_fit(poly, Bfit, Mfit)
        Ms, kHF, alpha = popt
        Mcorr = poly(Bfit,Ms,kHF,alpha)

        ax.plot(Bfit, Mcorr, '-', color='darkred', lw=1.5)

    return Ms, Mrs, Bc, sHF, kHF, alpha

def plotVSMLT(T,M,norm=False,ylim=()):

    M = np.array(M)

    ax = plt.subplot()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(visible=1, which='major', axis='both')

    ax.set_xlabel('Temperature (K)')
    if norm == False:
        ax.set_ylabel('sIRM (A m2)')
        ax.plot(T, M, 'k-', lw=0.5, marker='o', ms=4, mew=0.5)
        if ylim != 0:
            ax.set_ylim(ylim[0],ylim[1])
    elif norm == True:
        ax.set_ylabel('Normalized sIRM')
        ax.set_ylim(0,1.05)
        ax.plot(T, M/M[0], 'k-', lw=0.5, marker='o', ms=4, mew=0.5)

    return



