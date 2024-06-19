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

def plotVSMdata(B,M,type,mass=0,der='',ylim=0,newfig='y'):

    if newfig == 'y':
        fig = plt.figure(figsize=(4,3))
    else:
        fig = newfig

    ax = plt.subplot()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(visible=1, which='major', axis='both')

    if type == 'IRM':
        if der == 'y':
            B = np.array([B[k] for k in np.arange(len(B)) if B[k]>0])
            M = np.array([M[k] for k in np.arange(len(B)) if B[k]>0])
            ax.tick_params(axis="x")
            ax.tick_params(axis="y")
            ax.set_xscale('log')
            ax.set_ylabel('IRM (A m2)')
            ax.plot(B*1000, M, 'k-', lw=0.5, marker='.', ms=3, mew=0.5)

            cs = CubicSpline(B*1000, M)
            xs = np.logspace(np.min(np.log10(B[3:]*1000)),np.max(np.log10(B[3:]*1000)),30)
            ax2 = ax.twinx()
            ax2.plot(xs, cs(xs, 1), ls='-', color='darkred',lw=2)
            ax2.tick_params(axis="y")
            ax.set_xlabel('Field (mT)')
            ax2.set_ylabel('dM/dB (A m2 T-1)')
        else:
            ax.set_xlabel('Field (T)')
            ax.set_ylabel('IRM (A m2)')
            ax.plot(B, M, 'k-', lw=2)

    elif type == 'Bcr':
        ax.set_xlabel('Field (T)')
        ax.set_ylabel('IRM (A m2)')
        ax.plot(B, M, 'k-', lw=2)
    else:
        ax.set_xlabel('Field (T)')
        if mass != 0:
            ax.set_ylabel('Moment (A m2 kg-1)')
            ax.plot(B, M/mass, 'k-', lw=2)
        else:
            ax.set_ylabel('Moment (A m2)')
            ax.plot(B, M, 'k-', lw=2)

    if ylim != 0:
        ax.set_ylim(-ylim,ylim)

    fig.tight_layout()
    return fig

def plotVSMLTdata(T,M,norm=False,ylim=0,newfig='y'):

    M = np.array(M)

    if newfig == 'y':
        fig = plt.figure()
    else:
        fig = newfig

    ax = plt.subplot()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(visible=1, which='major', axis='both')
    ax.set_xlabel('Temperature (K)', fontsize=13)
    if norm == False:
        ax.set_ylabel('sIRM (Am2)', fontsize=13)
        ax.plot(T, M, 'k-', lw=0.5, marker='o', ms=5, mew=0.5)
        if ylim != 0:
            ax.set_ylim(-ylim,ylim)
    elif norm == True:
        ax.set_ylabel('sIRM / sIRM_0', fontsize=13)
        ax.set_ylim(0,1.05)
        ax.plot(T, M/M[0], 'k-', lw=0.5, marker='o', ms=5, mew=0.5)

    fig.tight_layout()
    return fig


def getIRMacq(B, M, Bvalue):
    id1 = Get_closest_id(B, Bvalue)
    if B[id1] < Bvalue:
        B1, B2, M1, M2 = B[id1], B[id1+1], M[id1], M[id1+1]
    else:
        B1, B2, M1, M2 = B[id1-1], B[id1], M[id1-1], M[id1]
    slope, intercept = np.polyfit([B1,B2], [M1,M2], 1)
    IRMacq = slope*Bvalue+intercept
    return IRMacq

def getBcr(B, M):
    print()
    id1 = Get_closest_id(M, 0)
    if M[id1] > 0:
        B1, B2, M1, M2 = B[id1], B[id1+1], M[id1], M[id1+1]
    else:
        B1, B2, M1, M2 = B[id1-1], B[id1], M[id1-1], M[id1]
    slope, intercept = np.polyfit([B1,B2], [M1,M2], 1)
    Bcr = -intercept/slope
    return Bcr

def plotCorrHysteresis(B, M, type, mass=0, linear=True, interval=0.3, ylim=0, newfig='y', shownocorr=True):

    maxB = np.min([np.max(B), 1.5])
    Btmp = B
    B = np.array([B[k] for k in np.arange(len(B)) if abs(Btmp[k]) < maxB])
    M = np.array([M[k] for k in np.arange(len(Btmp)) if abs(Btmp[k]) < maxB])

    Bhalf = np.array([B[k] for k in np.arange(int(len(B)/2))])
    Mhalf = np.array([M[k] for k in np.arange(int(len(M)/2))])
    Bfit = np.array([Bhalf[k] for k in np.arange(len(Bhalf)) if Bhalf[k] > maxB - interval])
    Mfit = np.array([Mhalf[k] for k in np.arange(len(Bhalf)) if Bhalf[k] > maxB - interval])

    if linear == True:
        popt, pcov = curve_fit(line, Bfit, Mfit)
        Mcorr = M-popt[0]*B
        Mcorrhalf = Mhalf-popt[0]*Bhalf
        idb = Get_closest_id(Bhalf,0)
        Ms = Mcorrhalf[0]
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

        if mass != 0:
            print('Ms = '+f'{Ms/mass:.3e}'+' A m2 kg-1')
            print('Mrs = '+f'{Mrs/mass:.3e}'+' A m2 kg-1')
            print('Bc = ' + f'{Bc * 1000:.2f}' + ' mT')
            print('HF slope = ' + f'{popt[0]:.2e}' + ' A m2 T-1')
            print('HF slope = ' + f'{popt[0] * 4 * np.pi * 1e-7/mass:.2e}' + ' m3 kg-1')
        else:
            print('Ms = '+f'{Ms:.3e}'+' A m2')
            print('Mrs = '+f'{Mrs:.3e}'+' A m2')
            print('Bc = ' + f'{Bc * 1000:.2f}' + ' mT')
            print('HF slope = ' + f'{popt[0]:.2e}' + ' A m2 T-1')
            print('HF slope = ' + f'{popt[0] * 4 * np.pi * 1e-7:.2e}' + ' m3')

        if shownocorr == True:
            fig = plotVSMdata(B, M, type=type, mass=mass, ylim=ylim, newfig=newfig)
            plt.xlim(-1,1)
            plt.plot(B, Mcorr/mass, 'r-', lw=2)
        elif shownocorr == False:
            fig = plotVSMdata(B, Mcorr, type=type, mass=mass, ylim=ylim, newfig=newfig)
        fig.tight_layout()

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
        Ms, Khi, alpha = popt
        Mcorr = poly(Bfit,Ms,Khi,alpha)
        if mass != 0:
            print('Ms = '+f'{Ms/mass:.3e}'+' A m2 kg-1')
            print('Mrs = '+f'{Mrs/mass:.3e}'+' A m2')
            print('Bc = '+f'{Bc*1000:.2f}'+' mT')
        else:
            print('Ms = '+f'{Ms:.3e}'+' A m2')
            print('Mrs = ' + f'{Mrs:.3e}' + ' A m2')
            print('Bc = ' + f'{Bc * 1000:.2f}' + ' mT')
        print('HF slope = ' + f'{Khi:.3e}' + ' A m2 T-1')
        print('alpha = ' + f'{alpha:.3e}' + ' A m2 T-2'+'\n')
        fig = plotVSMdata(B, M, type=type, mass=mass, ylim=ylim, newfig=newfig)
        plt.plot(Bfit, Mcorr, 'r-', lw=1)
        fig.tight_layout()




