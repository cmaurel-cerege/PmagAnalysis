import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from returnEmpiricalFactors import *
import random

def calc_Mlost_vector(Mx,My,Mz,AFM,id1,id2,delta):

    Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)
    Mlost = np.sqrt((Mx[id1:id2] - Mx[id1]) ** 2 + (My[id1:id2] - My[id1]) ** 2 + (Mz[id1:id2] - Mz[id1]) ** 2) + delta
    AF = AFM[id1:id2]

    return list(Mlost), list(AF)

def calc_Mlost_norm(Mx,My,Mz):

    Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)
    Mlost = []
    for k in np.arange(len(Mx)-1):
        Mlost.append(np.sqrt((Mx[k+1]-Mx[k])**2+(My[k+1]-My[k])**2+(Mz[k+1]-Mz[k])**2))

    return Mlost

def plot_NRM_vs_Mlost_vector(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, mass=True, annot=False):

    NRMlost, AFlost, Mlost = [], [], []
    comp = int(eval(input('Number of components?  ')))
    id = []
    for n in np.arange(comp):
        print('Component '+str(n+1))
        id.append(int(eval(input('First datapoint?  '))))
    id.append(len(NRMx))

    for k in np.arange(len(id)-1):
        if len(NRMlost) == 0:
            NRMlost = calc_Mlost_vector(NRMx, NRMy, NRMz, AF, id[k], id[k + 1], 0)[0]
            Mlost = calc_Mlost_vector(Mx, My, Mz, AF, id[k], id[k + 1], 0)[0]
        else:
            NRMlost += calc_Mlost_vector(NRMx, NRMy, NRMz, AF, id[k], id[k+1], NRMlost[-1])[0]
            Mlost += calc_Mlost_vector(Mx, My, Mz, AF, id[k], id[k+1], Mlost[-1])[0]

    fig = plt.figure(figsize=(6,3))
    #plt.xlim(0,6e-5)
    #plt.ylim(0,5e-7)
    if mass == True:
        plt.xlabel(type+r' lost (A m$^{2}$ kg$^{-1}$)')
        plt.ylabel('NRM'+r' lost (A m$^{2}$ kg$^{-1}$)')
    else:
        plt.xlabel(type+r' lost (A m$^{2}$)')
        plt.ylabel('NRM'+r' lost (A m$^{2}$)')
    plt.plot(Mlost, NRMlost, 'ko', ms=7, lw=0)
    plt.ticklabel_format(axis='both', style='sci')
    if annot == True:
        for i in np.arange(len(Mlost)):
            plt.text(Mlost[i], NRMlost[i], str(i), fontsize=8)

    return id, NRMlost, Mlost, AFlost

def plot_NRM_vs_Mlost_norm(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, mass=True, annot=False):

    NRMlost, AFlost, Mlost = [], [], []
    comp = int(eval(input('Number of components?  ')))
    id = []
    for n in np.arange(comp):
        print('Component '+str(n+1))
        id.append(int(eval(input('First datapoint?  '))))
    id.append(len(NRMx))

    NRMlost, Mlost = [], []
    NRMlost_norm = calc_Mlost_norm(NRMx, NRMy, NRMz)
    NRMlost_tmp = 0
    for k in np.arange(len(NRMlost_norm)):
        NRMlost_tmp += NRMlost_norm[k]
        NRMlost.append(NRMlost_tmp)

    Mlost_norm = calc_Mlost_norm(Mx, My, Mz)
    Mlost_tmp = 0
    for k in np.arange(len(Mlost_norm)):
        Mlost_tmp += Mlost_norm[k]
        Mlost.append(Mlost_tmp)

    AFlost = AF

    fig = plt.figure(figsize=(6,3))
    plt.title('Norms added')
    if mass == True:
        plt.xlabel(type+r' lost (A m$^{2}$ kg$^{-1}$)', fontsize=13)
        plt.ylabel('NRM'+r' lost (A m$^{2}$ kg$^{-1}$)', fontsize=13)
    else:
        plt.xlabel(type+r' lost (A m$^{2}$)', fontsize=13)
        plt.ylabel('NRM'+r' lost (A m$^{2}$)', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.scatter(Mlost, NRMlost, s=50, c='lightgray', marker='o',edgecolor='k',linewidths=0.5)
    plt.ticklabel_format(axis='both', style='sci')
    if annot == True:
        for i in np.arange(len(Mlost)):
            plt.text(Mlost[i], NRMlost[i], str(i), fontsize=8)

    return id, NRMlost, Mlost, AFlost

def plot_REMprime(NRMx, NRMy, NRMz, Mx, My, Mz, AF, annot=True):

    NRM = np.array([np.sqrt((NRMx[k+1]-NRMx[k])**2+(NRMy[k+1]-NRMy[k])**2+(NRMz[k+1]-NRMz[k])**2) for k in np.arange(len(NRMx)-1)])
    M = np.array([np.sqrt((Mx[k+1]-Mx[k])**2+(My[k+1]-My[k])**2+(Mz[k+1]-Mz[k])**2) for k in np.arange(len(Mx)-1)])

    REMprime = NRM/M
    print(len(REMprime),len(AF))

    fig = plt.figure(figsize=(5, 5))
    plt.yscale("log")
    plt.xlabel('AF step (mT)', fontsize=13)
    plt.ylabel('REM prime', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.ylim(1e-3,1e-1)
    plt.scatter(AF, REMprime, s=50, c='lightgray', marker='o', edgecolor='k', linewidths=0.5)
    if annot == True:
        for i in np.arange(len(REMprime)):
            plt.text(AF[i], REMprime[i], str(i), fontsize=8)


def calc_REMslope_vector(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, tcrm, mineral, domain,biasfield=100, calib=False, mass=True, annot=False):

    if calib == True:
        idcomp = 17
        id, NRMlost, Mlost, AFlost = plot_NRM_vs_Mlost_vector(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, mass, annot)
        plt.draw()
        plt.show(block=False)
        res = stats.linregress(Mlost[idcomp:], NRMlost[idcomp:])
        REMp = res.slope
        plt.plot(Mlost[idcomp:], res.slope * np.array(Mlost[idcomp:]) + res.intercept, 'r-', lw=1)

        print("R^2 = " + f'{res.rvalue:.2f}')
        print("Slope (x1000) = " + f'{res.slope * 1000:.2f}' + ' +/- ' + f'{2 * res.stderr * 1000:.2f}' + ' (2 sigma)')
        field = float(eval(input("Field applied (uT)? –> ")))
        if type == 'IRM':
            print("IRM proportionality constant = " + str(field / REMp))
        if type == 'ARM':
            print("ARM proportionality constant = " + str(100 * REMp / field))

    else:
        id, NRMlost, Mlost, AFlost = plot_NRM_vs_Mlost_vector(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, mass, annot)
        plt.draw()
        plt.show(block=False)

        REMp, RMEp2se, REMpsd = [], [], []
        for k in np.arange(len(id)-1):
            res = stats.linregress(Mlost[id[k]:id[k+1]], NRMlost[id[k]:id[k+1]])
            plt.plot(Mlost[id[k]:id[k + 1]], res.slope * np.array(Mlost[id[k]:id[k + 1]]) + res.intercept, 'r-', lw=1)
            REMp.append(res.slope)
            RMEp2se.append(2*res.stderr)
            REMpsd.append(res.stderr*np.sqrt(len(Mlost[id[k]:id[k + 1]])-2))

        seg = int(eval(input("Segment for paleointensity? (1,2,3...)")))

        print('\n'+str(type))
        print(RMEp2se[seg-1])
        print('Slope = '+f'{REMp[seg-1]:.5f}'+' +/- '+f'{REMpsd[seg-1]:.5f}'+' (1 s.d.)')

        file = type+'_empirical_factors.xlsx'
        sheet = tcrm+' '+mineral
        empiricalFactor = returnFactor(file,sheet,domain)
        mulogEmpiricalFactor = np.mean(np.log(np.array(empiricalFactor)))
        sdlogEmpiricalFactor = np.std(np.log(np.array(empiricalFactor)))

        print(mulogEmpiricalFactor,sdlogEmpiricalFactor)

        print('Geometric mean empirical factor = '+f'{np.exp(mulogEmpiricalFactor):.2f}')
        print('Geometric SD empirical factor = ' + f'{np.exp(sdlogEmpiricalFactor):.2f}')

        paleointensity = []
        for k in np.arange(10000):
            logEF = np.random.normal(mulogEmpiricalFactor, sdlogEmpiricalFactor, 1)[0]
            EF = np.exp(logEF)
            slope = np.random.normal(REMp[seg-1], REMpsd[seg-1], 1)[0]
            if type == 'IRM':
                paleointensity.append(EF*slope)
            elif type == 'ARM':
                paleointensity.append(biasfield/EF * slope)

        min95 = sorted(paleointensity)[249]
        max95 = sorted(paleointensity)[9750]
        print('95% confidence interval = [' + f'{min95:.0f}'+' uT, '+ f'{max95:.0f}'+' uT]')
        print('Mean paleointensity = '+f'{(min95+max95)/2:.0f}'+' uT\n')

    return paleointensity



def calc_REMslope_norm(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, mass=True, annot=True):

    id, NRMlost, Mlost, AFlost = plot_NRM_vs_Mlost_norm(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, mass, annot)
    plt.draw()
    plt.show(block=False)

    for k in np.arange(len(id)-1):
        res = stats.linregress(Mlost[id[k]:id[k+1]], NRMlost[id[k]:id[k+1]])
        REMp = res.slope
        plt.plot(Mlost[id[k]:id[k+1]], res.slope * np.array(Mlost[id[k]:id[k+1]]) + res.intercept, 'r-', lw=1)
        print(res.slope)
        print("R^2 = " + f'{res.rvalue:.2f}')
        print("Slope (x1000) = "+f'{res.slope*1000:.2f}'+' +/- '+f'{2*res.stderr*1000:.2f}'+' (2 sigma)')
        if type == 'IRM':
            print("IRM paleofield estimate = "+str(2500*REMp)+r" uT")
        if type == 'ARM':
            print("ARM paleofield estimate = "+str((100/3.3)*REMp)+r" uT")

    return