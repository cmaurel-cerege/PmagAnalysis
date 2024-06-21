import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def Plot_AF_demag(Mx, My, Mz, AF, type, color='lightgray', marker='o'):

    Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)
    M = np.sqrt(Mx**2 + My**2 + Mz**2)

    plt.xlabel('AF level (mT)')
    plt.ylabel('Normalized moment')
    plt.xlim(0, max(AF)+2)
    plt.ylim(0,1.1)
    plt.plot(AF, M/M[0], marker=marker, color='k', mec='k', mfc=color, ms=8, mew=0.5, lw=0.5, label=type)
    plt.legend()

    return


def Plot_thermal_demag(Mx, My, Mz, T, norm=False, color='lightgray', marker='o'):

    Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)
    NRM = np.sqrt(Mx**2 + My**2 + Mz**2)

    fig = plt.figure(figsize=(6,3))
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Moment (A m2)')
    plt.xlim(0, np.max(T)+10)
    plt.ylim(0, 1.1*np.max(NRM))
    plt.plot(T,NRM,marker=marker,mec='k',mfc=color,ms=5,lw=0.5)

    return


def Plot_Thellier(Mx, My, Mz, Thstep, type):

    NRMx, NRMy, NRMz, pTRMx, pTRMy, pTRMz, cHx, cHy, cHz, tHx, tHy, tHz = [], [], [], [], [], [], [], [], [], [], [], []
    zStep, iStep, cStep, tStep = [], [], [], []
    for k in np.arange(len(type)):
        if type[k] == 'Z':
            NRMx.append(Mx[k])
            NRMy.append(My[k])
            NRMz.append(Mz[k])
            zStep.append(Thstep[k])
        elif type[k] == 'I':
            pTRMx.append(Mx[k])
            pTRMy.append(My[k])
            pTRMz.append(Mz[k])
            iStep.append(Thstep[k])
        elif type[k] == 'C':
            cHx.append(Mx[k])
            cHy.append(My[k])
            cHz.append(Mz[k])
            cStep.append([Thstep[k],zStep[-1]])
        elif type[k] == 'T':
            tHx.append(Mx[k])
            tHy.append(My[k])
            tHz.append(Mz[k])
            tStep.append(Thstep[k])
    NRMx, NRMy, NRMz, pTRMx, pTRMy, pTRMz, cHx, cHy, cHz, tHx, tHy, tHz = np.array(NRMx), np.array(NRMy), np.array(NRMz), np.array(pTRMx), np.array(pTRMy), np.array(pTRMz), np.array(cHx), np.array(cHy), np.array(cHz), np.array(tHx), np.array(tHy), np.array(tHz)
    NRM = np.sqrt(NRMx**2 + NRMy**2 + NRMz**2)

    pTRMgained = np.concatenate([np.array([0]),np.sqrt((pTRMx-NRMx[1:])**2+(pTRMy-NRMy[1:])**2+(pTRMz-NRMz[1:])**2)])

    fig = plt.figure(figsize=(6,4))
    plt.xlabel('Temperature (C)')
    plt.ylabel('Normalized moment')
    plt.xlim(10, max(zStep) + 10)
    plt.ylim(0,1.1)
    plt.plot(zStep,NRM/NRM[0],color='k',marker='o',ms=8,mfc='darkred',mec='k',mew=0.5,lw=0.5,label='Moment remaining')
    plt.plot([zStep[0]]+iStep,pTRMgained/NRM[0],color='k',marker='o',ms=8,mfc='darkblue',mec='k',mew=0.5,lw=0.5,label='pTRM gained')
    plt.legend()

    return


def Plot_Aray(Mx, My, Mz, Thstep, type):

    NRMx, NRMy, NRMz, pTRMx, pTRMy, pTRMz, cHx, cHy, cHz, tHx, tHy, tHz = [], [], [], [], [], [], [], [], [], [], [], []
    zStep, iStep, cStep, tStep = [], [], [], []
    for k in np.arange(len(type)):
        if type[k] == 'Z':
            NRMx.append(Mx[k])
            NRMy.append(My[k])
            NRMz.append(Mz[k])
            zStep.append(Thstep[k])
        elif type[k] == 'I':
            pTRMx.append(Mx[k])
            pTRMy.append(My[k])
            pTRMz.append(Mz[k])
            iStep.append(Thstep[k])
        elif type[k] == 'C':
            cHx.append(Mx[k])
            cHy.append(My[k])
            cHz.append(Mz[k])
            cStep.append([Thstep[k],zStep[-1]])
        elif type[k] == 'T':
            tHx.append(Mx[k])
            tHy.append(My[k])
            tHz.append(Mz[k])
            tStep.append(Thstep[k])
    NRMx, NRMy, NRMz, pTRMx, pTRMy, pTRMz, cHx, cHy, cHz, tHx, tHy, tHz = np.array(NRMx), np.array(NRMy), np.array(NRMz), np.array(pTRMx), np.array(pTRMy), np.array(pTRMz), np.array(cHx), np.array(cHy), np.array(cHz), np.array(tHx), np.array(tHy), np.array(tHz)
    NRM = np.sqrt(NRMx ** 2 + NRMy ** 2 + NRMz ** 2)

    pTRMgained = np.concatenate([np.array([0]),np.sqrt((pTRMx-NRMx[1:])**2+(pTRMy-NRMy[1:])**2+(pTRMz-NRMz[1:])**2)])

    fig = plt.figure(figsize=(6,4))
    plt.xlabel('Normalized pTRM gained')
    plt.ylabel('Normalized NRM')
    plt.xlim(-0.02, 1.05)
    plt.ylim(-0.02, 1.05)
    plt.plot(pTRMgained / NRM[0], NRM / NRM[0], color='k', marker='o', mfc='darkred', mec='k', mew=0.5, ms=8, lw=0.5)

    cHgained = []
    for k in np.arange(len(cStep)):
        id1 = zStep.index(cStep[k][1])
        cHgained.append(np.sqrt((cHx[k]-NRMx[id1])**2 + (cHy[k]-NRMy[id1])**2 + (cHz[k]-NRMz[id1])**2))
    if cHgained != []:
        for k in np.arange(len(cStep)):
            id1 = zStep.index(cStep[k][1])
            id2 = zStep.index(cStep[k][0])
            plt.scatter(cHgained[k]/NRM[0], NRM[id2]/NRM[0], s=75, facecolors='none', marker='^', edgecolors='k', zorder=3)
            plt.hlines(y=NRM[id1]/NRM[0], xmin=cHgained[k]/NRM[0], xmax=pTRMgained[id1]/NRM[0], lw=0.5, color='k', ls='-')
            plt.vlines(x=cHgained[k]/NRM[0], ymin=NRM[id1]/NRM[0], ymax=NRM[id2]/NRM[0], lw=0.5, color='k', ls='-')

    tHgained = []
    for k in np.arange(len(tStep)):
        id = zStep.index(tStep[k])
        tHgained.append(np.sqrt((tHx[k]-NRMx[id])**2 + (tHy[k]-NRMy[id])**2 + (tHz[k]-NRMz[id])**2))
    if tHgained != []:
         plt.scatter(np.array(tHgained)/NRM[0], NRM/NRM[0], s=65, c='w', marker='s', edgecolors='k', zorder=3)

    return NRMx, NRMy, NRMz, NRM, pTRMgained, cHgained, zStep, cStep

def Stat_Thellier(NRM,pTRMgained,cHgained,zStep,cStep,field):

    idstart = int(eval(input("Start step for analysis? ")))
    idend = int(eval(input("Start step for analysis? (max=99) ")))
    print(len(NRM))

    if idend == 99:
        idend = len(NRM)

    res = stats.linregress(pTRMgained[idstart:idend], NRM[idstart:idend])

    fig = plt.figure()
    plt.xlabel('pTRM gained normalized by NRM0')
    plt.ylabel('NRM normalized by NRM0')
    plt.xlim(-0.02, 1.05)
    plt.ylim(-0.02, 1.05)
    plt.plot(pTRMgained / NRM[0], NRM / NRM[0], color='k', marker='o', markerfacecolor='darkred', markeredgecolor='k',
             ms=6, lw=0.5)
    plt.plot(pTRMgained[idstart:idend]/NRM[0], res.slope*pTRMgained[idstart:idend]/NRM[0] + res.intercept/NRM[0], 'g-', lw=1.2)
    print("Paleointensity = " + f'{-field * res.slope:.2f}' + " +/- " + f'{field * res.stderr:.2f}' + " (1 s.e.)")

    xprime = [(NRM[k]-res.intercept)/res.slope for k in np.arange(idstart,idend)]
    yprime = [res.slope*pTRMgained[k]+res.intercept for k in np.arange(idstart,idend)]
    dxprime = abs(xprime[0] - xprime[-1])
    dyprime = abs(yprime[0] - yprime[-1])

    f = dyprime / abs(res.intercept)
    beta = res.stderr / abs(res.slope)
    g = 1-np.sum([(yprime[k+1]-yprime[k])**2 for k in np.arange(len(yprime)-1)])/dyprime**2
    q = f * g / beta

    print("f = " + f'{f:.3f}')
    print("beta = " + f'{beta:.3f}')
    print("g = " + f'{g:.3f}')
    print("q = " + f'{q:.1f}')

    if cHgained != []:
        DRATS = np.sum([abs(cHgained[k]-pTRMgained[zStep.index(cStep[k][1])]) for k in np.arange(len(cStep))])/pTRMgained[-1]
        print("DRATS = "+f'{DRATS*100:.1f}'+'%')

    return

def Plot_thellier_arm(Mx, My, Mz, step, color='k', line=0.5, marker='o', path=''):

    Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)
    NRM = [np.sqrt(Mx[0]**2+My[0]**2+Mz[0]**2),np.sqrt(Mx[1]**2+My[1]**2+Mz[1]**2)]
    T = step[0:2]
    for k in np.arange(2,len(step)):
        if step[k]>step[k-1] and step[k]>step[k-2]:
            NRM.append(np.sqrt(Mx[k] ** 2 + My[k] ** 2 + Mz[k] ** 2))
            T.append(step[k])
    NRM = np.array(NRM)
    print(NRM)
    fig = plt.figure()
    plt.xlabel('Temperature (C)')
    plt.ylabel('ARM normalized to ARM0')
    plt.xlim(10, max(step) + 10)
    #plt.ylim(0.6,1.4)
    plt.plot(T, NRM/NRM[0], c='k', marker='o', markeredgecolor='k', markerfacecolor='darkred', ms=6, lw=0.5, zorder=3)
    fig.tight_layout()












# def Plot_ratio_demag(Mxa, Mya, Mza, Mxb, Myb, Mzb, step, color='k', label, path):
#
#     Mxa, Mya, Mza, Mxb, Myb, Mzb = np.array(Mxa), np.array(Mya), np.array(Mza), np.array(Mxb), np.array(Myb), np.array(Mzb)
#
#     Ma = np.sqrt(Mxa ** 2 + Mya ** 2 + Mza ** 2)
#     Mb = np.sqrt(Mxb ** 2 + Myb ** 2 + Mzb ** 2)
#
#     ax = plt.subplot()
#     ax.set_xlabel('AF level (mT)')
#     ax.set_ylabel('M / Mlab')
#     ax.set_xlim(-2, max(AFstep) + 2)
#     #ax.set_ylim(0,0.6)
#     #ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.plot(AFstep, Ma/Mb, color=color, marker='o', markeredgecolor='k', ms=5, lw=0, label=label)
#     #res = stats.linregress(np.log(AFstep),np.log(Ma/Mb))
#     #ax.plot(AFstep, np.exp((res.slope * np.log(AFstep) + res.intercept)*np.log(2.7)), 'k--', lw=1)
#     #print('M/Mlab = f(AFstep) slope: '+str(res.slope))
#
#     plt.legend()
#     if path != '':
#         plt.savefig(str(path), format='png', dpi=200, bbox_inches="tight")
