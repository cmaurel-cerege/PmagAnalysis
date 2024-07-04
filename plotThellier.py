import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PCAZijderveld import *

def get_Thellier_quantities(Mx, My, Mz, Thstep, Thtype):

    ## NRM left (NRM), pTRM gained (pTRM), pTRM check (cH), pTRM tail check (tH), temperatures for Z, I, pTRM check and pTRM tail check steps.
    NRMx, NRMy, NRMz, pTRMx, pTRMy, pTRMz, cHx, cHy, cHz, tHx, tHy, tHz, zStep, iStep, cStep, tStep, izziseq = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for k in np.arange(len(Thtype)):
        if Thtype[k] == 'Z':
            NRMx.append(Mx[k])
            NRMy.append(My[k])
            NRMz.append(Mz[k])
            zStep.append(Thstep[k])
            if k == 0:
                izziseq.append('I')
            else:
                izziseq.append(Thtype[k-1])
        elif Thtype[k] == 'I':
            pTRMx.append(Mx[k])
            pTRMy.append(My[k])
            pTRMz.append(Mz[k])
            iStep.append(Thstep[k])
        elif Thtype[k] == 'C':
            cHx.append(Mx[k])
            cHy.append(My[k])
            cHz.append(Mz[k])
            cStep.append([Thstep[k], zStep[-1]])
        elif Thtype[k] == 'T':
            tHx.append(Mx[k])
            tHy.append(My[k])
            tHz.append(Mz[k])
            tStep.append(Thstep[k])

    print(NRMx)
    print(pTRMx)

    NRMx, NRMy, NRMz, pTRMx, pTRMy, pTRMz, cHx, cHy, cHz, tHx, tHy, tHz = np.array(NRMx), np.array(NRMy), np.array(NRMz), np.array(pTRMx), np.array(pTRMy), np.array(pTRMz), np.array(cHx), np.array(cHy), np.array(cHz), np.array(tHx), np.array(tHy), np.array(tHz)

    pTRMgainx = np.array([0]+list(pTRMx-NRMx[1:]))
    pTRMgainy = np.array([0]+list(pTRMy-NRMy[1:]))
    pTRMgainz = np.array([0]+list(pTRMz-NRMz[1:]))

    cHgainx, cHgainy, cHgainz = [], [], []
    for k in np.arange(len(cStep)):
        id = zStep.index(cStep[k][1])
        cHgainx.append(cHx[k]-NRMx[id])
        cHgainy.append(cHy[k]-NRMy[id])
        cHgainz.append(cHz[k]-NRMz[id])
    cHgainx, cHgainy, cHgainz = np.array(cHgainx), np.array(cHgainy), np.array(cHgainz)

    tHgainx, tHgainy, tHgainz = [], [], []
    for k in np.arange(len(tStep)):
        id = zStep.index(tStep[k])
        tHgainx.append(tHx[k]-NRMx[id])
        tHgainy.append(tHy[k]-NRMy[id])
        tHgainz.append(tHz[k]-NRMz[id])
    tHgainx, tHgainy, tHgainz = np.array(tHgainx), np.array(tHgainy), np.array(tHgainz)

    return NRMx, NRMy, NRMz, pTRMgainx, pTRMgainy, pTRMgainz, cHgainx, cHgainy, cHgainz, tHgainx, tHgainy, tHgainz, zStep, izziseq, iStep, cStep, tStep

def Plot_Thellier(Mx, My, Mz, Thstep, type, colors='y'):

    NRMx, NRMy, NRMz, pTRMgainx, pTRMgainy, pTRMgainz, cHgainx, cHgainy, cHgainz, tHgainx, tHgainy, tHgainz, zStep, izziseq, iStep, cStep, tStep = get_Thellier_quantities(
        Mx, My, Mz, Thstep, type)

    NRM = np.sqrt(NRMx**2 + NRMy**2 + NRMz**2)
    pTRMgain = np.sqrt(pTRMgainx**2 + pTRMgainy**2 + pTRMgainz**2)
    cHgain = np.sqrt(cHgainx**2 + cHgainy**2 + cHgainz**2)
    tHgain = np.sqrt(tHgainx**2 + tHgainy**2 + tHgainz**2)

    if colors == 'y':
        mfcNRM, mfcpTRM = 'darkred', 'darkblue'
    else:
        mfcNRM, mfcpTRM = 'k', 'w'

    fig = plt.figure(figsize=(5,4))
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Normalized moment')
    plt.xlim(10, max(zStep) + 10)
    plt.ylim(0,1.1)
    plt.plot(zStep,NRM/NRM[0],color='k',marker='o',ms=6,mfc=mfcNRM,mec='k',mew=0.5,lw=0.5,label='Moment remaining')
    plt.plot([zStep[0]]+iStep,pTRMgain/NRM[0],color='k',marker='o',ms=6,mfc=mfcpTRM,mec='k',mew=0.5,lw=0.5,label='pTRM gained')
    plt.legend()

    return NRMx, NRMy, NRMz, pTRMgainx, pTRMgainy, pTRMgainz, cHgainx, cHgainy, cHgainz, tHgainx, tHgainy, tHgainz, zStep, izziseq, iStep, cStep, tStep


def Plot_Aray(Mx, My, Mz, Thstep, type, colors='y'):

    NRMx, NRMy, NRMz, pTRMgainx, pTRMgainy, pTRMgainz, cHgainx, cHgainy, cHgainz, tHgainx, tHgainy, tHgainz, zStep, izziseq, iStep, cStep, tStep = get_Thellier_quantities(
        Mx, My, Mz, Thstep, type)

    NRM = np.sqrt(NRMx**2 + NRMy**2 + NRMz**2)
    pTRMgain = np.sqrt(pTRMgainx**2 + pTRMgainy**2 + pTRMgainz**2)
    cHgain = np.sqrt(cHgainx**2 + cHgainy**2 + cHgainz**2)
    tHgain = np.sqrt(tHgainx**2 + tHgainy**2 + tHgainz**2)

    if colors == 'y':
        mfcZ, mfcI, mfcC, mfcT = 'w', 'darkred', 'darkblue', 'darkblue'
    else:
        mfcZ, mfcI, mfcC, mfcT = 'k', 'k', 'k', 'k'

    fig = plt.figure(figsize=(5,4))
    plt.xlabel('Normalized pTRM gained')
    plt.ylabel('Normalized NRM')
    plt.xlim(-0.02, 1.05)
    plt.ylim(-0.02, 1.05)

    if cHgain != []:
        plt.scatter(-10, -10, s=55, facecolors='none', marker='^', edgecolors=mfcC,zorder=3,label='pTRM check')

        print('pTRM checks:')
        for k in np.arange(len(cStep)):
            id1 = zStep.index(cStep[k][1])
            id2 = zStep.index(cStep[k][0])
            plt.scatter(cHgain[k]/NRM[0], NRM[id2]/NRM[0], s=55, facecolors='w', marker='^', edgecolors=mfcC, zorder=3)
            hxmin = cHgain[k]/NRM[0]
            hxmax = pTRMgain[id1]/NRM[0]
            vymin = np.min([NRM[id1]/NRM[0],NRM[id2]/NRM[0]])
            vymax = np.max([NRM[id1]/NRM[0],NRM[id2]/NRM[0]])
            plt.hlines(y=NRM[id1]/NRM[0], xmin=hxmin, xmax=hxmax, lw=0.75, color=mfcC, ls='-', zorder=0)
            plt.vlines(x=cHgain[k]/NRM[0], ymin=vymin, ymax=vymax, lw=0.75, color=mfcC, ls='-', zorder=0)
            print(' * d'+str(cStep[k][0])+','+str(cStep[k][1])+' = '+f'{cHgain[k]/NRM[0]-pTRMgain[id2]/NRM[0]:.3f}')
    if tHgain != []:
        plt.scatter(-10, -10, s=50, facecolors='none', marker='s', edgecolors=mfcT,zorder=3,label='pTRM tail check')
        print('pTRM tail checks:')
        for k in np.arange(len(tStep)):
            id = iStep.index(tStep[k])
            plt.scatter(pTRMgain[id]/NRM[0], tHgain[k]/NRM[0], s=50, facecolors='none', marker='s', edgecolors=mfcT, zorder=3)
            print(' * d' + str(tStep[k]) + ' = ' + f'{tHgain[k]/NRM[0]:.3f}')

    plt.plot(pTRMgain/NRM[0], NRM/NRM[0], color='k', ms=3, lw=0.5)
    NRM_Z = np.array([NRM[k] for k in np.arange(len(NRM)) if izziseq[k] == 'Z'])
    pTRMgain_Z = np.array([pTRMgain[k] for k in np.arange(len(pTRMgain)) if izziseq[k] == 'Z'])
    NRM_I = np.array([NRM[k] for k in np.arange(len(NRM)) if izziseq[k] == 'I'])
    pTRMgain_I = np.array([pTRMgain[k] for k in np.arange(len(pTRMgain)) if izziseq[k] == 'I'])

    plt.plot(pTRMgain_Z/NRM[0], NRM_Z/NRM[0], color='k', marker='o', mfc=mfcZ, mec='k', mew=0.5, ms=6, lw=0,label='After Z step')
    plt.plot(pTRMgain_I/NRM[0], NRM_I/NRM[0], color='k', marker='o', mfc=mfcI, mec='k', mew=0.5, ms=6, lw=0, label='After I step')

    if colors == 'y':
        plt.legend()



    return NRMx, NRMy, NRMz, pTRMgainx, pTRMgainy, pTRMgainz, cHgainx, cHgainy, cHgainz, tHgainx, tHgainy, tHgainz, zStep, izziseq, iStep, cStep, tStep


def Stat_Thellier(Mx, My, Mz, Thstep, type, field, id_i=[], id_f=[], colors='y'):

    if id_i == [] and id_f == []:
        nb_comp = int(eval(input('Number of components?  ')))
        id_i, id_f = [], []
        for n in np.arange(nb_comp):
            print('COMPONENT '+str(n+1))
            idi = int(eval(input('First datapoint?  ')))
            idf = input('Last datapoint?  (None = last of sequence)  ')
            if idf == '': idf = len(Mx)-1
            else: idf = int(eval(idf))
            id_i.append(idi)
            id_f.append(idf)
        print('\n')

    NRMx, NRMy, NRMz, pTRMgainx, pTRMgainy, pTRMgainz, cHgainx, cHgainy, cHgainz, tHgainx, tHgainy, tHgainz, zStep, izziseq, iStep, cStep, tStep = get_Thellier_quantities(
        Mx, My, Mz, Thstep, type)

    NRM = np.sqrt(NRMx**2 + NRMy**2 + NRMz**2)
    pTRMgain = np.sqrt(pTRMgainx**2 + pTRMgainy**2 + pTRMgainz**2)
    cHgain = np.sqrt(cHgainx**2 + cHgainy**2 + cHgainz**2)
    tHgain = np.sqrt(tHgainx**2 + tHgainy**2 + tHgainz**2)

    if colors == 'y':
        mfcZ, mfcI, mfcC, mfcT = 'w', 'darkred', 'darkblue', 'darkblue'
    else:
        mfcZ, mfcI, mfcC, mfcT = 'k', 'k', 'k', 'k'

        fig = plt.figure(figsize=(5,4))
    plt.xlabel('Normalized pTRM gained')
    plt.ylabel('Normalized NRM')
    plt.xlim(-0.02, 1.05)
    plt.ylim(-0.02, 1.05)

    plt.plot(pTRMgain/NRM[0], NRM/NRM[0], color='k', ms=3, lw=0.5)
    NRM_Z = np.array([NRM[k] for k in np.arange(len(NRM)) if izziseq[k] == 'Z'])
    pTRMgain_Z = np.array([pTRMgain[k] for k in np.arange(len(pTRMgain)) if izziseq[k] == 'Z'])
    NRM_I = np.array([NRM[k] for k in np.arange(len(NRM)) if izziseq[k] == 'I'])
    pTRMgain_I = np.array([pTRMgain[k] for k in np.arange(len(pTRMgain)) if izziseq[k] == 'I'])

    plt.plot(pTRMgain_Z/NRM[0], NRM_Z/NRM[0], color='k', marker='o', mfc=mfcZ, mec='k', mew=0.5, ms=6, lw=0,label='After Z step')
    plt.plot(pTRMgain_I/NRM[0], NRM_I/NRM[0], color='k', marker='o', mfc=mfcI, mec='k', mew=0.5, ms=6, lw=0, label='After I step')

    if cHgain != []:
        plt.scatter(-10, -10, s=55, facecolors='none', marker='^', edgecolors=mfcC,zorder=3,label='pTRM check')
        for k in np.arange(len(cStep)):
            id1 = zStep.index(cStep[k][1])
            id2 = zStep.index(cStep[k][0])
            plt.scatter(cHgain[k]/NRM[0], NRM[id2]/NRM[0], s=55, facecolors='none', marker='^', edgecolors=mfcC, zorder=3)
            plt.hlines(y=NRM[id1]/NRM[0], xmin=cHgain[k]/NRM[0], xmax=pTRMgain[id1]/NRM[0], lw=0.75, color=mfcC, ls='-', zorder=0)
            plt.vlines(x=cHgain[k]/NRM[0], ymin=NRM[id1]/NRM[0], ymax=0.985*NRM[id2]/NRM[0], lw=0.75, color=mfcC, ls='-', zorder=0)

    if colors == 'y':
        plt.legend()

    best_fit_lines, paleoint_mean, paleoint_2se = [], [], []
    for k in np.arange(len(id_i)):

        NRMcx = NRMx[id_i[k]:id_f[k]+1]
        NRMcy = NRMy[id_i[k]:id_f[k]+1]
        NRMcz = NRMz[id_i[k]:id_f[k]+1]
        pTRMcx = pTRMgainx[id_i[k]:id_f[k]+1]
        pTRMcy = pTRMgainy[id_i[k]:id_f[k]+1]
        pTRMcz = pTRMgainz[id_i[k]:id_f[k]+1]

        NRMc = NRM[id_i[k]:id_f[k]+1]
        pTRMgainc = pTRMgain[id_i[k]:id_f[k]+1]

        res = stats.linregress(pTRMgainc, NRMc)
        best_fit_lines.append(res)
        plt.plot(pTRMgainc/NRM[0], (res.slope*pTRMgainc+res.intercept)/NRM[0], color='darkgray', ls='--', lw=1)
        paleoint_mean.append(-field*res.slope)
        paleoint_2se.append(2*field*res.stderr)

    res = best_fit_lines[-1]

    ## Scatter statistics beta
    beta = res.stderr/abs(res.slope)

    ## fvsd
    xprime = 0.5*(pTRMgain+(NRM-res.intercept)/res.slope)
    yprime = 0.5*(NRM+res.slope*pTRMgain+res.intercept)
    dxprime = np.absolute(xprime[0]-xprime[-1])
    dyprime = np.absolute(yprime[0]-yprime[-1])
    VDS = NRM[-1]
    for j in np.arange(len(NRM)-1):
        VDS += np.sqrt((NRMx[j+1]-NRMx[j])**2+(NRMy[j+1]-NRMy[j])**2+(NRMz[j+1]-NRMz[j])**2)
    fvds = dyprime/VDS

    ## Gap factor g
    g = 1
    for j in np.arange(len(NRM) - 1):
        g -= (yprime[j+1]-yprime[j])**2/dyprime**2

    ## Quality factor q
    f = dyprime/np.absolute(res.intercept)
    q = f*g/beta

    ## CDRATprime
    if cHgain != []:
        dpTRM = [cHgain[j]-pTRMgain[zStep.index(cStep[j][1])] for j in np.arange(len(cStep))]
        CDRAT = np.absolute(np.sum(dpTRM))/np.sqrt(dxprime**2+dyprime**2)*100

    print('Statistics on the HT component: ')
    print(' * Paleointensity = ' + f'{paleoint_mean[-1]:.2f}' + ' +/- ' + f'{paleoint_2se[-1]:.2f}' + ' (2 s.e.)')
    print(" * Scatter statistics; beta = " + f'{beta:.3f}')
    print(" * VDS of NRM fraction; f_vds = " + f'{fvds:.3f}')
    print(" * Quality factor; q = " + f'{q:.1f}')
    if cHgain != []:
        print(" * CDRAT = "+f'{CDRAT:.1f}')

    return best_fit_lines, paleoint_mean, paleoint_2se, beta, fvds, q, CDRAT
