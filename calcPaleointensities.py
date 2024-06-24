import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from returnEmpiricalFactors import *
import random

def Merge_AF_lists(NRMx,NRMy,NRMz,NRMAF,Mx,My,Mz,MAF):

    NRMAF, MAF = sorted(NRMAF), sorted(MAF)
    AF = list(set(NRMAF)&set(MAF))

    NRMx = [NRMx[k] for k in np.arange(1,len(NRMx)) if (NRMAF[k] in AF and NRMAF[k] not in NRMAF[0:k])]
    NRMy = [NRMy[k] for k in np.arange(len(NRMy)) if (NRMAF[k] in AF and NRMAF[k] not in NRMAF[0:k])]
    NRMz = [NRMz[k] for k in np.arange(len(NRMz)) if (NRMAF[k] in AF and NRMAF[k] not in NRMAF[0:k])]
    Mx = [Mx[k] for k in np.arange(len(Mx)) if (MAF[k] in AF and MAF[k] not in MAF[0:k])]
    My = [My[k] for k in np.arange(len(My)) if (MAF[k] in AF and MAF[k] not in MAF[0:k])]
    Mz = [Mz[k] for k in np.arange(len(Mz)) if (MAF[k] in AF and MAF[k] not in MAF[0:k])]

    return NRMx,NRMy,NRMz,Mx,My,Mz,AF

def calc_Mlost(Mx,My,Mz,AFM,id1,id2,delta):

    Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)
    Mlost = list(np.sqrt((Mx[id1:id2+1]-Mx[id1])**2 + (My[id1:id2+1]-My[id1])**2 + (Mz[id1:id2+1]-Mz[id1])**2) + delta)
    AF = list(AFM[id1:id2+1])

    return Mlost, AF

def plot_Mlost(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, mass=1, annot=False):

    ## Recording the different components and completing the sequence if needed
    nb_comp = int(eval(input('Number of components?  ')))
    id_i, id_f = [], []
    for n in np.arange(nb_comp):
        print('Component '+str(n+1))
        idi = int(eval(input('First datapoint?  ')))
        idf = input('Last datapoint?  (None = last of sequence)  ')
        if idf == '': idf = len(NRMx)-1
        else: idf = int(eval(idf))
        id_i.append(idi)
        id_f.append(idf)
    id_i_full, id_f_full, show_comp = [], [], []
    if id_i[0] != 0:
        id_i_full.append(0)
        id_f_full.append(id_i[0]-1)
        show_comp.append(False)
    id_i_full.append(id_i[0])
    id_f_full.append(id_f[0])
    show_comp.append(True)

    for k in np.arange(1,len(id_i)):
        if id_i[k] != id_f[k-1]+1:
            id_i_full.append(id_f[k-1]+1)
            id_f_full.append(id_i[k]-1)
            show_comp.append(False)
        id_i_full.append(id_i[k])
        id_f_full.append(id_f[k])
        show_comp.append(True)

    id_i, id_f = id_i_full, id_f_full

    NRMlost = calc_Mlost(NRMx, NRMy, NRMz, AF, id_i[0], id_f[0], 0)[0]
    Mlost = calc_Mlost(Mx, My, Mz, AF, id_i[0], id_f[0], 0)[0]
    AFlost = calc_Mlost(Mx, My, Mz, AF, id_i[0], id_f[0], 0)[1]
    for k in np.arange(1,len(id_i)):
        NRMlost += calc_Mlost(NRMx, NRMy, NRMz, AF, id_i[k], id_f[k], NRMlost[-1])[0]
        Mlost += calc_Mlost(Mx, My, Mz, AF, id_i[k], id_f[k], Mlost[-1])[0]
        AFlost += calc_Mlost(Mx, My, Mz, AF, id_i[k], id_f[k], Mlost[-1])[1]

    fig = plt.figure(figsize=(6,3))
    plt.ticklabel_format(axis='both', style='sci')
    if mass != 1:
        unit = 'A m2 kg-1'
    else:
        unit = 'A m2'
    plt.xlabel(type+r' lost ('+unit+')')
    plt.ylabel('NRM'+r' lost ('+unit+')')
    plt.xlim(-0.1*np.max(Mlost[-1]),1.1*np.max(Mlost[-1]))
    plt.ylim(-0.1*np.max(NRMlost[-1]), 1.1 * np.max(NRMlost[-1]))
    plt.scatter(Mlost, NRMlost, s=45, c='lightgray', marker='o', edgecolor='k', linewidths=0.5)

    if annot == True:
        for i in np.arange(len(Mlost)):
            plt.text(Mlost[i], NRMlost[i], str(i), fontsize=8)

    return NRMlost, Mlost, AFlost, id_i, id_f, show_comp


def calc_paleointensity(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, tcrm, mineral, domain, biasfield=100, mass=1, annot=False):

    ## Get the empirical factors
    if mineral == 'm':
        mineral = 'magnetite'
    elif  mineral == 'p':
        mineral = 'pyrrhotite'
    elif  mineral == 'f':
        mineral = 'FeNi'

    file = type + '_empirical_factors.xlsx'
    sheet = tcrm + ' ' + mineral
    empiricalFactor = returnFactor(file, sheet, domain)
    mulogEmpiricalFactor = np.mean(np.log(np.array(empiricalFactor)))
    sdlogEmpiricalFactor = np.std(np.log(np.array(empiricalFactor)))

    print('\nEMPIRICAL FACTOR:')
    print('* Geometric mean = ' + f'{np.exp(mulogEmpiricalFactor):.2f}')
    print('* Geometric SD = ' + f'{np.exp(sdlogEmpiricalFactor):.2f}\n')

    NRMlost, Mlost, AFlost, id_i, id_f, show_comp = plot_Mlost(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, mass, annot)
    plt.draw()
    plt.show(block=False)

    count = 0
    REMp_mean, REMp_2se, REMp_sd, paleointensity = [], [], [], []
    for k in np.arange(len(id_i)):

        res = stats.linregress(np.array(Mlost[id_i[k]:id_f[k]]), np.array(NRMlost[id_i[k]:id_f[k]]))
        REMp_mean.append(res.slope)
        REMp_2se.append(2*res.stderr)
        REMp_sd.append(res.stderr*np.sqrt(len(Mlost[id_i[k]:id_f[k]])))

        if show_comp[k] == True:
            count += 1
            plt.plot(Mlost[id_i[k]:id_f[k]], res.slope*np.array(Mlost[id_i[k]:id_f[k]])+res.intercept, color='r', ls='--', lw=1.5)

            print('COMPONENT '+str(count))
            print(' * Slope = '+f'{REMp_mean[-1]:.5f}'+' +/- '+f'{REMp_sd[-1]:.5f}'+' (1 s.d.) +/- '+f'{REMp_2se[-1]:.5f}'+' (2 s.e.)')

            paleointensity = []
            for k in np.arange(10000):
                logEF = np.random.normal(mulogEmpiricalFactor, sdlogEmpiricalFactor, 1)[0]
                EF = np.exp(logEF)
                slope = np.random.normal(REMp_mean[-1], REMp_sd[-1], 1)[0]
                if type == 'IRM':
                    paleointensity.append(EF*slope)
                elif type == 'ARM':
                    paleointensity.append(biasfield/EF*slope)

            min95 = sorted(paleointensity)[249]
            max95 = sorted(paleointensity)[9750]
            print(' * Mean paleointensity = '+f'{(min95+max95)/2:.0f}'+' uT')
            print(' * 95% confidence interval = ['+f'{min95:.0f}'+' uT, '+f'{max95:.0f}'+' uT]')

    return paleointensity


def plot_REMp(NRMx, NRMy, NRMz, Mx, My, Mz, AF, annot=False):

    NRM = np.array([np.sqrt((NRMx[k+1]-NRMx[k])**2+(NRMy[k+1]-NRMy[k])**2+(NRMz[k+1]-NRMz[k])**2) for k in np.arange(len(NRMx)-1)])
    M = np.array([np.sqrt((Mx[k+1]-Mx[k])**2+(My[k+1]-My[k])**2+(Mz[k+1]-Mz[k])**2) for k in np.arange(len(Mx)-1)])

    REMp = NRM/M

    fig = plt.figure(figsize=(6,3))
    plt.yscale("log")
    plt.xlabel('AF step (mT)')
    plt.ylabel('REM prime')
    plt.ylim(1e-4,1)
    plt.scatter(AF[1:], REMp, s=45, c='lightgray', marker='o', edgecolor='k', linewidths=0.5)
    if annot == True:
        for i in np.arange(len(REMp)):
            plt.text(AF[i], REMp[i], str(i), fontsize=8)

    return


# #### CALIBBBBB
#
# def calc_REMslope_vector(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, tcrm, mineral, domain, biasfield=100, calib=False,
#                          mass=True, annot=False):
#
#     if calib == True:
#         idcomp = 17
#         id, NRMlost, Mlost, AFlost = plot_NRM_vs_Mlost_vector(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, mass, annot)
#         plt.draw()
#         plt.show(block=False)
#         res = stats.linregress(Mlost[idcomp:], NRMlost[idcomp:])
#         REMp = res.slope
#         plt.plot(Mlost[idcomp:], res.slope * np.array(Mlost[idcomp:]) + res.intercept, 'r-', lw=1)
#
#         print("R^2 = " + f'{res.rvalue:.2f}')
#         print(
#             "Slope (x1000) = " + f'{res.slope * 1000:.2f}' + ' +/- ' + f'{2 * res.stderr * 1000:.2f}' + ' (2 sigma)')
#         field = float(eval(input("Field applied (uT)? –> ")))
#         if type == 'IRM':
#             print("IRM proportionality constant = " + str(field / REMp))
#         if type == 'ARM':
#             print("ARM proportionality constant = " + str(100 * REMp / field))
#
#     else:
#         id, NRMlost, Mlost, AFlost = plot_NRM_vs_Mlost_vector(NRMx, NRMy, NRMz, Mx, My, Mz, AF, type, mass, annot)
#         plt.draw()
#         plt.show(block=False)
#
#         REMp, RMEp2se, REMpsd = [], [], []
#         for k in np.arange(len(id) - 1):
#             res = stats.linregress(Mlost[id[k]:id[k + 1]], NRMlost[id[k]:id[k + 1]])
#             plt.plot(Mlost[id[k]:id[k + 1]], res.slope * np.array(Mlost[id[k]:id[k + 1]]) + res.intercept, 'r-',
#                      lw=1)
#             REMp.append(res.slope)
#             RMEp2se.append(2 * res.stderr)
#             REMpsd.append(res.stderr * np.sqrt(len(Mlost[id[k]:id[k + 1]]) - 2))
#
#         seg = int(eval(input("Segment for paleointensity? (1,2,3...)")))
#
#         print('\n' + str(type))
#         print(RMEp2se[seg - 1])
#         print('Slope = ' + f'{REMp[seg - 1]:.5f}' + ' +/- ' + f'{REMpsd[seg - 1]:.5f}' + ' (1 s.d.)')
#
#         file = type + '_empirical_factors.xlsx'
#         sheet = tcrm + ' ' + mineral
#         empiricalFactor = returnFactor(file, sheet, domain)
#         mulogEmpiricalFactor = np.mean(np.log(np.array(empiricalFactor)))
#         sdlogEmpiricalFactor = np.std(np.log(np.array(empiricalFactor)))
#
#         print(mulogEmpiricalFactor, sdlogEmpiricalFactor)
#
#         print('Geometric mean empirical factor = ' + f'{np.exp(mulogEmpiricalFactor):.2f}')
#         print('Geometric SD empirical factor = ' + f'{np.exp(sdlogEmpiricalFactor):.2f}')
#
#         paleointensity = []
#         for k in np.arange(10000):
#             logEF = np.random.normal(mulogEmpiricalFactor, sdlogEmpiricalFactor, 1)[0]
#             EF = np.exp(logEF)
#             slope = np.random.normal(REMp[seg - 1], REMpsd[seg - 1], 1)[0]
#             if type == 'IRM':
#                 paleointensity.append(EF * slope)
#             elif type == 'ARM':
#                 paleointensity.append(biasfield / EF * slope)
#
#         min95 = sorted(paleointensity)[249]
#         max95 = sorted(paleointensity)[9750]
#         print('95% confidence interval = [' + f'{min95:.0f}' + ' uT, ' + f'{max95:.0f}' + ' uT]')
#         print('Mean paleointensity = ' + f'{(min95 + max95) / 2:.0f}' + ' uT\n')
#
#     return paleointensity