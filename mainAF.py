
#### /!\ IMPORTANT NOTE /!\ ####

# For the code to work flawlessly:
#   NRM demagnetation DAT file or txt file must contain "NRM"
#   ARM demagnetation DAT file or txt file must contain "ARM"
#   IRM demagnetation DAT file or txt file must contain "IRM"

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from plotDemag import *
from plotZijderveld import *
from plotEqualArea import *
import PCAZijderveld as pca
from calcPaleointensities import *


def Get_closest_id(L,value):
    return list(L).index(min(L, key=lambda x:abs(x-value)))

NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep, ARMatAFx, ARMatAFy, ARMatAFz, ARMatAFstep, \
IRMatAFx, IRMatAFy, IRMatAFz, IRMatAFstep = [], [], [], [], [], [], [], [], [], [], [], []

save = input('Save the figures? (y/N)')

if len(sys.argv) == 1:
    input('No files...')
    sys.exit()

files = sys.argv[1:]
path = ''
for k in np.arange(len(files[0].split('/'))-1):
    path += str(files[0].split('/')[k])+'/'
if not os.path.exists(path+'Plots'):
    os.makedirs(path+'Plots')

sample_name, type_of_file = [], []
for file in files:
    sample_name.append(file.split('/')[-1].split('.')[0])

    if (('ARM' not in file) and ('IRM' not in file)) or 'NRM' in file:
        type_of_file.append('NRM')
    elif 'ARM' in file:
        type_of_file.append('ARM')
    elif 'IRM' in file:
        type_of_file.append('IRM')

    fp = open(str(file),'r')
    Mx, My, Mz, step = [], [], [], []
    if file[len(file)-3:] == 'txt':
        ## This assumes moment in A m2, field in mT
        for j, line in enumerate(fp):
            cols = line.split(',')
            Mx.append(float(cols[1]))
            My.append(float(cols[2]))
            Mz.append(float(cols[3]))
            step.append(int(cols[0]))
        fp.close()

    else:
        for j, line in enumerate(fp):
            ## This assumes moment in emu, field in G
            if j > 0:
                cols = line.split()
                Mx.append(float(cols[1])*1e-3)
                My.append(float(cols[2])*1e-3)
                Mz.append(float(cols[3])*1e-3)
                step.append(int(cols[-1])*0.1)
        fp.close()

    if type_of_file[-1] == 'NRM':
        NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep = np.array(Mx), np.array(My), np.array(Mz), np.array(step)
    if type_of_file[-1] == 'ARM':
        ARMatAFx, ARMatAFy, ARMatAFz, ARMatAFstep = np.array(Mx), np.array(My), np.array(Mz), np.array(step)
    if type_of_file[-1] == 'IRM':
        IRMatAFx, IRMatAFy, IRMatAFz, IRMatAFstep = np.array(Mx), np.array(My), np.array(Mz), np.array(step)

########################
## Mass normalization ##
########################
normalize_by_mass = input('Mass normalize? (y/N)')
if normalize_by_mass != 'y':
    unit = 'A m2'
    massNRM, massIRM = 1, 1
else:
    unit = 'A m2 kg-1'
    massNRM = float(eval(input('Mass of the NRM sample (g) ?')))*1e-3
    if len(NRMatAFx) != 0:
        NRMatAFx, NRMatAFy, NRMatAFz = NRMatAFx/massNRM, NRMatAFy/massNRM, NRMatAFz/massNRM
    if len(ARMatAFx) != 0:
        massARM = input('Mass of the ARM sample (g) ? (default = same as NRM)')
        if massARM != '':
            massARM = float(eval(massARM))*1e-3
            ARMatAFx, ARMatAFy, ARMatAFz = ARMatAFx/massARM, ARMatAFy/massARM, ARMatAFz/massARM
        else:
            ARMatAFx, ARMatAFy, ARMatAFz = ARMatAFx/massNRM, ARMatAFy/massNRM, ARMatAFz/massNRM
    if len(IRMatAFx) != 0:
        massIRM = input('Mass of the IRM sample (g) ? (default = same as NRM)')
        if massIRM != '':
            massIRM = float(eval(massIRM))*1e-3
            IRMatAFx, IRMatAFy, IRMatAFz = IRMatAFx/massIRM, IRMatAFy/massIRM, IRMatAFz/massIRM
        else:
            IRMatAFx, IRMatAFy, IRMatAFz = IRMatAFx/massNRM, IRMatAFy/massNRM, IRMatAFz/massNRM

########################
## Zijderveld Diagram ##
########################
zijd = str(input('Plot Zijderveld? (Y/n)  '))
if zijd != 'n':
    if len(NRMatAFx) != 0:
        plt.figure(figsize=(5, 5))
        Plot_Zijderveld(NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep,unit=unit,title='NRM@AF',color='k')
        if save == 'y':
            id = type_of_file.index('NRM')
            plt.savefig(path + 'Plots/' + sample_name[id] + '-ZIJD.pdf', format='pdf', dpi=200, bbox_inches="tight")
        plt.figure(figsize=(5, 5))
        Plot_Zijderveld(NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep, unit=unit, title='NRM@AF', gui='guiX', color='k')
    if len(ARMatAFx) != 0:
        plt.figure(figsize=(5, 5))
        Plot_Zijderveld(ARMatAFx,ARMatAFy,ARMatAFz,ARMatAFstep,unit=unit,title='ARM@AF',color='k')
    if len(IRMatAFx) != 0:
        plt.figure(figsize=(5, 5))
        Plot_Zijderveld(IRMatAFx,IRMatAFy,IRMatAFz,IRMatAFstep,unit=unit,title='IRM@AF',color='k')
plt.show()


######################
## Equal area plots ##
######################
eqarea = str(input('Plot equal area? (Y/n)  '))
if eqarea != 'n':
    if len(NRMatAFx) != 0:
        fig = plt.figure(figsize=(5,5))
        plot_equal_area_sequence(NRMatAFx,NRMatAFy,NRMatAFz,fig,'NRM@AF',color='k')
        if save == 'y':
            id = type_of_file.index('NRM')
            plt.savefig(path+'Plots/'+sample_name[id]+'-EQAREA.pdf', format='pdf', dpi=200, bbox_inches="tight")
    if len(ARMatAFx) != 0:
        fig = plt.figure(figsize=(5, 5))
        plot_equal_area_sequence(ARMatAFx,ARMatAFy,ARMatAFz,fig,'ARM@AF')
    if len(IRMatAFx) != 0:
        fig = plt.figure(figsize=(5, 5))
        plot_equal_area_sequence(IRMatAFx,IRMatAFy,IRMatAFz,fig,'IRM@AF')
plt.show(block=False)


#####################
## Intensity demag ##
#####################
intdemag = str(input('Plot intensity demag? (Y/n)  '))
if intdemag != 'n':
    fig = plt.figure(figsize=(6,3))
    if len(NRMatAFx) != 0:
        Plot_AF_demag(NRMatAFx,NRMatAFy,NRMatAFz,NRMatAFstep, 'NRM', color='r')
    if len(ARMatAFx) != 0:
        Plot_AF_demag(ARMatAFx,ARMatAFy,ARMatAFz,ARMatAFstep,'ARM', color='g')
    if len(IRMatAFx) != 0:
        Plot_AF_demag(IRMatAFx,IRMatAFy,IRMatAFz,IRMatAFstep,'IRM', color='b')
    if save == 'y':
        id = type_of_file.index('NRM')
        plt.savefig(path + 'Plots/' + sample_name[id][0:-4] + '-AFINT.pdf', format='pdf', dpi=200, bbox_inches="tight")

plt.show(block=False)


##############################
## Median destructive field ##
##############################
if len(NRMatAFx) != 0:
    NRMatAF = np.sqrt(NRMatAFx**2+NRMatAFy**2+NRMatAFz**2)
    MDF_NRM = NRMatAFstep[Get_closest_id(NRMatAF,NRMatAF[0]/2)]
    print("\n * MDF NRM = "+str(MDF_NRM) + " mT")
if len(ARMatAFx) != 0:
    ARMatAF = np.sqrt(ARMatAFx ** 2 + ARMatAFy ** 2 + ARMatAFz ** 2)
    MDF_ARM = ARMatAFstep[Get_closest_id(ARMatAF, ARMatAF[0] / 2)]
    print(" * MDF ARM = " + str(MDF_ARM) + " mT")
if len(IRMatAFx) != 0:
    IRMatAF = np.sqrt(IRMatAFx ** 2 + IRMatAFy ** 2 + IRMatAFz ** 2)
    MDF_IRM = IRMatAFstep[Get_closest_id(IRMatAF, IRMatAF[0] / 2)]
    print(" * MDF IRM = " + str(MDF_IRM) + " mT")
print('\n')


##################
## PCA analysis ##
##################
dopca = str(input('Run PCA analysis? (Y/n)  '))
if dopca != 'n':

    if len(NRMatAFx) != 0 and len(ARMatAFx) != 0:
        Mcx, Mcy, Mcz, Mcd, Mci, Mcmax, MAD, DANG, MAD95, id_i, id_f = pca.PCA_analysis(NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep, rem='ARM', remdata=ARMatAF, mass=massNRM)
    elif len(NRMatAFx) != 0 and len(IRMatAFx) != 0:
        Mcx, Mcy, Mcz, Mcd, Mci, Mcmax, MAD, DANG, MAD95, id_i, id_f = pca.PCA_analysis(NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep, rem='IRM', remdata=IRMatAF, mass=massNRM)
    else:
        Mcx, Mcy, Mcz, Mcd, Mci, Mcmax, MAD, DANG, MAD95, id_i, id_f = pca.PCA_analysis(NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep, mass=massNRM)


####################
## Paleointensity ##
####################
nrmlost = str(input('Run REM prime analysis? (Y/n)  '))
if nrmlost != 'n':

    tcrm = input('TRM, CRM? (default = TRM)  ')
    if tcrm == '': tcrm = 'TRM'
    mineral = input('Mineral? (default = m: magnetite; p: pyrrhotite; f: Fe-Ni)  ')
    if mineral == '': mineral = 'm'
    domain = input('SD, PSD, MD? (default = SD/PSD)  ')
    if domain == '': domain = 'SD/PSD'
    ARMbiasfield = input('ARM bias field? (default = 100 mT)  ')
    if ARMbiasfield == '': ARMbiasfield = 100
    else: ARMbiasfield = int(eval(ARMbiasfield))

    paleointensityARM = []
    if len(NRMatAFx) != 0 and len(ARMatAFx) != 0:

        print('** ARM PALEOINTENSITIES **')

        NRMatAFx, NRMatAFy, NRMatAFz, ARMatAFx, ARMatAFy, ARMatAFz, AFARM = Merge_AF_lists(NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep, ARMatAFx, ARMatAFy, ARMatAFz, ARMatAFstep)

        paleointensityARM = calc_paleointensity(NRMatAFx, NRMatAFy, NRMatAFz, ARMatAFx, ARMatAFy, ARMatAFz, AFARM, type='ARM', tcrm=tcrm, mineral=mineral, domain=domain, biasfield=ARMbiasfield, mass=massNRM)
        if save == 'y':
            id = type_of_file.index('NRM')
            plt.savefig(path + 'Plots/' + sample_name[id] + '-VS-ARM-lost.pdf', format='pdf', dpi=200, bbox_inches="tight")

    paleointensityIRM = []
    if len(NRMatAFx) != 0 and len(IRMatAFx) != 0:

        print('** IRM PALEOINTENSITIES **')

        NRMatAFx, NRMatAFy, NRMatAFz, IRMatAFx, IRMatAFy, IRMatAFz, AFIRM = Merge_AF_lists(NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep, IRMatAFx, IRMatAFy, IRMatAFz, IRMatAFstep)

        plot_REMp(NRMatAFx, NRMatAFy, NRMatAFz, IRMatAFx, IRMatAFy, IRMatAFz, AFIRM, annot=False)
        if save == 'y':
            id = type_of_file.index('NRM')
            plt.savefig(path + 'Plots/' + sample_name[id] + '_REMp.pdf', format='pdf', dpi=200, bbox_inches="tight")

        paleointensityIRM = calc_paleointensity(NRMatAFx, NRMatAFy, NRMatAFz, IRMatAFx, IRMatAFy, IRMatAFz, AFIRM, type='IRM', tcrm=tcrm, mineral=mineral, domain=domain, mass=massIRM)
        if save == 'y':
            id = type_of_file.index('NRM')
            plt.savefig(path + 'Plots/' + sample_name[id] + '-VS-IRM-lost.pdf', format='pdf', dpi=200, bbox_inches="tight")

        plt.figure(figsize=(6, 3))
        plt.xlabel('Paleointensity (uT)')
        plt.ylabel('Probability density')
        if len(paleointensityARM) != 0:
            plt.hist(paleointensityARM, bins=20, density=True, color='lightgreen', edgecolor='darkgreen',label='ARM method')
        if len(paleointensityIRM) != 0:
            plt.hist(paleointensityIRM, bins=20, density=True, color='lightblue', edgecolor='darkblue',label='IRM method')
        plt.legend(loc=1)

plt.show()

