import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from plotDemag import *
from plotZijderveld import *
from plotEqualArea import *
import PCAZijderveld as pca
from calcPaleointensities import *

colors = ['r','g','b','m','c','gray','violet','gold','saddlebrown','lighsteelblue','darkred','darkgreen','darkblue','greenyellow','moccasin']
def Get_closest_id(L,value):
    return list(L).index(min(L, key=lambda x:abs(x-value)))

MatAFx, MatAFy, MatAFz, MatAFstep = [], [], [], []

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
        Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)

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
        Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)

    MatAFx.append(Mx)
    MatAFy.append(My)
    MatAFz.append(Mz)
    MatAFstep.append(step)

########################
## Zijderveld Diagram ##
########################
zijd = str(input('Plot Zijderveld? (Y/n)  '))
if zijd != 'n':
    for k in np.arange(len(MatAFstep)):
        plt.figure(figsize=(5, 5))
        Plot_Zijderveld(MatAFx[k], MatAFy[k], MatAFz[k], MatAFstep[k], unit='A m2', color='AF')
        if save == 'y':
            plt.savefig(path + 'Plots/' + sample_name[k] + '-ZIJD.pdf', format='pdf', dpi=200, bbox_inches="tight")
plt.show(block=False)


######################
## Equal area plots ##
######################
eqarea = str(input('Plot equal area? (Y/n)  '))
if eqarea != 'n':
    for k in np.arange(len(MatAFstep)):
        fig = plt.figure(figsize=(5, 5))
        plot_equal_area_sequence(MatAFx[k], MatAFy[k], MatAFz[k], MatAFstep[k], fig=fig,  color='AF')
        if save == 'y':
            plt.savefig(path + 'Plots/' + sample_name[k] + '-EQAREA.pdf', format='pdf', dpi=200, bbox_inches="tight")
plt.show(block=False)


#####################
## Intensity demag ##
#####################
intdemag = str(input('Plot intensity demag? (Y/n)  '))
if intdemag != 'n':
    normalize = input('Normalize moment? (Y/n)  ')
    if normalize == 'n':
        plt.figure(figsize=(6, 3))
        for k in np.arange(len(MatAFstep)):
            Plot_AF_demag(MatAFx[k], MatAFy[k], MatAFz[k], MatAFstep[k], sample_name[k], norm=False,color=colors[k])
    else:
        plt.figure(figsize=(6, 3))
        for k in np.arange(len(MatAFstep)):
            Plot_AF_demag(MatAFx[k], MatAFy[k], MatAFz[k], MatAFstep[k], sample_name[k], norm=True, color=colors[k])
    if save == 'y':
        plt.savefig(path + 'Plots/' + sample_name[k][0:-4] + '-AFINT.pdf', format='pdf', dpi=200, bbox_inches="tight")
plt.show()


##############################
## Median destructive field ##
##############################
for k in np.arange(len(MatAFstep)):
    NRMatAF = np.sqrt(MatAFx[k]**2+MatAFy[k]**2+MatAFz[k]**2)
    MDF_NRM = MatAFstep[k][Get_closest_id(NRMatAF,NRMatAF[0]/2)]
    print('\n * MDF '+sample_name[k]+' = '+str(MDF_NRM) + ' mT')



