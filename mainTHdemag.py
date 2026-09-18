
#### /!\ IMPORTANT NOTE /!\ ####

# For the code to work flawlessly, the temperature step decimals must represent the type of step:
#   if it's a zero-field step, you must enter temperature+".00" (e.g., 150.00)
#   if it's an in-field step, you must enter temperature+".01" (e.g., 150.01)
#   if it's a pTRM check, you must enter temperature+".02" (e.g., 150.02)
#   if it's a pTRM tail check, you must enter temperature+".03" (e.g., 150.03)

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from plotDemag import *
from plotZijderveld import *
from plotEqualArea import *
from PCAZijderveld import *
from calcPaleointensities import *
from builtins import *
import pandas as pd

save = input('Save the figures? (y/N)')

if len(sys.argv) == 1:
    input('No files...')
    sys.exit

file = sys.argv[1]
path = ''
fp = open(file,'r')
for k in np.arange(len(file.split('/'))-1):
    path += str(file.split('/')[k])+'/'
if not os.path.exists(path+'Plots'):
    os.makedirs(path+'Plots')

sample = file.split('/')[-1].split('.')[0]

Mx, My, Mz, Thstep = [], [], [], []
count = 0
if file[len(file) - 3:] == 'DAT':
    unit = 1e-3
    df = pd.read_csv(file, sep='\t', header=0, usecols=[1, 2, 3, 15], names=["Mx", "My", "Mz", "step"], engine="python")
else:
    count += 1
    df = pd.read_csv(file, sep=None, usecols=[0, 1, 2, 3], names=["step", "Mx", "My", "Mz"], engine="python")
    if count == 1:
        unit = input('Unit = A m2? (y/N) ')
        if unit == 'y': unit = 1
        else: unit = 1e-3
        print('\nFirst line starts with: ' + str(df.step[0]) + '\n')
        header = input('Header? (y/N) ')
        if header == 'y': header = 1
        else: header = 0
    df = pd.read_csv(file, sep=None, header=header, usecols=[0, 1, 2, 3], names=["step", "Mx", "My", "Mz"], engine="python")

Thstep, Mx, My, Mz = np.array(df.step), np.array(df.Mx) * unit, np.array(df.My) * unit, np.array(df.Mz) * unit
M = np.sqrt(Mx**2 + My**2+ Mz**2)
print('NRM = ' + str(M[0])+' A m2')

fig = plt.figure(figsize=(6,3))
Plot_TH_demag(Mx, My, Mz, Thstep, norm=True)
if save == 'y':
    plt.savefig(path+'Plots/'+sample + '-Therm-int.pdf', format='pdf', dpi=400, bbox_inches="tight")

fig = plt.figure(figsize=(7,7))
Plot_Zijderveld(Mx, My, Mz, Thstep, unit='A m2', title='NRM@TH', color='k')
if save == 'y':
    plt.savefig(path+'Plots/'+sample + '-Therm-zijd.pdf', format='pdf', dpi=400, bbox_inches="tight")
fig = plt.figure(figsize=(7,7))
Plot_Zijderveld(Mx, My, Mz, Thstep, unit='A m2', title='NRM@TH', color='k', gui='guiX',annot='X')

fig = plt.figure(figsize=(5,5))
plot_equal_area_sequence(Mx, My, Mz, Thstep, fig=fig, title='NRM@TH', color='k')

if save == 'y':
    plt.savefig(path+'Plots/'+sample + '-Therm-eqarea.pdf', format='pdf', dpi=400, bbox_inches="tight")
plt.show(block=False)

#massNRM = float(eval(input('Mass of the NRM sample (mg) ?'))) * 1e-6
angular_jump_origin = Angular_jump_origin_analysis(Mx, My, Mz, Thstep, demag='TH')
#angular_jump_seq = Angular_jump_sequence_analysis(Mx, My, Mz, Thstep, demag='TH')

Mxmodel,Mymodel,Mzmodel = np.logspace(-11, -2, 100), np.logspace(-11, -2, 100), np.logspace(-11, -2, 100)
Mmodel = np.sqrt(Mxmodel ** 2 + Mymodel ** 2 + Mzmodel ** 2)

noise_int, D, I = 2e-11, 0, 0
angjumpmodelorigin,angjumpmodelseq = [],[]
for k in np.arange(len(Mmodel)):
    Mxr, Myr, Mzr = [Mxmodel[k]], [Mymodel[k]], [Mzmodel[k]]
    for j in np.arange(1000):
        Mxr_tmp, Myr_tmp, Mzr_tmp = Mxmodel[k], Mymodel[k], Mzmodel[k]
        Dr = random.sample(list(np.linspace(0, 380, 1000)), 1)
        Ir = random.sample(list(np.linspace(-90, 90, 1000)), 1)
        Xr, Yr, Zr = dir2cart([Dr[0], Ir[0], 1])
        Mxr.append(Mxr_tmp + Xr*noise_int)
        Myr.append(Myr_tmp + Yr*noise_int)
        Mzr.append(Mzr_tmp + Zr*noise_int)
    angjumpmodelorigin.append(3.5 + np.mean(Angular_jump_origin_analysis(Mxr, Myr, Mzr, np.arange(len(Mxr)))))
    #angjumpmodelseq.append(3.5 + np.mean(Angular_jump_sequence_analysis(Mxr, Myr, Mzr, np.arange(len(Mxr)))))

fig, ax = plt.subplots(1,1,figsize=(6,3))
#plt.xlim(0,600)
plt.ylim(0,50)#180)
plt.xlabel('Temperature (°C)')
plt.ylabel('Angular jump between consecutive points (°)')
plt.xscale('log')
#plt.xlim(1e-8, 1e-3)
plt.plot(M[1:], angular_jump_origin, 'k-', lw=0.5, marker='o', ms=5,zorder=3)
cursor = FollowDotCursor(ax, M[1:], angular_jump_origin, M[1:], Thstep[1:])
plt.plot(Mmodel, angjumpmodelorigin,'r-')

if save == 'y':
    plt.savefig(path + 'Plots/' + sample + '-angjumporigin.pdf', format='pdf', dpi=400, bbox_inches="tight")
plt.show(block=False)

dopca = input('Run PCA analysis? (Y/n)  ')
if dopca != 'n':
    Mcx, Mcy, Mcz, Mcd, Mci, Mcmax, MAD, DANG, MAD95, id_i, id_f = PCA_analysis(Mx, My, Mz, Thstep, demag='TH')
    fig = plt.figure(figsize=(5, 5))
    plot_equal_area_sequence(Mcx, Mcy, Mcz, Thstep, fig=fig, title='PC@TH', color='k')


plt.show()