import numpy as np
import matplotlib.pyplot as plt
import sys
from forc_area import *

forc_file = sys.argv[1]

mass = float(input("Mass of the sample (kg)? (default = 1)  "))
Ms = float(input("Ms (A m2 kg-1)?  "))
Bc = float(input("Bc (mT)?  "))

Et_delta, Ehyst = get_energy_parameters(forc_file)
sigma_hyst = np.log(Ehyst/(4*Ms*mass*Bc*1e-3))

print(Et_delta/Ehyst,sigma_hyst)

#
# BSi, MSi = [], []
# for k in np.arange(1,len(Bforc)):
#     if Bforc[::-1][k-1] > Bforc[::-1][k]:
#         BSi.append(Bforc[::-1][k])
#         MSi.append(Mforc[::-1][k])
#     else:
#         break
#
# Bhyplus = np.array(Bhy[0:int(len(Bhy)/2)][::-1])
# Mhyplus = np.array(Mhy[0:int(len(Mhy)/2)][::-1])
#
# Bfit = np.linspace(np.min(Bhyplus),np.max(Bhyplus),1000)
#
# csHyplus = CubicSpline(Bhyplus, Mhyplus)
# Mhyplusfit = csHyplus(Bfit,0)
#
# Bhyminus = np.array(Bhy[int(len(Bhy)/2):])
# Mhyminus = np.array(Mhy[int(len(Bhy)/2):])
# csHyminus = CubicSpline(Bhyminus, Mhyminus)
# Mhyminusfit = csHyminus(Bfit,0)
#
# deltaBhy = Bhyplus
# deltaMhy = Mhyplus-Mhyminus
# csHydelta = CubicSpline(deltaBhy, deltaMhy)
# Mhydeltafit = csHydelta(Bfit,0)
#
# BSi = np.array(BSi[::-1])
# MSi = np.array(MSi[::-1])
# csFORC = CubicSpline(BSi, MSi)
# BSifit = np.linspace(np.min(BSi),np.max(BSi),500)
# MSifit = csFORC(BSifit,0)
#
# deltaMSi = Mhyplusfit[int(len(Mhyplusfit)/2):]-MSifit
# csSi = CubicSpline(BSifit, deltaMSi)
# MSideltafit = csSi(BSifit,0)
#
# Etdelta = 2*csSi.integrate(np.min(BSifit),np.max(BSifit))
# Ehys = csHydelta.integrate(np.min(Bfit),np.max(Bfit))
#
# plt.plot(Bfit,Mhyplusfit,'r')
# plt.plot(BSifit,MSifit,'b')
#
# print(BSi)

# plt.show()