import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import interpolate

fp = open(sys.argv[1],'r',encoding="utf8", errors='ignore')
save = input('Save the figures? (y/N)')
path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'
sample = sys.argv[1].split('/')[-1].split('.')[0]

T, K = [], []
for j,line in enumerate(fp):
    cc = line.split(' ')
    cols = [c for c in cc if c != '']
    if j >0:
        T.append(float(cols[0]))
        K.append(float(cols[1]))
fp.close()

Theat = np.array(T[1:T.index(np.max(T))+1])
Kheat = np.array(K[1:T.index(np.max(T))+1])
Tcool = np.array(T[T.index(np.max(T)):])[::-1]
Kcool = np.array(K[T.index(np.max(T)):])[::-1]

Tpe, Kpe = [], []
if len(sys.argv) > 2:
    fppe = open(sys.argv[2], 'r', encoding="utf8", errors='ignore')
    for j, line in enumerate(fppe):
        cc = line.split(' ')
        cols = [c for c in cc if c != '']
        if j > 0:
            Tpe.append(float(cols[0]))
            Kpe.append(float(cols[1]))
    fppe.close()

    Tpeheat = np.array(Tpe[:Tpe.index(np.max(Tpe)) + 1])
    Kpeheat = np.array(Kpe[:Tpe.index(np.max(Tpe)) + 1])
    Tpecool = np.array(Tpe[Tpe.index(np.max(Tpe)):])[::-1]
    Kpecool = np.array(Kpe[Tpe.index(np.max(Tpe)):])[::-1]

    tck_heat = interpolate.splrep(Theat, Kheat, s=0)
    tck_cool = interpolate.splrep(Tcool, Kcool, s=0)
    tckpe_heat = interpolate.splrep(Tpeheat, Kpeheat, s=0)
    tckpe_cool = interpolate.splrep(Tpecool, Kpecool, s=0)


    T_i = np.linspace(40, np.min([Theat[-1],Tpeheat[-1]]), 80)

    Kheat_i = interpolate.splev(T_i, tck_heat, der=0)
    Kpeheat_i = interpolate.splev(T_i, tckpe_heat, der=0)
    Kcool_i = interpolate.splev(T_i, tck_cool, der=0)
    Kpecool_i = interpolate.splev(T_i, tckpe_cool, der=0)

    mass = float(eval(input('Mass of the sample (g)?  ')))
    Khi_heat = (Kheat_i-Kpeheat_i) * 10e-6 / (mass * 1e-3)
    Khi_cool = (Kcool_i - Kpecool_i) * 10e-6 / (mass * 1e-3)

else:
    T_i = np.linspace(40, Theat[-1], 80)

    tck_heat = interpolate.splrep(Theat, Kheat, s=0)
    tck_cool = interpolate.splrep(Tcool, Kcool, s=0)
    Kheat_i = interpolate.splev(T_i, tck_heat, der=0)
    Kcool_i = interpolate.splev(T_i, tck_cool, der=0)

    mass = float(eval(input('Mass of the sample (g)?  ')))
    Khi_heat = (Kheat_i) * 10e-6 / (mass * 1e-3)
    Khi_cool = (Kcool_i) * 10e-6 / (mass * 1e-3)

fig = plt.figure()
plt.title(sys.argv[1].split('/')[-1].split('.')[0])
plt.xlabel('Temperature (C)')
plt.ylabel('Susceptibility (10-6 m3 kg-1)')
plt.xlim(0,710)
plt.ylim(0,2.4)
plt.plot(T_i,Khi_heat,'r-',marker='.',ms='4',lw=0.5)
plt.plot(T_i,Khi_cool,'b-',marker='.',ms='4',lw=0.5)
fig.tight_layout()

if save == 'y':
    plt.savefig(path + sample + '-MFKHT.pdf', format='pdf', dpi=400, bbox_inches="tight")


plt.show()

