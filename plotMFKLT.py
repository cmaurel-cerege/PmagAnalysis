import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import interpolate

fp = open(sys.argv[1],'r',encoding="utf8", errors='ignore')
T, K = [], []
for j,line in enumerate(fp):
    cc = line.split(' ')
    cols = [c for c in cc if c != '']
    if j >0:
        T.append(float(cols[0]))
        K.append(float(cols[1]))
fp.close()

TT, KK = [], []
for k in np.arange(1, len(T)):
    if T[k] > T[k-1]:
        TT.append(T[k])
        KK.append(K[k])

T_i = np.linspace(-194, 0, 80)
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

    TTpe, KKpe = [], []
    for k in np.arange(1,len(Tpe)):
        if Tpe[k] != Tpe[k-1]:
            TTpe.append(Tpe[k])
            KKpe.append(Kpe[k])

    KK, KKpe = np.array(KK), np.array(KKpe)
    tck = interpolate.splrep(TT, KK, s=0)
    tckpe = interpolate.splrep(TTpe, KKpe, s=0)
    K_i = interpolate.splev(T_i, tck, der=0)
    Kpe_i = interpolate.splev(T_i, tckpe, der=0)
    mass = float(eval(input('Mass of the sample (g)?  ')))
    Khi = (K_i-Kpe_i) * 10e-6 / (mass * 1e-3)

    tckprime = interpolate.splrep(T_i, Khi, s=0)
    Khi_prime = interpolate.splev(T_i, tckprime, der=1)

else:
    KK = np.array(KK)
    tck = interpolate.splrep(TT, KK, s=0)
    K_i = interpolate.splev(T_i, tck, der=0)
    mass = float(eval(input('Mass of the sample (g)?  ')))
    Khi = K_i * 10e-6 / (mass * 1e-3)

    tckprime = interpolate.splrep(T_i, Khi, s=0)
    Khi_prime = interpolate.splev(T_i, tckprime, der=1)

fig, ax1 = plt.subplots(figsize=(5,4))
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel('Susceptibility (10-6 m3 kg-1)')
ax1.set_xlim(75,275)
ax1.set_ylim(0,np.max(Khi)+0.1*np.max(Khi))
ax1.plot(T_i+273.15,Khi,'k-',marker='.',ms='4',lw=0.5)

# ax2 = ax1.twinx()
# ax2.set_ylabel('Derivative of susceptibility (10-6 m3 kg-1 K-1)')
# ax2.set_ylim(-np.max(np.absolute(Khi_prime))-0.1*np.max(np.absolute(Khi_prime)),np.max(np.absolute(Khi_prime))+0.1*np.max(np.absolute(Khi_prime)))
# plt.plot(T_i+273.15,Khi_prime,color='r',marker='.',ms='4',lw=0.5)
fig.tight_layout()

#plt.show()

