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
    if j > 0:
        T.append(float(cols[0].split('-')[-1][:-1]))
        K.append(float(cols[6])*1e6)
fp.close()

T, K = np.array(T), np.array(K)
mass = float(eval(input('Mass of the sample (g)?  ')))
K = K * 10e-6 / (mass * 1e-3)

fig = plt.figure()
plt.xlabel('Temperature (C)')
plt.ylabel('Susceptibility (10-6 m3 kg-1)')
plt.plot(T,K,'k-',marker='.',ms='4',lw=0.5)
fig.tight_layout()

if save == 'y':
    plt.savefig(path + sample + '.pdf', format='pdf', dpi=400, bbox_inches="tight")


plt.show()

