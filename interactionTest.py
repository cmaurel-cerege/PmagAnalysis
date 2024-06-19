import numpy as np
import matplotlib.pyplot as plt
import sys

Nfields = 5

fp = open(sys.argv[1],'r')
ARM, IRM = [0], 0
for j, line in enumerate(fp):
    if j > 0 and j <= Nfields:
        cols = line.split()
        ARM.append(float(cols[4]))
    elif j > 0 and j > Nfields:
        cols = line.split()
        IRM = float(cols[4])
fp.close()
Barm = [0.0,0.1,0.2,0.3,0.4,0.5]
ARM = np.array(ARM)

plt.xlim(0,0.5)
plt.ylim(0,0.7)
plt.plot(Barm,ARM/IRM,c='k',lw=0.75,marker='o',ms=6,markeredgecolor='k',markerfacecolor='w')

plt.show()