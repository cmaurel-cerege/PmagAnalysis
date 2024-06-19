import numpy as np
import matplotlib.pyplot as plt
from plotREMprime import *
import sys
from plotDemag import *

M1x, M1y, M1z, M2x, M2y, M2z, step1, step2 = [], [], [], [], [], [], [], []
sample = sys.argv[1].split('/')[-1].split('.')[0]
fp1 = open(str(sys.argv[1]),'r')
for j, line in enumerate(fp1):
    if j > 0:
        cols = line.split()
        M1x.append(float(cols[1])*1e-3)
        M1y.append(float(cols[2])*1e-3)
        M1z.append(float(cols[3])*1e-3)
        step1.append(int(cols[-1])*0.1)
fp1.close()

fp2 = open(str(sys.argv[2]),'r')
for j, line in enumerate(fp2):
    if j > 0:
        cols = line.split()
        M2x.append(float(cols[1])*1e-3)
        M2y.append(float(cols[2])*1e-3)
        M2z.append(float(cols[3])*1e-3)
        step2.append(int(cols[-1])*0.1)
fp2.close()

# massM1 = float(eval(input('Mass of the sample 1 (g)?'))) * 1e-3
# massM2 = float(eval(input('Mass of the sample 2 (g)?'))) * 1e-3
# M1x, M1y, M1z = np.array(M1x)*massM1, np.array(M1y)*massM1, np.array(M1z)*massM1
# M2x, M2y, M2z = np.array(M2x)*massM2, np.array(M2y)*massM2, np.array(M2z)*massM2

#M1x, M1y, M1z, M2x, M2y, M2z, step = Merge_AF_demag(M1x, M1y, M1z, step1, M2x, M2y, M2z, step2)
#calc_REMslope_vector(M1x, M1y, M1z, M2x, M2y, M2z, step, 'IRM')


# for k in np.arange(len(sys.argv)-1):
#     Mx, My, Mz, step = [], [], [], []
#     fp = open(str(sys.argv[k+1]),'r')
#     for j, line in enumerate(fp):
#         if j > 0:
#             cols = line.split()
#             Mx.append(float(cols[1])*1e-3)
#             My.append(float(cols[2])*1e-3)
#             Mz.append(float(cols[3])*1e-3)
#             step.append(int(cols[-1])*0.1)
#     fp.close()
#
#     sample = sys.argv[k+1].split('/')[-1].split('.')[0]
#
#     Plot_AF_demag(Mx, My, Mz, step, sample, marker='o')
#
#
# plt.savefig('Initial_demag.pdf', format='pdf', dpi=200, bbox_inches="tight")

plt.show()
