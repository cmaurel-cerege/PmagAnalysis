## Based on Jelinek: the statistical theory of measuring anisotropy of
## magnetic susceptibility of rocks and its application

import numpy as np
import sys
from cart2dir import *
from dir2cart import *
#np.set_printoptions(suppress=True)
from plotEqualArea import *

## KLY2 15 measurement directions (decl, incl, int=1):
Pos1 = np.array([135,0,1])
Pos2 = np.array([45,0,1])
Pos3 = np.array([0,0,1])
Pos4 = np.array([315,0,1])
Pos5 = np.array([225,0,1])
Pos6 = np.array([270,45,1])
Pos7 = np.array([90,45,1])
Pos8 = np.array([90,0,1])
Pos9 = np.array([90,-45,1])
Pos10 = np.array([270,-45,1])
Pos11 = np.array([0,-45,1])
Pos12 = np.array([0,45,1])
Pos13 = np.array([0,90,1])
Pos14 = np.array([180,45,1])
Pos15 = np.array([180,-45,1])
Pos = np.array([Pos1, Pos2, Pos3, Pos4, Pos5, Pos6, Pos7, Pos8, Pos9, Pos10, Pos11, Pos12, Pos13, Pos14, Pos15])

## Read file (a la main de Jerome)
path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'
name = sys.argv[1].split('/')[-1].split('.')[0]

Khi = []
fp = open(sys.argv[1], 'r')
for j, line in enumerate(fp):
    cols = line.split()
    Khi.append(float(cols[0]))
fp.close()

if len(Khi) != 15:
    print('Not 15 measurements in file...')
    sys.exit()

## Calculate design matrix A
DM = [0]*len(Pos)
for k in np.arange(len(Pos)):
    d1, d2, d3 = dir2cart(Pos[k])
    DM[k] = [d1**2,d2**2,d3**2,2*d1*d2,2*d2*d3,2*d1*d3]
DM = np.matrix(DM)
Khi = np.matrix(Khi)

## Calculate coefficients anisotropy matrix K
DMLSQ = np.matmul(np.linalg.inv(np.matmul(DM.T,DM)),DM.T)
coeffs = np.ravel(np.matmul(DMLSQ,Khi.T))
k11,k22,k33,k12,k23,k13 = coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4],coeffs[5]
Kvec = np.matrix([k11,k22,k33,k12,k23,k13])
AM = np.matrix([[k11,k12,k13],[k12,k22,k23],[k13,k23,k33]])
print('Anisotropy matrix : ')
print(str(AM)+'\n')

## Calculate model ARM and RSS
Khi_model = np.matmul(DM,Kvec.T)
RSS = np.sum((np.ravel(Khi)-np.ravel(Khi_model))**2)

## Diagonalize anisotropy matrix K to get eigenvectors and eigenvalues
eigvals, eigvecs = np.linalg.eig(AM)
valvec = [[eigvals[i], np.transpose(eigvecs)[i]] for i in np.arange(len(eigvals))]
valvec.sort(key=lambda j: (j[0], -j[1]), reverse=True)
eigvals = np.array([valvec[i][0] for i in np.arange(len(valvec))])
eigvecs = np.array([np.ravel(valvec[i][1]) for i in np.arange(len(valvec))])
eigvals_mean = (eigvals[0]+eigvals[1]+eigvals[2])/3

alpha1, alpha2, alpha3 = eigvals[0]/eigvals_mean, eigvals[1]/eigvals_mean, eigvals[2]/eigvals_mean
L, F, P = alpha1 / alpha2, alpha2 / alpha3, alpha1 / alpha3
T = (2*np.log(alpha2)-np.log(alpha1)-np.log(alpha3))/(np.log(alpha1)-np.log(alpha3))

## 95% confidence intervals on eigenvalues
## sqrt(0.4)*S*t9 where t9 is the quantile of a Student distribution with probablity 0.95 and 9 DoF
t9 = 1.833
S = np.sqrt(1/(len(Pos)-6)*RSS)
alpha_err = np.sqrt(0.4)*S*t9/eigvals_mean

## 95% confidence ellipses on eigenvectors
## F29 is the quantile of a F distribution of 2 and 9 DoF with probability 95%.
F29 = 4.26
e1_2 = np.arctan(S*np.sqrt(2*F29)*2*np.absolute(alpha1-alpha2))*180/np.pi
e1_3 = np.arctan(S*np.sqrt(2*F29)*2*np.absolute(alpha1-alpha3))*180/np.pi
e1 = np.min([e1_2,e1_3])

e2_1 = np.arctan(S*np.sqrt(2*F29)*2*np.absolute(alpha1-alpha2))*180/np.pi
e2_3 = np.arctan(S*np.sqrt(2*F29)*2*np.absolute(alpha2-alpha3))*180/np.pi
e2 = np.min([e2_1, e2_3])

e3_1 = np.arctan(S*np.sqrt(2*F29)*2*np.absolute(alpha1-alpha3))*180/np.pi
e3_2 = np.arctan(S*np.sqrt(2*F29)*2*np.absolute(alpha2-alpha3))*180/np.pi
e3 = np.min([e3_1, e3_2])

print('Eigenvalues (normalized):')
print(f'{alpha1:.3f}'+', '+f'{alpha2:.3f}'+', '+f'{alpha3:.3f}'+'\n')
print('95% confidence interval on eigenvalues:')
print(' +/- '+f'{alpha_err:.3f}'+'\n')
print('Eigenvectors and 95% confidence angle:')
print('('+f'{cart2dir(eigvecs[0])[0]:.1f}'+', '+f'{cart2dir(eigvecs[0])[1]:.1f}'+')'+'  '+f'{e1:.0f}')
print('('+f'{cart2dir(eigvecs[1])[0]:.1f}'+', '+f'{cart2dir(eigvecs[1])[1]:.1f}'+')'+'  '+f'{e2:.0f}')
print('('+f'{cart2dir(eigvecs[2])[0]:.1f}'+', '+f'{cart2dir(eigvecs[2])[1]:.1f}'+')'+'  '+f'{e3:.0f}'+'\n')
print("Degree of anisotropy P")
print(f'{P:.3f}'+'\n')
print("Shape parameter T")
print(f'{T:.3f}'+'\n')

fig = plt.figure()
plot_confidence_ellipse_indiv(eigvecs[0][0],eigvecs[0][1],eigvecs[0][2],e1,e1,fig,'r','E1')
plot_confidence_ellipse_indiv(eigvecs[1][0],eigvecs[1][1],eigvecs[1][2],e2,e2,fig,'g','E2')
plot_confidence_ellipse_indiv(eigvecs[2][0],eigvecs[2][1],eigvecs[2][2],e3,e3,fig,'b','E3')

plt.show()
