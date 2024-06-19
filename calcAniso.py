## Based on Jelinek: the statistical theory of measuring anisotropy of
## magnetic susceptibility of rocks and its application

import numpy as np
import sys
from cart2dir import *
from dir2cart import *
#np.set_printoptions(suppress=True)

## AGICO 18 measurement directions (decl, incl, int=1):
U1 = np.array([0,0,1])
U2 = np.array([180,0,1])
U3 = np.array([90,0,1])
U4 = np.array([270,0,1])
U5 = np.array([0,90,1])
U6 = np.array([0,-90,1])
U7 = np.array([45,0,1])
U8 = np.array([225,0,1])
U9 = np.array([315,0,1])
U10 = np.array([135,0,1])
U11 = np.array([90,45,1])
U12 = np.array([270,-45,1])
U13 = np.array([90,-45,1])
U14 = np.array([270,45,1])
U15 = np.array([0,45,1])
U16 = np.array([180,-45,1])
U17 = np.array([180,45,1])
U18 = np.array([0,-45,1])
U = np.array([U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18])

## Read file
X_meas_tmp = []
fp = open(sys.argv[1], 'r')
for j, line in enumerate(fp):
    if j > 0:
        cols = line.split()
        X_meas_tmp.append(float(cols[1]))
fp.close()
nb_meas = len(X_meas_tmp)

idpos, idmeas = [], []
if nb_meas == 18:
    nb_pos = int(eval(input('How many do you want to use (9, 15, 18)?  ')))
    if nb_pos == 9:
        idpos = np.array([1,3,5,7,15,11,10,17,14])-1
        idmeas = idpos
    elif nb_pos == 15:
        idpos = np.array([10,7,1,9,8,14,11,3,13,12,18,15,5,17,16])-1
        idmeas = idpos
    elif nb_meas == 18:
        idpos = np.array([1,2,3,4,5,67,8,9,10,11,12,13,14,15,16,17,18])-1
        idmeas = idpos
elif nb_meas == 15:
    nb_pos = int(eval(input('How many do you want to use (9, 15)?  ')))
    if nb_pos == 9:
        idpos = np.array([1,3,5,7,15,11,10,17,14])-1
        idmeas = np.array([3,8,13,2,12,7,1,14,6])-1
    elif nb_pos == 15:
        idpos = np.array([10,7,1,9,8,14,11,3,13,12,18,15,5,17,16])-1
        idmeas = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])-1
elif nb_meas == 9:
    idpos = np.array([1,3,5,7,15,11,10,17,14])-1
    idmeas = np.array([1,2,3,4,5,6,7,8,9])-1
else:
    print('Weird number of measurements...')
    sys.exit()

## Vector with all measurement directions
P = []
for i in idpos:
    P.append(U[i])
P = np.array(P)

## Select appropriate measurements
X_meas = []
for i in idmeas:
    X_meas.append(X_meas_tmp[i])

## Calculate design matrix A
A = [0]*len(P)
for k in np.arange(len(P)):
    d1, d2, d3 = dir2cart(P[k])
    A[k] = [d1**2,d2**2,d3**2,2*d1*d2,2*d2*d3,2*d1*d3]
A = np.array(A)

## Calculate coefficients anisotropy matrix K
ALSQ = np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T)
coeffs = np.matmul(ALSQ,X_meas.T)
k11,k22,k33,k12,k23,k13 = coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4],coeffs[5]
Kvec = np.array([k11,k22,k33,k12,k23,k13])
K = np.array([[k11,k12,k13],[k12,k22,k23],[k13,k23,k33]])
print('Anisotropy matrix:')
print(str(K)+'\n')

## Calculate model ARM and RSS
X_model = np.matmul(A,Kvec)
RSS = np.sum((X_meas-X_model)**2)

## Diagonalize anisotropy matrix K to get eigenvectors and eigenvalues
eigvals, eigvecs = np.linalg.eig(K)
valvec = [[eigvals[i], np.transpose(eigvecs)[i]] for i in np.arange(len(eigvals))]
valvec.sort(key=lambda j: (j[0], -j[1]), reverse=True)
eigvals = np.array([valvec[i][0] for i in np.arange(len(valvec))])
eigvecs = np.array([valvec[i][1] for i in np.arange(len(valvec))])
eigvals_mean = (eigvals[0]+eigvals[1]+eigvals[2])/3

alpha1, alpha2, alpha3 = eigvals[0]/eigvals_mean, eigvals[1]/eigvals_mean, eigvals[2]/eigvals_mean
L, F, P = alpha1 / alpha2, alpha2 / alpha3, alpha1 / alpha3
T = (2*np.log(alpha2)-np.log(alpha1)-np.log(alpha3))/(np.log(alpha1)-np.log(alpha3))

## 95% confidence intervals on eigenvalues
## sqrt(0.4)*S*t9 where t9 is the quantile of a Student distribution with probablity 0.95 and 9 DoF
t9 = 1.833
S = np.sqrt(1/(len(D)-6)*RSS)
alpha_err = np.sqrt(0.4)*S*t9/eigvals_mean

print('Eigenvalues (normalized):')
print(f'{alpha1:.3f}'+', '+f'{alpha2:.3f}'+', '+f'{alpha3:.3f}'+'\n')
print('95% confidence interval on eigenvalues:')
print(' +/- '+f'{alpha_err:.3f}'+'\n')
print("Degree of anisotropy P")
print(f'{P:.3f}'+'\n')
print("Shape parameter T")
print(f'{T:.3f}'+'\n')
