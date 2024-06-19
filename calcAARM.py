## Based on Jelinek: the statistical theory of measuring anisotropy of
## magnetic susceptibility of rocks and its application

import numpy as np
import sys
from cart2dir import *
from dir2cart import *
#np.set_printoptions(suppress=True)
from plotEqualArea import *

save = input('Save figs? (y/N)')

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
path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'
name = sys.argv[1].split('/')[-1].split('.')[0]

Mxtmp, Mytmp, Mztmp = [], [], []
fp = open(sys.argv[1], 'r')
for j, line in enumerate(fp):
    if j > 0:
        cols = line.split()
        Mxtmp.append(float(cols[1]) * 1e-3)
        Mytmp.append(float(cols[2]) * 1e-3)
        Mztmp.append(float(cols[3]) * 1e-3)
fp.close()
nb_meas = len(Mxtmp)

idpos, idmeas = [], []
if nb_meas == 18:
    nb_pos = int(eval(input('How many do you want to use (6, 9, 15, 18)?  ')))
    if nb_pos == 6:
        idpos = np.array([1,2,3,4,5,6])-1
        idmeas = idpos
    elif nb_pos == 9:
        idpos = np.array([1,3,5,7,15,11,10,17,14])-1
        idmeas = idpos
    elif nb_pos == 15:
        idpos = np.array([10,7,1,9,8,14,11,3,13,12,18,15,5,17,16])-1
        idmeas = idpos
    elif nb_meas == 18:
        idpos = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])-1
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
    nb_pos = nb_meas
    idpos = np.array([1,3,5,7,15,11,10,17,14])-1
    idmeas = np.array([1,2,3,4,5,6,7,8,9])-1
elif nb_meas == 6:
    nb_pos = nb_meas
    idpos = np.array([1,2,3,4,5,6])-1
    idmeas = idpos
else:
    print('Weird number of measurements...')
    sys.exit()


## Vector with all measurement directions
P = []
for i in idpos:
    P.append(U[i])
P = np.array(P)

## Select appropriate measurements
Mx, My, Mz = [], [], []
for i in idmeas:
    Mx.append(Mxtmp[i])
    My.append(Mytmp[i])
    Mz.append(Mztmp[i])

## Change the measurement reference frame to the agico reference frame
refframe = input('Are the 2G frame and Agico frame the same? (Y/n)')
if refframe == 'n':
    xxx = input('x_agico = ? (x,y,z,-x,-y,-z)\n')
    if xxx == 'x':
        Mrotx = np.array([1, 0, 0])
    elif xxx == '-x':
        Mrotx = np.array([-1, 0, 0])
    if xxx == 'y':
        Mrotx = np.array([0, 1, 0])
    elif xxx == '-y':
        Mrotx = np.array([0, -1, 0])
    if xxx == 'z':
        Mrotx = np.array([0, 0, 1])
    elif xxx == '-z':
        Mrotx = np.array([0, 0, -1])
    yyy = input('y_agico = ? (x,y,z,-x,-y,-z)\n')
    if yyy == 'x':
        Mroty = np.array([1, 0, 0])
    elif yyy == '-x':
        Mroty = np.array([-1, 0, 0])
    if yyy == 'y':
        Mroty = np.array([0, 1, 0])
    elif yyy == '-y':
        Mroty = np.array([0, -1, 0])
    if yyy == 'z':
        Mroty = np.array([0, 0, 1])
    elif yyy == '-z':
        Mroty = np.array([0, 0, -1])
    Mrotz = np.cross(Mrotx,Mroty)
    Mrot = np.matrix([Mrotx,Mroty,Mrotz])
else:
    Mrot = np.array([[1,0,0],[0,1,0],[0,0,1]])
print('2G to agico rotation matrix:')
print(Mrot)

## Calculate magnetization along the measured direction
Mxrot, Myrot, Mzrot, X_meas = [], [], [], []
for k in np.arange(len(Mx)):
    d1, d2, d3 = dir2cart(P[k])
    Magico = np.array(np.matmul(Mrot, np.array([Mx[k], My[k], Mz[k]])))
    Mxrot.append(Magico[0])
    Myrot.append(Magico[1])
    Mzrot.append(Magico[2])
    X_meas.append(Magico[0] * d1 + Magico[1] * d2 + Magico[2] * d3)

## Decimal
dec = np.absolute(np.floor(np.log10(np.absolute(X_meas[0]))))
X_meas = np.array(X_meas)*10**(dec)

if nb_pos == 6:
    ## Build anisotropy matrix
    MXX = (Mxrot[0]-Mxrot[1])/2
    MYY = (Myrot[2]-Myrot[3])/2
    MZZ = (Mzrot[4]-Mzrot[5])/2
    MXY = MYX = ((Myrot[0]-Myrot[1])/2+(Mxrot[2]-Mxrot[3])/2)/2
    MXZ = MZX = ((Mzrot[0]-Mzrot[1])/2+(Mxrot[4]-Mxrot[5])/2)/2
    MYZ = MZY = ((Mzrot[2]-Mzrot[3])/2+(Myrot[4]-Myrot[5])/2)/2

    K = np.array([[MXX, MXY, MXZ], [MYX, MYY, MYZ], [MZX, MZY, MZZ]])
    print('Anisotropy matrix:')
    print(str(K)+'\n')

    eigvals, eigvecs = np.linalg.eig(K)
    valvec = [[eigvals[i], np.transpose(eigvecs)[i]] for i in np.arange(len(eigvals))]
    valvec.sort(key=lambda k: (k[0], -k[1]), reverse=True)
    eigvals = np.array([valvec[i][0] for i in np.arange(len(valvec))])
    eigvecs = np.array([valvec[i][1] for i in np.arange(len(valvec))])
    eigvals_mean = (eigvals[0]+eigvals[1]+eigvals[2])/3
    alpha1, alpha2, alpha3 = eigvals[0]/eigvals_mean, eigvals[1]/eigvals_mean, eigvals[2]/eigvals_mean
    L, F, P = alpha1 / alpha2, alpha2 / alpha3, alpha1 / alpha3
    T = (2*np.log(alpha2)-np.log(alpha1)-np.log(alpha3))/(np.log(alpha1)-np.log(alpha3))

    print('Eigenvalues (normalized):')
    print(f'{alpha1:.3f}'+', '+f'{alpha2:.3f}'+', '+f'{alpha3:.3f}'+'\n')
    print("Degree of anisotropy P")
    print(f'{P:.3f}'+'\n')
    print("Shape parameter T")
    print(f'{T:.3f}'+'\n')

else:
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
    print('Anisotropy matrix (x1e-'+str(dec)+'): ')
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
    S = np.sqrt(1/(len(U)-6)*RSS)
    alpha_err = np.sqrt(0.4)*S*t9/eigvals_mean

    ## 95% confidence ellipses on eigenvectors
    ## F29 is the quantile of a F distribution of 2 and 9 DoF with probability 95%.
    F29 = 4.26
    e1_2 = np.arctan2(S*np.sqrt(2*F29),2*np.absolute(alpha1-alpha2))*180/np.pi
    e1_3 = np.arctan2(S*np.sqrt(2*F29),2*np.absolute(alpha1-alpha3))*180/np.pi
    e1 = np.min([e1_2,e1_3])

    e2_1 = np.arctan2(S*np.sqrt(2*F29),2*np.absolute(alpha1-alpha2))*180/np.pi
    e2_3 = np.arctan2(S*np.sqrt(2*F29),2*np.absolute(alpha2-alpha3))*180/np.pi
    e2 = np.min([e2_1, e2_3])

    e3_1 = np.arctan2(S*np.sqrt(2*F29),2*np.absolute(alpha1-alpha3))*180/np.pi
    e3_2 = np.arctan2(S*np.sqrt(2*F29),2*np.absolute(alpha2-alpha3))*180/np.pi
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
    plot_indiv_confidence_ellipse(eigvecs[0][0], eigvecs[0][1], eigvecs[0][2], e1, e1, 'r', fig,'E1')
    plot_indiv_confidence_ellipse(eigvecs[1][0], eigvecs[1][1], eigvecs[1][2], e2, e2, 'g', fig,'E2')
    plot_indiv_confidence_ellipse(eigvecs[2][0], eigvecs[2][1], eigvecs[2][2], e3, e3, 'b', fig,'E3')
    if save == 'y':
        plt.savefig(path+name+'.pdf', format='pdf', dpi=200, bbox_inches='tight')




plt.show()