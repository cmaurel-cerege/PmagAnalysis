import numpy as np
import sys

def cart2dir(cart):
    X, Y, Z = cart[0], cart[1], cart[2]
    R = np.sqrt(X**2+Y**2+Z**2)
    decl = (np.arctan2(Y,X)*180/np.pi) % 360
    if R == 0.0:
        incl = 90.0
    else:
        incl = np.arcsin(Z/R)*180/np.pi
    return decl, incl

def dir2cart(dir):
    decl, incl, intensity = dir[0], dir[1], dir[2]
    X = intensity*np.cos(decl*np.pi/180.0)*np.cos(incl*np.pi/180.0)
    Y = intensity*np.sin(decl*np.pi/180.0)*np.cos(incl*np.pi/180.0)
    Z = intensity*np.sin(incl*np.pi/180.0)
    return X, Y, Z

fp = open(sys.argv[1],'r')
name, Kmean, k1, k2, k3, d1, i1, d2, i2, d3, i3 = [], [], [], [], [], [], [], [], [], [], []
for j, line in enumerate(fp):
    if j > 0:
        cols = line.split()
        name.append(str(cols[0]))
        Kmean.append(float(cols[1])*1e-11)
        k1.append(float(cols[2]))
        k2.append(float(cols[3]))
        k3.append(float(cols[4]))
        d1.append(float(cols[5]))
        i1.append(float(cols[6]))
        d2.append(float(cols[7]))
        i2.append(float(cols[8]))
        d3.append(float(cols[9]))
        i3.append(float(cols[10]))
fp.close()

e1 = [np.array(dir2cart([d1[l],i1[l],1])) for l in np.arange(len(name))]
e2 = [np.array(dir2cart([d2[l],i2[l],1])) for l in np.arange(len(name))]
e3 = [np.array(dir2cart([d3[l],i3[l],1])) for l in np.arange(len(name))]

for k in np.arange(0,len(name)-1,2):

    A = np.asmatrix(np.column_stack((e1[k],e2[k],e3[k])))
    AH = np.asmatrix(np.column_stack((e1[k+1],e2[k+1],e3[k+1])))
    D = np.diag([k1[k]*Kmean[k],k2[k]*Kmean[k],k3[k]*Kmean[k]])
    DH = np.diag([k1[k+1]*Kmean[k+1],k2[k+1]*Kmean[k+1],k3[k+1]*Kmean[k+1]])

    Maniso = np.matmul(A,np.matmul(D,np.linalg.inv(A)))
    ManisoH = np.matmul(AH,np.matmul(DH,np.linalg.inv(AH)))

    ManisoDiff = ManisoH-Maniso

    eigvals, eigvecs = np.linalg.eig(ManisoDiff)
    valvec = [[eigvals[i], np.transpose(np.asarray(eigvecs))[i]] for i in np.arange(len(eigvals))]
    valvec.sort(key=lambda n: (n[0], -n[1]), reverse=True)
    eigvals = np.array([valvec[i][0] for i in np.arange(len(valvec))])
    eigvecs = np.array([valvec[i][1] for i in np.arange(len(valvec))])
    eigvals_mean = (eigvals[0]+eigvals[1]+eigvals[2])/3
    alpha1, alpha2, alpha3 = eigvals[0]/eigvals_mean, eigvals[1]/eigvals_mean, eigvals[2]/eigvals_mean
    L, F, P = alpha1 / alpha2, alpha2 / alpha3, alpha1 / alpha3
    T = (2*np.log(alpha2)-np.log(alpha1)-np.log(alpha3))/(np.log(alpha1)-np.log(alpha3))


    print('*****  '+name[k]+'  *****\n')
    print("Non-heated anisotropy matrix:")
    print(str(Maniso)+'\n')
    print("Heated anisotropy matrix:")
    print(str(ManisoH)+'\n')
    print("Differential anisotropy matrix:")
    print(str(ManisoDiff)+'\n')
    print("k_mean:")
    print(f'{eigvals_mean:.3e}\n')
    print("e1, e2, e3:")
    print('('+f'{cart2dir(eigvecs[0])[0]:.0f}'+', '+f'{cart2dir(eigvecs[0])[1]:.0f}'+')')
    print('('+f'{cart2dir(eigvecs[1])[0]:.0f}'+', '+f'{cart2dir(eigvecs[1])[1]:.0f}'+')')
    print('('+f'{cart2dir(eigvecs[2])[0]:.0f}'+', '+f'{cart2dir(eigvecs[2])[1]:.0f}'+')\n')
    print("k1, k2, k3:")
    print(f'{alpha1:.3f}'+', '+f'{alpha2:.3f}'+', '+f'{alpha3:.3f}'+'\n')
    print("Degree of anisotropy P")
    print(f'{P:.3f}'+'\n')
    print("Shape parameter T")
    print(f'{T:.3f}'+'\n')
