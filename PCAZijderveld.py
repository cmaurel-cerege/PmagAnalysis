import numpy as np
from cart2dir import *

def Define_segment(Mx, My, Mz):

    id1 = np.int(eval(input("Index of the first datapoint?")))
    id2 = np.int(eval(input("Index of the last datapoint? (use 99 for very last)")))
    if id2 == 99:
        id2 = len(Mx)-1
    Mxc, Myc, Mzc = Mx[id1:id2+1], My[id1:id2+1], Mz[id1:id2+1]

    return np.array(Mxc), np.array(Myc), np.array(Mzc), id1, id2

def Calc_PCA(Mx, My, Mz):
    Mxp = np.array(Mx) - np.mean(Mx)
    Myp = np.array(My) - np.mean(My)
    Mzp = np.array(Mz) - np.mean(Mz)

    COV = 1/len(Mxp)*np.array([[np.sum(Mxp**2), np.sum(Mxp*Myp), np.sum(Mxp*Mzp)],
                            [np.sum(Myp*Mxp), np.sum(Myp**2), np.sum(Myp*Mzp)],
                            [np.sum(Mzp*Mxp), np.sum(Mzp*Myp), np.sum(Mzp**2)]])

    val, vec = np.linalg.eig(COV)
    vals = sorted(val)[::-1]
    vecs = [vec[:,list(val).index(v)] for v in vals]

    ## Choose the right direction for the PCA vector
    control = np.array([Mx[0]-Mx[-1],My[0]-My[-1],Mz[0]-Mz[-1]])
    dot = np.dot(control,vecs[0])
    if dot < 0:
        vecs[0][0], vecs[0][1], vecs[0][2] = -vecs[0][0], -vecs[0][1], -vecs[0][2]

    return vals, vecs


def Calc_MAD_and_DANG(Mx, My, Mz, AF, mass=True):

    Mxc, Myc, Mzc, id1, id2 = Define_segment(Mx, My, Mz)
    Mcmax = np.array([Mxc[0]-Mxc[-1],Myc[0]-Myc[-1],Mzc[0]-Mzc[-1]])
    print("\nRange: "+str(int(AF[id1]))+' mT - '+str(int(AF[id2]))+' mT')
    print('N = '+str(len(Mxc)))

    if mass == True:
        print("Magnetic moment: "+f'{np.linalg.norm(Mcmax):.3e}'+r' A m2 kg-1')
    else:
        print("Magnetic moment: " + f'{np.linalg.norm(Mcmax):.3e}' + r' A m2')
    vals, vecs = Calc_PCA(Mxc, Myc, Mzc)
    print("PC vector (x, y, z): " + f'{vecs[0][0]:.2e}' + ", " + f'{vecs[0][1]:.2e}' + ", " + f'{vecs[0][2]:.2e}')
    print("PC vector (decl/incl): "+f'{cart2dir(vecs[0])[0]:.2f}'+"°  "+f'{cart2dir(vecs[0])[1]:.2f}'+'°')

    #print("Prolate (l3 >> l1, l2) or oblate (l3 = l2 >> l1)? (p/o)")
    #type = str(input())

    type = 'p'

    MAD = 0
    if type == 'p':
        MAD = np.arctan2(np.sqrt(vals[1] + vals[2]), np.sqrt(vals[0])) * 180.0 / np.pi
    elif type == 'o':
        MAD = np.arctan2(np.sqrt(vals[2]), np.sqrt(vals[0] + vals[1])) * 180.0 / np.pi

    print("MAD = "+f'{MAD:.2f}'+'°')

    if len(Mxc) == 3: CMAD = 7.69;
    if len(Mxc) == 4: CMAD = 3.90;
    if len(Mxc) == 5: CMAD = 3.18;
    if len(Mxc) == 6: CMAD = 2.88;
    if len(Mxc) == 7: CMAD = 2.71;
    if len(Mxc) == 8: CMAD = 2.63;
    if len(Mxc) == 9: CMAD = 2.57;
    if len(Mxc) == 10: CMAD = 2.54;
    if len(Mxc) == 11: CMAD = 2.51;
    if len(Mxc) == 12: CMAD = 2.48;
    if len(Mxc) == 13: CMAD = 2.46;
    if len(Mxc) == 14: CMAD = 2.44;
    if len(Mxc) == 15: CMAD = 2.43;
    if len(Mxc) == 16: CMAD = 2.43;
    if len(Mxc) > 16: CMAD = 2.37;

    MAD95 = CMAD*MAD
    #print("alpha 95 = CMAD*MAD = " + f'{MAD95:.2f}')

    CoM = np.array([np.mean(Mx), np.mean(My), np.mean(Mz)])
    DANG = np.arccos(np.dot(Mcmax/np.linalg.norm(Mcmax),CoM/np.linalg.norm(CoM)))*180/np.pi
    print("DANG = " + f'{DANG:.2f}'+'°\n')

    return MAD, DANG, vecs[0], MAD95, Mcmax, id1, id2
