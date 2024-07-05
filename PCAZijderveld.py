import numpy as np
from cart2dir import *
from numpy.linalg import *

def Calc_PCA(Mx, My, Mz, anc='free'):

    if anc == 'anc':
        Mxp = np.array(Mx)
        Myp = np.array(My)
        Mzp = np.array(Mz)
    elif anc == 'free':
        Mxp = np.array(Mx) - np.mean(Mx)
        Myp = np.array(My) - np.mean(My)
        Mzp = np.array(Mz) - np.mean(Mz)

    COV = np.array([[np.sum(Mxp**2), np.sum(Mxp*Myp), np.sum(Mxp*Mzp)],\
                    [np.sum(Myp*Mxp), np.sum(Myp**2), np.sum(Myp*Mzp)],\
                    [np.sum(Mzp*Mxp), np.sum(Mzp*Myp), np.sum(Mzp**2)]])

    val, vec = np.linalg.eig(COV)
    vals = sorted(val)[::-1]
    vecs = [vec[:,list(val).index(v)] for v in vals]

    ## Choose the right direction for the PCA vector
    control = np.array([Mxp[0]-Mxp[-1],Myp[0]-Myp[-1],Mzp[0]-Mzp[-1]])
    dot = np.dot(control,vecs[0])
    if dot < -1:
        dot = -1
    elif dot > 1:
        dot = 1
    if np.arccos(dot) > np.pi/2:
        vecs[0][0], vecs[0][1], vecs[0][2] = -vecs[0][0], -vecs[0][1], -vecs[0][2]

    return vals, vecs


def PCA_analysis(Mx, My, Mz, AF, rem='', remdata=[], mass=1, demag='AF'):

    if mass == 1:
        unit = 'A m2'
    else:
        unit = 'A m2 kg-1'
    if demag == 'AF':
        dem = 'mT'
    elif demag == 'TH':
        dem = '°C'

    nb_comp = input('Number of components?  (default = 1)')
    if nb_comp == '':
        nb_comp = 1
    else:
        nb_comp = int(eval(nb_comp))

    id_i, id_f = [], []
    for n in np.arange(nb_comp):
        print('COMPONENT '+str(n+1))
        idi = input('First datapoint?  (default = 0)  ')
        idf = input('Last datapoint?  (default = last of sequence)  ')
        if idi == '': idi = 0
        else: idi = int(eval(idi))
        if idf == '': idf = len(Mx)-1
        else: idf = int(eval(idf))
        id_i.append(idi)
        id_f.append(idf)
    print('\n')

    Mcx, Mcy, Mcz, Mcd, Mci, Mcmax, MAD, DANG, MAD95 = [], [], [], [], [], [], [], [], []
    for k in np.arange(len(id_i)):

        Cx = Mx[id_i[k]:id_f[k]+1]
        Cy = My[id_i[k]:id_f[k]+1]
        Cz = Mz[id_i[k]:id_f[k]+1]

        print('COMPONENT '+str(k+1))
        print('Principal component vector:')
        Mcmax.append(np.array([Cx[0]-Cx[-1],Cy[0]-Cy[-1],Cz[0]-Cz[-1]]))
        print(" * Range: "+str(int(AF[id_i[k]]))+' '+dem+' - '+str(int(AF[id_f[k]]))+' '+dem)
        print(' * N = '+str(len(Cx)))
        print(" * Magnetic moment: "+f'{np.linalg.norm(Mcmax[-1]):.3e}'+r' '+unit)

        vals, vecs = Calc_PCA(Cx, Cy, Cz)
        Mcx.append(vecs[0][0])
        Mcy.append(vecs[0][1])
        Mcz.append(vecs[0][2])
        Mcd.append(cart2dir(vecs[0])[0])
        Mci.append(cart2dir(vecs[0])[1])

        print(" * PCx, PCy, PCz = " + f'{Mcx[-1]:.3f}' + ", " + f'{Mcy[-1]:.3f}' + ", " + f'{Mcz[-1]:.3f}')
        print(" * PCd, PCi = "+f'{Mcd[-1]:.1f}'+"°  "+f'{Mci[-1]:.1f}'+'°')

        MAD.append(np.arctan2(np.sqrt(vals[1]+vals[2]), np.sqrt(vals[0]))*180.0/np.pi)
        print(" * MAD = "+f'{MAD[-1]:.2f}'+'°')

        lenCx = [3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        CMADS = [7.69,3.9,3.18,2.88,2.71,2.63,2.57,2.54,2.48,2.46,2.44,2.43,2.43]
        for i in np.arange(len(lenCx)):
            if len(Cx) == lenCx[i]:
                CMAD = CMADS[i]
            elif len(Cx) > lenCx[i]:
                CMAD = 2.37

        MAD95.append(CMAD*MAD[-1])
        #print(" * alpha 95 = CMAD*MAD = " + f'{MAD95[-1]:.2f}')

        Mcvec = np.array([Mcx[-1], Mcy[-1], Mcz[-1]])
        CoM = np.array([np.mean(Mx), np.mean(My), np.mean(Mz)])
        DANG.append(np.arccos(np.dot(Mcvec,CoM)/(norm(CoM)*norm(Mcvec)))*180/np.pi)
        print(" * DANG = " + f'{DANG[-1]:.2f}'+'°')

        if rem != '':
            print(' * REM '+rem+': '+f'{norm(Mcmax[-1])/remdata[id_i[k]]:.5f}\n')

    return Mcx, Mcy, Mcz, Mcd, Mci, Mcmax, MAD, DANG, MAD95, id_i, id_f

