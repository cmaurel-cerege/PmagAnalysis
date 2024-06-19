import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
from plotZijderveld import *

save = input('Save figure? (y/N)')

path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'
sample = sys.argv[1].split('/')[-1].split('.')[0].split('_')[0]

field = 57.0
mass = float(eval(input('Mass of the sample (g)?')))*1e-3

tacq, Mxacq, Myacq, Mzacq, tdec, Mxdec, Mydec, Mzdec = [], [], [], [], [], [], [], []
for k in np.arange(1,len(sys.argv)):
    fp = open(sys.argv[k],'r')
    if 'VRMACQtime' in sys.argv[k]:
        for j, line in enumerate(fp):
            cols = line.split()
            tacq.append(float(cols[0]))
        fp.close()
        tacq = np.array(tacq)
    elif 'VRMACQ' in sys.argv[k]:
        for j, line in enumerate(fp):
            if j > 0:
                cols = line.split()
                Mxacq.append(float(cols[1]) * 1e-3)
                Myacq.append(float(cols[2]) * 1e-3)
                Mzacq.append(float(cols[3]) * 1e-3)
        fp.close()
        Mxacq, Myacq, Mzacq = np.array(Mxacq), np.array(Myacq), np.array(Mzacq)
    elif 'VRMDECtime_cut' in sys.argv[k]:
        for j, line in enumerate(fp):
            cols = line.split()
            tdec.append(float(cols[0]))
        fp.close()
        tdec = np.array(tdec)
    elif 'VRMDEC_cut' in sys.argv[k]:
        for j, line in enumerate(fp):
            if j > 0:
                cols = line.split()
                Mxdec.append(float(cols[1]) * 1e-3)
                Mydec.append(float(cols[2]) * 1e-3)
                Mzdec.append(float(cols[3]) * 1e-3)
        fp.close()
        Mxdec, Mydec, Mzdec = np.array(Mxdec), np.array(Mydec), np.array(Mzdec)

plt.figure(1)
plt.xlabel('log(Time [s])')
plt.ylabel('VRM [A m2 kg-1 uT-1]')
plt.xlim(1,8)
plt.ylim(0,3e-7)

### DECAY ###
if len(tdec) != 0:
    VRMxdec, VRMydec, VRMzdec = Mxdec-Mxacq[0], Mydec-Myacq[0], Mzdec-Mzacq[0]
    VRMdec = np.sqrt(VRMxdec**2 + VRMydec**2 + VRMzdec**2)/(mass*field)
    logtdec = np.log10(tdec)
    resdec = stats.linregress(logtdec, VRMdec)

    plt.figure(1)
    plt.plot(logtdec, VRMdec, color='k', lw=0.5, marker='o', ms=8, markeredgecolor='k', markerfacecolor='w')
    plt.plot(logtdec, (resdec.slope*logtdec + resdec.intercept), color='r', ls='--', lw=1)

    Sd = resdec.slope
    print("Sd = " + str(Sd) + " A m2 kg-1 log(s)-1 uT-1")

if len(tacq) != 0:
    NRM = np.sqrt(Mxacq[0]**2+Myacq[0]**2+Mzacq[0]**2)
    VRMxacq, VRMyacq, VRMzacq = Mxacq[1:]-Mxacq[0], Myacq[1:]-Myacq[0], Mzacq[1:]-Mzacq[0]

    VRMacq = np.sqrt(VRMxacq**2 + VRMyacq**2 + VRMzacq**2)/(mass*field)
    logtacq = np.log10(tacq[1:])
    if len(tdec) != 0:
        VRMacq += Sd*np.log10(23)
    resacq = stats.linregress(logtacq, VRMacq)

    plt.figure(1)
    plt.plot(logtacq, VRMacq, color='k', lw=0.5, marker='o', ms=8, markeredgecolor='k', markerfacecolor='k')
    plt.plot(logtacq, (resacq.slope*logtacq + resacq.intercept), color='r', ls='--', lw=1)

    if save == 'y':
        plt.savefig(path+sample+'-VRM.pdf', format='pdf', dpi=400, bbox_inches="tight")

    print("Sa = " + str(resacq.slope) + " A m2 kg-1 log(s)-1 uT-1")

    VRMacqprop = resacq.slope * np.log10(1e5*3600*24*365)*45.
    print("After 10,000 years in a 45 uT field:")
    print("VRM acquired: " + f'{VRMacqprop:.2e}' + " A m2 kg-1")
    print("VRM acquired (% NRM): " + f'{VRMacqprop/NRM:.0f}' + " %")

    Plot_Zijderveld(VRMxacq, VRMyacq, VRMzacq, tacq, newfig='y', unit='mom', gui='nogui', color='k', path='')
    if save == 'y':
        plt.savefig(path+sample+'-VRMacqZijd.pdf', format='pdf', dpi=400, bbox_inches="tight")

    if len(tdec) != 0:
        print("Sa/Sd = " + str(np.absolute(resacq.slope / resdec.slope)))
        Plot_Zijderveld(VRMxdec, VRMydec, VRMzdec, tdec, newfig='y', unit='mom', gui='nogui', color='k', path='')
        if save == 'y':
            plt.savefig(path + sample + '-VRMdecZijd.pdf', format='pdf', dpi=400, bbox_inches="tight")


plt.show()