## All VRM data
import numpy as np

# time = [0,
#         14*60,
#         1*3600+26*60,
#         3*3600+47*60,
#         23*3600+18*60,
#         1*24*3600+4*3600+0*60,
#         6*24*3600+4*3600+24*60,
#         9*24*3600+1*3600+2*60,
#         13*24*3600+1*3600+8*60,
#         20*24*3600+0*3600+56*60,
#         33*24*3600+4*3600+35*60,
#         36*24*3600+5*3600+50*60,
#         49*24*3600+1*3600+24*60,
#         83*24*3600+1*3600+48*60,
#         159*24*3600+23*2600+22*60,
#         379*24*3600+23*3600+9*60]

time = [23+k*17 for k in np.arange(0,19)]+\
       [6*60+k*16-23 for k in np.arange(0,1)]+\
       [12*60+k*16-23 for k in np.arange(0,1)]+\
       [18*60+k*16-23 for k in np.arange(0,1)]+\
       [23*60+k*16-23 for k in np.arange(0,1)]+\
       [29*60+k*16-23 for k in np.arange(0,1)]+\
       [35*60+k*16-23 for k in np.arange(0,1)]+\
       [41*60+k*16-23 for k in np.arange(0,1)]+\
       [47*60+k*16-23 for k in np.arange(0,1)]+\
       [53*60+k*16-23 for k in np.arange(0,1)]+\
       [59*60+k*16-23 for k in np.arange(0,1)]+\
       [1*3600+4*60+k*16-23 for k in np.arange(0,1)]+\
       [1*3600+10*60+k*16-23 for k in np.arange(0,1)]+\
       [1*3600+16*60+k*16-23 for k in np.arange(0,1)]+\
       [1*3600+22*60+k*16-23 for k in np.arange(0,1)]+\
       [1*3600+28*6+k*16-23 for k in np.arange(0,1)]+\
       [1*3600+34*60+k*16-23 for k in np.arange(0,1)]+\
       [1*3600+40*60+k*16-23 for k in np.arange(0,1)]+\
       [1*3600+46*60+k*16-23 for k in np.arange(0,1)]+\
       [1*3600+52*60+k*16-23 for k in np.arange(0,1)]+\
       [1*3600+58*60+k*16-23 for k in np.arange(0,1)]+\
       [2*3600+4*60+k*16-23 for k in np.arange(0,1)]+\
       [2*3600+10*60+k*16-23 for k in np.arange(0,1)]+\
       [2*3600+16*60+k*16-23 for k in np.arange(0,1)]+\
       [2*3600+22*60+k*16-23 for k in np.arange(0,1)]+\
       [2*3600+34*60+k*16-23 for k in np.arange(0,1)]+\
       [2*3600+42*60+k*16-23 for k in np.arange(0,1)]+\
       [2*3600+48*60+k*16-23 for k in np.arange(0,1)]+\
       [2*3600+54*60+k*16-23 for k in np.arange(0,1)]+\
       [3*3600+0*60+k*16-23 for k in np.arange(0,1)]+\
       [3*3600+6*60+k*16-23 for k in np.arange(0,1)]+\
       [3*3600+12*60+k*16-23 for k in np.arange(0,1)]+\
       [3*3600+18*60+k*16-23 for k in np.arange(0,1)]+\
       [3*3600+25*60+k*16-23 for k in np.arange(0,1)]+\
       [5*3600+3*60+k*16-23 for k in np.arange(0,1)]+\
       [5*3600+14*60+k*16-23 for k in np.arange(0,1)]+\
       [6*3600+33*60+k*16-23 for k in np.arange(0,1)]+\
       [7*3600+38*60+k*16-23 for k in np.arange(0,1)]+\
       [15*3600+18*60+k*16-23 for k in np.arange(0,1)]+\
       [19*3600+53*60+k*16-23 for k in np.arange(0,1)]+\
       [23*3600+51*60+k*16-23 for k in np.arange(0,1)]+\
       [3*24*3600+16*3600+37*60+k*16-23 for k in np.arange(0,1)]+\
       [7*24*3600+17*3600+10*60+k*16-23 for k in np.arange(0,1)]#+\
       #[29*24*3600+2*3600+29*60+k*16-23 for k in np.arange(0,1)]

print(len(time))
fp = open('EC002S7A1-0_VRMDECtime_cut.dat','w')
for k in np.arange(len(time)):
    fp.write(str(time[k])+'\n')
fp.close()
