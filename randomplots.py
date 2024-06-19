import numpy as np
import matplotlib.pyplot as plt
from plotEqualArea import *

# ratioARM = [0.44,0.46,0.28,0.19,0.27,0.13,0.44,0.13]
# ratioIRM1H = [0.15,0.01,0.15,0.28,0.16,0.06,0.28,0.12]
# ratioIRM2H = [0.16,0.03,0.33,0.28,0.19,0.06,0.37,0.38]
#
# plt.figure(1)
# plt.xlabel('Ratio ARM before and after 1-h heating')
# plt.ylabel('Ratio IRM before and after 1-h heating')
# plt.xlim(0,0.7)
# plt.ylim(0,0.7)
# plt.plot(ratioARM,ratioIRM1H,'bo',ms=7,lw=0,label='IRM heated 1h')
# plt.plot(ratioARM,ratioIRM2H,'ro',ms=5,lw=0,label='IRM heated 3h')
# plt.plot([0,0.7],[0,0.7],'k--',lw=0.5)
# plt.legend()
#
#
#
# fp = open('../../../../Desktop/ARMfactordatabase.csv','r')
# ARMdomain, ARMfactor= [], []
# for j, line in enumerate(fp):
#     if j > 0:
#         cols=line.split(',')
#         ARMdomain.append(str(cols[0]))
#         ARMfactor.append(1/float(cols[1]))
# fp.close()
# fp = open('../../../../Desktop/IRMfactordatabase.csv','r')
# IRMdomain, IRMfactor= [], []
# for j, line in enumerate(fp):
#     if j > 0:
#         cols=line.split(',')
#         IRMdomain.append(str(cols[0]))
#         IRMfactor.append(float(cols[1]))
# fp.close()
#
# IRMfactorSD = [IRMfactor[k] for k in np.arange(len(IRMfactor)) if IRMdomain[k] == 'SD/PSD']
# ARMfactorSD = [ARMfactor[k] for k in np.arange(len(ARMfactor)) if ARMdomain[k] == 'SD/PSD']
#
# IRMfactorMD = [IRMfactor[k] for k in np.arange(len(IRMfactor)) if IRMdomain[k] == 'MD']
# ARMfactorMD = [ARMfactor[k] for k in np.arange(len(ARMfactor)) if ARMdomain[k] == 'MD']
#
#
# samples = ['SIG', 'BUR', 'BAR03', 'BAR04', 'ROB01', 'FBL', 'Synt1', 'Synt2']
# fARM = [1/0.557, 1/0.548, 1/0.939, 1/0.523, 1/1.099, 1/1.603, 1/1.377, 1/3.042]
# aIRM = [1185, 2975, 5220, 6084, 5919, 2665, 4191, 3287]
#
# print(len(IRMfactorSD))
# print(len(ARMfactorSD))
#
# plt.figure(2,figsize=(4,3))
# plt.xlabel('IRM factor')
# plt.xlim(0,8000)
# plt.ylim(0,0.25)
# plt.hist(IRMfactorSD,bins=40,weights=[1./len(IRMfactorSD)]*len(IRMfactorSD),color='darkred',alpha=0.5,edgecolor='darkred', label='SD/PSD')
# #plt.hist(IRMfactorMD,bins=40,color='darkblue',alpha=0.5,edgecolor='darkblue', label='MD')
# #plt.legend()
# plt.hist(aIRM,bins=30,weights=[1./len(aIRM)]*len(aIRM),color='darkblue',alpha=0.5,edgecolor='darkblue')
# #plt.savefig('../IRMhist.pdf', format='pdf', dpi=300, bbox_inches="tight")
#
# plt.figure(3,figsize=(4,3))
# plt.xlabel("ARM factor")
# plt.xlim(0,2)
# plt.ylim(0,0.25)
# plt.hist(ARMfactorSD,bins=15,weights=[1./len(ARMfactorSD)]*len(ARMfactorSD),color='darkred',alpha=0.5,edgecolor='darkred', label='SD/PSD')
# #plt.hist(ARMfactorMD,bins=40,color='darkblue',alpha=0.5,edgecolor='darkblue', label='MD')
# #plt.legend()
# plt.hist(fARM,bins=40,weights=[1./len(fARM)]*len(fARM),color='darkblue',alpha=0.5,edgecolor='darkblue')
# #plt.savefig('../ARMhist.pdf', format='pdf', dpi=300, bbox_inches="tight")
#
# plt.figure(4,figsize=(6,3))
# plt.xlim(-0.5,len(fARM)-0.5)
# plt.ylim(0,4)
# plt.xticks(np.arange(len(fARM)),samples)
# plt.xlabel('Sample')
# plt.ylabel('ARM factor f')
# plt.plot([-0.5,len(fARM)-0.5],[np.mean(ARMfactorSD),np.mean(ARMfactorSD)], 'k--', lw=1)
# plt.plot(np.arange(len(fARM)), fARM, color='darkred', marker='o',ms=5, lw=0)
#
#
# plt.figure(5,figsize=(6,3))
# plt.xlim(-0.5,len(aIRM)-0.5)
# plt.ylim(0,5000)
# plt.xticks(np.arange(len(aIRM)),samples)
# plt.xlabel('Sample')
# plt.ylabel('IRM factor a')
# plt.plot([-0.5,len(aIRM)-0.5],[np.mean(IRMfactorSD),np.mean(IRMfactorSD)], 'k--', lw=1)
# plt.plot(np.arange(len(aIRM)), aIRM, color='darkblue', marker='o',ms=5, lw=0)
#
# #####
# colors = ['darkred', 'darkgreen', 'darkblue', 'coral', 'gray', 'violet']
#
# samples100 = ['SIG', 'BUR', 'BAR03', 'ROB01', 'FBL', 'CuPPy']
# fARM100 = [0, 0.557, 0.548, 0.523, 1.099, 3.042]
# aIRM100 = [1185, 2975, 5220, 5919, 2665, 3287]
#
# samples50 = ['SIG', 'BUR', 'BAR03', 'ROB01', 'FBL', 'CuPPy']
# fARM50 = [0, 0.224, 0.328, 0.441, 1.241, 2.370]
# aIRM50 = [1960, 3697, 3841, 2460, 1193, 2332]
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.set_figheight(4)
# fig.set_figwidth(8)
# ax1.set_aspect('equal')
# ax2.set_aspect('equal')
# ax1.set_xlabel("fARM 100 uT")
# ax1.set_ylabel("fARM 50 uT")
# ax2.set_xlabel("aIRM 100 uT")
# ax2.set_ylabel("aIRM 50 uT")
# ax2.set_xlim(0,6500)
# ax2.set_ylim(0,6500)
# ax1.set_xlim(0,3.5)
# ax1.set_ylim(0,3.5)
# for k in np.arange(len(fARM100)):
#     ax1.plot(fARM100[k],fARM50[k],lw=0,marker='o',ms=5,color=colors[k],label=samples100[k])
#     ax2.plot(aIRM100[k],aIRM50[k],lw=0,marker='o',ms=5,color=colors[k])
# ax1.plot([0,3.5],[0,3.5],'k-')
# ax2.plot([0,6500],[0,6500],'k-')
#
#
# fig.tight_layout()
# ax1.legend()
#
#
#
#
#
# ## Type 3 tetra
# names = ['Semarkona','LAR06279', 'GRO06054','Krymka','Chainpur', 'Hallingeberg','Khohar','Mezo-Madaras','Dhajala']
# types = ['3.00','3.05/1','3.1','3.2','3.4','3.4','3.6','3.7','3.8']
# colors = ['darkred', 'darkgreen', 'darkblue', 'coral', 'gold', 'gray', 'violet','k','cyan']
#
# coerpeaks = [[54],[69],[70,342],[267,93,29],[185,14],[480,132,23],[42],[185,14],[296,123,61]]
# pct_HC = [0,0,8,27,93,22,0,93,44]
#
# fig = plt.figure()
# for k in np.arange(len(names)):
#     for j in np.arange(len(coerpeaks[k])):
#         if j == 0:
#             plt.plot(k,coerpeaks[k][j],lw=0,marker='o',ms=6,color=colors[k],label=names[k])
#         else:
#             plt.plot(k, coerpeaks[k][j], lw=0, marker='o', ms=6, color=colors[k])
# plt.xticks(np.arange(len(names)),types)
# plt.xlim(-1,9)
# plt.ylim(0,500)
# plt.xlabel("Petrologic type")
# plt.ylabel("Coercivity peak (mT)")
# plt.xticks(np.arange(0,9),types)
# plt.legend(loc=2)
# fig.tight_layout()
#
# fig = plt.figure()
# plt.xlim(-1,9)
# plt.ylim(-1,100)
# plt.xlabel('Petrologic type')
# plt.ylabel('Percent of coercivity from HC mineral')
# plt.xticks(np.arange(0,9),types)
# for k in np.arange(len(names)):
#     plt.plot(np.arange(len(pct_HC))[k], pct_HC[k], lw=0, marker='o',color=colors[k],label=names[k])
# plt.legend()
# fig.tight_layout()



S7a11 = [9.7e-6,-1.7e-5,7.4e-6]
S7a12 = [1.6e-5,-2.65e-5,-1.4e-5]
S7a13 = [1.75e-5,-2.85e-5,-8.63e-7]
S7a14 = [1.73e-5,-2.88e-5,-3.46e-6]
S7a16 = [1.25e-5,-2.46e-5,2.32e-7]
S7as = [8.4e-5,-1.5e-4,-2.9e-5]

S7 = [S7a11,S7a12,S7a13,S7a14,S7a16,S7as]

Mx = [ss[0] for ss in S7]
My = [ss[1] for ss in S7]
Mz = [ss[2] for ss in S7]

for k in np.arange(len(Mx)):
    print(cart2dir([Mx[k],My[k],Mz[k]]))


plot_equal_area_sequence(Mx,My,Mz, [1]*len(S7))

S7e1 = [2.39e-6,-9.69e-6,1.9e-7]
S7e2 = [-5.91e-7,7.35e-6,2.68e-6]
S7e3 = [-4.8e-6,-1.15e-7,-1.61e-6]
S7e4 = [9.95e-6,-1.65e-5,-1.32e-5]
S7e5 = [6.57e-7,1.25e-5,1.04e-6]
S7e = [1.36e-6,-2.53e-5,-2.91e-5]

S7e = [S7e1,S7e2,S7e3,S7e4,S7e5,S7e]

Mx = [ss[0] for ss in S7e]
My = [ss[1] for ss in S7e]
Mz = [ss[2] for ss in S7e]

plot_equal_area_sequence(Mx,My,Mz, [1]*len(S7e))



plt.show()