import numpy as np
import sys

MatAFx, MatAFy, MatAFz, MatAFstep = [], [], [], []

files = sys.argv[1:]
for file in files:
    path = ''
    for k in np.arange(len(file.split('/'))-1):
        path += str(file.split('/')[k])+'/'
    name = str(file.split('/')[-1].split('.')[0])
    ext = str(file.split('/')[-1].split('.')[1])
    fp = open(path+name+'.'+ext,'r')

    Mx, My, Mz, step = [], [], [], []
    for j, line in enumerate(fp):
        if j > 0:
            cols = line.split()
            Mx.append(float(cols[1])*1e-3)
            My.append(float(cols[2])*1e-3)
            Mz.append(float(cols[3])*1e-3)
            step.append(int(cols[-1])*0.1)
    fp.close()

    fpn = open(path+name+'.txt','w')
    fpn.write(name+'\n')
    fpn.write('step\tX\tY\tZ\n')
    for j in np.arange(len(Mx)):
        fpn.write(str(int(step[j]))+'\t'+'{:.4e}'.format(Mx[j])+'\t'+'{:.4e}'.format(My[j])+'\t'+'{:.4e}'.format(Mz[j])+'\n')
    fpn.close()



# if len(sys.argv) > 1:
#
#     path = ''
#     for k in np.arange(len(sys.argv[1].split('/'))-1):
#         path += str(sys.argv[1].split('/')[k])+'/'
#
#     sample, type = [], []
#     for k in np.arange(1,len(sys.argv)):
#
#         sample.append(sys.argv[k].split('/')[-1].split('.')[0])
#
#         if (('ARM' not in sys.argv[k]) and ('IRM' not in sys.argv[k])) or 'NRM' in sys.argv[k]:
#             type.append('NRM')
#         elif 'ARM' in sys.argv[k]:
#             type.append('ARM')
#         elif 'IRM' in sys.argv[k]:
#             type.append('IRM')
#
#         fp = open(str(sys.argv[k]),'r')
#         name, Mx, My, Mz, step = [], [], [], [], []
#         if sys.argv[k][len(sys.argv[k])-3:] == 'txt':
#             for j, line in enumerate(fp):
#                 cols = line.split(',')
#                 if name == []:
#                     name.append(str(cols[0]))
#                 Mx.append(float(cols[5])*1e-8)
#                 My.append(float(cols[6])*1e-8)
#                 Mz.append(float(cols[7])*1e-8)
#                 step.append(int(cols[1])*0.1)
#             fp.close()
#
#         else:
#             for j, line in enumerate(fp):
#                 if j > 0:
#                     cols = line.split()
#                     if name == []:
#                         name.append(str(cols[0]))
#                     Mx.append(float(cols[1])*1e-3)
#                     My.append(float(cols[2])*1e-3)
#                     Mz.append(float(cols[3])*1e-3)
#                     step.append(int(cols[-1])*0.1)
#             fp.close()
#
#         if type[-1] == 'NRM':
#             NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep = np.array(Mx), np.array(My), np.array(Mz), np.array(step)
#         if type[-1] == 'ARM':
#             ARMatAFx, ARMatAFy, ARMatAFz, ARMatAFstep = np.array(Mx), np.array(My), np.array(Mz), np.array(step)
#         if type[-1] == 'IRM':
#             IRMatAFx, IRMatAFy, IRMatAFz, IRMatAFstep = np.array(Mx), np.array(My), np.array(Mz), np.array(step)
#
# ## Normalization
# unit = 'mom'
# normalize_by_mass = str(input('Normalize by mass? (y/N)'))
# if normalize_by_mass == 'y':
#     mass = True
#     unit = 'mag'
#     massNRM = float(eval(input('Mass of the sample for NRM (g)?'))) * 1e-3
#     if len(NRMatAFx) != 0:
#         NRMatAFx, NRMatAFy, NRMatAFz = NRMatAFx / massNRM, NRMatAFy / massNRM, NRMatAFz / massNRM
#     if len(ARMatAFx) != 0:
#         ARMatAFx, ARMatAFy, ARMatAFz = ARMatAFx / massNRM, ARMatAFy / massNRM, ARMatAFz / massNRM
#     if len(IRMatAFx) != 0:
#         massIRM = float(eval(input('Mass of the sample for IRM (g)?'))) * 1e-3
#         IRMatAFx, IRMatAFy, IRMatAFz = IRMatAFx / massIRM, IRMatAFy / massIRM, IRMatAFz / massIRM
# else:
#     mass = False
#     massNRM = massIRM = 1.
#
# ## Zijderveld
# zijd = str(input('Plot Zijderveld? (Y/n)'))
# if zijd != 'n':
#     if len(NRMatAFx) != 0:
#         Plot_Zijderveld(NRMatAFx,NRMatAFy,NRMatAFz,NRMatAFstep,unit=unit,title='NRM@AF',newfig='y',gui='gui',color='k')
#         Plot_Zijderveld(NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep,unit=unit,title='NRM@AF',newfig='y',color='k')
#         if save == 'y':
#             id = type.index('NRM')
#             plt.savefig(path + 'Plots/' + sample[id]+'-ZIJD.pdf', format='pdf', dpi=400, bbox_inches="tight")
#     if len(ARMatAFx) != 0:
#         lastpoint = input('Subtract last point of previous demag? (y/N)')
#         if lastpoint == 'y':
#             ARMatAFx, ARMatAFy, ARMatAFz = ARMatAFx-NRMatAFx[-1],ARMatAFy-NRMatAFy[-1],ARMatAFz-NRMatAFz[-1]
#         Plot_Zijderveld(ARMatAFx,ARMatAFy,ARMatAFz,ARMatAFstep,unit=unit,title='ARM@AF',newfig='y',color='k')
#     if len(IRMatAFx) != 0:
#         Plot_Zijderveld(IRMatAFx,IRMatAFy,IRMatAFz,IRMatAFstep,unit=unit,title='IRM@AF',newfig='y',color='k')
# plt.show()
#
# ## Equal area AF demag
# afdemag = str(input('Plot AF demag? (Y/n)'))
# if afdemag != 'n':
#     if len(NRMatAFx) != 0:
#         plot_equal_area_sequence(NRMatAFx,NRMatAFy,NRMatAFz,NRMatAFstep, newfig='y', title='NRM@AF', path='')
#         if save == 'y':
#             id = type.index('NRM')
#             plt.savefig(path + 'Plots/' + sample[id]+'-EQAREA.pdf', format='pdf', dpi=400, bbox_inches="tight")
#     if len(ARMatAFx) != 0:
#         plot_equal_area_sequence(ARMatAFx,ARMatAFy,ARMatAFz,ARMatAFstep,newfig='y', title='ARM@AF', path='')
#     if len(IRMatAFx) != 0:
#         plot_equal_area_sequence(IRMatAFx,IRMatAFy,IRMatAFz,IRMatAFstep,newfig='y', title='IRM@AF', path='')
# plt.show()
#
# ## Intensity demag
# intdemag = str(input('Plot intensity demag? (Y/n)'))
# if intdemag != 'n':
#     fig = plt.figure(figsize=(5,5))
#     if len(NRMatAFx) != 0:
#         Plot_AF_demag(NRMatAFx,NRMatAFy,NRMatAFz,NRMatAFstep, 'NRM', color='r', marker='o')
#     if len(ARMatAFx) != 0:
#         Plot_AF_demag(ARMatAFx,ARMatAFy,ARMatAFz,ARMatAFstep,'ARM', color='g', marker='s')
#     if len(IRMatAFx) != 0:
#         Plot_AF_demag(IRMatAFx,IRMatAFy,IRMatAFz,IRMatAFstep,'IRM', color='b', marker='^')
#     if save == 'y':
#         id = type.index('NRM')
#         plt.savefig(path + 'Plots/' + sample[id][0:-4] + '-AFINT.pdf', format='pdf', dpi=400, bbox_inches="tight")
#     fig.tight_layout()
#
# plt.show()
#
# ## Median destructive field
# if len(NRMatAFx) != 0:
#     NRMatAF = np.sqrt(NRMatAFx**2+NRMatAFy**2+NRMatAFz**2)
#     MDF_NRM = NRMatAFstep[Get_closest_id(NRMatAF,NRMatAF[0]/2)]
#     print("\nMedian destructive field NRM = "+str(MDF_NRM) + " mT")
# if len(ARMatAFx) != 0:
#     ARMatAF = np.sqrt(ARMatAFx ** 2 + ARMatAFy ** 2 + ARMatAFz ** 2)
#     MDF_ARM = ARMatAFstep[Get_closest_id(ARMatAF, ARMatAF[0] / 2)]
#     print("Median destructive field ARM = " + str(MDF_ARM) + " mT")
# if len(IRMatAFx) != 0:
#     IRMatAF = np.sqrt(IRMatAFx ** 2 + IRMatAFy ** 2 + IRMatAFz ** 2)
#     MDF_IRM = IRMatAFstep[Get_closest_id(IRMatAF, IRMatAF[0] / 2)]
#     print("Median destructive field IRM = " + str(MDF_IRM) + " mT")
#
# ## PCA analysis
# dopca = str(input('Run PCA analysis? (Y/n)'))
# if dopca != 'n':
#     MAD, DANG, vec, MAD95, Mcmax, id1, id2 = pca.Calc_MAD_and_DANG(NRMatAFx,NRMatAFy,NRMatAFz,NRMatAFstep,mass=mass)
#
#     ## REM
#     if len(NRMatAFx) != 0 and len(IRMatAFx) != 0:
#         print("REM IRM  " + f'{np.linalg.norm(Mcmax)/IRMatAF[id1]:.5f}')
#     if len(NRMatAFx) != 0 and len(ARMatAFx) != 0:
#         print("REM ARM  "+ f'{np.linalg.norm(Mcmax)/ARMatAF[id1]:.5f}')
#
# ## BPCA analysis
# #dobpca = str(input('Run BPCA analysis? (Y/n)'))
# #if dobpca != 'n':
# #    MD0_squared, QW_mean, Qmu_mean, Qcov = bpca.Calc_confidence_BPCA(NRMatAFx,NRMatAFy,NRMatAFz, 100)
# #Mxc, Myc, Mzc, id1, id2 = pca.Define_segment(NRMatAFx,NRMatAFy,NRMatAFz)
# #print(bpca2.BPCA_postprocess(np.array([Mxc, Myc, Mzc])))
#
#
#
# ## REM' and paleointensity estimates
# nrmlost = str(input('Run REM prime analysis? (Y/n)'))
# if nrmlost != 'n':
#
#     calib = False
#     tcrm = 'CRM'
#     mineral = 'magnetite'
#     domain = 'SD/PSD'
#     ARMbiasfield = 150
#
#     paleointensityARM = []
#     if len(NRMatAFx) != 0 and len(ARMatAFx) != 0:
#
#         NRMatAFx, NRMatAFy, NRMatAFz, ARMatAFx, ARMatAFy, ARMatAFz, AFARM = Merge_AF_demag(NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep, ARMatAFx, ARMatAFy, ARMatAFz, ARMatAFstep)
#
#         paleointensityARM = calc_REMslope_vector(NRMatAFx, NRMatAFy, NRMatAFz, ARMatAFx, ARMatAFy, ARMatAFz, AFARM, type='ARM', tcrm=tcrm, mineral=mineral, domain=domain, biasfield=ARMbiasfield, calib=calib, mass=mass)
#         if save == 'y':
#             id = type.index('NRM')
#             plt.savefig(path + 'Plots/' + sample[id] + '-VS-ARM-lost-vector.pdf', format='pdf', dpi=400, bbox_inches="tight")
#
#     paleointensityIRM = []
#     if len(NRMatAFx) != 0 and len(IRMatAFx) != 0:
#
#         NRMatAFx, NRMatAFy, NRMatAFz, IRMatAFx, IRMatAFy, IRMatAFz, AFIRM = Merge_AF_demag(NRMatAFx, NRMatAFy, NRMatAFz, NRMatAFstep, IRMatAFx, IRMatAFy, IRMatAFz, IRMatAFstep)
#
#         plot_REMprime(NRMatAFx, NRMatAFy, NRMatAFz, IRMatAFx, IRMatAFy, IRMatAFz, AFIRM, annot=True)
#         if save == 'y':
#             id = type.index('NRM')
#             plt.savefig(path + 'Plots/' + sample[id] + '-VS-IRM-REMprime.pdf', format='pdf', dpi=400, bbox_inches="tight")
#
#         paleointensityIRM = calc_REMslope_vector(NRMatAFx, NRMatAFy, NRMatAFz, IRMatAFx, IRMatAFy, IRMatAFz, AFIRM, type='IRM', tcrm=tcrm, mineral=mineral, domain=domain, calib=calib, mass=mass)
#         if save == 'y':
#             id = type.index('NRM')
#             plt.savefig(path + 'Plots/' + sample[id] + '-VS-IRM-lost-vector.pdf', format='pdf', dpi=400, bbox_inches="tight")
#
#         plt.figure(figsize=(5, 5))
#         plt.xlabel('Paleointensity (uT)')
#         plt.ylabel('Probability density')
#         if len(paleointensityARM) != 0:
#             plt.hist(paleointensityARM, bins=20, density=True, color='lightgreen', edgecolor='darkgreen',label='ARM method')
#         if len(paleointensityIRM) != 0:
#             plt.hist(paleointensityIRM, bins=20, density=True, color='lightblue', edgecolor='darkblue',label='IRM method')
#         plt.legend(loc=1)
#
#
#
# plt.show()
#
