import numpy as np
import sys

fp = open(str(sys.argv[1]), 'r')

Mx, My, Mz, step = [], [], [], []
for j, line in enumerate(fp):
    if j > 0:
        cols = line.split()
        # Ajouter une vérification pour 'NA'
        if 'NA' not in cols[1:4]:
            Mx.append(float(cols[1]) * 1e-3)
            My.append(float(cols[2]) * 1e-3)
            Mz.append(float(cols[3]) * 1e-3)
            step.append(int(cols[-1]) * 0.1)
fp.close()

Mxa = [Mx[0]]
Mya = [My[0]]
Mza = [Mz[0]]
stepa = [0]

# for j in np.arange(1,len(Mx),3):
#     Mxa.append((Mx[j]+Mx[j+1]+Mx[j+2])/3)
#     Mya.append((My[j]+My[j+1]+My[j+2])/3)
#     Mza.append((Mz[j]+Mz[j+1]+Mz[j+2])/3)
#     stepa.append(step[j])

for j in np.arange(1, len(Mx), 1):
    Mxa.append(Mx[j])
    Mya.append(My[j])
    Mza.append(Mz[j])
    stepa.append(step[j])

# for j in np.arange(1,26,1):
#     Mxa.append(Mx[j])
#     Mya.append(My[j])
#     Mza.append(Mz[j])python
#     stepa.append(step[j])
#     print(step[j])
# for j in np.arange(26,len(Mx),3):
#     Mxa.append((Mx[j]+Mx[j+1]+Mx[j+2])/3)
#     Mya.append((My[j]+My[j+1]+My[j+2])/3)
#     Mza.append((Mz[j]+Mz[j+1]+Mz[j+2])/3)
#     stepa.append(step[j])

# stepa, Mxa,Mya,Mza=step,Mx,My,Mz

fpw = open('..' + str(sys.argv[1]).split('.')[2] + '.txt', 'w')
for j in np.arange(len(Mxa)):
    fpw.write(str(int(stepa[j])) + ' ' + str(Mxa[j]) + ' ' + str(Mya[j]) + ' ' + str(Mza[j]) + ' ' + '\n')
fpw.close()
