import numpy as np
import sys

path = ''
for k in np.arange(len(sys.argv[1].split('/')) - 1):
    path += str(sys.argv[1].split('/')[k]) + '/'
sample = sys.argv[1].split('/')[-1].split('-')[0].split('.')[0]

fp = open(sys.argv[1],'r',encoding="utf8", errors='ignore')
field, IRM = [], []
for j, line in enumerate(fp):
    cols = line.split(',')
    field.append(float(cols[0]))
    IRM.append(float(cols[1]))
fp.close()

x = 0.005
fField, fIRM = [field[0]],[IRM[0]]
f = 1
for k in np.arange(len(IRM)-1):
    if f > 1:
        f -= 1
        print('f='+str(f))
        continue
    print('k='+str(k))
    while (IRM[k+f]-IRM[k]) <= x*IRM[-1]:
        print('f=' + str(f))
        if k+f == len(IRM)-1:
            break
        else:
            f += 1
    fField.append(field[k+f])
    fIRM.append(IRM[k+f])

print("Number of data: "+str(len(fIRM)))

fp = open(path+sample+'_filter.txt','w')
for k in np.arange(len(fIRM)):
    fp.write(str(fField[k])+', '+str(fIRM[k])+'\n')
fp.close()

