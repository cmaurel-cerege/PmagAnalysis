import numpy as np

NRMAF = sorted([0, 0, 0, 3, 7, 10, 5,6,7,8,10])
test = np.arange(len(NRMAF))
MAF = [0, 3, 7,8,10]

AF = sorted(list(set(NRMAF) & set(MAF)))
test = [test[k] for k in np.arange(len(test)) if (NRMAF[k] in AF and NRMAF[k] not in NRMAF[0:k])]

print(AF)
print(NRMAF)
print(test)

#[NRMx[k] for k in np.arange(len(NRMx)) if (NRMAF[k] in AF and NRMAF[k] not in NRMAF[0:k])]