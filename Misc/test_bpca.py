import numpy as np
import pmagpy.pmag as pmag

## From Tauxe's book - sample eba24a
D = np.array([339.9,325.7,321.3,314.8,310.3,305.0,303.9,303.0,303.6,299.8,292.5,297.0,299.3])
I = np.array([57.9,49.1,45.9,41.7,38.7,37.0,34.7,32.3,32.4,30.8,31.0,25.6,11.3])
intensity = np.array([9.2830e-05,7.5820e-05,6.2920e-05,5.2090e-05,4.4550e-05,3.9540e-05,3.2570e-05,2.5670e-05,2.2520e-05,1.9820e-05,1.3890e-05,1.2570e-05,0.5030e-05])

x, y, z = [], [], []
for k in np.arange(len(D)):
    x.append(pmag.dir2cart([D[k],I[k],intensity[k]])[0])
    y.append(pmag.dir2cart([D[k],I[k],intensity[k]])[1])
    z.append(pmag.dir2cart([D[k],I[k],intensity[k]])[2])
Mx, My, Mz = x[6:14], y[6:14], z[6:14]

w_mean, w_cov, mu_mean, MD2 = BPCA.BPCA(np.array([Mx, My, Mz]))
print("MD2 = "+str(MD2))
print("W_mean = "+str(pmag.cart2dir(np.ravel(w_mean))))
