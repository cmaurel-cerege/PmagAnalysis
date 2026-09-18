import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import special

def lognormCDF(x,mu,sigma,amp):
    return amp*0.5*(1+special.erf((np.log(x)-mu)/(sigma*np.sqrt(2))))

X = np.logspace(0,3.2,1000)

mu = 2.345#float(input("Mean of the lognorm distribution: "))
sigma = 0.21#float(input("Standard deviation of the lognorm distribution: "))
amp = 0.00000603

IRMpyr = []
for x in X:
    IRMpyr.append(lognormCDF(x,mu,sigma,amp))

mu = 1.972#float(input("Mean of the lognorm distribution: "))
sigma = 0.21#float(input("Standard deviation of the lognorm distribution: "))
amp = 0.00002649

IRMmag = []
for x in X:
    IRMmag.append(lognormCDF(x,mu,sigma,amp))

plt.xscale('log')
plt.plot(X,IRMpyr,'r')
plt.plot(X,IRMmag,'b')




plt.show()

