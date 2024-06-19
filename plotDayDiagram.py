import numpy as np
import matplotlib.pyplot as plt

names = ['SIG', 'BUR', 'BAR03', 'BAR04', 'ROB01', 'FBL', 'CUKPY', 'CUPPY']
colors = ['darkred', 'darkgreen', 'darkblue', 'coral', 'gold', 'gray', 'k','violet']

## H350 - 5h - argon - 100 uT: SIG, BUR, BAR03, BAR04, ROB01, FBL, CUKPY, CUPPY
Ms = np.array([2.88E-03, 6.06E-03, 2.42E-03, 1.52E-03, 2.70E-03, 8.16E-02, 2.80E-03, 3.90E-02])
Mrs = np.array([3.22E-04, 7.29E-04, 4.74E-04, 5.14E-04, 5.48E-04, 7.07E-03, 3.20E-04, 7.95E-03])
Bc = np.array([5.4, 6.4, 12.9, 25.1, 10.1, 4.9, 8.0, 14.8])
Bcr = np.array([36.3, 36.4, 43, 57.6, 42.5, 20.3, 29.1, 43])

plt.figure(1)
ax = plt.subplot()
ax.set_xscale("log")
ax.set_yscale("log")
plt.xlim(1,20)
plt.ylim(0.01,1.)
plt.xlabel('Bcr/Bc',fontsize=13)
plt.ylabel('Mrs/Ms',fontsize=13)
for k in np.arange(len(Ms)):
    plt.plot(Bcr[k]/Bc[k], Mrs[k]/Ms[k], color=colors[k], marker='o', ms=6, lw=0, label=names[k])
plt.plot([2,2],[0,1],'k-',lw=0.5)
plt.plot([5,5],[0,1],'k-',lw=0.5)
plt.plot([0,20],[0.02,0.02],'k-',lw=0.5)
plt.plot([0,20],[0.5,0.5],'k-',lw=0.5)
plt.legend()


# ## H350 - 10h - air : SIG, BUR, BAR03, BAR04, ROB01, FBL, CUKPY, CUPPY
# Ms = np.array([2.98E-03, 2.41E-02, 1.34E-03, 2.46E-03, 4.47E-03, 3.97E-02, 4.31E-03, 5.96E-02])
# Mrs = np.array([1.42E-04, 1.21E-03, 2.12E-04, 5.83E-04, 6.64E-04, 1.22E-03, 7.29E-04, 1.10E-02])
# Bc = np.array([3.80, 3.4, 9.1, 16.0, 7.6, 1.4, 12.6, 10.7])
# Bcr = np.array([53.6, 27.7, 43.9, 45.2, 35.0, 10.0, 35.0, 27.0])
#
# plt.figure(1)
# for k in np.arange(len(Ms)):
#     plt.plot(Bcr[k]/Bc[k], Mrs[k]/Ms[k], color=colors[k], marker='d', ms=6, lw=0)




## H350 - 5h - argon - 50 uT : SIG, BUR, BAR03, BAR04, ROB01, FBL, CUKPY, CUPPY
names = ['SIG', 'BAR03', 'BAR04', 'ROB01', 'FBL', 'CUKPY', 'CUPPY']
colors = ['darkred', 'darkgreen', 'darkblue', 'coral', 'gold', 'gray', 'k','violet']
Ms = np.array([2.29E-03,2.14E-03,2.10E-03,1.94E-03,9.47E-02,2.22E-03,4.82E-02])
Mrs = np.array([2.49E-04,3.37E-04,6.51E-04,3.35E-04,7.39E-03,2.85E-04,9.70E-03])
Bc = np.array([5.9,3.4,26.4,7.2,4.8,7.8,12.6])
Bcr = np.array([36.9,38.3,64.2,35.9,21.2,28.3,35.4])

plt.figure(2)
ax = plt.subplot()
ax.set_xscale("log")
ax.set_yscale("log")
#plt.xlim(1,20)
#plt.ylim(0.01,1.)
plt.xlabel('Bcr/Bc',fontsize=13)
plt.ylabel('Mrs/Ms',fontsize=13)
for k in np.arange(len(Ms)):
    plt.plot(Bcr[k]/Bc[k], Mrs[k]/Ms[k], color=colors[k], marker='o', ms=6, lw=0, label=names[k])
plt.plot([2,2],[0,1],'k-',lw=0.5)
plt.plot([5,5],[0,1],'k-',lw=0.5)
plt.plot([0,20],[0.02,0.02],'k-',lw=0.5)
plt.plot([0,20],[0.5,0.5],'k-',lw=0.5)
plt.legend()


## Type 3 tetrataenite
names = ['Chainpur', 'Dhajala', 'Hallingeberg', 'Khohar', 'Krymka', 'Mezo-Madaras', 'Semarkona']
colors = ['darkred', 'darkgreen', 'darkblue', 'coral', 'gold', 'gray', 'violet']
Ms = np.array([8.56E+00, 3.76E+01, 1.70E+01, 1.22E+01, 2.21E+00, 1.59E+01, 3.73E+00])
Mrs = np.array([2.88E-01, 3.43E-01, 3.26E-01, 2.19E-01, 1.92E-01, 1.75E-01, 3.99E-01])
Bc = np.array([9.4, 3.2, 6.3, 3.7, 24.6, 2.5, 17.8])
Bcr = np.array([91.4, 120.8, 92.6, 26.8, 76.0, 36.6, 48.3])

fig = plt.figure(3)
ax = plt.subplot()
ax.set_xscale("log")
ax.set_yscale("log")
plt.xlim(1,60)
plt.ylim(0.005,0.6)
plt.xlabel('Bcr/Bc',fontsize=13)
plt.ylabel('Mrs/Ms',fontsize=13)
for k in np.arange(len(Ms)):
    plt.plot(Bcr[k]/Bc[k], Mrs[k]/Ms[k], color=colors[k], marker='o', ms=6, lw=0, label=names[k])
plt.plot([2,2],[0,1],'k-',lw=0.5)
plt.plot([5,5],[0,1],'k-',lw=0.5)
plt.plot([1,60],[0.02,0.02],'k-',lw=0.5)
plt.plot([1,60],[0.5,0.5],'k-',lw=0.5)
plt.legend(loc=1)
fig.tight_layout()

plt.show()