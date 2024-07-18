import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def Plot_AF_demag(Mx, My, Mz, AF, type, norm=True, color='lightgray', marker='o'):

    Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)
    M = np.sqrt(Mx**2 + My**2 + Mz**2)

    plt.xlabel('AF level (mT)')
    plt.xlim(0, max(AF)+2)
    if norm == False:
        plt.ylabel('Moment (A m2)')
        plt.ylim(0, 1.1*np.max(M))
        plt.plot(AF, M, marker=marker, color='k', mec='k', mfc=color, ms=6, mew=0.5, lw=0.5, label=type)
    elif norm == True:
        plt.ylabel('Normalized moment')
        plt.ylim(0,1.1)
        plt.plot(AF, M/M[0], marker=marker, color='k', mec='k', mfc=color, ms=6, mew=0.5, lw=0.5, label=type)

    plt.legend()

    return


def Plot_TH_demag(Mx, My, Mz, T, label='', norm=False, color='lightgray', marker='o'):

    Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)
    M = np.sqrt(Mx ** 2 + My ** 2 + Mz ** 2)

    plt.xlabel('Temperature (°C)')
    plt.xlim(0, np.max(T) + 10)
    if norm == False:
        plt.ylabel('Moment (A m2)')
        plt.ylim(0, 1.1 * np.max(M))
        plt.plot(T, M, color='k', marker=marker, mec='k', mfc=color, ms=6, lw=0.5, mew=0.5,label=label)
    elif norm == True:
        plt.ylabel('Normalized moment')
        plt.ylim(0, 1.1)
        plt.plot(T, M / M[0], color='k', marker=marker, mec='k', mfc=color, ms=6, lw=0.5, mew=0.5, label=label)
    plt.legend()

    return
