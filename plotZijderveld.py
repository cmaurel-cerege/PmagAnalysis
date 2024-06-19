import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from followDotCursor import *

def Set_frame_Zijderveld(ax, x1, x2, y1, y2, xlim, ylim):
    # set the x-spine and y-spine
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # turn off the box
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    plt.gca().set_aspect('equal', adjustable='box')

    # calculate the limits
    if xlim == [] or ylim == []:
        X = np.max([np.max(abs(x1)),np.max(abs(y1)), np.max(abs(y2))]) + 0.1*np.max([np.max(abs(x1)),np.max(abs(y1)), np.max(abs(y2))])
        Y = np.max([np.max(abs(x1)),np.max(abs(y1)), np.max(abs(y2))]) + 0.1*np.max([np.max(abs(x1)),np.max(abs(y1)), np.max(abs(y2))])
        xlim = [-X,X]
        ylim = [-Y,Y]

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    return xlim, ylim


def Plot_Zijderveld(Mx, My, Mz, AFlevel, xlim=[], ylim=[], unit='mom', newfig='y', title='', gui='', color='AF', path=''):

    if newfig == 'y':
        fig = plt.figure()
    if color == 'AF':
        colors = Create_color_scale(np.arange(len(AFlevel)), 'Blues')
        ax = plt.subplot()
        xlim, ylim = Set_frame_Zijderveld(ax, My, My, Mx, Mz, xlim, ylim)
        ## North - East
        ax.plot(My, Mx, 'k-', lw=0.5)
        ax.scatter(My, Mx, c=colors, marker='o', edgecolor='k', linewidth=0.25, s=50, zorder=3, label='X-Y')
        ## Up - East
        ax.plot(My, Mz, 'k-', lw=0.5)
        ax.scatter(My, Mz, c='w', marker='o', edgecolor=colors, linewidth=2, s=40, zorder=3, label='Z-Y')
    else:
        Mz = -Mz
        ax = plt.subplot()
        xlim, ylim = Set_frame_Zijderveld(ax, My, My, Mx, Mz, xlim, ylim)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.plot(My, Mx, 'k-', lw=0.5)
        ax.scatter(My, Mx, marker='o', edgecolor='k', c=color, linewidth=0.5, s=50, zorder=3, label='X-Y')
        ax.plot(My, Mz, 'k-', lw=0.5)
        ax.scatter(My, Mz, marker='o', edgecolor=color, c='w', linewidth=0.5, s=50, zorder=3, label='Z-Y')

    if unit == 'mag':
        plt.text(0.95*xlim[1], 0.02*ylim[1], 'Y ' + r' (A m$^{2}$ kg$^{-1}$)', horizontalalignment='left',fontsize=11)
        plt.text(0.01*xlim[1], ylim[1]-0.05*ylim[1], 'X,Z' + r' (A m$^{2}$ kg$^{-1}$)', fontsize=11)
    elif unit == 'mom':
        plt.text(0.95*xlim[1], 0.02*ylim[1], 'Y'+ r' (A m$^{2}$)', horizontalalignment='left',fontsize=11)
        plt.text(0.01*xlim[1], ylim[1]-0.05*ylim[1], 'X,Z' + r' (A m$^{2}$)', fontsize=11)

    if gui == 'gui':
        cursor = FollowDotCursor(ax, My, Mz, My, AFlevel, Mx)

    if title != '':
        plt.title(title)
    if newfig == 'y':
        fig.tight_layout()
    if gui == 'gui':
        cursor = FollowDotCursor(ax, My, Mz, My, AFlevel, Mx)

    if title != '':
        plt.title(title)
    if newfig == 'y':
        fig.tight_layout()