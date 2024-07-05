import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from followDotCursor import *

def Create_color_scale(variable,colormap):
    variable = np.array(variable)
    norm = (variable - variable.min()) / (variable.max() - variable.min())
    cmap = plt.get_cmap(colormap)
    colors = [cmap(tl) for tl in norm]
    return colors,colormap

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

    if xlim == ():
        lim = np.max([np.max(abs(x1)), np.max(abs(y1)), np.max(abs(y2))]) + 0.1*np.max(
            [np.max(abs(x1)), np.max(abs(y1)), np.max(abs(y2))])
        xlim=(-lim,lim)
    if ylim == ():
        lim = np.max([np.max(abs(x1)),np.max(abs(y1)), np.max(abs(y2))]) + 0.1*np.max([np.max(abs(x1)),np.max(abs(y1)), np.max(abs(y2))])
        ylim = (-lim, lim)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    return xlim, ylim


def Plot_Zijderveld(Mx, My, Mz, step, xlim=(), ylim=(), unit='', title='', color='k', gui=''):

    if color == 'AF':
        colors,colormap = Create_color_scale(np.arange(len(step)), 'Blues')
        label = 'AF step (mT)'
    elif color == 'TH':
        colors,colormap = Create_color_scale(np.arange(len(step)), 'Reds')
        label = 'Temperature step (°C)'
    else:
        colors = color
    ax = plt.subplot()
    xlim, ylim = Set_frame_Zijderveld(ax, My, My, Mx, Mz, xlim, ylim)
    ax.plot(My, Mx, 'k-', lw=0.5)
    ax.plot(My, Mz, 'k-', lw=0.5)
    if  color == 'AF' or color == 'TH':
        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_clim(vmin=step[0], vmax=step[-1])
        ax.scatter(My, Mx, c=colors, marker='o', ec='k', lw=0.25, s=40, zorder=3, label='X-Y')
        ax.scatter(My, Mz, c='w', marker='o', ec=colors, lw=2, s=30, zorder=3, label='Z-Y')
        plt.colorbar(sm,ax=plt.gca(),pad=0.05,orientation='horizontal',location='bottom',shrink=0.4,aspect=15, label=label, ticks=[step[0],int(step[-1]/2),step[-1]])
    else:
        ax.scatter(My, Mx, marker='o', ec='k', c=colors, lw=0.5, s=40, zorder=3, label='X-Y')
        ax.scatter(My, Mz, marker='o', ec=colors, c='w', lw=0.5, s=40, zorder=3, label='Z-Y')

    plt.text(0.95*xlim[1], 0.02*ylim[1], 'Y ' + r' ('+unit+')', horizontalalignment='left',fontsize=10)
    plt.text(0.01*xlim[1], ylim[1]-0.05*ylim[1], 'X,Z' + r' ('+unit+')', fontsize=10)
    plt.title(title,loc='right')

    if gui == 'guiX':
        cursor = FollowDotCursor(ax, My, Mz, My, step, Mx)
    if gui == 'guiZ':
        cursor = FollowDotCursor(ax, My, Mx, My, step, Mz)

    return