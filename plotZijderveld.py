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

def set_limits_one_extra_tick(ax, x, *y_arrays, ratio_threshold=20, force_equal_step=None):
    """
    x: array-like, shared x data
    *y_arrays: one or more y arrays plotted on the same axes
    ratio_threshold: if max(range)/min(range) exceeds this, fall back to
                      independent tick steps/aspect instead of forcing them equal
    force_equal_step: override auto-detection — True/False to force behavior manually
    """
    y_combined = np.concatenate([np.ravel(y) for y in y_arrays])
    x = np.ravel(x)

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y_combined), np.max(y_combined)

    range_x = x_max - x_min
    range_y = y_max - y_min
    ratio = max(range_x, range_y) / min(range_x, range_y)

    use_same_step = force_equal_step if force_equal_step is not None else (ratio <= ratio_threshold)

    if use_same_step:
        # --- Same graduation step + equal aspect ---
        overall_min = min(x_min, y_min)
        overall_max = max(x_max, y_max)
        locator = ax.xaxis.get_major_locator()
        ticks = locator.tick_values(overall_min, overall_max)
        step_x = step_y = ticks[1] - ticks[0]
    else:
        # --- Fallback: independent steps, no forced equal aspect ---
        locator_x = ax.xaxis.get_major_locator()
        ticks_x = locator_x.tick_values(x_min, x_max)
        step_x = ticks_x[1] - ticks_x[0]

        locator_y = ax.yaxis.get_major_locator()
        ticks_y = locator_y.tick_values(y_min, y_max)
        step_y = ticks_y[1] - ticks_y[0]

        print(f"[set_limits_one_extra_tick] Range ratio {ratio:.1f} exceeds "
              f"threshold {ratio_threshold} -> using independent tick steps "
              f"(x step={step_x}, y step={step_y}), equal aspect disabled.")

    for axis, data_min, data_max, step in (('x', x_min, x_max, step_x), ('y', y_min, y_max, step_y)):
        lower = np.floor(data_min / step) * step - step
        upper = np.ceil(data_max / step) * step + step

        if axis == 'x':
            ax.set_xlim(lower, upper)
            ax.xaxis.set_ticks(np.arange(lower, upper + step/2, step))
        else:
            ax.set_ylim(lower, upper)
            ax.yaxis.set_ticks(np.arange(lower, upper + step/2, step))

    if use_same_step:
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_aspect('auto')

# def set_limits_one_extra_tick(ax, x, *y_arrays):
#     """
#     x: array-like, the shared x data
#     *y_arrays: one or more y arrays (e.g. y1, y2, ...) plotted on the same axes
#     """
#     y_combined = np.concatenate([np.ravel(y) for y in y_arrays])
#
#     for axis, data in (('x', x), ('y', y_combined)):
#         data_min, data_max = np.min(data), np.max(data)
#         locator = ax.xaxis.get_major_locator() if axis == 'x' else ax.yaxis.get_major_locator()
#
#         ticks = locator.tick_values(data_min, data_max)
#         step = ticks[1] - ticks[0]
#
#         lower = np.floor(data_min / step) * step - step
#         upper = np.ceil(data_max / step) * step + step
#
#         if axis == 'x':
#             ax.set_xlim(lower, upper)
#             ax.xaxis.set_ticks(np.arange(lower, upper + step/2, step))
#         else:
#             ax.set_ylim(lower, upper)
#             ax.yaxis.set_ticks(np.arange(lower, upper + step/2, step))
#
#     ax.set_aspect('equal', adjustable='box')
#
def Set_frame_Zijderveld(ax, x1, x2, y1, y2, xlim, ylim):
    # set the x-spine and y-spine
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # turn off the box
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    ax.set_aspect('equal', adjustable='box')

    if xlim == () and ylim == ():
        set_limits_one_extra_tick(ax, x1, y1, y2)
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    return xlim, ylim


def Plot_Zijderveld(Mx, My, Mz, step, xlim=(), ylim=(), unit='', title='', color='k', gui='',annot='X'):

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
        ax.scatter(My, Mz, c='w', marker='o', ec=colors, lw=1, s=30, zorder=3, label='Z-Y')
        plt.colorbar(sm,ax=plt.gca(),pad=0.05,orientation='horizontal',location='bottom',shrink=0.4,aspect=15, label=label, ticks=[step[0],int(step[-1]/2),step[-1]])
    else:
        ax.scatter(My, Mx, marker='o', ec='k', c=colors, lw=0.5, s=40, zorder=3, label='X-Y')
        ax.scatter(My, Mz, marker='o', ec=colors, c='w', lw=0.5, s=40, zorder=3, label='Z-Y')

    if annot == 'X':
        for k in np.arange(0,len(step)-3,3):
            ax.annotate(str(int(step[k])), (My[k], Mx[k]), fontsize=7,textcoords="offset points", xytext=(4, 4), color="dimgray")
        ax.annotate(str(int(step[-1])), (My[-1], Mx[-1]), fontsize=7, textcoords="offset points", xytext=(4, 4),color="dimgray")
    elif annot == 'Y':
        for k in np.arange(0, len(step) - 3, 3):
            ax.annotate(str(int(step[k])), (My[k], Mz[k]), fontsize=7, textcoords="offset points", xytext=(4, 4), color="dimgray")
        ax.annotate(str(int(step[-1])), (My[-1], Mx[-1]), fontsize=7, textcoords="offset points", xytext=(4, 4),color="dimgray")

    #plt.text(0.95*xlim[1], 0.02*ylim[1], 'Y ' + r' ('+unit+')', horizontalalignment='left',fontsize=10)
    #plt.text(0.01*xlim[1], ylim[1]-0.05*ylim[1], 'X,Z' + r' ('+unit+')', fontsize=10)
    plt.title(title,loc='right')

    if gui == 'guiX':
        cursor = FollowDotCursorZijd(ax, My, Mz, My, step, Mx)
    if gui == 'guiZ':
        cursor = FollowDotCursorZijd(ax, My, Mx, My, step, Mz)

    return