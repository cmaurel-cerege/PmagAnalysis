import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
from dir2cart import *
from cart2dir import *
from followDotCursor import *

def Create_color_scale(variable,colormap):
    variable = np.array(variable)
    norm = (variable - variable.min()) / (variable.max() - variable.min())
    cmap = plt.get_cmap(colormap)
    colors = [cmap(tl) for tl in norm]
    return colors,colormap


def equal_area_coord_from_cart(cart):

    norm = sqrt(cart[0]**2+cart[1]**2+cart[2]**2)
    x, y, z = cart[0]/norm, cart[1]/norm, cart[2]/norm
    if sqrt(x**2+y**2) == 0.0:
        xeq, yeq = 0.0, 0.0
    else:
        R = sqrt(1.0-absolute(z))/sqrt(x**2+y**2)
        xeq, yeq = x*R, y*R
    orient = sign(z)

    return array(yeq), array(xeq), -array(orient)

def plot_frame_equal_area(fig):

    ax = fig.gca()
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    plt.axis('equal')

    # Inclinations
    incls = [[0]*361, [15]*361, [30]*361, [45]*361, [60]*361, [75]*361, [90]*361]
    for j in arange(len(incls)):
        Xincl, Yincl = [], []
        for k in arange(0, 361):
            X, Y, orient = equal_area_coord_from_cart(list(dir2cart([float(k), incls[j][k], 1])))
            Xincl.append(X)
            Yincl.append(Y)
        if j == 0:
            plt.plot(Xincl, Yincl, color='gray', lw=0.8)
        else:
            plt.plot(Xincl, Yincl, color='lightgray', ls=':', lw=0.8)

    decls = arange(12) * 30
    for d in decls:
        X, Y, orient = equal_area_coord_from_cart(list(dir2cart([d, 0.0, 1])))
        plt.plot([0, X], [0, Y], color='lightgray', ls=':', lw=0.8)

    plt.annotate('0°', (-0.03, 1.03), fontsize=10, color='gray')
    plt.annotate('90°', (1.03, -0.03), fontsize=10, color='gray')
    plt.annotate('30°', (0.01, 0.72), fontsize=10, color='gray')
    plt.annotate('60°', (0.01, 0.38), fontsize=10, color='gray')

    return ax


def plot_equal_area_sequence(Mx, My, Mz, step, fig, title='', color='k', ms=40, lw=0.5):

    ax = plot_frame_equal_area(fig)
    plt.title(title,loc='right')

    x, y, orient = [], [], []
    for k in arange(len(Mx)):
        x.append(equal_area_coord_from_cart([Mx[k], My[k], Mz[k]])[0])
        y.append(equal_area_coord_from_cart([Mx[k], My[k], Mz[k]])[1])
        orient.append(equal_area_coord_from_cart([Mx[k], My[k], Mz[k]])[2])

    if color == 'AF':
        colors,colormap = Create_color_scale(np.arange(len(step)),'Blues')
        label = 'AF step (mT)'
    elif color == 'TH':
        colors,colormap = Create_color_scale(np.arange(len(step)), 'Reds')
        label = 'Temperature step (°C)'
    else: colors = [color]*len(Mx)
    plt.plot(x, y, 'k-', ms=0, lw=lw)
    x_up = np.array([x[j] for j in np.arange(len(x)) if orient[j] == 1])
    y_up = np.array([y[j] for j in np.arange(len(y)) if orient[j] == 1])
    x_down = np.array([x[j] for j in np.arange(len(x)) if orient[j] == -1])
    y_down = np.array([y[j] for j in np.arange(len(y)) if orient[j] == -1])

    if color == 'AF' or color == 'TH':
        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_clim(vmin=step[0], vmax=step[-1])
        plt.scatter(x_up, y_up, marker='o', s=ms, ec=colors, c='w', lw=1.5)
        plt.scatter(x_down, y_down, marker='o', s=ms, c=colors, ec='k', lw=0.5)
        plt.colorbar(sm,ax=plt.gca(),pad=0.05,orientation='horizontal',location='bottom',shrink=0.4,aspect=15, label=label, ticks=[step[0],int(step[-1]/2),step[-1]])
    else:
        plt.scatter(x_up, y_up, marker='o', s=ms, ec=color, c='w', lw=1.5)
        plt.scatter(x_down, y_down, marker='o', s=ms, c=color, ec='k', lw=0.5)

    fig.tight_layout()

    return

def calculate_alpha95(Mx, My, Mz):

    Mx, My, Mz = np.array(Mx), np.array(My), np.array(Mz)

    N = len(Mx)
    R = np.sqrt(np.sum(Mx)**2+np.sum(My)**2+np.sum(Mz)**2)

    Mxmean, Mymean, Mzmean = 1/R*np.sum(Mx), 1/R*np.sum(My), 1/R*np.sum(Mz)
    alpha95 = 180/np.pi*np.arccos(1-(N-R)/R*((1/0.05)**(1/(N-1))-1))

    return [Mxmean, Mymean, Mzmean], alpha95


def create_confidence_ellipse(Mmean, alpha, beta):

    alpha *= np.pi/180
    beta *= np.pi/180

    P1 = list(cart2dir(Mmean))
    P2 = [P1[0], P1[1] - sign(P1[1]) * 90.0]
    P3 = [P1[0] + 90.0, 0.0]

    X1, Y1, Z1 = dir2cart([P1[0],P1[1],1])
    X2, Y2, Z2 = dir2cart([P2[0],P2[1],1])
    X3, Y3, Z3 = dir2cart([P3[0],P3[1],1])

    # Ellipse on sphere at North Pole
    E_init = [asarray([sin(alpha)*cos(t), sin(beta)*sin(t), sqrt(1-(sin(alpha)*cos(t))**2-(sin(beta)*sin(t))**2)]) for t in np.arange(0, 361)*np.pi/180]

    M = array([[X2, X3, X1], [Y2, Y3, Y1], [Z2, Z3, Z1]])

    Ellipse = []
    for k in arange(361):
        Ellipse.append(dot(M, E_init[k]))

    return Ellipse


def plot_confidence_ellipse(Mx, My, Mz, fig, color):

    Mmean, alpha95 = calculate_alpha95(Mx, My, Mz)
    Ellipse = create_confidence_ellipse(Mmean,alpha95,alpha95)

    plot_equal_area_sequence(Mx, My, Mz, fig, color=color, ms=2)

    xe, ye, ze = [], [], []
    for k in arange(len(Ellipse)):
        xe.append(equal_area_coord_from_cart([Ellipse[k][0], Ellipse[k][1], Ellipse[k][2]])[0])
        ye.append(equal_area_coord_from_cart([Ellipse[k][0], Ellipse[k][1], Ellipse[k][2]])[1])
        ze.append(equal_area_coord_from_cart([Ellipse[k][0], Ellipse[k][1], Ellipse[k][2]])[2])

    below = array(ze) <= 0.0
    above = array(ze) > 0.0
    plt.plot(ma.masked_where(below, xe), ma.masked_where(below, ye), color=color, lw=1.5, ls='--')
    plt.plot(ma.masked_where(above, xe), ma.masked_where(above, ye), color=color, lw=1.5, ls='-')

    if sign(Mmean[2]) >= 0.0:
        plt.plot([equal_area_coord_from_cart(Mmean)[0]], [equal_area_coord_from_cart(Mmean)[1]], marker='*', ms=7, lw=0, mew=1.5, mfc='w', mec=color)
    else:
        plt.plot([equal_area_coord_from_cart(Mmean)[0]], [equal_area_coord_from_cart(Mmean)[1]], marker='*', ms=7, lw=0, mew=1.5, mfc=color, mec=color)
    plt.legend(loc=3, scatterpoints=1)

    return


def plot_confidence_ellipse_indiv(Mx, My, Mz, alpha, beta, fig, color, label=''):

    M = [Mx, My, Mz]
    Ellipse = create_confidence_ellipse(M,alpha,beta)
    plot_frame_equal_area(fig)

    xe, ye, ze = [], [], []
    for k in arange(len(Ellipse)):
        xe.append(equal_area_coord_from_cart([Ellipse[k][0], Ellipse[k][1], Ellipse[k][2]])[0])
        ye.append(equal_area_coord_from_cart([Ellipse[k][0], Ellipse[k][1], Ellipse[k][2]])[1])
        ze.append(equal_area_coord_from_cart([Ellipse[k][0], Ellipse[k][1], Ellipse[k][2]])[2])

    below = array(ze) <= 0.0
    above = array(ze) > 0.0
    plt.plot(ma.masked_where(below, xe), ma.masked_where(below, ye), color=color, ls='--')
    plt.plot(ma.masked_where(above, xe), ma.masked_where(above, ye), color=color, ls='-')

    if sign(M[2]) >= 0.0:
        plt.plot([equal_area_coord_from_cart(Mmean)[0]], [equal_area_coord_from_cart(Mmean)[1]], marker='*', ms=7, lw=0, mew=1.5, mfc='w', mec=color, label=label)
    else:
        plt.plot([equal_area_coord_from_cart(Mmean)[0]], [equal_area_coord_from_cart(Mmean)[1]], marker='*', ms=7, lw=0, mew=1.5, mfc=color, mec=color, label=label)
    plt.legend(loc=3, scatterpoints=1)

    return