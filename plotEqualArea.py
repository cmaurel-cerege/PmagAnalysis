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
    return colors


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

    a = fig.gca()
    a.set_frame_on(False)
    a.set_xticks([])
    a.set_yticks([])
    plt.axis('off')
    plt.axis('equal')

    # Inclinations
    incls = [[0.0] * 361, [15.0] * 361, [30.0] * 361, [45.0] * 361, [60.0] * 361, [75.0] * 361, [90.0] * 361]
    for j in arange(len(incls)):
        Xincl = []
        Yincl = []
        for k in arange(0, 361):
            X, Y, orient = equal_area_coord_from_cart(list(dir2cart([float(k), incls[j][k], 1])))
            Xincl.append(X)
            Yincl.append(Y)
        if j == 0:
            plt.plot(Xincl, Yincl, color='k', lw=0.8, alpha=0.5)
        else:
            plt.plot(Xincl, Yincl, color='gray', ls=':', lw=0.8, alpha=0.5)

    ## Declinations
    decls = arange(12) * 30
    deg = "\u00b0"
    for d in decls:
        X, Y, orient = equal_area_coord_from_cart(list(dir2cart([d, 0.0, 1])))
        plt.plot([0, X], [0, Y], color='gray', ls=':', lw=0.8, alpha=0.5)

        plt.annotate('N', (-0.03, 1.02), fontsize=12, alpha=0.1, color='k')
        plt.annotate('E', (1.02, -0.03), fontsize=12, alpha=0.1, color='k')
        plt.annotate('S', (-0.03, -1.08), fontsize=12, alpha=0.1, color='k')
        plt.annotate('W', (-1.1, -0.03), fontsize=12, alpha=0.1, color='k')

        ## inclinations
        plt.annotate('30' + deg, (0.01, 0.72), fontsize=11, alpha=0.1, color='gray')
        plt.annotate('60' + deg, (0.01, 0.38), fontsize=11, alpha=0.1, color='gray')


def plot_equal_area_sequence(Mx, My, Mz, AFlevel, newfig='', title='', color='k', path=''):

    if newfig == 'y':
        fig = plt.figure(figsize=(5,5))
        plot_frame_equal_area(fig)
    else:
        fig = plt.figure(100,figsize=(5,5))
        plot_frame_equal_area(fig)

    x, y, orient = [], [], []
    for k in arange(len(Mx)):
        x.append(equal_area_coord_from_cart([Mx[k], My[k], Mz[k]])[0])
        y.append(equal_area_coord_from_cart([Mx[k], My[k], Mz[k]])[1])
        orient.append(equal_area_coord_from_cart([Mx[k], My[k], Mz[k]])[2])

    if color == 'AF':
        colors = Create_color_scale(np.arange(len(AFlevel)),'Blues')
        ax = plt.subplot()
        ax.plot(x, y, 'k-', ms=0, lw=0.5)
        for k in arange(len(x)):
            if orient[k] == 1:
                plt.scatter(x[k], y[k], color=colors[k], marker='o', edgecolor='k', linewidth=0.25, s=50, zorder=3)
            else:
                plt.scatter(x[k], y[k], color='w', marker='o', edgecolor=colors[k], linewidth=2, s=40, zorder=3,)

    else:
        plt.plot(x, y, 'k-', ms=0, lw=0.5)
        for k in arange(len(x)):
            if orient[k] == 1:
                plt.scatter(x[k], y[k], marker='s', s=30, edgecolor='k', linewidths=0.5, c='white', zorder=3)
            else:
                plt.scatter(x[k], y[k], marker='s', s=30, edgecolor='k', linewidths=0.5, c='k', zorder=3)
    if title != '':
        plt.title(title)
    if path != '':
        plt.savefig(path + '_stereo.png', format='png', dpi=200, bbox_inches="tight")


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


def plot_confidence_ellipse_equal_area(Mx, My, Mz, color, fig, path):

    Mmean, alpha95 = calculate_alpha95(Mx, My, Mz)

    Ellipse = create_confidence_ellipse(Mmean,alpha95,alpha95)

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

    if sign(Mmean[2]) >= 0.0:
        plt.scatter(equal_area_coord_from_cart(Mmean)[0], equal_area_coord_from_cart(Mmean)[1], s=200, lw=2.0, marker='*',facecolors='none', edgecolors=color)
    else:
        plt.scatter(equal_area_coord_from_cart(Mmean)[0], equal_area_coord_from_cart(Mmean)[1], s=200, lw=2.0, color=color,marker='*', edgecolors='none')
    plt.legend(loc=3, scatterpoints=1, fontsize=13)

    if path != '':
        plt.savefig(str(path), format='png', dpi=200, bbox_inches='tight')

def plot_indiv_confidence_ellipse(Mx, My, Mz, alpha, beta, color, fig, label=''):

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
        plt.scatter(equal_area_coord_from_cart(M)[0], equal_area_coord_from_cart(M)[1], s=200, lw=2.0, marker='*',facecolors='none', edgecolors=color, label=label)
    else:
        plt.scatter(equal_area_coord_from_cart(M)[0], equal_area_coord_from_cart(M)[1], s=200, lw=2.0, color=color,marker='*', edgecolors='none', label=label)

    plt.legend(loc=3, scatterpoints=1, fontsize=13)