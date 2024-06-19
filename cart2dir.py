import numpy as np

def cart2dir(cart):
    X, Y, Z = cart[0], cart[1], cart[2]
    R = np.sqrt(X**2+Y**2+Z**2)
    decl = (np.arctan2(Y,X)*180/np.pi) % 360

    if R == 0.0:
        incl = 90.0
    else:
        incl = np.arcsin(Z/R)*180/np.pi

    return decl, incl
