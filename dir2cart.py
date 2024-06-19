import numpy as np

def dir2cart(dir):
    # dir = [decl, incl, int] (angles in deg) ; return normalized cartesian coordinates

    decl, incl, intensity = dir[0], dir[1], dir[2]

    X = intensity*np.cos(decl*np.pi/180.0)*np.cos(incl*np.pi/180.0)
    Y = intensity*np.sin(decl*np.pi/180.0)*np.cos(incl*np.pi/180.0)
    Z = intensity*np.sin(incl*np.pi/180.0)

    return X, Y, Z