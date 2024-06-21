import numpy as np
from numpy import transpose as tr
from numpy.linalg import inv
import pmagpy.pmag as pmag
import pandas as pd
import scipy.optimize as opt
from scipy.special import erf
import scipy as sp
from scipy.integrate import quad #import quadrature function
from scipy.integrate import nquad #import quadrature function

def BPCA(y):
    """
    Performs Bayesian PCA on Bishop 1999 (doi:10.1049/cp:19991160) refered to as B99
    Code is a modified version of the bppca package (https://github.com/cangermueller/ppca) 

    Parameters:
    ____________
        y : demagnetization array (X,Y,Z), one step per column

    Returns
    ________
    w_mean, w_cov, mu_mean
        w_mean = 3d mean of Gaussian distribution of principal vector
        w_cov = covariance of 3d Gaussian distribution of principal vector
        mu_mean = 3d mean of Gaussian distribution of vector mean
           
    """
    #initialization of variables: place in main call to function before running updates    
    q = 1 #find leading principal component
    p=y.shape[0] #number of data points
    n=y.shape[1] #number of data dimensions
    
    #KEY PARAMETERS
    #x = distribution of pca scores
    #w = distribution of the principal direction of the data
    #mu = distribution of the mean of the data
    
    #Initialize prior distributions for model parameters
    np.random.seed(0) #set seed for reproducibility 
    x_mean = np.random.normal(0.0, 1.0,q*n).reshape(q,n) #score distribution mean 
    x_cov = np.eye(q) #score distribution covariance
    w_mean = np.random.normal(0.0, 1.0,p*q).reshape(p,q) #direction direction mean
    w_cov = np.eye(q) #direction distribution covariance
    mu_mean = np.random.normal(0.0, 1.0,p) #mean distribution mean
    mu_cov = np.eye(p) #mean distribution covariance
    
    #define priors on alpha (hyperparameters) according to gamma distribution (B99:Eqn7)
    alpha_a = 1.0 
    alpha_b = np.ones(q) #typically q=1

    #gamma corresponds to tau in B99 (inverse of noise variance) according to gamma distribution (B99:Eqn7)
    gamma_a = 1.0
    gamma_b = 1.0
    
    #set up hyperparameters (tau is estimated from the data to aid convergence)
    V,D = np.linalg.eig(np.cov(y))
    tau=1/np.mean(V[1:3])
    H_alpha_a = 1.0
    H_alpha_b = 1.0
    H_gamma_a = tau
    H_gamma_b = 1.0
    H_beta = 1.0

    #Enter updates phases (converages quickly, so only 20 updates are performed)
    for i in range(20):
        mu_mean, mu_cov = BPCA_update_mu(y,q,w_mean,x_mean,gamma_a,gamma_b,H_beta) #update mu distribution
        w_mean, w_cov = BPCA_update_w(y,q,x_mean,x_cov,mu_mean,gamma_a,gamma_b,alpha_a,alpha_b) #update w distribution
        x_mean,x_cov = BPCA_update_x(y,q,mu_mean,w_mean,gamma_a,gamma_b) #update x distribution
        alpha_a,alpha_b = BPCA_update_alpha(y,q,w_mean,H_alpha_a,H_alpha_b) #update alpha distribution
        gamma_a, gamma_b = BPCA_update_gamma(y,q,mu_mean,w_mean,x_mean,H_gamma_a,H_gamma_b) #update gamma distribution
    
    #Test if origin is consistent with fit
    MD2 =  BPCA_test_origin(mu_mean,mu_cov,w_mean,w_cov)

    return w_mean, w_cov, mu_mean, MD2

def BPCA_xmin(x0,Rw,Rmu):
    """
    Find value of score (x), which brings fit to point closest to origin

    Parameters:
    ____________
        x0 : value of score to be tested
        Rw : random draw from w distribution
        Rmu : random draw from mu distribution

    Returns
    ________
    D
    D = Euclidean distance between point on fitted line and origin  
    """ 
    
    D = np.linalg.norm(Rw*x0+Rmu)

    return D

def BPCA_test_origin(mu_mean,mu_cov,w_mean,w_cov,niter=2000):
    """
    Monte Carlo routine to test if fit is consistent with the origin.
    MD2 < 7.81 indicates fit is consistent with the origin.

    Parameters:
    ____________
        mu_mean : 3d mean of Gaussian distribution of vector mean
        mu_cov : covariance of Gaussian distribution of vector mean
        w_mean : 3d mean of Gaussian distribution of principal vector
        w_cov : covariance of 3d Gaussian distribution of principal vector
        niter : number of MC iterations

    Returns
    ________
    MD2
    MD2 : squared Mahalanobis distance to the origin.  
    """     
    Z = np.empty((niter,3))
    #build Z estimates
    for i in range(niter):
        Rmu = np.random.multivariate_normal(np.squeeze(mu_mean),np.eye(3)*mu_cov)[:,np.newaxis]
        Rw = np.random.multivariate_normal(np.squeeze(w_mean),np.eye(3)*w_cov)[:,np.newaxis]
        
        result = opt.minimize(BPCA_xmin,0, args=(Rw,Rmu))
        Z[i,:] = np.squeeze(Rw*result.x+Rmu)

    MD2 = -np.mean(Z,axis=0) @ np.linalg.inv(np.cov(Z.T)) @ -np.mean(Z,axis=0).T
    
    return MD2

    

def BPCA_update_mu(y,q,w_mean,x_mean,gamma_a,gamma_b,H_beta):
    """
    mu distribution update according to Bishop 1999 (doi:10.1049/cp:19991160) refered to as B99
    Code is a modified version of the bppca package (https://github.com/cangermueller/ppca) 

    Parameters:
    ____________
        y : demagnetization array (X,Y,Z), one step per column
        q : dimension of solution (q = 1)
        w_mean : 3d mean of Gaussian distribution of principal vector
        x_mean : 3d mean of Gaussian distribution of vector scores
        gamma_a : first gamma distribution term on error distribution
        gamma_b : second gamma distribution term on error distribution
        H_beta : hyperparameter on mu_cov

    Returns
    ________
    mu_mean, mu_cov
        mu_mean = 3d mean of Gaussian distribution of vector mean
        mu_cov = covariance of 3d Gaussian distribution of vector mean  
    """    
    p=y.shape[0]
    n=y.shape[1]
    
    gamma_mean = gamma_a / gamma_b
    mu_cov = (H_beta + n * gamma_mean)**-1 * np.eye(p) #B99:Eqn14
    mu_mean = np.sum(y - w_mean.dot(x_mean), 1) #B99:Eqn13
    mu_mean = gamma_mean * mu_cov.dot(mu_mean) #B99:Eqn13
    return mu_mean, mu_cov

def BPCA_update_w(y,q,x_mean,x_cov,mu_mean,gamma_a,gamma_b,alpha_a,alpha_b):    
    """
    w distribution update according to Bishop 1999 (doi:10.1049/cp:19991160) refered to as B99
    Code is a modified version of the bppca package (https://github.com/cangermueller/ppca) 

    Parameters:
    ____________
        y : demagnetization array (X,Y,Z), one step per column
        q : dimension of solution (q = 1)
        x_mean : 3d mean of Gaussian distribution of vector scores
        x_cov : covariance of 3d Gaussian distribution of vector scores
        mu_mean : 3d mean of Gaussian distribution of vector mean
        gamma_a : first gamma distribution term on error distribution
        gamma_b : second gamma distribution term on error distribution
        alpha_a : first gamma distribution term on alpha hyperparameter
        alpha_b : second gamma distribution term on alpha hyperparameter

    Returns
    ________
    w_mean, w_cov
        w_mean = 3d mean of Gaussian distribution of principal vector
        w_cov = covariance of 3d Gaussian distribution of principal vector  
    """    
    
    
    x_cov = np.zeros((q,q))
    for n in range(y.shape[1]):
        xn = x_mean[:, n]
        x_cov += xn[:, np.newaxis].dot(np.array([xn])) #expectation of xxT
        gamma_mean = gamma_a / gamma_b
        w_cov = np.diag(alpha_a / alpha_b) + gamma_mean * x_cov #B99:Eqn16
        w_cov = inv(w_cov) #B99:Eqn16
        # mean
        yc = y - mu_mean[:, np.newaxis] #B99:Eqn16
        w_mean = gamma_mean * w_cov.dot(x_mean.dot(tr(yc))) #B99:Eqn16
        w_mean = tr(w_mean) #B99:Eqn16
    
    return w_mean, w_cov

def BPCA_update_x(y,q,mu_mean,w_mean,gamma_a,gamma_b):
    """
    x distribution update according to Bishop 1999 (doi:10.1049/cp:19991160) refered to as B99
    Code is a modified version of the bppca package (https://github.com/cangermueller/ppca) 

    Parameters:
    ____________
        y : demagnetization array (X,Y,Z), one step per column
        q : dimension of solution (q = 1)
        mu_mean : 3d mean of Gaussian distribution of vector mean        
        w_mean = 3d mean of Gaussian distribution of principal vector
        gamma_a : first gamma distribution term on error distribution
        gamma_b : second gamma distribution term on error distribution

    Returns
    ________
    x_mean, x_cov
        x_mean : 3d mean of Gaussian distribution of vector scores
        x_cov : covariance of 3d Gaussian distribution of vector scores 
    """ 
    gamma_mean = gamma_a / gamma_b
    x_cov = inv(np.eye(q) + gamma_mean * tr(w_mean).dot(w_mean)) #B99:Eqn12
    x_mean = gamma_mean * x_cov.dot(tr(w_mean)).dot(y - mu_mean[:, np.newaxis]) #B99:Eqn11
    return x_mean,x_cov 

def BPCA_update_alpha(y,q,w_mean,H_alpha_a,H_alpha_b):
    """
    alpha distribution update according to Bishop 1999 (doi:10.1049/cp:19991160) refered to as B99
    Code is a modified version of the bppca package (https://github.com/cangermueller/ppca) 

    Parameters:
    ____________
        y : demagnetization array (X,Y,Z), one step per column
        q : dimension of solution (q = 1)        
        w_mean = 3d mean of Gaussian distribution of principal vector
        H_alpha_a : first gamma distribution term on alpha hyperparameter
        H_alpha_b : second gamma distribution term on alpha hyperparameter

    Returns
    ________
    alpha_a, alpha_b
        alpha_a : first gamma distribution term on alpha hyperparameter
        alpha_b : second gamma distribution term on alpha hyperparameter
    """
    p=y.shape[0]
    alpha_a = H_alpha_a + 0.5 * p #B99:Eqn17
    alpha_b = H_alpha_b + 0.5 * np.linalg.norm(w_mean, axis=0)**2 #B99:Eqn18
    return alpha_a,alpha_b

def BPCA_update_gamma(y,q,mu_mean,w_mean,x_mean,H_gamma_a,H_gamma_b):
    """
    gamma distribution update according to Bishop 1999 (doi:10.1049/cp:19991160) refered to as B99
    Code is a modified version of the bppca package (https://github.com/cangermueller/ppca) 

    Parameters:
    ____________
        y : demagnetization array (X,Y,Z), one step per column
        q : dimension of solution (q = 1)        
        mu_mean : 3d mean of Gaussian distribution of vector mean
        w_mean : 3d mean of Gaussian distribution of principal vector
        x_mean : 3d mean of Gaussian distribution of vector scores
        H_gamma_a : first gamma distribution term on gamma hyperparameter
        H_gamma_b : second gamma distribution term on gamma hyperparameter

    Returns
    ________
    gamma_a, gamma_b
        gamma_a : first gamma distribution term on gamma
        gamma_b : second gamma distribution term on gamma
    """
    p=y.shape[0]
    n=y.shape[1]
    
    gamma_a = H_gamma_a + 0.5 * n * p #B99:Eqn19
    gamma_b = H_gamma_b #from here down is B99:Eqn20
    ww = tr(w_mean).dot(w_mean)
    for n in range(y.shape[1]):
        yn = y[:, n]
        xn = x_mean[:, n]
        gamma_b += yn.dot(yn) + mu_mean.dot(mu_mean)
        gamma_b += np.trace(ww.dot(xn[:, np.newaxis].dot([xn])))
        gamma_b += 2.0 * mu_mean.dot(w_mean).dot(xn[:, np.newaxis])
        gamma_b -= 2.0 * yn.dot(w_mean).dot(xn)
        gamma_b -= 2.0 * yn.dot(mu_mean)
    
    return gamma_a, gamma_b

def oaG_cdf(theta,zeta):
    """
    Off axis gaussian cumulative distribution based on Love & Constable 2003: Eqn E8 

    Parameters
    ----------
    theta : angle from axis [degrees]
    zeta : distribution mean length / standard deviation 

    Returns
    -------
    p
    p : cumulative probability

    """    
    #
    #zeta = F/sigma, where zeta**2 approximates Fisher's kappa (Love & Constable 2003: Eqn E3)
    theta = np.deg2rad(theta)
    term1 = 1+erf(zeta/np.sqrt(2))
    term2 = np.cos(theta)*np.exp(-0.5*zeta**2*np.sin(theta)**2)
    term3 = 1+erf(zeta*np.cos(theta)/np.sqrt(2))
    p = 0.5*(term1-term2*term3)
    return p

def BPCA_postprocess(y):
    """
    Gets average direction using Bayesian PCA method of Heslop and Roberts (2016) 

    Parameters
    ----------
    y : nest list of data: [[X,Y,Z],...]

    Returns
    -------
    I,D,b95,MD2
    I : inclination of principal component, W
    D : declination of principal component, W
    b95 : beta95 - circle of 95% confidence about D,I
    MD2 : squared Mahalanobis distance to the origin (MD2 < 7.81 indicates fit is consistent with the origin).
    """

    w_mean, w_cov, mu_mean, MD2 = BPCA(y) #fit single component, return directional distribution and mean
    
    #check direction of component and flip if required
    if pmag.angle(pmag.cart2dir(tr(w_mean)),pmag.cart2dir(tr(mu_mean)))>90.0:
        w_mean *= -1.0

    DI=pmag.cart2dir(tr(w_mean)) #convert mean direction to dec and inc
    D=DI[0][0] #extract dec
    I=DI[0][1] #extract inc
    zeta = np.linalg.norm(w_mean)/np.sqrt(w_cov) #calculate zeta
    mod=opt.minimize(lambda theta: (oaG_cdf(theta,zeta)-0.95)**2,1) #find 95th percentile in cdf
    b95=mod.x #record off axis angle as b95
    return I,D,b95,MD2
