"""ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes

This software has been developped by Carole Clastres under the supervision of
David Mary (Lagrange institute, University of Nice) and ported to python by
Laure Piqueras (CRAL).

The project is funded by the ERC MUSICOS (Roland Bacon, CRAL). Please contact
Carole for more info at carole.clastres@univ-lyon1.fr



Test version.
Origin.py must be run as script for the moment.
It is not installed as a mpdaf.package
"""

from astropy.table import Table, Column, join
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os.path
from scipy.io import loadmat
from scipy.ndimage import measurements, morphology
from scipy import signal, stats, special

#from ...obj import Cube, Image, Spectrum
#from ...sdetect import Source
from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.sdetect import Source

from numpy.fft import rfftn, irfftn
from scipy.signal.signaltools import _next_regular, _centered

import time
import sys

__version__ = 'ORIGIN_18122015'

def Compute_PSF(wave, Nz, Nfsf, beta, fwhm1, fwhm2, lambda1, lambda2,
                step_arcsec): 
    """Compute PSF with a Moffat function

    Parameters
    ----------
    wave    : mpdaf.obj.WaveCoord
              Spectral coordinates
    Nz      : int
              Number of spectral channels
    Nfsf    : int
              Spatial dimension of the FSF
    beta    : float
              Moffat Beta parameter
    fwhm1   : float
              fwhm en arcsec of the first point
    fwhm2   : float
              fwhm en arcsec of the second point
    lambda1 : float
              wavelength in angstrom of the first point
    lambda2 : float
              wavelength in angstrom of the second point

    Returns
    -------
    PSF_Moffat : array (Nz, Nfsf, Nfsf)
                 MUSE PSF 
    fwhm_pix   : array (Nz)
                 fwhm of the PSF in pixels
    fwhm_arcsec : array (Nz)
                  fwhm of the PSF in arcsec

    Date  : Dec, 11 2015 
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Compute_PSF'
    t0 = time.time()
    wavelengths = wave.coord(unit=u.angstrom)
    
    slope = (fwhm2 - fwhm1)/(lambda2 - lambda1)
    ordon_or = fwhm2 - slope*lambda2
    # fwhm curve in arcsec
    fwhm_arcsec = slope * wavelengths + ordon_or
    # conversion fwhm arcsec -> pixels
    fwhm_pix = fwhm_arcsec/step_arcsec

    # x vector
    x = np.arange(-np.floor(Nfsf/2),np.floor(Nfsf/2)+1)
    # y vector
    y = np.arange(-np.floor(Nfsf/2),np.floor(Nfsf/2)+1)
    # alpha coefficient in pixel
    alpha = fwhm_pix/(2*np.sqrt(2**(1/beta)-1))

    #figure,plot(wavelengths,alpha)
    PSF_Moffat = np.empty((Nz,Nfsf,Nfsf))
    
    for i in range(Nz):
        for dx in range(Nfsf):
            for dy in range(Nfsf):
                PSF_Moffat[i,dx,dy] = (1 + (np.sqrt(x[dx]**2 + y[dy]**2)\
                                                        /alpha[i])**2)**(-beta)
    # Normalization
    PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat, axis=(1,2))\
                                                    [:, np.newaxis, np.newaxis]
    print '    %0.1fs'%(time.time()-t0)
    return PSF_Moffat, fwhm_pix ,fwhm_arcsec


def Spatial_Segmentation(Nx, Ny, NbSubcube):
    """Function to compute the limits in pixels for each zone. 
    Each zone is computed from the left to the right and the top to the bottom 
    First pixel of the first zone has coordinates : (row,col) = (Nx,1).

    Parameters
    ----------
    Nx        : integer
                Number of columns
    Ny        : integer
                Number of rows
    NbSubcube : integer
                Number of subcubes for the spatial segmentation
                
    Returns
    -------
    intx, inty : integer, integer
                  limits in pixels of the columns/rows for each zone

    Date  : Dec,10 2015 
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Spatial_Segmentation'
    t0 = time.time()
    # Segmentation of the rows vector in Nbsubcube parts from the right to the
    # left
    inty = np.linspace(Ny, 0, NbSubcube + 1, dtype=np.int)
    # Segmentation of the columns vector in Nbsubcube parts from the left to
    # the right
    intx = np.linspace(0, Nx, NbSubcube + 1, dtype=np.int)
    print '    %0.1fs'%(time.time()-t0)
    return inty, intx
    
    
def Compute_PCA_SubCube(NbSubcube, cube_std, intx, inty, Edge_xmin, Edge_xmax,
                        Edge_ymin, Edge_ymax):
    """Function to compute the PCA on each zone of a data cube. 

    Parameters
    ----------
    NbSubcube : integer
                Number of subcubes for the spatial segmentation
    cube_std  : array
                Cube data weighted by the standard deviation
    intx      : integer
                limits in pixels of the columns for each zone
    inty      : integer
                limits in pixels of the rows for each zone
    Edge_xmin : int
                Minimum limits along the x-axis in pixel
                of the data taken to compute p-values
    Edge_xmax : int
                Maximum limits along the x-axis in pixel
                of the data taken to compute p-values
    Edge_ymin : int
                Minimum limits along the y-axis in pixel
                of the data taken to compute p-values
    Edge_ymax : int
                Maximum limits along the y-axis in pixel
                of the data taken to compute p-values

    Returns
    -------
    A       : dict
              Projection of the data on the eigenvectors basis
    V       : dict
              Eigenvectors basis
    eig_val : dict
              Eigenvalues computed for each spatio-spectral zone
    nx      : array
              Number of columns for each spatio-spectral zone
    ny      : array
              Number of rows for each spatio-spectral zone
    nz      : array
              Number of spectral channels for each spatio-spectral zone

    Date  : Dec,7 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Compute_PCA_SubCube'
    t0 = time.time()
    #Initialization
    nx = np.empty((NbSubcube, NbSubcube), dtype=np.int)
    ny = np.empty((NbSubcube, NbSubcube), dtype=np.int)
    nz = np.empty((NbSubcube, NbSubcube), dtype=np.int)
    eig_val = {}
    V = {}
    A = {}

    #Spatial segmentation
    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            # limits of each spatial zone
            x1 = intx[numx]
            x2 = intx[numx+1]
            y2 = inty[numy]
            y1 = inty[numy+1]
            # Data in this spatio-spectral zone
            cube_temp = cube_std[:, y1:y2, x1:x2]

            # Edges are excluded for PCA computing
            x1 = max(x1, Edge_xmin+1)
            x2 = min(x2, Edge_xmax)
            y1 = max(y1, Edge_ymin+1)
            y2 = min(y2, Edge_ymax)
            cube_temp_edge = cube_std[:, y1:y2, x1:x2]

            #Dimensions of each subcube of each spatio-spectral zone
            nx[numx,numy] = cube_temp.shape[2]
            ny[numx,numy] = cube_temp.shape[1]
            nz[numx,numy] = cube_temp.shape[0]

            # PCA on each subcube
            A_c, V_c,lambda_c = Compute_PCA_edge(cube_temp, cube_temp_edge)
            # eigenvalues for each spatio-spectral zone
            eig_val[(numx, numy)] = lambda_c
            # Eigenvectors basis for each spatio-spectral zone
            V[(numx, numy)]= V_c
            # Projection of the data on the eigenvectors basis
            # for each spatio-spectral zone
            A[(numx,numy)]  = A_c
         
    print '    %0.1fs'%(time.time()-t0)
    return A, V, eig_val, nx, ny, nz

def Compute_PCA_edge(cube, cube_edge):
    """Function to compute the PCA the spectra of a data cube by excluding
    the undesired spectra.
    
    Parameters
    ----------
    cube      : array
                cube data weighted by the standard deviation
    cube_edge : array
                cube data weighted by the standard deviation without the
                undesired spectra (ie those on the edges).

    Returns
    -------
    A       : array
              Projection of the data cube on the eigenvectors basis
    eig_vec : array
              Eigenvectors basis corrsponding to the eigenvalues
    eig_val : array
              Eigenvalues computed for each spatio-spectral zone

    Date  : Dec,3 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # data cube converted to dictionary of spectra
    cube_v = cube.reshape(cube.shape[0], cube.shape[1]*cube.shape[2])
    # data cube without undesired spectra converted to dictionary of spectra
    cube_ve = cube_edge.reshape(cube_edge.shape[0],
                                cube_edge.shape[1]*cube_edge.shape[2])
    # Spectral covariance of the desired spectra
    C = np.cov(cube_ve)
    # Eigenvalues (ascending order) and Eigenvectors basis
    eig_val, eig_vec = np.linalg.eigh(C)
    # Projection of the data cube on the eigenvectors basis
    A = eig_vec.T.dot(cube_v)
    return A, eig_vec, eig_val
    
def Compute_Number_Eigenvectors_Zone(NbSubcube, list_r0, eig_val, plot_lambda):
    """Function to compute the number of eigenvectors to keep for the
    projection for each zone by calling the function
    Compute_Number_Eigenvectors.

    Parameters
    ----------
    NbSubcube   : float
                  Number of subcube in the spatial segementation
    list_r0     : array
                  List of the determination coefficient for each zone
    eig_val     : dict 
                  eigenvalues of each spatio-spectral zone
    plot_lambda : bool
                  if True, plot and save the eigenvalues and the separation
                  point
    
    Returns
    -------
    nbkeep : array
             number of eigenvalues for each zone used to compute the projection

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Compute_Number_Eigenvectors_Zone'
    t0 = time.time()
    #Initialization
    nbkeep = np.empty((NbSubcube, NbSubcube), dtype=np.int)
    zone = 0
    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            
            # Eigenvalues for this zone
            lambdat = eig_val[(numx,numy)]
                   
            # Number of eigenvalues per zone
            nbkeep[numx, numy] = Compute_Number_Eigenvectors(lambdat,
                                                             list_r0[zone])
            zone = zone + 1

    # plot the ln of the eigenvalues and the separation point for each zone
    if plot_lambda == 1:
        plt.figure()
        zone = 0
        for numy in range(NbSubcube):
            for numx in range(NbSubcube):
                lambdat = eig_val[(numx,numy)]
                nbt = nbkeep[numx,numy]
                zone = zone + 1
                plt.subplot(NbSubcube, NbSubcube, zone)
                plt.semilogy(lambdat)
                plt.semilogy(nbt, lambdat[nbt], 'r+')
                plt.title('zone %d'%zone)
   
    print '    %0.1fs'%(time.time() - t0)
    return nbkeep
    

def Compute_Number_Eigenvectors(eig_val, r0):
    """Function to compute the number of eigenvectors to keep for the
    projection with a linear regression and its associated determination
    coefficient

    Parameters
    ----------
    eig_val : array
              eigenvalues of each zone
    r0      : float
              Determination coefficient value set by the user

    Returns
    -------
    nbkeep : float
             number of eigenvalues used to compute the projection

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # Initialization
    nl = eig_val.shape[0]
    coeffr = np.zeros(nl - 4)

    # Start with the 5 first eigenvalues for the linear regression
    for r in range(5, nl+1):
        Y = np.log(eig_val[:r] + 0j)
        #Y[np.isnan(Y)] = 0
        X = np.array([np.ones(r), eig_val[:r]])
        beta = np.linalg.lstsq(X.T,Y)[0]
        Yest = np.dot(X.T, beta)
        # Determination coefficient
        coeffr[r-5] = 1 - (np.sum((Y - Yest)**2)/np.sum((Y - np.mean(Y))**2))

    # Find the coefficient closer of r0
    rt = 4 + np.where(coeffr >= r0)[0]
    if rt.shape[0]==0:
        return 0
    else:
        return rt[-1]

def Compute_Proj_Eigenvector_Zone(nbkeep, NbSubcube, Nx, Ny, Nz, A, V,
                                  nx, ny, nz, inty, intx):
    """Function to compute the projection on the selected eigenvectors of the
    data cube in the original basis by calling the function
    Compute_Proj_Eigenvector.

    Parameters
    ----------
    nbkeep    : array
                number of eigenvalues for each zone used to compute the
                projection
    NbSubcube : int
                Number of subcube in the spatial segementation
    Nx        : int
                Size of the cube along the x-axis
    Ny        : int
                Size of the cube along the z-axis
    Nz        : int
                Size of the cube along the spectral axis
    A         : dict
                Projection of the data on the eigenvectors basis
    V         : dict
                Eigenvectors basis
    nx        : array
                Number of columns for each spatio-spectral zone
    ny        : array
                Number of rows for each spatio-spectral zone
    nz        : array
                Number of spectral channels for each spatio-spectral zone
    intx      : array
                limits in pixels of the columns for each zone
    inty      : array
                limits in pixels of the rows for each zone
    
    Returns
    -------
    cube_faint : array
                 Projection on the eigenvectors associated to the lower
                 eigenvalues of the data cube (reprensenting the faint signal)
    cube_cont  : array
                 Projection on the eigenvectors associated to the higher
                 eigenvalues of the data cube (representing the continuum)

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Compute_Proj_Eigenvector_Zone'
    t0 = time.time()
    # initialization
    cube_faint = np.zeros((Nz, Ny, Nx))
    cube_cont =  np.zeros((Nz, Ny, Nx))

    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            # limits of each spatial zone
            x1 = intx[numx]
            x2 = intx[numx+1]
            y2 = inty[numy]
            y1 = inty[numy+1]
            At = A[(numx,numy)]
            Vt = V[(numx,numy)]
            r = nbkeep[numx, numy]
            cube_proj_faint_v , cube_proj_cont_v = \
                                            Compute_Proj_Eigenvector(At, Vt, r)
            # Resize the subcube
            cube_faint[:, y1:y2, x1:x2] = \
                                     cube_proj_faint_v.reshape((nz[numx,numy],
                                                                ny[numx,numy],
                                                                nx[numx,numy]))
            cube_cont[:, y1:y2, x1:x2] = \
                                     cube_proj_cont_v.reshape((nz[numx,numy],
                                                               ny[numx,numy],
                                                               nx[numx,numy]))
    print '    %0.1fs'%(time.time() - t0)
    return cube_faint, cube_cont  
    
def Compute_Proj_Eigenvector(A, V, r):
    """Function to compute the projection of the data in the original basis
    keepping the desired number eigenvalues.

    Parameters
    ----------
    A : array
        Projection of the data on the eigenvectors basis
    V : array
        Eigenvectors basis
    r : float
        Number of eigenvalues to keep for the projection

    Returns
    -------
    cube_proj_low_v  : array
                       Projection on the eigenvectors associated to the lower
                       eigenvalues of the spectra. 
    cube_proj_high_v : array
                       Projection on the eigenvectors associated to the higher
                       eigenvalues of the spectra.

    Date  : Dec,7 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # initialization
    cube_proj_low_v = np.dot(V[:,:r+1], A[:r+1,:])
    cube_proj_high_v = np.dot(V[:,r+1:], A[r+1:,:])
    return cube_proj_low_v , cube_proj_high_v
    

def Correlation_GLR_test(cube, sigma, PSF_Moffat, Dico):
    """Function to compute the cube of GLR test values obtained with the given
    PSF and dictionary of spectral profile.

    Parameters
    ----------
    cube       : array
                 data cube on test
    sigma      : array
                 MUSE covariance
    PSF_Moffat : array
                 FSF for this data cube
    Dico       : array
                 Dictionary of spectral profiles to test 

    Returns
    -------
    correl  : array
              cube of T_GLR values
    profile : array
              Number of the profile associated to the T_GLR 

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Correlation_GLR_test'
    t0 = time.time()
    # data cube weighted by the MUSE covariance
    cube_var = cube / np.sqrt(sigma)
    # Inverse of the MUSE covariance
    inv_var = 1. / sigma

    # Dimensions of the data
    shape = cube_var.shape
    Nz = cube_var.shape[0]
    Ny = cube_var.shape[1]
    Nx = cube_var.shape[2]

    # Spatial convolution of the weighted data with the zero-mean FSF
    PSF_Moffat_m = PSF_Moffat \
                   - np.mean(PSF_Moffat, axis=(1,2))[:, np.newaxis, np.newaxis]
    cube_fsf = np.empty(shape)
    
    for i in range(Nz):
        cube_fsf[i,:,:] = signal.fftconvolve(cube_var[i,:,:],
                                  PSF_Moffat_m[i,:,:][::-1, ::-1], mode='same')                           
    del cube_var

    fsf_square = PSF_Moffat_m**2
    del PSF_Moffat_m
    # Spatial part of the norm of the 3D atom
    norm_fsf = np.empty(shape)
    for i in range(Nz):
        norm_fsf[i,:,:] = signal.fftconvolve(inv_var[i,:,:],
                                  fsf_square[i,:,:][::-1, ::-1], mode='same')
    del fsf_square, inv_var
    
    # First cube of correlation values
    # initialization with the first profile
    profile = np.zeros(shape, dtype=np.int)

    # First spectral profile
    k0 = 0
    d_j = Dico[:,k0]
    # zero-mean spectral profile
    d_j = d_j - np.mean(d_j)
    # Compute the square of the spectral profile
    profile_square = d_j**2
    
    ygrid, xgrid = np.mgrid[0:Ny,0:Nx]
    xgrid = xgrid.flatten()
    ygrid = ygrid.flatten()

    cube_profile = np.empty(shape)
    norm_profile = np.empty(shape)

    s1 = np.array(cube_fsf.shape[0])
    s2 = np.array(d_j.shape)
    
    shape = s1 + s2 - 1
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    fshape = [_next_regular(int(d)) for d in shape]
    
    d_j_fft =  rfftn(d_j, fshape)
    profile_square_fft = rfftn(profile_square, fshape)
    
    cube_fsf_fft = []
    norm_fsf_fft = []    
    for y in range(Ny):
        for x in range(Nx):
            cube_fsf_fft.append(rfftn(cube_fsf[:,y,x], fshape))
            norm_fsf_fft.append(rfftn(norm_fsf[:,y,x], fshape))
    
    i = 0
    for y in range(Ny):
        for x in range(Nx):
            # Spectral convolution of the weighted data cube spreaded
            # by the FSF and the spectral profile : correlation between the
            # data and the 3D atom
#            cube_profile[:,y,x] = signal.fftconvolve(cube_fsf[:,y,x], d_j,
#                                                  mode = 'same')
            ret = irfftn(cube_fsf_fft[i] * d_j_fft, fshape)[fslice].copy()
            cube_profile[:,y,x] = _centered(ret, s1)
            # Spectral convolution between the spatial part of the norm of the
            # 3D atom and the spectral profile : The norm of the 3D atom
#            norm_profile[:,y,x] = signal.fftconvolve(norm_fsf[:,y,x],
#                                                  profile_square,
#                                                  mode = 'same')
            ret = irfftn(norm_fsf_fft[i] *
                         profile_square_fft, fshape)[fslice].copy()
            norm_profile[:,y,x] = _centered(ret, s1)
            i = i+1
    
    # Set to the infinity the norm equal to 0
    norm_profile[norm_profile<=0] = np.inf
    # T_GLR values with constraint  : cube_profile>0
    GLR = np.zeros((Nz, Ny, Nx, 2))
    GLR[:,:,:,0] = cube_profile/np.sqrt(norm_profile)
    
    for k in range(1, Dico.shape[1]):
        # Second cube of correlation values
        d_j = Dico[:,k]
        d_j = d_j - np.mean(d_j)
        profile_square = d_j**2
        
        d_j_fft =  rfftn(d_j, fshape)
        profile_square_fft = rfftn(profile_square, fshape)
        
        i = 0
        for y in range(Ny):
            for x in range(Nx):
#                cube_profile[:,y,x] = signal.fftconvolve(cube_fsf[:,y,x], d_j,
#                                                  mode = 'same')
#                norm_profile[:,y,x] = signal.fftconvolve(norm_fsf[:,y,x],
#                                                  profile_square,
#                                                  mode = 'same')
                ret = irfftn(cube_fsf_fft[i] * d_j_fft, fshape)[fslice].copy()
                cube_profile[:,y,x] = _centered(ret, s1)
                ret = irfftn(norm_fsf_fft[i] *
                             profile_square_fft, fshape)[fslice].copy()
                norm_profile[:,y,x] = _centered(ret, s1)
                i = i+1
                                                  
        norm_profile[norm_profile<=0] = np.inf
        GLR[:,:,:,1] = cube_profile/np.sqrt(norm_profile)
        
        # maximum over the fourth dimension
        PROFILE_MAX = np.argmax(GLR, axis=3)
        correl = np.amax(GLR, axis=3)
        # Number of corresponding real profile
        profile[PROFILE_MAX == 1] = k
        # Set the first cube of correlation values correspond
        # to the maximum of the two previous ones
        GLR[:,:,:,0] = correl
        # Display the number of the spectral profile already done
        output = '\r%d/%d'%(k,Dico.shape[1]-1)
        sys.stdout.write("\r\x1b[K" + output.__str__())
        sys.stdout.flush()
        
    print '    %0.1fs'%(time.time()-t0)
    return correl, profile
    
def Compute_pval_correl_zone(correl, intx, inty, NbSubcube, Edge_xmin,
                             Edge_xmax, Edge_ymin, Edge_ymax, threshold):
    """Function to compute the p-values associated to the
    T_GLR values for each zone

    Parameters
    ----------
    correl    : array
                cube of T_GLR values (correlations)
    intx      : array
                limits in pixels of the columns for each zone
    inty      : array
                limits in pixels of the rows for each zone
    NbSubcube : int
                Number of subcube in the spatial segementation
    Edge_xmin : int
                Minimum limits along the x-axis in pixel
                of the data taken to compute p-values
    Edge_xmax : int
                Maximum limits along the x-axis in pixel
                of the data taken to compute p-values
    Edge_ymin : int
                Minimum limits along the y-axis in pixel
                of the data taken to compute p-values
    Edge_ymax : int
                Maximum limits along the y-axis in pixel
                of the data taken to compute p-values
    threshold : float
                The threshold applied to the p-values cube

    Returns
    -------
    cube_pval_correl : array
                       cube of thresholded p-values associated
                       to the T_GLR values

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Compute_pval_correl_zone'
    t0 = time.time()
    # initialization
    cube_pval_correl = np.ones(correl.shape)
    
    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            # limits of each spatial zone
            x1 = intx[numx]
            x2 = intx[numx+1]
            y2 = inty[numy]
            y1 = inty[numy+1]
            
            # Edges are excluded for computing parameters of the
            # distribution of the T_GLR (mean and std)
            x1 = max(x1, Edge_xmin+1)
            x2 = min(x2, Edge_xmax)
            y1 = max(y1, Edge_ymin+1)
            y2 = min(y2, Edge_ymax)
            correl_temp_edge = correl[:, y1:y2, x1:x2]
            
            # Cube of pvalues for each zone
            cube_pval_correl_temp = Compute_pval_correl(correl_temp_edge)
            cube_pval_correl[:, y1:y2, x1:x2] = cube_pval_correl_temp

    # Threshold the pvalues
    threshold_log = 10**(-threshold);
    cube_pval_correl = cube_pval_correl * (cube_pval_correl < threshold_log)
    # The pvalues equals to zero correspond to the values flag to zero because 
    # they are higher than the threshold so actually they have to be set to 1 
    cube_pval_correl[cube_pval_correl == 0] = 1
    print '    %0.1fs'%(time.time()-t0)
    return cube_pval_correl
    
    
def Compute_pval_correl(correl_temp_edge):
    """Function to compute distribution of the T_GLR values with 
    hypothesis : T_GLR are distributed according a normal distribution

    Parameters
    ----------
    correl_temp_edge : T_GLR values with edges excluded
    
    Returns
    -------
    cube_pval_correl : array
                       p-values asssociated to the T_GLR values 

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    moy_est = np.mean(correl_temp_edge)
    std_est = np.std(correl_temp_edge)
    # hypothesis : T_GLR are distributed according a normal distribution
    rv = stats.norm(loc=moy_est, scale=std_est)
    cube_pval_correl = 1 - rv.cdf(correl_temp_edge)

    # Set the pvalues equals to zero to an arbitrary very low value, but not
    # zero (eps in Matlab =~ 2.2204.10^(-16))
    cube_pval_correl[cube_pval_correl==0] = np.spacing(1)**6

    return cube_pval_correl

def Compute_pval_channel_Zone(cube_pval_correl, intx, inty, NbSubcube,
                              mean_est):
    """Function to compute the p-values associated to the number of
    thresholded p-values of the correlations per spectral channel for
    each zone by calling the function Compute_pval_channel

    Parameters
    ----------
    cube_pval_correl : array
                       cube of thresholded p-values associated
                       to the T_GLR values
    intx             : array
                       limits in pixels of the columns for each zone
    inty             : array
                       limits in pixels of the rows for each zone
    NbSubcube        : int
                       Number of subcube in the spatial segmentation
    mean_est         : float
                       Estimated mean of the distribution

    Returns
    -------
    cube_pval_channel : array
                        cube of p-values associated to the number of 
                        thresholded p-values of the correlations per spectral
                        channel for each zone

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Compute_pval_channel_Zone'
    t0 = time.time()
    # initialization
    cube_pval_channel = np.zeros(cube_pval_correl.shape)
    # The p-values higher than the thresholded are previously set to 1, here
    # we set them to 0 because we want to count the number of pvalues 
    # thresholded.
    cube_pval_correl_threshold = cube_pval_correl.copy()
    cube_pval_correl_threshold[cube_pval_correl == 1] = 0
    
    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            # limits of each spatial zone
            x1 = intx[numx]
            x2 = intx[numx+1]
            y2 = inty[numy]
            y1 = inty[numy+1]

            X = cube_pval_correl_threshold[:,y1:y2,x1:x2]

            # How many thresholded pvalues in each spectral channel
            n_lambda = np.sum(np.array(X!=0, dtype=np.int), axis=(1,2))
            # pvalues computed for each spectral channel
            pval_channel_temp = Compute_pval_channel(X, n_lambda, mean_est)
            # cube of p-values
            cube_pval_channel[:,y1:y2,x1:x2] = pval_channel_temp[:, np.newaxis,
                                                                 np.newaxis]
    print '    %0.1fs'%(time.time()-t0)
    return cube_pval_channel
            
def Compute_pval_channel(X, n_lambda, mean_est):
    """Function to compute the p-values associated to the
    number of thresholded p-values of the correlations per spectral channel

    Parameters
    ----------
    X        : array
               number of thresholded p-values associated to the T_GLR values
               per spectral channel
    n_lambda : int
               How many thresholded pvalues in each spectral channel
    mean_est : float
               Estimated mean of the distribution given by the FWHM of the FSF.

    Returns
    -------
    cube_pval_channel : array
                        cube of p-values for each spectral channel

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # initialization        
    N = np.sum(np.array(X!=0, dtype=np.int))
    # Estimation of p parameter with the mean of the distribution set by the
    # FSF size
    p_est = mean_est / N
    # Hypothesis : Binomial distribution for each channel
    pval_channel = special.bdtr(N-1, N, p_est) - \
                   special.bdtr(n_lambda, N, p_est)
    # Set the pvalues equals to zero to an arbitrary very low value, but not
    # zero (eps in Matlab ~= 2.2204.10^(-16))
    pval_channel[pval_channel <= 0] = np.spacing(1)**6
    return pval_channel
    
  
def Compute_pval_final(cube_pval_correl, cube_pval_channel, threshold):
    """Function to compute the final p-values which are the thresholded
    pvalues associated to the T_GLR values divided by twice the pvalues
    associated to the number of thresholded p-values of the correlations
    per spectral channel for each zone

    Parameters
    ----------
    cube_pval_correl  : array
                        cube of thresholded p-values associated
                        to the T_GLR values
    cube_pval_channel : array
                        cube of p-values
    threshold         : float
                        The threshold applied to the p-values cube

    Returns
    -------
    cube_pval_final : array
                      cube of final thresholded p-values 

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Compute_pval_final'
    t0 = time.time()
    # probability : Pr(line|not nuisance) = Pr(line)/Pr(not nuisance)
    probafinale = cube_pval_correl/cube_pval_channel
    # pvalue = probability/2
    cube_pval_final = probafinale/2
    # Set the nan to 1
    cube_pval_final[np.isnan(cube_pval_final)] = 1
    # Threshold the p-values
    threshold_log = 10**(-threshold)
    cube_pval_final = cube_pval_final*(cube_pval_final<threshold_log)
    # The pvalues equals to zero correspond to the values flag to zero because 
    # they are higher than the threshold so actually they have to be set to 1
    cube_pval_final[cube_pval_final == 0] = 1
    print '    %0.1fs'%(time.time()-t0)
    return cube_pval_final
    
def Compute_Connected_Voxel(cube_pval_final, threshold, neighboors):
    """Function to compute the groups of connected voxels with a
    flood-fill algorithm.

    Parameters
    ----------
    cube_pval_final : array
                      cube of final thresholded p-values 
    threshold       : float
                      The threshold applied to the p-values cube
    neighboors      : int
                      Number of connected components

    Returns
    -------
    labeled_cube : array
                   An integer array where each unique feature in 
                   cube_pval_final has a unique label.
    Ngp          : integer
                   Number of groups

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Compute_Connected_Voxel'
    t0 = time.time()
    threshold_log = 10**(-threshold)
    # The p-values higher than the thresholded are previously set to 1, here we
    # set them to 0 because we want to merge in group pvalues thresholded.
    cube_pval_final = cube_pval_final*(cube_pval_final<threshold_log)

    # connected components
    conn = (neighboors + 1)**(1/3.)
    s = morphology.generate_binary_structure(3, conn)
    labeled_cube, Ngp = measurements.label(cube_pval_final, structure=s)
    # Maximum number of voxels in one group
    print '    %0.1fs'%(time.time()-t0)
    return labeled_cube, Ngp

    
def Compute_Referent_Voxel(correl, profile, cube_pval_correl,
                           cube_pval_channel, cube_pval_final, Ngp,
                           labeled_cube):
    """Function to compute refrerent voxel of each group of connected voxels
    using the voxel with the higher T_GLR value.

    Parameters
    ----------
    correl            : array
                        cube of T_GLR values
    profile           : array
                        Number of the profile associated to the T_GLR 
    cube_pval_correl  : array
                        cube of thresholded p-values associated
                        to the T_GLR values
    cube_pval_channel : array
                        cube of p-values
    cube_pval_final   : array
                        cube of final thresholded p-values 
    Ngp               : int
                        Number of groups
    labeled_cube      : array
                        An integer array where each unique feature in 
                        cube_pval_final has a unique label.
                        
    Returns
    -------
    Cat_ref : astropy.Table
              Catalogue of the referent voxels corrdinates for each group
              Columns of Cat_ref : x y z T_GLR profile pvalC pvalS pvalF

    Date  : Dec,16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print ' Compute_Referent_Voxel'
    t0 = time.time()
    grp = measurements.find_objects(labeled_cube)
    argmax = [np.argmax(correl[grp[i]]) for i in range(Ngp)]
    correl_max = np.array([np.ravel(correl[grp[i]])[argmax[i]] for i in range(Ngp)])
    z, y, x = np.meshgrid(range(correl.shape[0]), range(correl.shape[1]),
                          range(correl.shape[2]), indexing='ij')
    zpixRef = np.array([np.ravel(z[grp[i]])[argmax[i]] for i in range(Ngp)])
    ypixRef = np.array([np.ravel(y[grp[i]])[argmax[i]] for i in range(Ngp)])
    xpixRef = np.array([np.ravel(x[grp[i]])[argmax[i]] for i in range(Ngp)])
    profile_max = profile[zpixRef, ypixRef, xpixRef]
    pvalC = cube_pval_correl[zpixRef, ypixRef, xpixRef]
    pvalS = cube_pval_channel[zpixRef, ypixRef, xpixRef]
    pvalF = cube_pval_final[zpixRef, ypixRef, xpixRef]
    # Catalogue of referent pixels
    Cat_ref = Table([xpixRef, ypixRef, zpixRef, correl_max,
                     profile_max, pvalC, pvalS, pvalF],
                    names=('x', 'y', 'z', 'T_GLR',
                    'profile', 'pvalC', 'pvalS', 'pvalF'))
    # Catalogue sorted along the Z axis
    Cat_ref.sort('z')
    print '    %0.1fs'%(time.time()-t0)
    return Cat_ref

def Narrow_Band_Test(Cat0, cube_raw, Dico, PSF_Moffat, nb_ranges,
                     plot_narrow, wcs):
    """Function to compute the 2 narrow band tests for each detected
    emission line

    Parameters
    ----------
    Cat0        : astropy.Table
                  Catalogue of parameters of detected emission lines:
                  Columns of the Catalogue Cat0 :
                  x y z T_GLR profile pvalC pvalS pvalF
    cube_raw    : array
                  Raw data cube
    Dico        : array
                  Dictionary of spectral profiles to test 
    PSF_Moffat  : array
                  FSF for this data cube
    nb_ranges   : integer
                  Number of skipped intervals for computing control cube
    plot_narrow : boolean
                  If True, plot the narrow bands images
    wcs         : mpdaf.obj.WCS
                  Spatial coordinates

    Returns
    -------
    Cat1 : astropy.Table
           Catalogue of parameters of detected emission lines:
           Columns of the Catalogue Cat1 :
           x y z T_GLR profile pvalC pvalS pvalF T1 T2

    Date : Dec,16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Narrow_Band_Test'
    t0 = time.time()
    # Initialization
    T1 = []
    T2 = []

    for i in range(len(Cat0)):
        # Coordinates of the voxel
        x0 = Cat0[i]['x']
        y0 = Cat0[i]['y']
        z0 = Cat0[i]['z']
        # spectral profile
        num_prof = Cat0[i]['profile']
        profil0 = Dico[:, num_prof]
        # length of the spectral profile
        profil1 = profil0[profil0>1e-13]
        long0 = profil1.shape[0]
        # half-length of the spectral profile
        longz = long0/2
        # spectral range
        intz1 = max(0, z0 - longz)
        intz2 = min(cube_raw.shape[0], z0 + longz + 1)
        # Subcube on test
        longxy = PSF_Moffat.shape[1]/2  
        inty1 = max(0, y0 - longxy)
        inty2 = min(cube_raw.shape[1], y0 + longxy + 1)
        intx1 = max(0, x0 - longxy)
        intx2 = min(cube_raw.shape[2], x0 + longxy + 1)
        cube_test = cube_raw[intz1:intz2, inty1:inty2, intx1:intx2]
        # Larger spatial ranges for the plots
        longxy0 = 20
        y01 = max(0, y0 - longxy0)
        y02 = min(cube_raw.shape[1], y0 + longxy0 + 1)
        x01 = max(0, x0 - longxy0)
        x02 = min(cube_raw.shape[2], x0 + longxy0 + 1)
        # Coordinates in this window
        y00 = y0 - y01
        x00 = x0 - x01
        # subcube for the plot   
        cube_test_plot = cube_raw[intz1:intz2, y01:y02, x01:x02]
  
        # controle cube
        if (z0 + longz + nb_ranges*long0) < cube_raw.shape[0]:
            intz1c = intz1 + nb_ranges*long0
            intz2c = intz2 + nb_ranges*long0
        else:
            intz1c = intz1 - nb_ranges*long0
            intz2c = intz2 - nb_ranges*long0
        cube_controle = cube_raw[intz1c:intz2c, inty1:inty2, intx1:intx2]
        cube_controle_plot = cube_raw[intz1c:intz2c, y01:y02, x01:x02]
    
        # (1/sqrt(2)) * difference of the 2 sububes
        diff_cube = (1./np.sqrt(2)) * (cube_test - cube_controle)
        diff_cube_plot = (1./np.sqrt(2))*(cube_test_plot - cube_controle_plot)

        # Test 1
        s1 = np.ones_like(cube_test)
        s1 = s1/np.sqrt(np.sum(s1**2))
        T1.append(np.inner(diff_cube.flatten(),s1.flatten()))
    
        # Test 2
        atom = np.zeros((long0, PSF_Moffat.shape[1], PSF_Moffat.shape[2]))
    
        # Construction of the 3D atom corresponding to the spectral profile
        for k in range(long0):
            z = k + z0 - longz
            if z>=0 and z<cube_raw.shape[0]:
                atom[k,:,:] = profil1[k] * PSF_Moffat[z,:,:]

        # Normalization     
        atom = atom/np.sqrt(np.sum(atom**2))
        # Edges
        # The minimal coordinates corresponding to the spatio-spectral range
        # of the data cube
        if (x0 - longxy) <= 0:
            x1 = np.abs(x0 - longxy)
        else:
            x1 = 0
        if (y0 - longxy) <= 0:
            y1 = np.abs(y0 - longxy)
        else:
            y1 = 0
        if (z0 - longz) <= 0:
            z1 = np.abs(z0 - longz)
        else:
            z1 = 0
    
        # Part of the atom corresponding to the spatio-spectral range of the
        # data cube
        s2 = atom[z1:z1+intz2-intz1, y1:y1+inty2-inty1, x1:x1+intx2-intx1]

        # Test 2
        T2.append(np.inner(diff_cube.flatten(),s2.flatten()))
 
        # Plot the narrow bands images
        if plot_narrow:  
            plt.figure()
            plt.plot(x00, y00, 'm+')
            ima_test_plot = Image(data=cube_test_plot.sum(axis=0),
                                  wcs=wcs[y01:y02, x01:x02])
            title = 'cube test - (%d,%d)\n'%(x0, y0) + \
                    'T1=%.3f T2=%.3f\n'%(T1[i], T2[i]) + \
                    'lambda=%d int=[%d,%d['%(z0, intz1, intz2)
            ima_test_plot.plot(colorbar='v', title=title)
            
            plt.figure()
            plt.plot(x00, y00, 'm+')
            ima_controle_plot = Image(data=cube_controle_plot.sum(axis=0),
                                  wcs=wcs[y01:y02, x01:x02])
            title = 'check - (%d,%d)\n'%(x0, y0) + \
                    'T1=%.3f T2=%.3f\n'%(T1[i], T2[i]) + \
                    'int=[%d,%d['%(intz1c, intz2c)                      
            ima_controle_plot.plot(colorbar='v', title=title)

            plt.figure()
            plt.plot(x00, y00, 'm+')
            ima_diff_plot = Image(data=diff_cube_plot.sum(axis=0),
                                  wcs=wcs[y01:y02, x01:x02])
            title = 'Difference narrow band - (%d,%d)\n'%(x0, y0) + \
                    'T1=%.3f T2=%.3f\n'%(T1[i], T2[i]) + \
                    'int=[%d,%d['%(intz1c, intz2c)                      
            ima_diff_plot.plot(colorbar='v', title=title)

    col_t1 = Column(name='T1', data=T1)
    col_t2 = Column(name='T2', data=T2)
    Cat1 = Cat0.copy()
    Cat1.add_columns([col_t1, col_t2])
    print '    %0.2fs'%(time.time()-t0)
    return Cat1

    
def Narrow_Band_Threshold(Cat1, thresh_T1, thresh_T2):
    """Function to compute the 2 narrow band tests for each detected
    emission line

    Parameters
    ----------
    Cat1      : astropy.Table
                Catalogue of parameters of detected emission lines:
    thresh_T1 : float
                Threshold for the test 1
    thresh_T2 : float
                Threshold for the test 2

    Returns
    -------
    Cat1_T1 : astropy.Table
              Catalogue of parameters of detected emission lines selected with
              the test 1
    Cat1_T2 : astropy.Table
              Catalogue of parameters of detected emission lines selected with
              the test 2

    Columns of the Catalogues :
        [col line; row line; spectral channel line; T_GLR line ;
        spectral profile ; pval T_GLR;  pval channel;  final pval ; T1 ; T2];

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Narrow_Band_Threshold'
    t0 = time.time()
    # Catalogue with the rows corresponding to the lines with test values
    # greater than the given threshold
    Cat1_T1 = Cat1[Cat1['T1'] > thresh_T1]
    Cat1_T2 = Cat1[Cat1['T2'] > thresh_T2]
    print '    %0.1fs'%(time.time()-t0)
    return Cat1_T1, Cat1_T2
    
def Estimation_Line(Cat1_T, profile, Nx, Ny, Nz, sigma, cube_faint,
                    grid_dxy, grid_dz, PSF_Moffat, Dico):
    """Function to compute the estimated emission line and the optimal
    coordinates for each detected lines in a spatio-spectral grid. 

    Parameters
    ----------
    Cat1_T     : astropy.Table
                 Catalogue of parameters of detected emission lines selected
                 with a narrow band test.
                 Columns of the Catalogue Cat1_T:
                 x y z T_GLR profile pvalC pvalS pvalF T1 T2
    profile    : array
                 Number of the profile associated to the T_GLR 
    Nx         : int
                 Size of the cube along the x-axis
    Ny         : int
                 Size of the cube along the z-axis
    Nz         : int
                 Size of the cube along the spectral axis
    sigma      : array
                 MUSE covariance
    cube_faint : array
                 Projection on the eigenvectors associated to the lower
                 eigenvalues
    grid_dxy   : integer
                 Maximum spatial shift for the grid
    grid_dz    : integer
                 Maximum spectral shift for the grid
    PSF_Moffat : array
                 FSF for this data cube
    Dico       : array
                 Dictionary of spectral profiles to test 

    Returns
    -------
    Cat2             : astropy.Table
                       Catalogue of parameters of detected emission lines.
                       Columns of the Catalogue Cat2:
                       x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual
                       flux num_line   
    Cat_est_line_raw : list of arrays
                       Estimated lines in data space
    Cat_est_line_std : list of arrays
                       Estimated lines in SNR space

    Date  : Dec, 16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Estimation_Line'
    t0 = time.time()
    # Initialization
    Cat2_x = []
    Cat2_y = []
    Cat2_z = []
    Cat2_res_min = []
    Cat2_flux = []
    Cat_est_line_raw = []
    Cat_est_line_std = []
    longxy = PSF_Moffat.shape[1]/2
    # Loop on emission lines detected
    it = 0
    nit = len(Cat1_T)
    for x0, y0, z0 in zip(Cat1_T['x'], Cat1_T['y'], Cat1_T['z']):
        it = it + 1
        output = '\r%d/%d'%(it, nit)
        sys.stdout.write("\r\x1b[K" + output.__str__())
        sys.stdout.flush()
        # x0, y0, z0: Coordinates of the voxel
        # Spatio-spectral grid
        grid_x1 = max(0, x0 - grid_dxy)
        grid_x2 = min(Nx, x0 + grid_dxy + 1)
        grid_y1 = max(0, y0 - grid_dxy)
        grid_y2 = min(Ny, y0 + grid_dxy + 1)
        grid_z1 = max(0, z0 - grid_dz)
        grid_z2 = min(Nz, z0 + grid_dz + 1)
        
        # initialization
        ngrid = (grid_x2 - grid_x1) *  (grid_y2 - grid_y1) * (grid_z2 - grid_z1)
        line_est_raw = np.zeros((ngrid, Nz))
        line_est_std = np.zeros((ngrid, Nz))
        residual = np.zeros(ngrid)
        x_f = np.zeros(ngrid)
        y_f = np.zeros(ngrid)
        z_f = np.zeros(ngrid)
        flux = np.zeros(ngrid)
        
        # Estimation of a line on each voxel of the grid
        n = 0
        for z0t in range(grid_z1, grid_z2):
            for y0t in range(grid_y1, grid_y2):
                for x0t in range(grid_x1, grid_x2):
                    x_f[n] = x0t
                    y_f[n] = y0t
                    z_f[n] = z0t
                    f, res, lraw, lstd = Compute_Estim_Grid(x0t, y0t, z0t,
                                                            grid_dxy,
                                                            profile, Nx, Ny, Nz,
                                                            sigma,
                                                            cube_faint,
                                                            PSF_Moffat, longxy,
                                                            Dico)
                    flux[n] = f
                    residual[n] = res
                    line_est_raw[n,:] = lraw
                    line_est_std[n,:] = lstd
                    n = n+1

        # Take the estimated line with the minimum absolute value of the residual
        ind_n = np.argmin(np.abs(residual))
        Cat2_x.append(x_f[ind_n])
        Cat2_y.append(y_f[ind_n])
        Cat2_z.append(z_f[ind_n])
        Cat2_res_min.append(np.abs(residual[ind_n]))
        Cat2_flux.append(flux[ind_n])
        Cat_est_line_raw.append(line_est_raw[ind_n,:])
        Cat_est_line_std.append(line_est_std[ind_n,:])
    
    Cat2 = Cat1_T.copy()
    Cat2['x'] = Cat2_x
    Cat2['y'] = Cat2_y
    Cat2['z'] = Cat2_z
    col_res = Column(name='residual', data=Cat2_res_min)
    col_flux = Column(name='flux', data=Cat2_flux)
    col_num = Column(name='num_line', data=np.arange(len(Cat2)))
    Cat2.add_columns([col_res, col_flux, col_num])
    
    print '    %0.1fs'%(time.time()-t0)
    return Cat2, Cat_est_line_raw, Cat_est_line_std


def Compute_Estim_Grid(x0, y0, z0, grid_dxy, profile, Nx, Ny, Nz,
                       sigma, cube_faint, PSF_Moffat,longxy, Dico):
    """Function to compute the estimated emission line for each coordinate
    with the deconvolution model :
    subcube = FSF*line -> line_est = subcube*fsf/(fsf^2)

    Parameters
    ----------
    x0 : integer
         Column of the voxel to compute the estimated line
    y0 : integer
         Row of the voxel to compute the estimated line
    z0 : integer
         Spectral channel of the voxel to compute the estimated line
    grid_dxy : integer
               Maximum spatial shift for the grid
    profile  : array
               Number of the profile associated to the T_GLR 
    Nx         : int
                 Size of the cube along the x-axis
    Ny         : int
                 Size of the cube along the z-axis
    Nz         : int
                 Size of the cube along the spectral axis
    sigma      : array
                 MUSE covariance
    cube_faint : array
                 Projection on the eigenvectors associated to the lower
                 eigenvalues
    PSF_Moffat : array
                 FSF for this data cube
    longxy     : float
                 mid-size of the PSF
    Dico       : array
                 Dictionary of spectral profiles to test 

    Returns
    -------
    res          : float
                   Residual for this line estimation
    flux         : float
                   Flux of the estimated line in the data space
    line_est_raw : array
                   Estimated line in the data space
    line_est     : array
                   Estimated line in the SNR space
    

    Date  : Dec, 11 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # Covariance for the pixel under test
    sigmat = sigma[:,y0,x0]
    # spectral profile
    num_prof = profile[z0, y0, x0]
    profil0 = Dico[:, num_prof]
    profil1 = profil0[profil0>1e-20]
    long0 = profil1.shape[0]
    longz = long0/2
    
    # size of the 3D atom
    inty1 = max(0, y0 - longxy)
    inty2 = min(Ny, y0 + longxy + 1)
    intx1 = max(0, x0 - longxy)
    intx2 = min(Nx, x0 + longxy + 1)
    intz1 = max(0, z0 - longz)
    intz2 = min(Nz, z0 + longz + 1)
       
    # part of the cube at the same location
    cube_faint_t = cube_faint[intz1:intz2, inty1:inty2, intx1:intx2]
    # corresponding covariance
    sigma_t = sigma[intz1:intz2, inty1:inty2, intx1:intx2]

    # Initialization
    line_est = np.zeros(sigma.shape[0])
    # Pad subcube in case of the 3D atom is out of the cube data
    cube_faint_pad = np.zeros((cube_faint.shape[0],
                               cube_faint.shape[1] + 2*grid_dxy + 2*longxy,
                               cube_faint.shape[2] + 2*grid_dxy + 2*longxy))
    cube_faint_pad[0: cube_faint.shape[0],
                   grid_dxy+longxy: cube_faint.shape[1] + grid_dxy + longxy,
                   grid_dxy+longxy: cube_faint.shape[2] + grid_dxy + longxy] \
                   = cube_faint

    # Deconvolution 
    for k in range(intz1, intz2):       
        fsf = PSF_Moffat[k,:,:].flatten()
        subcube = cube_faint_pad[k,
                                 y0 + grid_dxy: y0 + 2*longxy + grid_dxy + 1,
                                 x0 + grid_dxy: x0 + 2*longxy + grid_dxy + 1]
        line_est[k] = np.inner(fsf, subcube.flatten()) \
                      / np.inner(fsf, fsf)

    # Estimated line in data space  
    line_est_raw = line_est * np.sqrt(sigmat)

    # Atome 3D corresponding to the estimated line
    atom_est = np.zeros((long0, PSF_Moffat.shape[1], PSF_Moffat.shape[2]))
    
    for k in range(long0):
            z = k + z0 - longz
            if z>=0 and z<cube_raw.shape[0]:
                atom_est[k,:,:] = line_est_raw[z] * PSF_Moffat[z,:,:]
                
    x1 = np.abs(min(0, x0 - longxy))
    y1 = np.abs(min(0, y0 - longxy))
    z1 = np.abs(min(0, z0 - longz))

    # Atom cut at the edges of the cube 
    atom_est_cut = atom_est[z1:z1+intz2-intz1, y1:y1+inty2-inty1,
                            x1:x1+intx2-intx1]
    # Estimated 3D atom in SNR space      
    atom_est_std = atom_est_cut / np.sqrt(sigma_t)
    # Norm of the 3D atom
    norm2_atom1 = np.inner(atom_est_std.flatten(), atom_est_std.flatten())
    # Estimated amplitude of the 3D atom
    # = (cube_faint_t(:)'*atom_est_std(:))/norm2_atom1;
    alpha_est = np.inner(cube_faint_t.flatten(), atom_est_std.flatten()) / norm2_atom1
    # Estimated detected emitters
    atom_alpha_est = alpha_est * atom_est_std
    # Residual of the estimation
    res = np.sum((cube_faint_t - atom_alpha_est)**2)
    # Flux
    flux = np.sum(line_est_raw)

    return flux, res, line_est_raw, line_est

def Spatial_Merging_Circle(Cat0, fwhm_fsf, Nx, Ny):
    """Construct a catalogue of sources by spatial merging of the detected
    emission lines in a circle with a diameter equal to the mean over the
    wavelengths of the FWHM of the FSF

    Parameters
    ----------
    Cat0     : astropy.Table
               catalogue
               Columns of Cat0:
               x y z T_GLR profile pvalC pvalS pvalF T1 T2
               residual flux num_line
    fwhm_fsf : float
               The mean over the wavelengths of the FWHM of the FSF
    Nx       : integer
               Size of the cube along the x-axis
    Ny       : integer
               Size of the cube along the y-axis
    
    Returns
    -------
    CatF : astropy.Table
           Columns of CatF:
           ID x_circle y_circle x_centroid y_centroid nb_lines 
           x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual flux num_line

    Date : Dec,16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Spatial_Merging_Circle'
    t0 = time.time()
    E = Cat0.copy()
    CatF = Table()
    num_source = 0
    # Add indices of lines
    col_id = Column(name='ID', data=np.arange(1,len(E)+1))
    E.add_column(col_id, index=0)

    while len(E) > 0:
        # Set the new indices
        E['ID'] = np.arange(1,len(E)+1)
        
        Cix, Ciy = np.mgrid[0:Nx,0:Ny]
        Cix = Cix.flatten()
        Ciy = Ciy.flatten()
        d = (E['x'][:,np.newaxis] - Cix[np.newaxis,:])**2 + \
            (E['y'][:,np.newaxis] - Ciy[np.newaxis,:])**2
        numi = np.where(d <= np.round(fwhm_fsf/2)**2)
        if len(numi[-1]) != 0:
            unique, count = np.unique(numi[-1], return_counts=True)
            ksel = np.where(count==max(count))
            pix = unique[ksel][0]
            C0 = [Cix[pix], Ciy[pix]]
            d = (E['x']- C0[0])**2 + (E['y'] - C0[1])**2
            num0 = np.where(d <= np.round(fwhm_fsf/2)**2)
            E0 = E[num0]
            
            if len(ksel[0]) > 1:
                # T_GLR values of the voxels in this group
                correl_temp  = E0['T_GLR']
                # Spatial positions of the voxels
                x_gp = E0['x']
                y_gp = E0['y']
                # Centroid weighted by the T_GLR of voxels in each group
                x_centroid = np.sum(correl_temp*x_gp) / np.sum(correl_temp)
                y_centroid = np.sum(correl_temp*y_gp) / np.sum(correl_temp)
                # Distances of the centroids to the center of the circle
                pix = unique[ksel]
                d_i = (x_centroid - Cix[pix])**2 + (y_centroid - Ciy[pix])**2
                # Keep the lower distance
                ksel2 = np.argmin(d_i)
                C0 = [Cix[pix][ksel2], Ciy[pix][ksel2]]
                d = (E['x']- C0[0])**2 + (E['y'] - C0[1])**2
                num0 = np.where(d <= np.round(fwhm_fsf/2)**2)
                E0 = E[num0]
        else:
            x0 = 0
            y0 = 0
            C0 = [x0, y0]
            # Distance from the center of the circle to the pixel
            d = (E['x']- C0[0])**2 + (E['y'] - C0[1])**2  # d**2 ???????
            # Indices of the voxel inside the circle
            num0 = np.where(d <= np.round(fwhm_fsf/2)**2)
            # subgroup containing only the voxels inside the cylinder
            E0 = E[num0]
              
        # Number of this source    
        num_source = num_source + 1
        # Number of lines for this source
        nb_lines = len(E0)
        # To fulfill each line of the catalogue
        n_S = np.resize(num_source, nb_lines)
        # Coordinates of the center of the circle
        x_c = np.resize(C0[0], nb_lines)
        y_c = np.resize(C0[1], nb_lines)
        # T_GLR values of the voxels in this group
        correl_temp  = E0['T_GLR']
        # Spatial positions of the voxels
        x_gp = E0['x']
        y_gp = E0['y']
        # Centroid weighted by the T_GLR of voxels in each group
        x_centroid = np.sum(correl_temp*x_gp) / np.sum(correl_temp)
        y_centroid = np.sum(correl_temp*y_gp) / np.sum(correl_temp)
        # To fulfill each line of the catalogue
        x_centroid = np.resize(x_centroid, nb_lines) 
        y_centroid = np.resize(y_centroid, nb_lines)
        # Number of lines for this source
        nb_lines = np.resize(int(nb_lines), nb_lines)
        # New catalogue of detected emission lines merged in sources
        CatF0 = E0.copy()
        CatF0['ID'] = n_S
        col_x = Column(name='x_circle', data=x_c)
        col_y = Column(name='y_circle', data=y_c)
        col_xc = Column(name='x_centroid', data=x_centroid)
        col_yc = Column(name='y_centroid', data=y_centroid)
        col_nlines = Column(name='nb_lines', data=nb_lines)
        CatF0.add_columns([col_x, col_y, col_xc, col_yc, col_nlines],
                          indexes=[1, 1, 1, 1, 1])
        if len(CatF)==0:
            CatF = CatF0
        else:
            CatF = join(CatF, CatF0, join_type='outer')
        # Suppress the voxels added in the catalogue
        for k in E0['ID']:
            E.remove_rows(E['ID']==k)
    print '    %0.1fs'%(time.time()-t0)
    return CatF

def Spectral_Merging(Cat, Cat_est_line_raw, deltaz=1):
    """Merge the detected emission lines distants to less than deltaz
    spectral channel in each group

    Parameters
    ---------
    Cat          : astropy.Table
                   Catalogue of detected emission lines
                   Columns of Cat:
                   ID x_circle y_circle x_centroid y_centroid nb_lines 
                   x y z T_GLR profile pvalC pvalS pvalF T1 T2
                   residual flux num_line
    Cat_est_line : list of array
                   Catalogue of estimated lines
    deltaz       : integer
                   Distance maximum between 2 different lines

    Returns
    -------
    CatF : astropy.Table
           Catalogue
           Columns of CatF:
           ID x_circle y_circle x_centroid y_centroid nb_lines 
           x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual flux num_line

    Date  : Dec,16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Spectral_Merging'
    t0 = time.time()
    # Initialization
    CatF = Table()

    # Loop on the group
    for i in np.unique(Cat['ID']):
        # Catalogue of the lines in this group
        E = Cat[Cat['ID'] == i]
        # Sort along the maximum of the estimated lines 
        z = [np.argmax(Cat_est_line_raw[k]) for k in E['num_line']]
        # Add the spectral channel of the maximum
        # Sort along the spectral channel of the maximum
        col_zp = Column(name='z2', data=z)
        Ez = E.copy()
        Ez.add_column(col_zp)
        Ez.sort('z2')
        
        ksel = np.where(Ez[1:]['z2'] - Ez[:-1]['z2'] > deltaz)
        indF = []
        for ind in np.split(np.arange(len(Ez)), ksel[0]+1):
            if len(ind) == 1:
                # if the 2 lines are not close and not in the catlaogue yet
                indF.append(ind[0])
            else:
                # Keep the estimated line with the highest flux
                indF.append(np.where(Ez['flux'] == max(Ez[ind]['flux']))[0][0])
        
        CatF_temp = Table(Ez[indF])
        nb_line = len(CatF_temp)
        
        # Set the new number of lines for each group
        CatF_temp['nb_lines'] = np.resize(int(nb_line), len(CatF_temp))
        if len(CatF)==0:
            CatF = CatF_temp
        else:
            CatF = join(CatF, CatF_temp, join_type='outer')

    CatF.remove_columns(['z2'])
    print '    %0.1fs'%(time.time()-t0)
    return CatF
    
def Add_radec_to_Cat(Cat, wcs):
    """Function to add corresponding RA/DEC to each referent pixel
    of each group

    Parameters
    ----------
    Cat : astropy.Table
          Catalogue of the detected emission lines:
          ID x_circle y_circle x_centroid y_centroid nb_lines 
          x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual flux num_line
    wcs : mpdaf.obj.WCS
          Spatial coordinates

    Returns
    -------
    Cat_radec : astropy.Table
                Catalogue of parameters of detected emission lines:
                ID x_circle y_circle x_centroid y_centroid nb_lines 
                x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual flux
                num_line RA DEC

    Date  : Dec,16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Add_radec_to_Cat'
    t0 = time.time()
    x = Cat['x_centroid']
    y = Cat['y_centroid']
    pixcrd = [[p,q] for p,q in zip(y,x)]
    skycrd = wcs.pix2sky(pixcrd)
    ra = skycrd[:,1]
    dec = skycrd[:,0]
    col_ra = Column(name='RA', data=ra)
    col_dec = Column(name='DEC', data=dec)
    Cat_radec = Cat.copy()
    Cat_radec.add_columns([col_ra, col_dec])
    print '    %0.1fs'%(time.time()-t0)
    return Cat_radec
    
def Construct_Object_Catalogue(Cat, Cat_est_line_raw, correl, wave, filename):
    """Function to create the final catlogue of sources with their parameters

    Parameters
    ----------
    Cat              : Catalogue of parameters of detected emission lines:
                       ID x_circle y_circle x_centroid y_centroid nb_lines 
                       x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual
                       flux num_line RA DEC
    Cat_est_line_raw : list of arrays
                       Catalogue of estimated lines
    correl            : array
                        Cube of T_GLR values
    wave              : mpdaf.obj.WaveCoord
                        Spectral coordinates
    filename          : string
                        Name of the cube

    Returns
    -------
    sources : list of mpdaf.sdetect.Source
              List of sources
              
    Date  : Dec, 16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    print 'Construct_Object_Catalogue'
    t0 = time.time()
    sources = []
    uflux = u.erg/(u.s * u.cm**2)
    unone = u.dimensionless_unscaled
    cols = ['LBDA_OBS','FWHM_OBS','FLUX_OBS','GLR','PVALC','PVALS','PVALF',
            'T1','T2','PROF']
    units = [u.Angstrom,u.Angstrom,uflux,unone,unone,unone,unone,unone,unone,
             unone]
    desc = None
    fmt = ['.2f','.2f','.1f','.1f','.1e','.1e','.1e','.1f','.1f','d']

    step_wave = wave.get_step(unit=u.angstrom)
    for i in np.unique(Cat['ID']):
        # Source = group
        E = Cat[Cat['ID']==i]
        origin = ('ORIGIN', 'V1.1', filename)
        src = Source.from_data(i, E['RA'][0], E['DEC'][0], origin)
        src.add_attr('x', E['x_centroid'][0], desc='x position in pixel',
                     unit=u.pix, fmt='d')
        src.add_attr('y', E['y_centroid'][0], desc='y position in pixel',
                     unit=u.pix, fmt='d')
        # Lines of this group
        wave_pix = E['z']
        GLR = E['T_GLR']
        num_profil = E['profile']
        pvalC = E['pvalC']
        pvalS = E['pvalS']
        pvalF = E['pvalF']
        T1 = E['T1']
        T2 = E['T2']
        
        # Number of lines in this group
        nb_lines = E['nb_lines'][0]
        for j in range(nb_lines):
            ksel = np.where(Cat_est_line_raw[E['num_line'][j]] != 0)
            z1 = ksel[0][0]
            z2 = ksel[0][-1]+1
            # Estimated line
            sp = Cat_est_line_raw[E['num_line'][j]][z1:z2]
            # Wavelength in angstrom of estimated line
            #wave_ang = wave.coord(ksel[0], unit=u.angstrom)
            # T_GLR centered around this line
            c = correl[z1:z2, E['y'][j], E['x'][j]]
            # FWHM in arcsec of the profile
            profile_num = num_profil[j]
            FWHM_list = np.linspace(2, 12, 20)
            profil_FWHM = step_wave * FWHM_list[profile_num]
            #profile_dico = Dico[:, profile_num]
            flux = E['flux'][j]
            w = wave.coord(wave_pix[j], unit=u.angstrom)
            vals = [w, profil_FWHM, flux, GLR[j], pvalC[j], pvalS[j],
                    pvalF[j], T1[j], T2[j], profile_num]
            src.add_line(cols, vals, units, desc, fmt)
            sp = Spectrum(wave=wave[z1:z2], data=sp)
            src.spectra['LINE{:04d}'.format(j+1)] = sp
            sp = Spectrum(wave=wave[z1:z2], data=c)
            src.spectra['CORR{:04d}'.format(j+1)] = sp
           
        sources.append(src)
    print '    %0.1fs'%(time.time()-t0)
    return sources



if __name__ == '__main__':
    
    ##########################################
    # Input parameters                       #
    ##########################################
    
    # Spatial dimension of the FSF
    PSF_Nfsf = 13
    PSF_beta = 2.6
    # fwhm en arcsec
    PSF_fwhm2 = 0.66
    PSF_fwhm1 = 0.76
    # wavelength in angstrom
    PSF_lambda2 = 7000
    PSF_lambda1 = 4750
    
    # Dictionary of spectral profile
    Dico =  loadmat('Dico_FWHM_2_12.mat')['Dico']
    #for i in range(20):
    #    plt.plot(np.arange(201), Dico[:,i])
    
    # name of the MUSE data cube
    filename = 'minicube.fits'
    
    # Number of subcubes for the spatial segmentation
#    NbSubcube = 4 # HDFS and UDF
    NbSubcube = 1 # minicube
    
    # Edges for PCA and p-values (matlab-1)
#    Edge_xmin = 12 # HDFS
#    Edge_xmax = 311 # HDFS
#    Edge_ymin  = 18 # HDFS
#    Edge_ymax = 316 # HDFS
#    Edge_xmin = 6 # UDF
#    Edge_xmax = 313 # UDF
#    Edge_ymin  = 5 # UDF
#    Edge_ymax = 315 # UDF
    Edge_xmin = 1 # minicube
    Edge_xmax = 37 # minicube
    Edge_ymin  = 2 # minicube
    Edge_ymax = 38 # minicube
    
    # Coefficient of determination for projection during PCA
    r0 = 0.67 # HDFS 0.31c et minicube
#    r0 = 0.63 # HDFS 1.35
#    UDF ?
    
    # threshold applied on pvalues
    threshold = 8
    
    # Connectivity of contiguous voxels
    neighboors = 26
    
    # Number of the spectral ranges skipped to compute the controle cube
    nb_ranges = 3
    
    # Estimation of each emission line 
    # =1 is very slow (16h HDFS)
    grid_dxy  = 0
    grid_dz = 0
    
    
    ##########################################
    # Detection                              #
    ##########################################
    
    # Read cube
    cube = Cube(filename)
    # Raw data cube
    # Set to 0 the Nan
    cube_raw = cube.data.filled(fill_value=0)
    # Covariance Sigma
    sigma = cube.var
    # RA-DEC coordinates
    wcs = cube.wcs
    # spectral coordinates
    wave = cube.wave
    
    del cube

    # Set to Inf the Nana
    sigma[np.isnan(sigma)] = np.inf
    # Weigthed data cube
    cube_std = cube_raw / np.sqrt(sigma)

    #Dimensions
    Nz = cube_std.shape[0]
    Ny = cube_std.shape[1]
    Nx = cube_std.shape[2]

    # Parameters
    wave0 = wave.get_start(unit=u.angstrom)
    step_wave = wave.get_step(unit=u.angstrom)
    # 1 pixel in arcsec 
    step_arcsec = wcs.get_step(unit=u.arcsec)[0]
    
    # PSF
    PSF_Moffat, fwhm_pix ,fwhm_arcsec = Compute_PSF(wave, Nz,
                                                    PSF_Nfsf, PSF_beta,
                                                    PSF_fwhm1, PSF_fwhm2,
                                                    PSF_lambda1, PSF_lambda2,
                                                    step_arcsec)

    # Parameters for projection during PCA
    list_r0  = np.resize(r0, NbSubcube**2)
    
    # mean of the fwhm of the FSF in pixel
    fwhm_fsf = np.mean(fwhm_arcsec)/step_arcsec
    # Estimated mean for p-values distribution related
    # to the Rayleigh criterium
    mean_est = fwhm_fsf**2
    
    #Spatial segmentation
    inty, intx = Spatial_Segmentation(Nx, Ny, NbSubcube)

    # Compute PCA results
    A, V, eig_val, nx, ny, nz = Compute_PCA_SubCube(NbSubcube, cube_std,
                                                    intx, inty,
                                                    Edge_xmin, Edge_xmax,
                                                    Edge_ymin, Edge_ymax)

    # Number of eigenvectors for each zone
    # Parameter set to 1 if we want to plot the results
    plot_lambda = False
    nbkeep = Compute_Number_Eigenvectors_Zone(NbSubcube, list_r0, eig_val, 
                                              plot_lambda)

    # Adaptive projection of the cube on the eigenvectors
    cube_faint, cube_cont = Compute_Proj_Eigenvector_Zone(nbkeep, NbSubcube,
                                                          Nx, Ny, Nz,
                                                          A, V, nx, ny, nz,
                                                          inty, intx)

    # TGLR computing (normalized correlations)
    correl, profile = Correlation_GLR_test(cube_faint, sigma, PSF_Moffat, Dico)

    # p-values of correlation values
    # Parameter set to 1 if we want to plot the results and associated folder
    plot_dist = 0
    cube_pval_correl = Compute_pval_correl_zone(correl, intx, inty, NbSubcube,
                                                Edge_xmin, Edge_xmax,
                                                Edge_ymin, Edge_ymax,
                                                threshold)

    # p-values of spectral channel 
    cube_pval_channel = Compute_pval_channel_Zone(cube_pval_correl, intx, inty,
                                                  NbSubcube, mean_est)

    # Final p-values 
    cube_pval_final = Compute_pval_final(cube_pval_correl, cube_pval_channel,
                                         threshold)
                                         
    # connected voxels 
    labeled_cube, Ngp = Compute_Connected_Voxel(cube_pval_final, threshold,
                                                neighboors)

    # Referent pixel 
    Cat0 = Compute_Referent_Voxel(correl, profile, cube_pval_correl,
                                  cube_pval_channel, cube_pval_final, Ngp,
                                  labeled_cube)

    # 2D map : maximum of the T_GLR values over the spectral channels.
    carte_2D_correl = np.amax(correl, axis=0)
    carte_2D_correl_ = Image(data=carte_2D_correl, wcs=wcs)

    plt.figure()
    zpix, ypix, xpix = np.where(labeled_cube!=0)
    plt.plot(xpix, ypix, 'b+')
    carte_2D_correl_.plot(vmin=0, vmax=30, title='Catalogue-0-gp-voxel')

    plt.figure()
    plt.plot(Cat0['x'], Cat0['y'], 'b+')
    carte_2D_correl_.plot(vmin=0, vmax=30, title='Catalogue-0-ref-voxel')

    # Narrow band tests 
    # Parameter set to 1 if we want to plot the results and associated folder
    plot_narrow = False

    Cat1 = Narrow_Band_Test(Cat0, cube_raw, Dico, PSF_Moffat, nb_ranges,
                            plot_narrow, wcs)
    
    # Thresholded narrow bands tests
    thresh_T1 = .2
    thresh_T2 = 2;

    Cat1_T1, Cat1_T2 = Narrow_Band_Threshold(Cat1, thresh_T1, thresh_T2)

    # 2D maps plot
    plt.figure()
    plt.plot(Cat1['x'], Cat1['y'], 'b+')
    carte_2D_correl_.plot(vmin=0, vmax=30, title='Catalogue-1')

    plt.figure()
    plt.plot(Cat1_T1['x'], Cat1_T1['y'], 'b+')
    carte_2D_correl_.plot(vmin=0, vmax=30, title='Catalogue-1-T1')
    
    plt.figure()
    plt.plot(Cat1_T2['x'], Cat1_T2['y'], 'b+')
    carte_2D_correl_.plot(vmin=0, vmax=30, title='Catalogue-1-T2')

    # Estimation of each emission line 
    # Estimation with the catalogue from the narrow band Test number 2 
    Cat2_T2, Cat_est_line_raw_T2, Cat_est_line_std_T2 = \
    Estimation_Line(Cat1_T2, profile, Nx, Ny, Nz, sigma, cube_faint,
                    grid_dxy, grid_dz, PSF_Moffat, Dico)
                    
    # Spatial merging
    Cat3 =  Spatial_Merging_Circle(Cat2_T2, fwhm_fsf, Nx, Ny)

    # 2D map plot
    plt.figure()
    plt.plot(Cat3['x_centroid'], Cat3['y_centroid'], 'k+')
    for x, y in zip(Cat3['x_circle'], Cat3['y_circle']):
        circle = plt.Circle((x, y), np.round(fwhm_fsf/2), color='k',
                            fill=False)
        plt.gcf().gca().add_artist(circle)
    carte_2D_correl_.plot(vmin=0, vmax=30, title='Catalogue-3-T2')

    plt.figure()
    plt.plot(Cat2_T2['x'], Cat2_T2['y'], 'k+')
    for x, y in zip(Cat3['x_circle'], Cat3['y_circle']):
        circle = plt.Circle((x, y), np.round(fwhm_fsf/2), color='k',
                            fill=False)
        plt.gcf().gca().add_artist(circle)
    carte_2D_correl_.plot(vmin=0, vmax=30, title='Catalogue-2-T2-circle')

    # Spectral merging
    Cat4 =  Spectral_Merging(Cat3, Cat_est_line_raw_T2) 
    
    plt.figure()
    plt.plot(Cat4['x_circle'], Cat4['y_circle'], 'k+')
    carte_2D_correl_.plot(vmin=0, vmax=30, title='Catalogue-final-T2')

    # Add RA-DEC to the catalogue
    CatF_radec = Add_radec_to_Cat(Cat4, wcs)
    
    # list of source objects
    sources = Construct_Object_Catalogue(CatF_radec, Cat_est_line_raw_T2,
                                         correl, wave,
                                         os.path.basename(filename))

    plt.show()