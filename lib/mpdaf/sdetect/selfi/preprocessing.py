"""Created on Fri Jun 14 11:28:41 2013
@author: Celine Meillier

Filtering functions for data preprocessing 
"""
import numpy as np
from scipy import ndimage, signal
import multiprocessing

from lsf import LSFmatrix

def Image_conv(arglist):
    """ Defines the convolution between an image and an array.
        Designed to be used with the multiprocessing function 'FSF_convolution_multiprocessing'
         
        Parameters
        ----------
        k     : integer
                index
        im    : array
                image
        tab   : array
                array containing the convolution kernel
         
        Returns
        -------
        out : array
    """
    k = arglist[0]
    im = arglist[1]
    tab = arglist[2]
    res = signal.convolve(im, tab, 'full')
    a,b = tab.shape
    return k, res[(a-1)/2 :im.shape[0] + (a-1)/2 , (b-1)/2 :im.shape[1]+(b-1)/2 ]


def FSF_cube(norme, LBDAmin=4800, LBDAmax=9299, dlbda=1.25, dx=0.2, FWHM = False):
    """Computes the FSF cube from the equation given on the wikiMUSE (available on https://musewiki.aip.de/fsf-model) for the DryRun data cube. It returns the FSF cube
        
        :param norme: 1 for l1-norm and 2 (or whatever) for l2-norm
        :type norm: int
        :param LBDAmin: value of the smallest wavelength if FSF is three-dimensional
        :type LBDAmin: float
        :param LBDAmax: value of the largest wavelength if FSF is three-dimensional
        :type LBDAmax: float
        :param FWHM: if True return an array containing the FWHM computed at each wavelength.
        :type FWHM: Boolean
        """
    # MUSE FSF parameters information available on https://musewiki.aip.de/fsf-model
    
    lbda = np.arange(LBDAmin, LBDAmax +1., dlbda)
    LBDA = len(lbda)
    s0 = 0.8
    sec = 1.2
    l0=22
    
    F0 = sec**0.6*(5000/lbda)**dx*s0
    r0 = 2.01*1e-5*lbda/F0
    
    
    beta = -np.log(2)/np.log(2.183*(r0/l0)**0.356)
    alpha = F0/np.sqrt(2**(1/beta))/2
    
    if FWHM == True:
        fwhm = 2*alpha*np.sqrt( 2**(1./beta) -1 )

    t = 6
    taille = np.arange(-t,t+1)*dx
    r2 = taille[:, np.newaxis]**2 + taille[np.newaxis, :]**2
    fsf = (1+r2[np.newaxis, :, :] / alpha[:, np.newaxis, np.newaxis]**2)**(-beta[:, np.newaxis, np.newaxis])
    ksel = np.where(np.sqrt(r2)>t*dx)
    fsf[:, ksel[0], ksel[1]] = 0
    masque = np.ones((LBDA,2*t+1,2*t+1))
    masque[:, ksel[0], ksel[1]] = 0
    fsf0 = fsf[:,t,0]
    fsf -= fsf0[:, np.newaxis, np.newaxis]
    fsf[:, ksel[0], ksel[1]] = 0
    if norme == 1:
        normalisation = np.sum(fsf, axis=(1,2))
    else:
        normalisation = np.sqrt(np.sum(fsf**2, axis=(1,2)))
    fsf /= normalisation[:, np.newaxis, np.newaxis]

    if FWHM == False:
        return fsf
    else:
        return fsf, fwhm
    
def FSF_convolution_multiprocessing(cube, fsf):
    """
    """
    cpu_count = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(processes=cpu_count)
    processlist = list()
    for k in range(cube.shape[0]):
        data = np.ma.filled(cube[k,:,:].data, np.ma.median(cube[k,:,:].data))
        processlist.append([k, data, fsf[k,:,:]])
        #processlist.append([k, cube[k,:,:].data.data, fsf[k,:,:]])
    processresult = pool.imap_unordered(Image_conv, processlist)
    pool.close()
    result = np.empty((cube.shape))
    for k,out in processresult:
        result[k,:,:] = out
    return result

def spectral_spread_adapted_filter(cube, fsf):
    """ Applies the 3D matched filter. It returns the result of the filtering.
         
         Parameters
         ----------
         cube : array
                Normalized data cube
         fsf  : array
                Spatial PSF (fsf) (with the spectral evolution of the FSF)
    """
    LBDA, P, Q = cube.shape
 
    convFSF = FSF_convolution_multiprocessing(cube, fsf=fsf)
 
    form = np.array([0.5, 1, 2, 5, 8, 10, 8, 5, 2, 1, 0.5])
     
    normalisation = np.dot(form, form)
    form /= np.sqrt(normalisation)
 
    tailleLSF = 7
    LBDAmin, LBDAmax = cube.wave.get_range()
    lsf = LSFmatrix(tailleLSF, LBDA, LBDAmin, LBDAmax)
     
    LSF = lsf.LSFconvolved(form)        
     
    filtered_im = np.zeros((LBDA, P, Q))
    for p in np.arange(0,P):
        for q in np.arange(0,Q):
            filtered_im[:,p,q] = np.dot(LSF,convFSF[:,p,q].squeeze() )
 
    return filtered_im

def fast_processing(im_max, P,Q, thresh_max):
    """ Fast processing
    
    Parameters
    ----------
    im_max     : array
                 result of the adapted filter and the max-test
    P          : int
                 number of lines of this image
    Q          : int
                 number of columns of this image
    thresh_max : float
                 threshold value for the max-test
                 
    Returns
    -------
    out : (Map, Map2)
          Map: tupple of arrays that contains coordinates of local maxima
          of areas of pixels above the threshold value 'thresh'
          Map2: tupple of arrays that contains coordinates
          of pixels above the threshold value 'thresh'
    """
    SE = np.ones((3,3))
    SE[1,1] = 0
    SE[0,0] = 0
    SE[0,2] = 0
    SE[2,2] = 0
    SE[2,0] = 0
    imdilate_max = ndimage.grey_dilation(im_max, footprint = SE)
    Map = np.where(((im_max>imdilate_max) & (im_max > thresh_max*np.ones((P,Q))) & (im_max == im_max)))
    Map2 = np.where((im_max > thresh_max*np.ones((P,Q))))
    return Map,Map2

def preprocessing_photometric(image, thresh):
    """ Applies a simple threshold for data following : y = X + noise
       (no adapted filter for detection).
       
       Parameters
       ----------
       image  : array
                Image to be preprocessed
       thresh : float       
                Threshold value
               
       Returns
       -------
       out : (Map, Map2)
             Map: tuple of arrays that contains coordinates of local maxima
                  of areas of pixels above the threshold value 'thresh'
             Map2: tuple of arrays that contains coordinates of pixels
             above the threshold value 'thresh'
        """
    
    SE = np.ones((3,3))
    SE[1,1] = 0

    imdilate = ndimage.grey_dilation(image, footprint=SE)
    Map = np.where((image>imdilate) & (image > thresh*np.ones_like(image)))
    Map2 = np.where(image > thresh*np.ones_like(image))
    
    return Map, Map2

def moving_average(ima, niter=3):
    P,Q = ima.shape
    image = ima.copy()
    step = 20
    sizeW = 71
    deltaW = (sizeW-1)/2
    for p in np.arange(0,P,step):
        for q in np.arange(0,Q, step):
            pmin = max(0, p-deltaW)
            pmax = min(P, p+deltaW)
            qmin = max(0, q-deltaW)
            qmax = min(Q, q+deltaW)

            m = np.ma.median(ima.data[pmin:pmax, qmin:qmax])
            image.data[pmin:pmax, qmin:qmax] = ima.data[pmin:pmax, qmin:qmax] - m

    tab = image.data
    for n in range(niter + 1):
        ksel = np.where(tab <= (np.ma.mean(tab) + 3 * np.ma.std(tab)))
        tab = tab[ksel]
    a = np.array([np.ma.mean(tab) , np.ma.std(tab)])

    image.data = image.data/(a[1]*np.ones((P,Q)))
    return image
