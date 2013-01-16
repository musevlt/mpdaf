from scipy import special
import numpy as np

def LSF(lbda,step,size):
    """ Returns a simple LSF model.
    
    This is a simple model where the LSF is supposed to be constant over the filed of view.
    It uses a simple parametric model of variation with wavelength.
    
    The model is a convolution of a step function with a gaussian.
    The resulting function is then sample by the pixel size.
    The slit width is assumed to be constant (2.09 pixels).
    The gaussian sigma parameter is a polynomial approximation of order 3 with wavelength.
    
    :param lbda: wavelength value in A
    :type lbda: float
    :param step: size of the pixel in A
    :type step: float
    :param size: number of pixels
    :type size: odd integer
    :rtype: np.array
    """
    T = lambda x: np.exp((-x**2)/2.0) + np.sqrt(2.0*np.pi)*x*special.erf(x/np.sqrt(2.0))/2.0
    
    c = np.array([-0.09876662, 0.44410609, -0.03166038, 0.46285363])
    sigma = lambda x: c[3] + c[2]*x + c[1]*x**2 + c[0]*x**3
    
    x = (lbda-6975.0)/4650.0
    h = 2.09
    sig = sigma(x)
    dy = step / 1.25
        
    k = size/2
    y = np.arange(-k, k+1)
        
    y1 = (y - h/2.0) / sig
    y2 = (y + h/2.0) / sig
    
    LSF = T(y2 + dy/2.0) - T(y2 - dy/2.0) - T(y1 + dy/2.0) + T(y1 - dy/2.0)
    LSF /= LSF.sum()
        
    return LSF