"""
Created on Wed Apr 10 10:59:56 2013

@author: Celine Meillier
"""
import numpy as np
from scipy import signal

from ...MUSE import LSF


class LSFmatrix(object):
    """ This class manages the LSFmatrix objects.
        
        Parameters
        ----------
        sizeLSF : int
                  Length of the LSF
        nbLBDA  : int
                  Number of wavelengths 
        LBDAmin : float
                  Minimal value of the wavelengths axis, in Angstrom.
        LBDAmax : float
                  Maximal value of the wavelengths axis, in Angstrom.
        
        Attributes
        ----------       
        sizeLSF : int
                  Length of the LSF
        nbLBDA  : int
                  Number of wavelengths 
        LSF     : array
                  nbLBDA x nbLBDA array containing the LSF definition.
                  Adapted to apply the LSF convolution by matrix product.
        LBDAmin : float
                  Minimal value of the wavelengths axis, in Angstrom.
        LBDAmax : float
                  Maximal value of the wavelengths axis, in Angstrom.
        lsf_tab : array
                  nbLBDA x sizeLSF array representing the evolution
                  of the LSF with wavelength 
    """
    
    def __init__(self, sizeLSF, nbLBDA, LBDAmin, LBDAmax):
        """ Creates a LSFmatrix objects.
            
        Parameters
        ----------
        sizeLSF : int
                  Length of the LSF
        nbLBDA  : int
                  Number of wavelengths 
        LBDAmin : float
                  Minimal value of the wavelengths axis, in Angstrom.
        LBDAmax : float
                  Maximal value of the wavelengths axis, in Angstrom.
        """
        self.sizeLSF = sizeLSF
        self.nbLBDA = nbLBDA
        self.LSF = np.zeros((nbLBDA, nbLBDA)) 
        self.LBDAmin = LBDAmin
        self.LBDAmax = LBDAmax
        dLBDA = (LBDAmax - LBDAmin) / nbLBDA
        self.lsf_tab = np.zeros((nbLBDA, self.sizeLSF))
        ind = 0
        #dLBDA = 1.25
        
        # LSF coeff
        normalisation = 0
        for i in (np.arange(nbLBDA)*dLBDA + LBDAmin):
            vec = LSF(type = 'qsim_v1').get_LSF(i, dLBDA, size=self.sizeLSF)
            normalisation = np.dot(vec, vec)
            self.lsf_tab[ind,:] = vec/np.sqrt(normalisation)       
            ind +=1
            
        # correlation matrix
        LSFtmp = np.zeros((self.nbLBDA + (self.sizeLSF -1), self.nbLBDA))   
        for i in np.arange(0, self.nbLBDA):
            LSFtmp[i:i+self.sizeLSF,i] = self.lsf_tab[i,:].T
        self.LSF = LSFtmp[(self.sizeLSF -1)/2:self.nbLBDA+(self.sizeLSF -1)/2,:]
        
    def LSFconvolved_tab(self, form):
        """ Returns an array with the convolution of a spectrum
        with the LSF for all the wavelengths, with an l2-normalization.
         
        Parameters
        ----------
        form : array
               spectrum
                
        Returns
        -------
        out : array
              nbLBDA x n array containing the convolution of the form
              with all the nbLBDA LSF expressions,
              n is the number of element in the form vector.
        """
        n = len(form)
        m = n + self.sizeLSF - 1
        tmp = np.zeros((self.nbLBDA, m))
        for i in np.arange(0, self.nbLBDA):
            tmp[i, :]= signal.convolve(self.lsf_tab[i,:].squeeze(), form, 'full')
            normalisation = np.dot(tmp[i,:], tmp[i,:])
            tmp[i, :] = tmp[i, :] / np.sqrt(normalisation)
        return tmp  
       
    def LSFconvolved(self, form):
        """ Returns a multi-diagonal square matrix with the convolution
        of a spectrum with the LSF for all the wavelengths,
        with an l2-normalization.
        
        Parameters
        ----------
        form : array
               spectrum
                
        Returns
        -------
        out : array
              nbLBDA x n array containing the convolution of the form
              with all the nbLBDA LSF expressions,
              n is the number of element in the form vector.
            """
         
        tmp = self.LSFconvolved_tab(form)
        n = len(form)
        m = n + self.sizeLSF - 1
        LSFtmp = np.zeros((self.nbLBDA + (m -1), self.nbLBDA))
         
        for i in np.arange(0, self.nbLBDA):
            LSFtmp[i:i+m,i] = tmp[i,:]
    
        tmp2 = LSFtmp[(m -1)/2: self.nbLBDA+(m -1)/2, :]
        return tmp2
