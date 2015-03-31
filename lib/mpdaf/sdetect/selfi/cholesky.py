"""
Created on Mon Jul 22 15:59:00 2013

@author: Celine Meillier 
"""
import numpy as np
from scipy import linalg 


class Cholesky_python(object):
    """ This class manages Cholesky_python object. It implements iterative 
            update procedure for Cholesky decomposition that reduces the 
            complexity comparing to the direct computation.
        
        
        Parameters
        ----------
        LBDA  : int
                Dimension of the data cube along the wavelength axis
        nbMax : int
                Maximal dimension of the Cholesky decomposition matrix
                1000 by default
               
        Attributes
        ----------
        LBDA               : int
                             Dimension of the data cube along the wavelength axis        
        Lprop              : array
                             Array containing the proposed lower-triangular Cholesky 
                             decomposition of the augmented (for birth case) or downdated (for 
                             death case) Gram matrix before the update validation.
        Lchol              : array
                             Array containg the validated lower-triangular Cholesky 
                             matrix
        DotVecSumProd      : array
                             Array of size n x (LBDA +1) containing the 
                             product of the lower-triangular Cholesky matrix with the data and 
                             the background mean
        DotVecSumProd_prop : array
                             Array of size n x (LBDA +1) containing the
                             product of the lower-triangular Cholesky matrix with the data and
                             the background mean before the update validation
        delta              : array
                             Array of size 1 x (2xLBDA + 1) containing the energy 
                             difference terms between the augmented or downdated configuration 
                             and the previous one
        n                  : int
                             Number of objects in the current configuration
    """
    def __init__(self, LBDA, nbMax = 1000):
        """ Creates a new Cholesky_python object.
            
        Parameters
        ----------
        LBDA  : int
                Dimension of the data cube along the wavelength axis
        nbMax : int
                Maximal dimension of the Cholesky decomposition matrix
                1000 by default
        """
        self.LBDA = LBDA
        self.Lprop = np.empty((nbMax,nbMax))
        self.Lchol = np.empty((nbMax,nbMax))
        self.DotVecSumProd = np.empty((nbMax, self.LBDA+1))
        self.delta = np.empty(2*self.LBDA + 1)
        self.n = int(0)
        self.DotVecSumProd_prop = np.empty((nbMax, self.LBDA+1))
    
        
    def copy(self):
        """ Copies a new Cholesky_python object.
        """
        copy = Cholesky_python(self.LBDA)
        copy.Lprop = self.Lprop.copy()
        copy.Lchol = self.Lchol.copy()
        copy.DotVecSumProd = self.DotVecSumProd.copy()
        copy.DotVecSumProd_prop = self.DotVecSumProd_prop.copy()
        copy.n = self.n
        copy.delta = self.delta.copy()
        
        return copy
        
    def zeros(self):
        """ Initializes a new Cholesky_python object.
        """
        self.Lprop[:]=0.
        self.Lchol[:]=0.
        self.DotVecSumProd[:]=0.
        self.delta[:]=0.
        self.DotVecSumProd_prop[:]=0.
    
    def replace(self, otherConfig):
        """ Replaces a Cholesky_python object by another Cholesky_python object.
             
        Parameters
        ----------
        otherConfig : class 'Cholesky_python
                      Cholesky_python object
        """
        self.Lprop[0:otherConfig.n, 0:otherConfig.n] = \
        otherConfig.Lprop[0:otherConfig.n, 0:otherConfig.n].copy()
        self.Lchol[0:otherConfig.n, 0:otherConfig.n] = \
        otherConfig.Lchol[0:otherConfig.n, 0:otherConfig.n].copy()
        self.DotVecSumProd[0:otherConfig.n, :] = \
        otherConfig.DotVecSumProd[0:otherConfig.n, :].copy()
        self.DotVecSumProd_prop[0:otherConfig.n, :] = \
        otherConfig.DotVecSumProd_prop[0:otherConfig.n, :].copy()
        self.delta = otherConfig.delta.copy()
        self.n = otherConfig.n   
        
    def propAugment(self, v, newSumProd):
        """ Proposes an new object 
            
        Parameters
        ----------
        v          : float array
                     Array containing interactions between the new object and 
                     the previous objects, it corresponds to the last column or row
                     of the Gram matrix
        newSumProd : float array
                     Array containing the scalar product between new 
                     object and the unit vector first and then with each band of the 
                     data cube
        """
        if self.n == 0:
            gam = np.array([1.] )
            self.Lprop[self.n,self.n] = gam
            self.DotVecSumProd_prop[0,:] = newSumProd/gam
        else:
            self.Lprop[0:self.n,0:self.n] = self.Lchol[0:self.n,0:self.n]
            vp = linalg.solve(self.Lprop[0:self.n,0:self.n],v[0:self.n],sym_pos=False,lower = True) # Exact solution until the 15th decimale
            gam = np.sqrt( v[self.n]-np.dot(vp,vp))
            # New line of the Cholesky lower matrix
            self.Lprop[self.n,0:self.n] = vp
            self.Lprop[self.n, self.n] = gam
            self.DotVecSumProd_prop[0:self.n,:] = self.DotVecSumProd[0:self.n,:]
            self.DotVecSumProd_prop[self.n,:] =(newSumProd - np.dot(vp,self.DotVecSumProd[0:self.n,:]))/gam
        self.delta[0:self.LBDA+1] = self.DotVecSumProd_prop[self.n,:]**2
        self.delta[self.LBDA+1:2*self.LBDA+2] = self.DotVecSumProd_prop[self.n,1:self.LBDA+1]*self.DotVecSumProd_prop[self.n,0]
        
       
    def confAugment(self, centre, cubeMap):
        """ Confirmes the augmentation of Gram matrix.
             
        Parameters
        ----------
        centre  : float tuple
                  Center position of the considered object
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the 
                  current object configuration
        """
        cubeMap.index_obj[centre] = self.n
        self.n = self.n + 1
        self.Lchol[0:self.n, 0:self.n] = self.Lprop[0:self.n, 0:self.n]
        self.DotVecSumProd[self.n-1,:] = self.DotVecSumProd_prop[self.n-1, :]
 
    def propRemove(self, centre, cubeMap):
        """ Proposes to remove an object of the configuration and computes the 
            downdating of the Cholesky decomposition of the Gram matrix.
             
        Parameters
        ----------
        centre  : float tuple
                  Center position of the considered object
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the 
                  current object configuration
        """
        ind = cubeMap.index_obj[centre]
        iind = 1
        tmp = self.DotVecSumProd[0:self.n, :].copy()
        self.Lprop[0:self.n, 0:self.n] = self.Lchol[0:self.n, 0:self.n]
        
        for j in xrange(ind+1,self.n):
            # Construct the Givens transformation in order to triangulate
            # the matrix L whose lines indexed by ind must be removed
            v1 = self.Lprop[j, j-iind]
            v2 = self.Lprop[j, j-iind+1]
             
            if abs(v1) > abs(v2):
                w = v2/v1
                q = np.sqrt(1 + w**2)
                c = np.sign(v1)/q
                s = w*c
                r = abs(v1)*q
                 
            elif v2 == 0:
                c =1
                s = 0
                r = 0
            else:
                w = v1/v2
                q = np.sqrt(1 + w**2)
                s = np.sign(v2)/q
                c = w*s
                r = abs(v2)*q
            self.Lprop[j, j-iind ] = r
            self.Lprop[j, j-iind+1] = 0
             
            vl = np.zeros(1 + len(np.arange(j+1, self.n)), dtype=int)                
            vl[0] = int(ind)
            vl[1:] = np.arange(j+1, self.n, dtype=int)
             
            w = self.Lprop[vl,j-iind]*c + self.Lprop[vl,j-iind+1]*s
            self.Lprop[vl,j-iind+1] = -self.Lprop[vl,j-iind]*s + self.Lprop[vl,j-iind+1] *c
            self.Lprop[vl,j-iind] = w  
            w = tmp[j-iind,:]*c + tmp[j-iind+1,:]*s
            tmp[j-iind+1,:] = - tmp[j-iind,:]*s + tmp[j-iind+1,:]* c
            tmp[j-iind,:] = w
         
        self.DotVecSumProd_prop[0:self.n,:] = tmp[0:self.n,:]
 
        if self.n > 1:
            index = np.zeros(self.n-1, dtype=int)
            index[0:ind] = np.arange(0, ind)
            index[ind:self.n] = np.arange(ind+1, self.n)
            self.Lprop[0:self.n-1,0:self.n-1] = self.Lprop[index,0:self.n-1]
        else:
            self.Lprop[0,0]=0
         
        self.delta[0:self.LBDA+1] = -(tmp[self.n-1,:].squeeze())**2
        self.delta[self.LBDA+1:self.LBDA*2+2] = -tmp[self.n-1,0]*tmp[self.n-1,1:self.LBDA+2]
       
    def confRemove(self,centre,cubeMap):
        """ Confirmes the downdating of Gram matrix.
             
        Parameters
        ----------
        centre  : float tuple
                  Center position of the considered object
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the 
                  current object configuration
        """
        for c in cubeMap.index_obj.keys():
            if cubeMap.index_obj[c] > cubeMap.index_obj[centre]:
                cubeMap.index_obj[c] = cubeMap.index_obj[c]-1
        self.n = self.n-1
        self.Lchol[0:self.n,0:self.n] = self.Lprop[0:self.n,0:self.n]
        self.Lchol[self.n,:] = 0.
        self.Lchol[:,self.n] = 0.
        self.Lprop[self.n,:] = 0.
        self.Lprop[:,self.n] = 0.
        self.DotVecSumProd[0:self.n,:] = self.DotVecSumProd_prop[0:self.n,:]
        del cubeMap.index_obj[centre]
