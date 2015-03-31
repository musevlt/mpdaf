"""
Created on Mon Jul 22 15:59:00 2013
    
@author: Celine Meillier
"""
import os
import numpy as np
from scipy import special, signal

from ...MUSE import LSF

from pvalues import p_values
from functor import Ellipse_scalar_product

 
class EllipseSersic(object):
    """ This class manages EllipseSersic object.
        
        Parameters
        ----------
        centre      : float tuple
                      Tuple containing the center position of the Ellipse
        fwhm_1      : float
                      FWHM of the Sersic profile along the first axis
        fwhm_2      : float
                      FWHM of the Sersic profile along the second axis
        angle       : float
                      Angle between the horizontal axis and the first ellipse axis
        convolution : float array
                      Array containing the spatial convolution mask
                      Use convolution = None if there is no convolution
        prop_energy : float
                      0.95 by default
        
        Attributes
        ----------
        
        centre            : float tuple
                            Position on the pixel grid of the ellipse's center
        p_max             : float
                            Confidence index for a max-test (not always necessary)
        pvalues           : p_valeur.p_values class
                            pvalues
        fwhm_1            : float
                            FWHM of the Sersic profile along the first axis
        fwhm_2            : float
                            FWHM of the Sersic profile along the second axis        
        n                 : float
                            Sersic index in {0.5, 1, 2}
        axe1              : float
                            Length of the first axis 
        axe2              : float
                            Length of the second axis
        angle             : float
                            Angle between the horizontal axis and the first ellipse axis
        length_fsf        : int
                            Radius of the fsf 
        profile_convolved : array
                            Array containing the Sersic profile convolved by the fsf
        positionList      : list
                            ...
        """
    
    
    def __init__(self, prng, centre=None, fwhm_1=None, fwhm_2=None, angle=None, convolution=None):
        """ Creates an EllipseSersic object.
            
        Parameters
        ----------
        prng        : numpy.random.RandomState
                      instance of numpy.random.RandomState() with potential chosen seed.
        centre      : float tuple
                      Tuple containing the center position of the Ellipse
        fwhm_1      : float
                      FWHM of the Sersic profile along the first axis
        fwhm_2      : float
                      FWHM of the Sersic profile along the second axis
        angle       : float
                      Angle between the horizontal axis and the first ellipse axis
        convolution : float array
                      Array containing the spatial convolution mask
                      Use convolution = None if there is no convolution
        """
        prop_energy = 0.95
        self.prng = prng
        self.centre = (float(centre[0]),float(centre[1]))
        self.p_max = 0 
        path = os.path.dirname(__file__) + '/p_val/'
        self.pvalues = p_values(path+'val.fits', path+'cdf_maxTest.fits')
        self.positionList = []
        
        self.fwhm_1 = fwhm_1
        self.fwhm_2 = fwhm_2
        
        if max(self.fwhm_1, self.fwhm_2) > 1.5:
            self.n = 0.5
        else:
            self.n =self.prng.randint(0,3)
            if self.n == 0:
                self.n = 0.5
    
        # Scale parameters for the Sersic profile
        alpha_1 = fwhm_1/np.log(2)**self.n
        alpha_2 = fwhm_2/np.log(2)**self.n
        
        tmp = special.gammaincinv(self.n,prop_energy)
        
        self.axe1 = max(1.,min(alpha_1*tmp**self.n,4*self.fwhm_1))
        self.axe2 = max(1.,min(alpha_2*tmp**self.n,4*self.fwhm_2))
        
        self.angle = angle
        
        R_search = self.getRadius()
        LiMin = int(self.centre[0])-int(R_search)
        LiMax = int(self.centre[0])+int(R_search+1)
        ColMin = int(self.centre[1])-int(R_search)
        ColMax = int(self.centre[1])+int(R_search+1)
        range_li = range(int(LiMin),int(LiMax))
        range_col = range(int(ColMin),int(ColMax))
        
        cost = np.cos(self.angle)
        sint = np.sin(self.angle)
        
        axis0 = np.array([ cost , -sint ]) / (self.axe2 )
        axis1 = np.array([ sint , cost ]) / (self.axe1 )
            
        a = cost**2/(alpha_1**2) + sint**2/(alpha_2**2)
        b = +np.sin(2*self.angle)/(2*alpha_1**2) - np.sin(2*self.angle)/(2*alpha_2**2)
        c = cost**2/(alpha_2**2) + sint**2/(alpha_1**2)
        
        profile2D = np.zeros((len(range_li), len(range_col)))
        
        p = 0
        for li in range_li:
            q = 0
            for col in range_col:
                point = np.array( [ [li - self.centre[0]] , [col - self.centre[1]] ] )
                ratio0 = np.dot( axis0 , point )
                ratio1 = np.dot( axis1 , point )
                ratio = ratio0**2 + ratio1**2
                if ratio <= 1 :        
                    x = li-self.centre[0]
                    y = col-self.centre[1]
                    
                    profile2D[p,q] = np.exp( - (np.sqrt( a*y**2 + 2*b*x*y + c*x**2 ))**(1./self.n) )
                
                q += 1
            p += 1
        
        if np.sum(convolution) == None:
            self.length_fsf = 0
            self.profile_convolved = profile2D
            total_energy = np.sqrt(np.sum(self.profile_convolved**2))
            self.profile_convolved = self.profile_convolved/ total_energy
        else:
            fsf = convolution
            self.length_fsf = (fsf.shape[0]-1)/2
            profile_convolved = signal.convolve(profile2D,fsf,'full')
            self.profile_convolved = profile_convolved
            total_energy = np.sqrt(np.sum(self.profile_convolved**2))
            self.profile_convolved = self.profile_convolved/ total_energy
            
    @classmethod
    def from_source(cls, prng, source):
        ima = source.ima['white']
        y, x = ima.wcs.sky2pix([source.dec, source.ra])[0]
        fwhm_1 = source.get_extra('SELFI fwhm1')
        fwhm_2 = source.get_extra('SELFI fwhm2')
        angle = source.get_extra('SELFI angle')
        el = cls(prng, centre=(y, x), fwhm_1=fwhm_1, fwhm_2=fwhm_2,
                 angle=angle, convolution=None)
        el.p_max = source.get_extra('SELFI pmax')
        el.length_fsf = source.get_extra('SELFI lfsf')
        el.profile_convolved = ima.data.data
        el.positionList = [(y,x)]
        return el
    
    def profile(self, LI, COL):
        """ Returns an image of the spatial intensity distribution of the ellipse.
         
        Parameters
        ----------
        LI  : int
              Number of lines of the image
        COL : int
              Number of columns of the image
        """
        R_search = self.getRadius() + self.length_fsf
        
        R0 = min(int(R_search), int(self.profile_convolved.shape[0]/2))
        R1 = min(int(R_search), int(self.profile_convolved.shape[1]/2))
        profile = np.zeros((LI, COL))
        
        profile[max(0, int(self.centre[0]) - R0) :\
                min(LI, int(self.centre[0]) + R0 + 1),\
                max(0, int(self.centre[1]) - R1) :\
                min(COL, int(self.centre[1]) + R1 + 1) ] = \
                self.profile_convolved[max(R0 - int(self.centre[0]), 0) :\
                                       min(2 * R0 + 1 ,
                                           R0 + (LI - int(self.centre[0]))),\
                                       max(R1 - int(self.centre[1]), 0) :\
                                       min(2 * R1 + 1 ,
                                           R1 + (COL -int(self.centre[1])))]
        return profile  
     
    def getRadius(self):
        """Returns the lenght of the largest axes.
        """
        return max(self.axe2, self.axe1)    
    
    def adapted_p_value_wavelength_calcul(self, m, sigma2, cube, index_map):
        """ Returns the p-value of the ellipse evaluating at the wavelength corresponding to the index_map parameter.
         
        Parameters
        ----------
        m         : float array
                    Array containing the mean of each spectral band of the data cube
        sigma2    : float array
                    Array containing the variance of each spectral band of the data cube
        cube      : float array
                    Data cube
        index_map : int
                    Index of the considered spectral band
        """
        index = index_map[int(self.centre[0]), int(self.centre[1])]
        p = self.profile(cube.shape[1], cube.shape[2])
        a = np.where(p > 0)
        pVal = (np.sum((cube[index, a[0], a[1]] - m[index]) * p[a[0], a[1]] )\
                /(np.sqrt(sigma2[index])))
        return self.pvalues.p_val_maxTest(pVal)
 
    def adapted_p_value_wavelength(self, m, sigma2, cube, index_map, len_lsf = 5):
        """ Returns the p-value of the ellipse evaluating at the wavelength
        corresponding to the index_map parameter, p-values are weighted by the LSF.
         
        Parameters
        ----------
        m         : float array
                    Array containing the mean of each spectral band of the data cube
        sigma2    : float array
                    Array containing the variance of each spectral band of the data cube
        cube      : float array
                    Data cube
        index_map : int
                    Index of the considered spectral band
        len_lsf   : int
                    Number of spectral band corresponding to the spreaf of the LSF
            """
        if np.sum(index_map*np.ones(index_map.shape)) == 0:
            return self.adapted_p_value_wavelength_calcul(m, sigma2, cube, index_map)
        else:
            index = index_map[int(self.centre[0]), int(self.centre[1])]
            p = self.profile(cube.shape[1], cube.shape[2])
            a = np.where(p > 0)
            li = []
            li2 = []
            lsf = LSF()
            lsf_val = lsf.get_LSF(index_map[int(self.centre[0]),
                                            int(self.centre[1])],
                                  1.25, len_lsf)
            j = -1
            normalisation = 0.
            for i in xrange(-(len_lsf-1)/2, (len_lsf-1)/2+1):
                j += 1
                if (index+i > 0 and index+i < cube.shape[0]):
                    pVal = (np.sum((cube[index+i, a[0], a[1]] - m[index+i])
                                   *p[a[0], a[1]])
                            /(np.sqrt(sigma2[index+i])))
                    li2.append(pVal)
                    li.append(pVal * lsf_val[j])
                    normalisation += lsf_val[j]
            return self.pvalues.p_val_maxTest(np.sum(li) / normalisation)        
            
    def applyToPoints(self,functor,LI,COL) :
        """ Applies a function to all the pixels that belongs to the ellipse support.
            
        Parameters
        ----------
        functor : class 'functor.functor'
                  Pixel-wise function
        LI      : int
                  Number of lines of the image
        COL     : int
                  Number of columns of the image
        """
        
        R_search = self.getRadius() + self.length_fsf
        LiMin = int(max(0, np.floor(self.centre[0] - R_search)))
        LiMax = int(min(LI, np.ceil(self.centre[0] + R_search + 1)))
        ColMin = int(max(0,np.floor(self.centre[1]-R_search)))
        ColMax = int(min(COL,np.ceil(self.centre[1]+R_search+1)))
        range_li = range(LiMin,LiMax)
        range_col = range(ColMin,ColMax)
        
        spatial_profile = self.profile(LI,COL)
        for li in range_li:
            
            for col in range_col:
                if spatial_profile[li,col] > 0:
                    functor.applyToPoint(self.centre, (li, col), spatial_profile[li,col])
                    
    def nbPixelShared(self,cubeMap, el):
        """ Returns pixels number shared with another ellipse 'el'
             
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the
                  current object configuration
        el      : class 'EllipseSersic'
                  EllipseSersic object
            """
        nbPix = 0
        R_search = self.getRadius()
        LiMin = int(max(0, np.floor(self.centre[0] - R_search)))
        LiMax = int(min(cubeMap.LI, np.ceil(self.centre[0] + R_search + 1)))
        ColMin = int(max(0, np.floor(self.centre[1] - R_search)))
        ColMax = int(min(cubeMap.COL, np.ceil(self.centre[1] + R_search + 1)))
        range_li = range(LiMin, LiMax)
        range_col = range(ColMin, ColMax)
        cost = np.cos(self.angle)
        sint = np.sin(self.angle)
         
        axis0 = [cost / (self.axe2 + 0.5) , -sint / (self.axe2 + 0.5)]
        axis1 = [sint / (self.axe1 + 0.5), cost / (self.axe1 + 0.5)]
         
        for li in range_li:
            for col in range_col:
                point = [li - self.centre[0] , col - self.centre[1]]
                ratio0 = axis0[0] * point[0] + axis0[1] * point[1]
                ratio1 = axis1[0] * point[0] + axis1[1] * point[1]
                ratio = ratio0**2 + ratio1**2
                if ratio <= 1 :
                    if el.centre in cubeMap.grille[li][col].vecShapes.keys():
                        nbPix += 1
        return nbPix

    def acceptability(self,cubeMap,hardcore, index = -np.ones((1,1))):
        """ Applies the acceptation criterion based on the maximum overlapping
        ratio of the new ellipse with the other object of the current
        configuration.
             
        Parameters
        ----------
        cubeMap  : class 'kernelDecorated.CubeMap'
                   CubeMap object containing the information about the
                   current object configuration
        hardcore : float
                   Ratio in [0,1] of the maximum overlapping ratio
                   authorized by the Raileigh criterion
        index    : array
                   Array containing the index of the wavelength
                   corresponding to the maximum value of each
                   spectrum of the data cube.
        """
        if index[0,0] == -1:
            index = np.zeros((cubeMap.LI, cubeMap.COL))
        functor = Ellipse_scalar_product(cubeMap)
        self.applyToPoints(functor, cubeMap.LI, cubeMap.COL)
    
        indic = True
        for centre in functor.dict.keys():
            if centre != self.centre:
                if (functor.dict[centre] > hardcore) :
                    indic = False
        return indic
     
    def acceptability_geometry(self, cubeMap, obj, hardcore,
                               index=-np.ones((1,1))):
        """ Applies the acceptation criterion based on the maximum
        overlapping ratio of the new ellipse with the other object
        of the current configuration in the case of a geometrical
        modification.
             
        Parameters
        ----------
        cubeMap  : class 'kernelDecorated.CubeMap'
                   CubeMap object containing the information about the
                   current object configuration
        hardcore : float
                   Ratio in [0,1] of the maximum overlapping ratio
                   authorized by the Raileigh criterion
        index    : array
                   Array containing the index of the wavelength corresponding
                   to the maximum value of each spectrum of the data cube.
            """
        if index[0,0] == -1:
            index = np.zeros((cubeMap.LI, cubeMap.COL))
        functor = Ellipse_scalar_product(cubeMap)
        self.applyToPoints(functor, cubeMap.LI, cubeMap.COL)
         
        indic = True
        for centre in functor.dict.keys():
            if centre != obj.centre and centre != self.centre :
                if (functor.dict[centre] > hardcore ) :
                    indic = False
        return indic
