"""
Created on Tue Oct 29 14:31:55 2013

@author: meillice
"""

import logging
import numpy as np

from objconfig import CubeMap, Picker, Backup
from ellipseSersic import EllipseSersic
from functor import addShapeToImageSF, rmvShapeToImageSF

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# import astropy.io.fits as pyfits


class KernelSersic(object):
    """ This class manages the KernelSersic objects. It inherits from KernelDecorated class, the object are defined by elliptical support and a Sersic profile.
        
        Parameters
        ----------
        cube             : class 'mpdaf.obj.Cube'
                           Cube object containing the data
        Shape_MaxDist2Bd : int
                           average size of the spatial PSF in pixels number
        PC1              : float
                           probability to propose a center from the class C1
        fsf              : float array or None             
                           Array containing FSF (spatial PSF)
                           If fsf = None, the FSF defines for the DryRun is used by default.
        MaxTranslation   : float
                           maximum radius (in number of pixels) of the translation field around the current position
        MinScaleFactor   : float
                           minimum value of the scale factor for a perturbation of the axes length
        MaxScaleFactor   : float
                           maximum value of the scale factor for a perturbation of the axes length
        Shape_MinRadius  : float
                           minimum size of the ellipse axes (before convolution by the PSF)
        Shape_MaxRadius  : float
                           maximum size of the ellipse axes (before convolution by the PSF)
        preprocessing    : tupple of arrays.
                           If no preprocessing step, preprocessing = None (default parameter).
                           Preprocessing result (proposition map, pixels classification, wavelength index map, etc)
                               
        Attributes
        ----------
        fsf            : float array
                         Array containing the spatial convolution mask. 
        MaxTranslation  : float
                          maximum radius (in number of pixels) of the translation field around the current position
        MinScaleFactor  : float
                          minimum value of the scale factor for a perturbation of the axes length
        MaxScaleFactor  : float
                          maximum value of the scale factor for a perturbation of the axes length
        Shape_MinRadius : float
                          minimum size of the ellipse axes (before convolution by the PSF)
        Shape_MaxRadius : float
                          maximum size of the ellipse axes (before convolution by the PSF)
        cube            : class 'mpdaf.obj.Cube'
                          Cube object containing the data                      
        picker          : class kernelDecorated.Picker
                          Picker object proposing the positions for new objects and the candidates to be modified or deleted.
        cubeMap         : class kernelDecorated.CubeMap
                          CubeMap object containing the information about the object configuration
        backup          : class kernelDecorated.Backup
                          Backup object contains the information about the maximum a posteriori estimate of the object configuration. 
        
    """
    
    def __init__(self, prng, cube, Shape_MaxDist2Bd, pC1, fsf, MaxTranslation, MinScaleFactor, MaxScaleFactor, Shape_MinRadius, Shape_MaxRadius, preprocessing=None):
        """ Creates a KernelSersic objects.
        
        Parameters
            ----------
            prng             : numpy.random.RandomState
                               instance of numpy.random.RandomState() with potential chosen seed.
            cube             : class 'mpdaf.obj.Cube'
                               Cube object containing the data
            Shape_MaxDist2Bd : int
                               average size of the spatial PSF in pixels number
            PC1              : float
                               probability to propose a center from the class C1
            fsf              : float array  
                               Array containing FSF (spatial PSF)
            MaxTranslation   : float
                               maximum radius (in number of pixels) of the translation field around the current position
            MinScaleFactor   : float
                               minimum value of the scale factor for a perturbation of the axes length
            MaxScaleFactor   : float
                               maximum value of the scale factor for a perturbation of the axes length
            Shape_MinRadius  : float
                               minimum size of the ellipse axes (before convolution by the PSF)
            Shape_MaxRadius  : float
                               maximum size of the ellipse axes (before convolution by the PSF)
            preprocessing    : tupple of arrays.
                               If no preprocessing step, preprocessing = None (default parameter).
                               Preprocessing result (proposition map, pixels classification, wavelength index map, etc)
        """
        self.logger = logging.getLogger('mpdaf corelib')
        d = {'class': 'KernelSersic', 'method': '__init__'}
        self.logger.info('SELFI - Defining the kernel as an elliptical support with a Sersic profile', extra=d)
        self.logger.info('\t\t FSF FWHM = %d spaxels'%Shape_MaxDist2Bd, extra=d)
        self.logger.info('\t\t pC1 probability = %0.1f'%pC1, extra=d)
        self.prng = prng
        self.cube = cube
        self.cubeMap = CubeMap(cube.shape[1], cube.shape[2], cube, preprocessing=preprocessing)
        self.picker = Picker(prng, self.cubeMap, Shape_MaxDist2Bd, pC1)
        self.backup = Backup(self.cubeMap)
        if len(fsf.shape) == 3:
            self.fsf = np.mean(fsf, axis = 0)
        else:
            self.fsf = fsf
        self.logger.info('\t\t Maximum radius for translation = %0.1f spaxel'%MaxTranslation, extra=d)
        self.logger.info('\t\t Minimum scaling factor for FWHM = %0.1f'%MinScaleFactor, extra=d)
        self.logger.info('\t\t Maximum scaling factor for FWHM = %0.1f'%MaxScaleFactor, extra=d)
        self.logger.info('\t\t Minimum half FWHM (before FSF convolution) = %0.1f'%Shape_MinRadius, extra=d)
        self.logger.info('\t\t Maximum half FWHM (before FSF convolution) = %0.1f'%Shape_MaxRadius, extra=d)
        self.MaxTranslation = MaxTranslation
        self.MinScaleFactor = MinScaleFactor
        self.MaxScaleFactor = MaxScaleFactor
        self.Shape_MinRadius = Shape_MinRadius
        self.Shape_MaxRadius = Shape_MaxRadius
        
    def plotConfig_support(self, image):
        """ Plots the object configuration on the background image. 
             
        Parameters
        ----------
        image : float array
                Array containing the background image
        """
         
        ells = [Ellipse(xy=(obj.centre[1],obj.centre[0]), width=2*obj.axe1, height=2*obj.axe2, angle=obj.angle*180/np.pi,linewidth=2, fill=False) for obj in self.cubeMap.list_obj.values()]
        fig = plt.figure()
         
        ax = fig.add_subplot(111)
         
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_edgecolor((1,0,0))
            
        background, std = image.background()
         
        LI = self.cubeMap.LI
        COL = self.cubeMap.COL
        ima = np.sum(self.cubeMap.cube, axis=0)
        ax.set_ylim(0, LI - 0.5)
        ax.set_xlim(0, COL -.5)
         
        plt.imshow(ima, cmap='binary', vmin=background-5*std, vmax=background+5*std)
 
    def plotConfig_marker(self, li_x,li_y):
        """ Plots the position of the objects center on a map.
             
        Parameters
        ----------
        li_x : list of float
               List of x coordinates
        li_y : list of float
               List of y coordinates
            """
         
        ells = [Ellipse(xy=(obj.centre[1],obj.centre[0]), width=2*obj.fwhm_1, height=2*obj.fwhm_2, angle=obj.angle*180/np.pi,linewidth=2, fill=False) for obj in self.cubeMap.list_obj.values()]
        fig = plt.figure()
         
        ax = fig.add_subplot(111, aspect='equal')
         
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_edgecolor(1,0,0)
         
        LI = self.cubeMap.image.data.shape[0]
        COL = self.cubeMap.image.data.shape[1]
        ax.set_xlim(0, LI-0.5)
        ax.set_ylim(0, COL-.5)
         
        plt.imshow(self.cubeMap.image.data,cmap='binary')
        ax.scatter(li_x,li_y,marker = '*', color = 'red')
        plt.show()         
   
    def addObject(self,obj):
        """ Adds an object to the current configuration
             
        Parameters
        ----------
        obj : class 'EllipseSersic'
              Object to add
            """
        functor = addShapeToImageSF(self.cubeMap)
        self.cubeMap.addObject(obj, functor)
        (p, q) = (np.floor(obj.centre[0]),
                  np.floor(obj.centre[1]))
        self.picker.inConfig[(p,q)] = 1         
 
    def removeObject(self,obj):
        """ Removes an object from the current configuration
             
        Parameters
        ----------
        obj : class 'EllipseSersic'
              Object to remove
            """
        functor = rmvShapeToImageSF(self.cubeMap)
        self.cubeMap.removeObject(obj,functor)
        (p,q) = (np.floor(obj.centre[0]), np.floor(obj.centre[1]))
        if (p,q) in self.picker.inConfig.keys():
            del self.picker.inConfig[(p,q)]
     
    def randomTranslation(self,obj):
        """ Proposes a new object which corresponds
        to a translation of an existant object
             
        Parameters
        ----------
        obj : class 'EllipseSersic'
              Object to add
               
        Returns
        -------
        out : EllipseSersic
              New object
        """
        u = self.prng.uniform(max(0, obj.centre[0]-self.MaxTranslation),
                              min( self.picker.LI , obj.centre[0] +
                                   self.MaxTranslation))
        v = self.prng.uniform(max(0, obj.centre[1]-self.MaxTranslation),
                              min( self.picker.COL , obj.centre[1] +
                                   self.MaxTranslation))
        centre = (u , v)
        proposed_obj = EllipseSersic(self.prng, centre, obj.fwhm_1,
                                     obj.fwhm_2, obj.angle,
                                     convolution=self.fsf)
        return proposed_obj  
     
    def randomRotation(self,obj):
        """ Proposes a new object which corresponds to a rotation 
        of an existant object 'obj'
         
        Parameters
        ----------
        obj : class 'EllipseSersic'
              Object to add
               
        Returns
        -------
        out : EllipseSersic
              New object
        """
        angle = self.prng.uniform(max(0, obj.angle * self.MinScaleFactor),
                                  min(np.pi/2, obj.angle * self.MaxScaleFactor))
        proposed_obj = EllipseSersic(self.prng, obj.centre, obj.fwhm_1,
                                     obj.fwhm_2, angle, convolution=self.fsf)
        return proposed_obj    
     
    def randomPerturbation(self,obj):
        """ Proposes a new object which corresponds to a modification
        of axes length of an existant object 'obj'
             
        Parameters
        ----------
        obj : class 'EllipseSersic'
              Object to add
               
        Returns
        -------
        out : EllipseSersic
              New object
        """
        fwhm_1 = self.prng.uniform(max(self.Shape_MinRadius,
                                       obj.fwhm_1 * self.MinScaleFactor),
                                   min( self.Shape_MaxRadius,
                                        obj.fwhm_1 * self.MaxScaleFactor))
        fwhm_2 = self.prng.uniform(max(self.Shape_MinRadius,
                                       obj.fwhm_2 * self.MinScaleFactor,
                                       fwhm_1/2.),
                                   min(self.Shape_MaxRadius,
                                       obj.fwhm_2 * self.MaxScaleFactor,
                                       2*fwhm_1))
        proposed_obj = EllipseSersic(self.prng, obj.centre, fwhm_1,
                                     fwhm_2, obj.angle, convolution=self.fsf)
        return proposed_obj
    
    def randomCreation(self, position):
        """ Proposed a new elliptic object at position given.
             
        Parameters
        ----------
        position : float tupple
                   Center's position of the object to create.
         
        Returns
        -------
        out : EllipseSersic
              New object
        """
        fwhm_1 = self.prng.uniform(self.Shape_MinRadius , self.Shape_MaxRadius)
        fwhm_2 = self.prng.uniform(max(fwhm_1/2., self.Shape_MinRadius),
                                   min( 2*fwhm_1,self.Shape_MaxRadius))
        angle = self.prng.uniform(0, np.pi/2)
        objet = EllipseSersic(self.prng, position, fwhm_1, fwhm_2, angle, convolution=self.fsf)
        return objet
