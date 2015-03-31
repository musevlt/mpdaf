import numpy as np
import datetime
import logging

from ...sdetect import Source
from ...obj import Image, Spectrum


class KernelBayesian(object):
    """ This class manages the KernelBayesian object
        
        Parameters
        ----------
        kernel                 : class 'KernelDecorated'
                                 A KernelDecorated object describing the objects (shape, intensity, etc)
        posterior_distribution : class 'posterior'
                                 Object containing the information on the posterior density and the methods to evaluate it.
        HardCore               : float
                                 Maximum ratio in [0,1] of energy shared by two close objects (Rayleigh criterion)
        max_thresh             : float
                                 Threshold value for the max-test in the preprocessing step.
        index                  : int array
                                 Possible array containing the wavelength index of each spectrum that constitutes the data cube.
        
        
        Attributes
        ----------
        objPattern   : class 'KernelDecorated'
                       A KernelDecorated object describing the objects (shape, intensity, etc)
        mCubeBg      : Array
                       Vector containing background value (mean computed by sigma-clipping).
        stdCubeBg    : Array
                       Vector containing background standard deviation (computed by sigma-clipping).
        index        : Array
                       Array containing the wavelength index of the maximum value of each spectrum that constitues the data cube.
                       If this information is not available or if the data cube is an image this array is set to zero.
        posterior    : class 'posterior'
                       Object containing the information on the posterior density and the methods to evaluate it.
        niter        : int
                       Iteration number
        list_pval    : list
                       List of p-values of the accepted object in the whole Markov chain
                       (to have an idea of the p-value under the H1-hypothesis)
        list_pval_H0 : list
                       List of p-values of the rejected object in the whole Markov chain
                       (to have an idea of the p-value under the H0-hypothesis)
        HardCore     : float
                       Maximum ratio in [0,1] of energy shared by two close objects (Rayleigh criterion)
        max_thresh   : float
                       Threshold value for the max-test in the preprocessing step.
        
        """

    def __init__(self,prng, kernel, posterior_distribution, HardCore, max_thresh, index = np.zeros((1,1))):
        """ Creates a KernelBayesian object.
        
        Parameters
        ----------
        prng                   : numpy.random.RandomState
                                 instance of numpy.random.RandomState() with potential chosen seed.
        kernel                 : class 'KernelDecorated'
                                 A KernelDecorated object describing the objects (shape, intensity, etc)
        posterior_distribution : class 'posterior'
                                 Object containing the information on the posterior density and the methods to evaluate it.
        HardCore               : float
                                 Maximum ratio in [0,1] of energy shared by two close objects (Rayleigh criterion)
        max_thresh             : float
                                 Threshold value for the max-test in the preprocessing step.
        index                  : int array
                                 Possible array containing the wavelength index of each spectrum that constitutes the data cube.
        """
        self.logger = logging.getLogger('mpdaf corelib')
        d = {'class': 'KernelBayesian', 'method': '__init__'}
        self.logger.info('SELFI - Defining the Bayesian model', extra=d)
        self.logger.info('\t\t Rayleigh criterion = %0.2f spaxels'%HardCore, extra=d)
        self.logger.info('\t\t threshold value for the max-test in the preprocessing step = %0.2f'%max_thresh, extra=d)
        
        self.prng = prng
        
        self.objPattern = kernel
        
        f = Image.background
        a = kernel.cube.loop_ima_multiprocessing(f, cpu = 8, verbose = True)
        a = np.array(list(a))
        self.mCubeBg = a[:, 0]/kernel.cube.fscale
        self.stdCubeBg = a[:, 1]/kernel.cube.fscale
        if kernel.cube.shape[0] == 1 or np.sum(index) == 0 :
            self.index = np.zeros((kernel.cube.shape[1], kernel.cube.shape[2]))
        else :
            self.index = index
        
        self.posterior = posterior_distribution
        self.niter = 0
        self.list_pval = []
        self.list_pval_H0 = []
        
        self.HardCore = HardCore
        self.max_thresh = max_thresh
        
    def plotConfig_marker(self,li_x,li_y):
        """ Plots the position of the objects center on a map.
             
        Parameters
        ----------
        li_x : list of float
               List of x coordinates
        li_y : list of float
               List of y coordinates
        """
        self.objPattern.plotConfig_marker(li_x,li_y)
     
     
    def plotConfig_support(self, image):
        """ Plots the object configuration on the background image.
             
        Parameters
        ----------
        image : float array
                Array containing the background image
            """
        self.objPattern.plotConfig_support(image)
    
    
    def initBirth(self, obj):
        """ Initializes the object configuration and the posterior density evaluation with an object
        
        Parameters
        ----------
        obj : class 'IShape'
              Object to be added to the configuration
        """
        
        self.posterior.updateBirthProp(self.niter,self.objPattern.cubeMap,obj)
        self.objPattern.addObject(obj)
        self.posterior.updateBirth(obj,self.objPattern.cubeMap)
        self.niter = self.niter + 1
        return True
    
    def tryBirth(self):
        """ Proposes a birth move to the RJMCMC algorithm and tests if it can be accepted in the current configuration
        """
 
        proposed_obj = self.objPattern.randomCreation(self.objPattern.picker.randomBirth())
         
        if proposed_obj.centre == (-1.0,-1.0):
            self.posterior.minimale_action(self.niter)
            self.niter = self.niter + 1
            return False            
        else:
            pval = proposed_obj.adapted_p_value_wavelength(self.mCubeBg,
                                                           self.stdCubeBg**2,
                                                           self.objPattern.cube.data.data,
                                                           self.index)
            if not proposed_obj.acceptability(self.objPattern.cubeMap, self.HardCore, self.index) :   
                self.posterior.minimale_action(self.niter)
                self.niter = self.niter + 1
                if (int(proposed_obj.centre[0]), int(proposed_obj.centre[1]))\
                 in self.objPattern.picker.bright.keys():
                    self.objPattern.picker.bright\
                    [(int(proposed_obj.centre[0]),\
                      int(proposed_obj.centre[1]))] = 0
                if (int(proposed_obj.centre[0]), int(proposed_obj.centre[1]))\
                 in self.objPattern.picker.intermed.keys():
                    self.objPattern.picker.intermed\
                    [(int(proposed_obj.centre[0]),\
                      int(proposed_obj.centre[1]))] = 0
                return False
            else:
                proposed_obj.p_max = self.objPattern.cubeMap.maxMap\
                [int(proposed_obj.centre[0]), int(proposed_obj.centre[1])]
                 
                if proposed_obj.p_max >= self.max_thresh:
                    self.posterior.updateBirthProp(self.niter,
                                                   self.objPattern.cubeMap,
                                                   proposed_obj)
                     
                    n = self.posterior.config.n + 1
                    tmp = np.dot(self.posterior.config.Lprop[0:n,0:n],
                                 self.posterior.config.DotVecSumProd_prop[0:n,:])
                    intensity = tmp[n-1,1: ] - self.mCubeBg*tmp[n-1,0]
 
                    if max(intensity) >= 0:
                        if len(self.objPattern.cubeMap.list_obj) == 0:
                            MHG = np.log(0.5) + \
                            self.posterior.birth_distribution(self.niter)
                        else:
                            MHG = self.posterior.birth_distribution(self.niter)
                        acceptanceRate = self.prng.uniform(0,1)
                         
                        if MHG >= np.log(acceptanceRate):
                            proposed_obj.positionList.append(proposed_obj.centre)
                            self.objPattern.picker.map_acceptation\
                            [int(proposed_obj.centre[0]), int(proposed_obj.centre[1])] += 1
                            self.objPattern.addObject(proposed_obj)
                            self.posterior.updateBirth(proposed_obj,
                                                       self.objPattern.cubeMap)
                             
                            self.niter = self.niter + 1
                            self.list_pval.append(pval)
                            if (int(proposed_obj.centre[0]),
                                int(proposed_obj.centre[1])) \
                                in self.objPattern.picker.bright.keys():
                                self.objPattern.picker.bright\
                                [(int(proposed_obj.centre[0]),\
                                  int(proposed_obj.centre[1]))] = 1
                            if (int(proposed_obj.centre[0]),
                                int(proposed_obj.centre[1])) \
                                in self.objPattern.picker.intermed.keys():
                                self.objPattern.picker.intermed\
                                [(int(proposed_obj.centre[0]),\
                                  int(proposed_obj.centre[1]))] = 1
                            return True
                        else:
                            self.list_pval_H0.append(pval)
                            if (int(proposed_obj.centre[0]),
                                int(proposed_obj.centre[1])) \
                                in self.objPattern.picker.bright.keys():
                                self.objPattern.picker.bright\
                                [(int(proposed_obj.centre[0]),\
                                  int(proposed_obj.centre[1]))] = 0
                            if (int(proposed_obj.centre[0]),
                                int(proposed_obj.centre[1])) \
                                in self.objPattern.picker.intermed.keys():
                                self.objPattern.picker.intermed\
                                [(int(proposed_obj.centre[0]),\
                                  int(proposed_obj.centre[1]))] = 0
                            self.posterior.minimale_action(self.niter)
                            self.niter = self.niter + 1
                            return False
                    else :
                        self.posterior.minimale_action(self.niter)
                        if (int(proposed_obj.centre[0]),
                            int(proposed_obj.centre[1])) \
                            in self.objPattern.picker.bright.keys():
                            self.objPattern.picker.bright\
                            [(int(proposed_obj.centre[0]),\
                              int(proposed_obj.centre[1]))] = 0
                        if (int(proposed_obj.centre[0]),
                            int(proposed_obj.centre[1])) \
                            in self.objPattern.picker.intermed.keys():
                            self.objPattern.picker.intermed\
                            [(int(proposed_obj.centre[0]),\
                              int(proposed_obj.centre[1]))] = 0
                        self.niter = self.niter + 1
                        return False
                else:
                    self.posterior.minimale_action(self.niter)
                    if (int(proposed_obj.centre[0]),
                        int(proposed_obj.centre[1])) \
                        in self.objPattern.picker.bright.keys():
                        self.objPattern.picker.bright\
                        [(int(proposed_obj.centre[0]),\
                          int(proposed_obj.centre[1]))] = 0
                    if (int(proposed_obj.centre[0]),
                        int(proposed_obj.centre[1])) \
                        in self.objPattern.picker.intermed.keys():
                        self.objPattern.picker.intermed\
                        [(int(proposed_obj.centre[0]),\
                          int(proposed_obj.centre[1]))] = 0
                    self.niter = self.niter + 1
                    return False
                                
    def tryDeath(self):
        """ Proposes a death move to the RJMCMC algorithm
        and tests if it can be accepted in the current configuration
        """
        if  len(self.objPattern.cubeMap.list_obj) == 0:
            self.posterior.minimale_action(self.niter)
            self.niter = self.niter + 1
            return False
        else:
            selected_centre = self.objPattern.picker.objChoice()
             
            if selected_centre == (-1, -1):
                self.posterior.minimale_action(self.niter)
                self.niter += 1
                return False
            else:
                selected_obj = self.objPattern.cubeMap.list_obj[selected_centre]
                 
                self.posterior.updateDeathProp(self.niter,
                                               self.objPattern.cubeMap,
                                               selected_obj)
                MHG = self.posterior.death_distribution(self.niter)
                if len(self.objPattern.cubeMap.list_obj) == 1:
                    MHG = MHG + np.log(2)
             
                if MHG >= np.log(self.prng.uniform(0,1)):
                    self.objPattern.removeObject(selected_obj)
                    self.posterior.updateDeath(selected_obj,
                                               self.objPattern.cubeMap)
             
                    if (int(selected_obj.centre[0]),
                        int(selected_obj.centre[1])) in\
                         self.objPattern.picker.bright.keys():
                        self.objPattern.picker.bright\
                        [(int(selected_obj.centre[0]),\
                          int(selected_obj.centre[1]))] = 0
                    if (int(selected_obj.centre[0]),
                        int(selected_obj.centre[1])) in \
                        self.objPattern.picker.intermed.keys():
                        self.objPattern.picker.intermed\
                        [(int(selected_obj.centre[0]),\
                          int(selected_obj.centre[1]))] = 0
                    self.niter = self.niter + 1
                    return True
                else:
                    self.posterior.minimale_action(self.niter)
                    self.niter = self.niter + 1
                    return False
 
    def tryPerturbation(self):
        """ Proposes a perturbation on an object axes of the current configuration and test if it can be accepted
            """
 
        selected_centre = self.objPattern.picker.objChoice()
        if selected_centre == (-1, -1):
            self.posterior.minimale_action(self.niter)
            self.niter += 1
            return False
        else:
            selected_obj = self.objPattern.cubeMap.list_obj[selected_centre]
            proposed_obj = self.objPattern.randomPerturbation(selected_obj)
             
            if  not proposed_obj.acceptability_geometry\
            (self.objPattern.cubeMap,selected_obj, self.HardCore, self.index):
                self.posterior.minimale_action(self.niter)
                self.niter = self.niter + 1
                return False
            else:
                proposed_obj.p_max = self.objPattern.cubeMap.maxMap\
                [int(proposed_obj.centre[0]), int(proposed_obj.centre[1])]
                 
                if proposed_obj.p_max >= self.max_thresh:   
                    n = self.posterior.config.n
                    tmp = np.dot(self.posterior.config.Lprop[0:n,0:n],
                                 self.posterior.config.DotVecSumProd_prop[0:n,:])
                    intensity = tmp[n-1,1: ] - self.mCubeBg*tmp[n-1,0]                    
                     
                    if max(intensity) >= 0:
                        qTest = -np.log(abs(
                                    min(self.objPattern.Shape_MaxRadius,
                                        selected_obj.axe1\
                                        * self.objPattern.MaxScaleFactor)\
                                    -max(self.objPattern.Shape_MinRadius,
                                          selected_obj.axe1\
                                        * self.objPattern.MinScaleFactor)))\
                                -np.log(abs(
                                    min(self.objPattern.Shape_MaxRadius,
                                        selected_obj.axe2\
                                        * self.objPattern.MaxScaleFactor)\
                                    -max(self.objPattern.Shape_MinRadius,
                                         selected_obj.axe2\
                                         * self.objPattern.MinScaleFactor)))
                        qPrev = -np.log(abs(
                                    min(self.objPattern.Shape_MaxRadius,
                                        proposed_obj.axe1\
                                        * self.objPattern.MaxScaleFactor)\
                                    -max(self.objPattern.Shape_MinRadius, 
                                          proposed_obj.axe1\
                                          * self.objPattern.MinScaleFactor)))\
                                -np.log(abs(
                                    min(self.objPattern.Shape_MaxRadius,
                                        proposed_obj.axe2 \
                                        * self.objPattern.MaxScaleFactor)\
                                    -max(self.objPattern.Shape_MinRadius,
                                         proposed_obj.axe2\
                                         * self.objPattern.MinScaleFactor)))      
                                         
                        MHG = self.posterior.updateGeometryProp\
                        (self.niter, self.objPattern.cubeMap, selected_obj,
                         proposed_obj) - qPrev + qTest
                         
                        if MHG >= np.log(self.prng.uniform(0,1)):
                            proposed_obj.positionList = selected_obj.positionList
                            self.objPattern.cubeMap.replace(self.posterior.updateGeometry())
                            self.list_pval.append(proposed_obj\
                                                  .adapted_p_value_wavelength\
                                                  (self.mCubeBg,
                                                   self.stdCubeBg**2,
                                                   self.objPattern.cube.data.data,
                                                   self.index))
                            self.niter = self.niter + 1
                            return True
                        else:
                            self.list_pval_H0.append(proposed_obj.\
                                                     adapted_p_value_wavelength\
                                                     (self.mCubeBg,
                                                      self.stdCubeBg**2,
                                                      self.objPattern.cube.data.data,
                                                      self.index))
                            self.posterior.minimale_action(self.niter)
                            self.niter = self.niter + 1
                            return False
                    else:
                        self.posterior.minimale_action(self.niter)
                        self.niter = self.niter + 1
                        return False
                else:
                    self.posterior.minimale_action(self.niter)
                    self.niter = self.niter + 1
                    return False

    def tryRotation(self):
        """ Proposes rotation of an object of the current configuration
        and test if it can be accepted
            """
         
        selected_centre = self.objPattern.picker.objChoice()
        if selected_centre == (-1, -1):
            self.posterior.minimale_action(self.niter)
            self.niter +=1
            return False
        else:
            selected_obj = self.objPattern.cubeMap.list_obj[selected_centre]
 
            proposed_obj = self.objPattern.randomRotation(selected_obj)
            if not proposed_obj.acceptability_geometry(self.objPattern.cubeMap,
                                                       selected_obj,
                                                       self.HardCore,
                                                       self.index):
                self.posterior.minimale_action(self.niter)
                self.niter = self.niter + 1
                return False
            else:
                proposed_obj.p_max = self.objPattern.cubeMap.maxMap\
                [int(proposed_obj.centre[0]), int(proposed_obj.centre[1])]
                 
                if proposed_obj.p_max >= self.max_thresh:
                    n = self.posterior.config.n
                    tmp = np.dot(self.posterior.config.Lprop[0:n,0:n],
                                 self.posterior.config.DotVecSumProd_prop[0:n,:])
                    intensity = tmp[n-1,1: ] - self.mCubeBg * tmp[n-1,0]

                    if max(intensity) >= 0:
                        if selected_obj.angle == 0:
                            qTest = 1
                            qPrev = 1
                        else:
                            qTest = -np.log(abs(min(np.pi/2,
                                                    selected_obj.angle\
                                                    * self.objPattern.MaxScaleFactor)\
                                                -max(0, selected_obj.angle\
                                                      * self.objPattern.MinScaleFactor)))
                            qPrev = -np.log(abs(min(np.pi/2,
                                                    proposed_obj.angle\
                                                    * self.objPattern.MaxScaleFactor)\
                                                -max(0, proposed_obj.angle\
                                                     * self.objPattern.MinScaleFactor)))
                 
                        MHG = self.posterior.updateGeometryProp(self.niter,
                                                                self.objPattern.cubeMap,
                                                                selected_obj,
                                                                proposed_obj)\
                                                                - qPrev + qTest            
                        if MHG >= np.log(self.prng.uniform(0,1)):
                            proposed_obj.positionList = selected_obj.positionList
                            self.list_pval.append(proposed_obj.\
                                                  adapted_p_value_wavelength\
                                                  (self.mCubeBg,
                                                   self.stdCubeBg**2,
                                                   self.objPattern.cube.data.data,
                                                   self.index))
                            self.objPattern.cubeMap.replace(self.posterior.updateGeometry())
                            self.niter = self.niter + 1
                            return True
                        else:
                            self.list_pval_H0.append(proposed_obj.\
                                                     adapted_p_value_wavelength\
                                                     (self.mCubeBg,
                                                      self.stdCubeBg**2,
                                                      self.objPattern.cube.data.data,
                                                      self.index))
                            self.posterior.minimale_action(self.niter)
                            self.niter = self.niter + 1
                            return False
                    else:
                        self.posterior.minimale_action(self.niter)
                        self.niter = self.niter + 1
                        return False
                else:
                    self.posterior.minimale_action(self.niter)
                    self.niter = self.niter + 1
                    return False
# 
    def tryTranslation(self):
        """ Proposes translation of an object of the current configuration
        and test if it can be accepted
            """
         
 
        selected_centre = self.objPattern.picker.objChoice()
        if selected_centre == (-1, -1):
            self.posterior.minimale_action(self.niter)
            self.niter +=1
            return False
        else:
            selected_obj = self.objPattern.cubeMap.list_obj[selected_centre]
             
            proposed_obj = self.objPattern.randomTranslation(selected_obj)
            if not proposed_obj.acceptability_geometry\
            (self.objPattern.cubeMap,selected_obj,
             self.HardCore, self.index):
                self.posterior.minimale_action(self.niter)
                self.niter = self.niter + 1
                return False
            else:
                n = self.posterior.config.n
                tmp = np.dot(self.posterior.config.Lprop[0:n,0:n],
                             self.posterior.config.DotVecSumProd_prop[0:n,:])
                intensity = tmp[n-1,1: ] - self.mCubeBg * tmp[n-1,0]                  
                if max(intensity) >= 0:
                    proposed_obj.p_max = self.objPattern.cubeMap.maxMap\
                    [int(proposed_obj.centre[0]), int(proposed_obj.centre[1])]
                     
                    if proposed_obj.p_max >= self.max_thresh:
                        qTest = -np.log(abs(
                                    min(self.objPattern.picker.LI,
                                        selected_obj.centre[0]\
                                        + self.objPattern.MaxTranslation)\
                                   -max(0, selected_obj.centre[0]-\
                                           self.objPattern.MaxTranslation)))\
                                -np.log(abs(
                                    min(self.objPattern.picker.LI,
                                        selected_obj.centre[1] \
                                        + self.objPattern.MaxTranslation)\
                                   -max(0, selected_obj.centre[1]\
                                          -self.objPattern.MaxTranslation)))
                        qPrev = -np.log(abs(
                                    min(self.objPattern.picker.LI,
                                        proposed_obj.centre[0]\
                                        + self.objPattern.MaxTranslation)\
                                   -max(0, proposed_obj.centre[0]-\
                                        self.objPattern.MaxTranslation)))\
                                -np.log(abs(
                                    min(self.objPattern.picker.LI,
                                        proposed_obj.centre[1]\
                                        + self.objPattern.MaxTranslation)\
                                   -max(0, proposed_obj.centre[1]\
                                        -self.objPattern.MaxTranslation)))
                 
                        MHG = self.posterior.updateGeometryProp(self.niter,self.objPattern.cubeMap,selected_obj, proposed_obj) - qPrev + qTest             
                        if MHG >= np.log(self.prng.uniform(0,1)):
                            proposed_obj.positionList = selected_obj.positionList
                            proposed_obj.positionList.append(proposed_obj.centre)
                            self.list_pval.append(proposed_obj.\
                                                  adapted_p_value_wavelength\
                                                  (self.mCubeBg,
                                                   self.stdCubeBg**2,
                                                   self.objPattern.cube.data.data,
                                                   self.index))
                            self.objPattern.cubeMap.replace(self.posterior.updateGeometry())
                            if (int(selected_obj.centre[0]),
                                int(selected_obj.centre[1])) != \
                                (int(proposed_obj.centre[0]),
                                 int(proposed_obj.centre[1])):
                                if (int(selected_obj.centre[0]),
                                    int(selected_obj.centre[1])) \
                                    in self.objPattern.picker.bright.keys():
                                    self.objPattern.picker.bright\
                                    [(int(selected_obj.centre[0]),
                                      int(selected_obj.centre[1]))] = 0
                                if (int(selected_obj.centre[0]),
                                    int(selected_obj.centre[1])) \
                                    in self.objPattern.picker.intermed.keys():
                                    self.objPattern.picker.intermed\
                                    [(int(selected_obj.centre[0]),
                                      int(selected_obj.centre[1]))] = 0
                            self.niter = self.niter + 1
                            return True
                        else:
                            self.list_pval_H0.append(proposed_obj\
                                                     .adapted_p_value_wavelength\
                                                     (self.mCubeBg,
                                                      self.stdCubeBg**2,
                                                      self.objPattern.cube.data.data,
                                                      self.index))
                            self.posterior.minimale_action(self.niter)
                            self.niter = self.niter + 1 
                            return False
                    else:
                        self.posterior.minimale_action(self.niter)
                        self.niter = self.niter + 1
                        return False
                else:
                    self.posterior.minimale_action(self.niter)
                    self.niter = self.niter + 1
                    return False         
         
    def tryHyperParameters(self):   
        self.posterior.updateHyperParameters(self.niter, self.objPattern.cubeMap)
        self.niter = self.niter + 1
        return True
    
    def get_catalog(self):
        """
        """
        list_obj = self.objPattern.backup.list_obj
        iteration = self.objPattern.backup.iteration
        m = self.posterior.m[iteration,:]
        wcs = self.objPattern.cube.wcs
        wave = self.objPattern.cube.wave
        LI = self.objPattern.cube.shape[1]
        COL = self.objPattern.cube.shape[2]
        LBDA = self.objPattern.cube.shape[0]
         
        X = self.objPattern.backup.matrix_configuration(LI, COL)
        index_obj = self.objPattern.backup.index_obj
         
        Gram = np.dot(X.T,X)
        invG = np.linalg.inv(Gram)
        A = np.dot(invG,X.T)
        w = np.empty((len(list_obj),LBDA))
        for l in xrange(LBDA):
            y = np.reshape(self.objPattern.cube.data[l,:,:] - m[l], LI*COL)
            w[:,l] = np.dot(A,y)
            
        sources = []

        for el in list_obj.values():
            # index
            n = index_obj[el.centre]
            #spectrum
            if LBDA>1:
                spe = {'tot': Spectrum(data=w[n,:], wave=wave)}
            else:
                spe = {}
            
            R_search = el.getRadius() + el.length_fsf
            pmin = max(0, int(el.centre[0]) - int(R_search))
            pmax = min(LI, int(el.centre[0]) + int(R_search) + 1)
            qmin = max(0, int(el.centre[1]) - int(R_search))
            qmax = min(COL, int(el.centre[1]) + int(R_search) + 1)
            im = el.profile_convolved
            sub_wcs = wcs[pmin:pmax, qmin:qmax]
            
            dec, ra = wcs.pix2sky([el.centre[0], el.centre[1]])[0]
            profile = Image(data=im, wcs=sub_wcs)
            
            extras = {}
            #Angle between the horizontal axis and the first ellipse axis
            extras['HIERARCH SELFI angle'] = \
            (el.angle,'Angle horizontal axis/first ellipse axis')
            #fwhm
            #step = self.objPattern.cube.get_step()
            extras['HIERARCH SELFI fwhm1'] = \
            (el.fwhm_1, 'Sersic profile FWHM along first axis')
            extras['HIERARCH SELFI fwhm2'] = \
            (el.fwhm_2, 'Sersic profile FWHM along first axis')
            #length_fsf
            extras['HIERARCH SELFI lfsf'] = \
            (el.length_fsf,'fsf radius')
            #p_max
            extras['HIERARCH SELFI pmax'] = \
            (el.p_max, 'Confidence index for a max-test')
            
            ima = {'white': profile}
            
            #create source object
            sources.append(Source(ID=n, ra=ra, dec=dec, origin='SELFI', 
                                  lines=[], spe=spe, ima=ima, extras=extras))
                           
        return sources

#     def print_result(self, filename, list_obj_white_image = {}):
#         """ Prints the catalog of detected galaxies in a .txt file and saves corresponding spectra and images in FITS format
#         
#         Parameters
#         ----------
#         filename             : string (.txt extension)
#                                File in which the object catalog will be stored
#         list_obj_white_image : dictionary with keys = IShape.centre and values = IShape instance
#                                List of the objects detected on the white image.
#                                Used to differentiate in the catalog if the object as been detected
#                                on the white image or with an emission line.
#             """
#         index = self.index
#         list_obj = self.objPattern.backup.list_obj
#         iteration = self.objPattern.backup.iteration
#         m = self.posterior.m[iteration,:]
#         sigma2 = self.posterior.sigma2[iteration,:]
#         wcs = self.objPattern.cube.wcs
#         wave = self.objPattern.cube.wave
#         LI = self.objPattern.cube.shape[1]
#         COL = self.objPattern.cube.shape[2]
#         LBDA = self.objPattern.cube.shape[0]
#         dim = self.objPattern.cube.get_range()
#         LBDAmin = dim[0,0]
#         step = self.objPattern.cube.get_step()[0]
#         
#         X = self.objPattern.backup.matrix_configuration(LI, COL)
#         index_obj = self.objPattern.backup.index_obj
#         
#         Gram = np.dot(X.T,X)
#         invG = np.linalg.inv(Gram)
#         A = np.dot(invG,X.T)
#         w = np.empty((len(list_obj),LBDA))
#         for l in xrange(LBDA):
#             y = np.reshape(self.objPattern.cube.data[l,:,:] - m[l], LI*COL)
#             w[:,l] = np.dot(A,y)
#         
#         file = open(filename, 'a')
#         file.write('#ID | RA | DEC | axe1 | axe2 | orientation | Sersic index | Determination step | lambda_max | confidence index # \n')
#         file.close()
#         
#         path = os.path.dirname(__file__)+'/p_val/'
#         pvalues = p_values(path+'val.fits', path+'cdf_maxTest.fits')
#         
#         for el in list_obj.values():
#             n = index_obj[el.centre]
#             if n < 10:
#                 N = '000'+str(n)
#             elif n < 100:
#                 N = '00'+str(n)
#             else:
#                 N = '0'+str(n)
#             im = el.profile_convolved
#             a,b = im.shape
#             pos = (a-1)/2
#             spe = Spectrum(data = w[n,:])
#             spe.wave = wave
#             spe.write('sp'+N+'.fits')
#             R_search = el.getRadius() + el.length_fsf
#             pmin = max(0,int(el.centre[0])-int(R_search))
#             pmax = min(LI,int(el.centre[0])+int(R_search)+1)
#             qmin = max(0,int(el.centre[1])-int(R_search))
#             qmax = min(COL,int(el.centre[1])+int(R_search)+1)
#             ima = Image(data = im)
#             ima.wcs = self.objPattern.cube[0,pmin:pmax, qmin:qmax].wcs.copy()
#             ima.write('im'+N +'.fits')
#             a = wcs.pix2sky([el.centre[0],el.centre[1]])
#             dec1 = a[0][0]
#             dec2 = a[0][1]
#             file = open(filename, 'a')
#             if el.centre in list_obj_white_image.keys():
#                 file.write(str(n) + ' ' + str(dec2) + ' ' + str(dec1)  +  ' ' + str(el.axe1)+  ' ' + str(el.axe2) +  ' ' + str(el.angle) +  ' ' + str(el.n) + ' ' + str(0) + ' ' + str(-1) + ' ' + str( pvalues.p_val_maxTest(el.p_max)) + '   \n' )
#             else:
#                 file.write(str(n) + ' ' + str(dec2) + ' ' + str(dec1)  +  ' ' + str(el.axe1)+  ' ' + str(el.axe2) +  ' ' + str(el.angle) +  ' ' + str(el.n) + ' ' + str(1) + ' ' + str(index[int(el.centre[0]),int(el.centre[1])]*step+LBDAmin) + ' ' + str( pvalues.p_val_maxTest(el.p_max)) + '   \n')
#             file.close()
