"""
Created on Tue Apr  2 10:00:37 2013

@author: Celine Meillier and Raphael Bacher

This file contains functors which describe posterior distribution : p(u,theta/y) 
on both the configuration and the parameter vector theta conditionnally
to the observation y. 

"""

import logging
import numpy as np
import multiprocessing
import code
import types
import math

from objconfig import CubeMap
from cholesky import Cholesky_python
from functor import addShapeToImageSF, rmvShapeToImageSF, CubeInteractions, Interactions


class iter_attributes(object):
    def __init__(self, posterior, delta2, nu, n, K,index = True):
        self.posterior = posterior
        self.LBDA = posterior.LBDA
        self.index = index
        self.delta2 = delta2
        self.K = K
        self.nu = nu
        self.B = self.posterior.sigma2[self.posterior.iteration]
        self.D = self.posterior.UnIdY
        self.E = self.posterior.YIdY
        self.F=self.posterior.config.DotVecSumProd[0:self.posterior.config.n,:]                    
        self.KxK=np.dot(K,K)
        self.H=self.posterior.cube
        self.Gshape=(nu+1)/2.
         
    def next(self):
        """Returns the values corresponding to the next wavelength"""
        if self.LBDA == 0:
            raise StopIteration
        self.LBDA -= 1
        if self.index is False:
            out = {}
            out['m'] = self.posterior.m[self.posterior.iteration, self.LBDA]
            out['sigma2'] = self.posterior.sigma2[self.posterior.iteration, self.LBDA]
            out['KxK'] = self.KxK
            out['KxF'] = np.dot(self.F[:,self.LBDA+1],self.K)
            out['HxH'] = self.posterior.HxH[self.LBDA]
            out['d'] = np.dot(self.F[:,self.LBDA+1],self.F[:,self.LBDA+1])
            out['D'] = self.D[self.LBDA]
            out['s2'] = self.delta2/self.nu*(self.E[self.LBDA] - out['d'] - self.delta2*(self.D[self.LBDA] - out['KxF'])**2)
            out['sumH'] = self.posterior.sumH[self.LBDA]
            out['mtilde'] = self.delta2*( out['D']- out['KxF'])
            out['sizeH'] = self.posterior.sizeH
            out['Gshape'] = self.Gshape
            out['mtilde_0'] = self.posterior.mtilde_0[self.LBDA]
            out['s2_0'] = self.posterior.s2_0[self.LBDA]
             
            return out
 
        else:
            out = {}
            out['m'] = self.posterior.m[self.posterior.iteration, self.LBDA]
            out['sigma2'] = self.posterior.sigma2[self.posterior.iteration, self.LBDA]
            out['KxK'] = self.KxK
            out['KxF'] = np.dot(self.F[:,self.LBDA+1],self.K)
            out['HxH'] = self.posterior.HxH[self.LBDA]
            out['d'] = np.dot(self.F[:,self.LBDA+1],self.F[:,self.LBDA+1])
            out['D'] = self.D[self.LBDA]
            out['s2'] = self.delta2/self.nu*(self.E[self.LBDA] - out['d'] - self.delta2*(self.D[self.LBDA] - out['KxF'])**2)
            out['sumH'] = self.posterior.sumH[self.LBDA]
            out['mtilde'] = self.delta2*( out['D']- out['KxF'])
            out['sizeH'] = self.posterior.sizeH
            out['Gshape'] = self.Gshape
            out['mtilde_0'] = self.posterior.mtilde_0[self.LBDA]
            out['s2_0'] = self.posterior.s2_0[self.LBDA]
            return ( out, self.LBDA)
 
    def __iter__(self):
        """Returns the iterator itself."""
        return self
 
def _process_attributes(arglist):
    try:
        obj = arglist[0]
         
        f = arglist[2]
         
        kargs = (arglist[4], arglist[3]['delta2'], arglist[3]['nu'],
                 arglist[3]['UnIdUn'], arglist[3]['n'], arglist[3]['q'],
                 arglist[3]['K'])
        if isinstance (f,types.FunctionType):
            obj_result = f(obj,*kargs)
        else:
            obj_result = getattr(obj, f)(*kargs)
        return (arglist[1],obj_result)
    except Exception as inst:
        raise type(inst) , str(inst)

class withoutIntensityReg(object):
    """ This class manages the WithoutIntensityReg object. Methods of this class make posssible to compute update of the configuration impact on the posterior density model developped in 2013 (cf. paper "Nonparametric Bayesian framework for detection of object configurations with large intensity dynamics in highly noisy hyperspectral data", C. Meillier, F. Chatelain, O. Michel, H. Ayasso, ICASSP 2014)
        
        Parameters
        ----------
        cube   : class 'mpdaf.obj.Cube'
                 Cube object containing the data
        NbMCMC : integer
                 Maximum number of iteration in the RJMCMC algorithm
        
        Attributes
        ----------
        
        cube             : array
                           Data cube
        LBDA             : int
                           Wavelength number in the data cube
        LI               : int
                           First spatial dimension of the data cube
        COL              : int
                           Second spatial dimension of the data cube        
        m                : array
                           Object containing the Markov chain of the sampled mean vector        
        sigma2           : array
                           Object containing the Markov chain of the sampled variance vector        
        cubeMap_tmp      : kernelDecorated.CubeMap
                           Object containing all the information about the object configuration.
                           Modification of the configuration are done in this temporary version of the CubeMap
                           before validation of the configuration perturbation.
        posterior_value  : array
                           Array containing the Markov chain of the evaluated posterior value      
        ratio_value      : array
                           Array containing the evaluated Metropolis-Hasting-Green ratio at each iteration
        cpu_count        : int
                           Number of CPU to use       
        UnIdUn           : int
                           Scalar product of the unit vector of size LIxCOL (number of pixels in one Image)       
        UnIdY            : array
                           Scalar product of the unit vector with the data 1' x [Y_1, ... , Y_LBDA]       
        YIdY             : array
                           Scalar product  [Y_1, ... , Y_LBDA]' x [Y_1, ... , Y_LBDA]      
        config           : Cholesky_python
                           Object that manages the iterative Cholesky decomposition of the Gram matrix X'X
                           containing information about the object configuration.
        tmp_config       : Cholesky_python
                           Temporary version of the iterative Cholesky decomposition of the Gram matrix X'X
                           for modification proposition        
        ratio            : float
                           Temporary variable to stock the Metropolis-Hastings-Green ratio
                           during the proposition and accepatation-reject process.       
        Delta_tmp        : float array
                           Array of size 1 x (2xLBDA + 1) containing the energy difference terms
                           between the augmented or downdated configuration and the previous one
        addShape         : functor.addShapeToImageSF
                           Functor initialized in the WithoutIntensityReg object,
                           used to apply the addition of a new object in the configuration.
        rmvShape         : functor.rmvShapeToImageSF
                           Functor initialized in the WithoutIntensityReg object,
                           used to apply the removing of an object of the configuration.
        cubeInteractions : functor.CubeInteractions
                           Functor initialized in the WithoutIntensityReg object,
                           used to evaluate the interaction of a proposed object with the data.
        interactions     : functor.Interactions
                           Functor initialized in the WithoutIntensityReg object,
                           used to evaluate the interaction of a proposed object
                           with the ones that constitute the current configuration. 
        q                : float
                           Parameter of the posterior distribution.      
        nu               : int
                           Parameter of the posterior distribution.        
        delta2_0         : float
                           Parameter of the posterior distribution evaluated for the initialization step.
        mtilde_0         : float
                           Parameter of the posterior distribution evaluated for the initialization step.
        s2_0             : float
                           Parameter of the posterior distribution evaluated for the initialization step.      
        sumH             : array
                           Scalar product of the unit vector with the data 1' x [Y_1, ... , Y_LBDA]       
        HxH              : array
                           l2-norm of each image of the data cube : 1' x [Y_1^2, ... , Y_LBDA^2]      
        sizeH            : int
                           Number of pixels of one image of the data cube.
        iteration        : int
                           Stock the current iteration number
        
        """
    def __init__(self, prng, cube, NbMCMC):
        """ Creates a WithoutIntensityReg objects. 
            
            Parameters
            ----------
            prng   : numpy.random.RandomState
                     instance of numpy.random.RandomState() with potential chosen seed.
            cube   : class 'mpdaf.obj.Cube'
                     Cube object containing the data
            NbMCMC : integer
                     Maximum number of iteration in the RJMCMC algorithm
            
        """
        self.logger = logging.getLogger('mpdaf corelib')
        d = {'class': 'withoutIntensityReg', 'method': '__init__'}
        self.logger.info('SELFI - Initializing the posterior density model', extra=d)
        
        self.prng = prng
        
        self.cube = cube.data.data
        dim = np.shape(cube)
        self.LBDA = dim[0]
        self.LI = dim[1]
        self.COL = dim[2]
        self.m = np.zeros((NbMCMC, self.LBDA))
        self.sigma2 = np.zeros((NbMCMC,self.LBDA))
        self.cubeMap_tmp = CubeMap(self.LI, self.COL, cube, preprocessing=None)
        self.posterior_value = np.zeros(NbMCMC)
        self.ratio_value = np.zeros(NbMCMC)
        
        
        self.UnIdUn = self.LI*self.COL
        self.UnIdY = np.sum(cube.data, axis=(1,2))
        self.YIdY = np.sum(cube.data**2, axis=(1,2))
        
        self.config = Cholesky_python(cube.shape[0])
        self.config.zeros()
        self.tmp_config = self.config.copy()
        self.ratio = 0.
        self.Delta_tmp = np.zeros((self.LBDA))
        
        self.addShape = addShapeToImageSF(self.cubeMap_tmp)
        self.rmvShape = rmvShapeToImageSF(self.cubeMap_tmp)
        self.cubeInteractions = CubeInteractions(self.cubeMap_tmp)
        self.interactions = Interactions(self.cubeMap_tmp)
        
        self.q = 1e3/(1+1e3)
        
        self.nu = self.LI * self.COL -1.
        self.delta2_0 = 1./self.UnIdUn
        
        self.mtilde_0= self.delta2_0 * self.UnIdY
        self.s2_0 = self.delta2_0/self.nu * (self.YIdY - self.delta2_0*self.UnIdY)
        
        if self.LBDA == 1:
            self.cpu_count = 1
        else:
            self.cpu_count = multiprocessing.cpu_count() - 1
        
        self.sumH = np.sum(cube.data.data, axis=(1,2))
        self.HxH = np.sum(cube.data.data**2, axis=(1,2))
        self.sizeH = self.LI*self.COL        
        
        # Initialization of the model parameters 
        
        res = np.zeros(self.LBDA)
        for l in np.arange(0,self.LBDA):
            a = cube[l,:,:].background(niter = 3)
            scale = 1./cube.fscale
            self.m[0,l] = a[0]*scale
            self.sigma2[0,l] = (a[1]*scale)**2 
            if (a[1]*scale)**2 < 1e-6 : 
                code.interact(local=locals())
        
            b = (np.reshape(cube.data[l,:,:], self.LI*self.COL)-self.m[0,l])
            d = np.dot(b.T,b)
            res[l] = -1./(2*self.sigma2[0,l])*d - (self.LI*self.COL/2. +1)*np.log(self.sigma2[0,l])
        self.posterior_value[:] = np.sum(res) +np.log(self.q)


        # temporary variables
        self.iteration = 0
        
    def initPosterior(self, m,sigma2):
        """ Initializes the modelParameter objects.
            
        Parameters
        ----------
        m         : Float array
                    1xLBDA array containing the mean vector
        sigma2    : Float array
                    1xLBDA array containing the variance vector
        """
        self.m[0,:] = m
        self.sigma2[0,:] = sigma2

    def birth_distribution(self,iteration):
        """ Computes the Metropolis-Hastings-Green ratio after the Cholesky
        decomposition update for birth move.
             
        Parameters
        ----------
        iteration : int
                    Iteration number
            """
        self.ratio = sum(self.Delta_tmp) + np.log(self.q)
        return self.ratio
     
    def death_distribution(self,iteration):
        """ Computes the Metropolis-Hastings-Green ratio after the Cholesky
        decomposition update for death move.
             
        Parameters
        ----------
        iteration : int
                    Iteration number
        """
        self.ratio = sum(self.Delta_tmp) + np.log(1./self.q)
        return self.ratio     
         
    def posterior_update_hyperparameters(self,iteration,cubeMap):
        """ Computes the posterior density for a hyperparameter update move
             
        Parameters
        ----------
        iteration : int
                    Iteration number
        cubeMap   : class 'kernelDecorated.CubeMap'
                    CubeMap object containing the information about the corresponding configuration
        """
         
        n = self.config.n
        self.posterior_value[iteration] = -(self.LI * self.COL / 2 + 1)\
        * np.sum(np.log(self.sigma2[iteration,:]))+ np.sum(self.Delta_tmp)\
        + math.log(math.factorial(n)) + (n+1)*np.log(self.q)
        self.iteration +=1
     
    def posterior_update_birth(self,iteration):
        """ Computes the posterior density for a birth move
             
        Parameters
        ----------
        iteration : int
                    Iteration number
        """
        self.ratio = np.sum(self.Delta_tmp) + np.log(self.q)
        if iteration == 0 :
            self.posterior_value[iteration] = self.posterior_value[0] \
                                            + self.ratio  \
                                            + np.log(self.config.n)
        else :
            self.posterior_value[iteration] = self.posterior_value[iteration-1]\
                                            + self.ratio\
                                            + np.log(self.config.n)
 
    def posterior_update_death(self,iteration):
        """ Computes the posterior density for a death move
             
        Parameters
        ----------
        iteration : int
                    Iteration number
        """
        self.ratio = sum(self.Delta_tmp) + np.log(1./self.q)
        self.ratio_value[iteration] = self.ratio
        if self.config.n > 1:
            self.posterior_value[iteration] = self.posterior_value[iteration-1]\
             + self.ratio - np.log(self.config.n-1)
        else:
            self.posterior_value[iteration] = self.posterior_value[iteration-1] \
            + self.ratio
 
    def posterior_update_geometry(self,iteration):
        """ Computes the posterior density for a simple geometrical move
           
        Parameters
        ----------
        iteration : int
                    Iteration number
        """
        self.ratio = np.sum(self.Delta_tmp)
        self.posterior_value[iteration] =\
        self.posterior_value[iteration-1] + self.ratio

    def minimale_action(self, iteration):
        if iteration > 0:
            self.sigma2[iteration, :] = self.sigma2[iteration-1, :]
            self.m[iteration, :] = self.m[iteration-1, :]
            
        
    def updateBirthProp(self, iteration, cubeMap, obj):
        """ Proposes an update of the configuration with the new object 'obj'
            
        Parameters
        ----------
        iteration : int
                    Iteration number
        cubeMap   : class 'kernelDecorated.CubeMap'
                    CubeMap object containing the information about the corresponding configurations
        obj       : class 'IShape'
                    Proposed object
        """
        self.iteration = iteration
        if iteration >0:
            self.sigma2[iteration,:] = self.sigma2[iteration-1,:]
            self.m[iteration,:] = self.m[iteration-1,:]
        self.interactions.zeros(cubeMap)
        obj.applyToPoints(self.interactions, self.LI, self.COL)
        v =self.interactions.tab_return(obj.centre)
        
        self.cubeInteractions.zeros(cubeMap)
        obj.applyToPoints(self.cubeInteractions, self.LI, self.COL)
        newSumAndProd =self.cubeInteractions.tab_return()
        
        self.config.propAugment(v, newSumAndProd )
        
        self.Delta_tmp[0:self.LBDA] = ( (self.m[iteration,0:self.LBDA])**2*self.config.delta[0] + self.config.delta[1:self.LBDA+1] -2*self.config.delta[self.LBDA+1:2*self.LBDA+1]*(self.m[iteration,0:self.LBDA]) )/(2.*(self.sigma2[iteration,0:self.LBDA]))

    def updateBirth(self, obj, cubeMap):
        """ Validates the birth proposition
             
        Parameters
        ----------
        cubeMap   : class 'kernelDecorated.CubeMap'
                    CubeMap object containing the information about the corresponding configurations
        obj       : class 'IShape'
                    Proposed object
        """
         
        self.config.confAugment(obj.centre, cubeMap)        
 
    def updateDeathProp(self, iteration, cubeMap, obj):
        """ Proposes an update of the configuration with the death of the object 'obj'
             
        Parameters
        ----------
        iteration : int
                    Iteration number
        cubeMap   : class 'kernelDecorated.CubeMap'
                    CubeMap object containing the information
                    about the corresponding configurations
        obj       : class 'IShape'
                    Proposed object
            """
        self.iteration = iteration
        if iteration > 0:
            self.sigma2[iteration, :] = self.sigma2[iteration-1, :]
            self.m[iteration, :] = self.m[iteration-1, :]
        self.config.propRemove(obj.centre, cubeMap)
        self.Delta_tmp[0:self.LBDA] = ( (self.m[iteration,0:self.LBDA])**2*self.config.delta[0] + self.config.delta[1:self.LBDA+1] -2*self.config.delta[self.LBDA+1:2*self.LBDA+1]*(self.m[iteration,0:self.LBDA]) )/(2.*(self.sigma2[iteration,0:self.LBDA]))
 
    def updateDeath(self,obj,cubeMap):
        """ Validates the death proposition
             
        Parameters
        ----------
        cubeMap   : class 'kernelDecorated.CubeMap'
                    CubeMap object containing the information about the corresponding configurations
        obj       : class 'IShape'
                    Proposed object
            """
        self.config.confRemove(obj.centre, cubeMap)
 
    def updateGeometryProp(self, iteration, cubeMap, selected_obj,
                           modified_obj):
        """ Proposes an update of the configuration with the modification
         of the object 'selected_obj'
          
        Parameters
        ----------
        iteration    : int
                       Iteration number
        cubeMap      : class 'kernelDecorated.CubeMap'
                       CubeMap object containing the information about the corresponding configurations
        obj          : class 'IShape'
                       Selected object
        modified_obj : class 'IShape'
                       Modified object
            """
        self.iteration = iteration
        self.tmp_config.replace(self.config)
        if iteration >0:
            self.sigma2[iteration, :] = self.sigma2[iteration-1, :]
            self.m[iteration, :] = self.m[iteration-1, :]
        self.cubeMap_tmp.replace(cubeMap)
        
        # -- Death step -- #
        self.tmp_config.propRemove(selected_obj.centre, self.cubeMap_tmp)         
        self.Delta_tmp[0:self.LBDA] = ((self.m[iteration, 0:self.LBDA])**2\
                                       *self.tmp_config.delta[0]\
                                       + self.tmp_config.delta[1:self.LBDA+1]\
                                       -2*self.tmp_config.delta[self.LBDA+1:\
                                                                2*self.LBDA+1]\
                                       *(self.m[iteration, 0:self.LBDA]))\
                                       /(2.*(self.sigma2[iteration, 0:self.LBDA]))
         
        tmpVect = self.Delta_tmp[0:self.LBDA].copy()   
        deathRatio = sum(self.Delta_tmp)
        self.tmp_config.confRemove(selected_obj.centre, self.cubeMap_tmp)
        self.rmvShape.zeros(self.cubeMap_tmp)
        self.cubeMap_tmp.removeObject(selected_obj, self.rmvShape)
         
        # -- Birth step -- #
        self.interactions.zeros(self.cubeMap_tmp)    
        modified_obj.applyToPoints(self.interactions, self.LI, self.COL)
        v =self.interactions.tab_return(modified_obj.centre)
        self.cubeInteractions.zeros(self.cubeMap_tmp)
        modified_obj.applyToPoints(self.cubeInteractions, self.LI, self.COL)  
        newSumAndProd =self.cubeInteractions.tab_return()   
        self.tmp_config.propAugment(v, newSumAndProd )
        self.Delta_tmp[0:self.LBDA] = ((self.m[iteration,0:self.LBDA])**2\
                                       *self.tmp_config.delta[0]\
                                       + self.tmp_config.delta[1:self.LBDA+1]\
                                       -2*self.tmp_config.delta[self.LBDA+1:2*self.LBDA+1]\
                                       *(self.m[iteration,0:self.LBDA]))\
                                       /(2.*(self.sigma2[iteration,0:self.LBDA]))    
        birthRatio = sum(self.Delta_tmp)
        self.tmp_config.confAugment(modified_obj.centre, self.cubeMap_tmp)
        self.addShape.zeros(self.cubeMap_tmp)
        self.cubeMap_tmp.addObject(modified_obj, self.addShape)
         
        MHG = deathRatio + birthRatio
        self.ratio_value[iteration] = MHG
        self.Delta_tmp[0:self.LBDA] = self.Delta_tmp[0:self.LBDA] + tmpVect
        return  MHG       
         
    def updateGeometry(self):
        """ Validates the object modification proposition
 
            """
        self.config.replace(self.tmp_config)
        return self.cubeMap_tmp

    def updateHyperParameters(self,iteration,cubeMap):
        """ Samples the model parameters value (m and sigma2).
          
        Parameters
        ----------
        iteration    : int
                       Iteration number
        cubeMap      : class 'kernelDecorated.CubeMap'
                       CubeMap object containing the information
                       about the corresponding configurations
            """
         
        self.iteration = iteration
        if len(cubeMap.list_obj) == 0:
            delta2 = self.delta2_0
        else:
            delta2 = 1./(self.UnIdUn -
                         np.sum(self.config.DotVecSumProd\
                                [0:len(cubeMap.list_obj), 0]**2))
         
        if self.LBDA > 1:
            f = evaluation_Delta
            Delta2  = self.loop_wavelength_multiprocessing(f, cpu=self.cpu_count,
                                                           delta2=delta2,
                                                           nu=self.nu,
                                                           UnIdUn=self.UnIdUn,
                                                           n=self.config.n,
                                                           q=self.q,
                                                           K=self.config.DotVecSumProd[0:self.config.n,0])
            n = self.config.n 
            self.m[iteration,:] = Delta2[:,1].T
            self.sigma2[iteration,:] = Delta2[:,2].T
            self.Delta_tmp[:] = Delta2[:,0]
            del Delta2
        else:
            if len(cubeMap.list_obj) == 0:
                mtilde = self.mtilde_0
                s2 = self.s2_0
            else:
                mtilde = delta2 * (self.UnIdY[0]
                                   - np.sum(self.config.DotVecSumProd\
                                            [0:len(cubeMap.list_obj),0]\
                                            *self.config.DotVecSumProd\
                                            [0:len(cubeMap.list_obj),1]))
                s2 = delta2 / self.nu * (self.YIdY[0]
                                         - np.sum(self.config.DotVecSumProd\
                                                  [0:len(cubeMap.list_obj), 1]\
                                                  *self.config.DotVecSumProd\
                                                  [0:len(cubeMap.list_obj), 1])\
                    - delta2*(self.UnIdY[0] - np.sum(self.config.DotVecSumProd\
                                                     [0:len(cubeMap.list_obj),0]\
                                                     *self.config.DotVecSumProd\
                                                     [0:len(cubeMap.list_obj), 1]))**2)
                if s2 <0 :
                    s2 = abs(s2)
            tmp = self.prng.standard_t(self.nu)
            self.m[iteration,0] = tmp * np.sqrt(s2) + mtilde
            self.sigma2[iteration,0] = 1/self.prng.gamma((self.LI * self.COL)/2,
                                                         scale=(2*delta2)*1/(self.nu*s2 + (self.m[iteration,0] - mtilde)**2))
            b = -1./(2.*(self.sigma2[iteration,0]))
            c = sum(sum(( self.cube[0,:,:] - self.m[iteration,0])**2))
            d = sum(self.config.DotVecSumProd[0:self.config.n, 1]**2)
            e = self.m[iteration,0]**2 * sum(self.config.DotVecSumProd[0:self.config.n, 0]**2)
            f = 2*self.m[iteration,0]* sum(self.config.DotVecSumProd[0:self.config.n, 0]*self.config.DotVecSumProd[0:self.config.n, 1])
             
            a = b * ( c - ( d + e  - f ))
            self.Delta_tmp[:] = a
            
    def loop_wavelength_multiprocessing(self, f, cpu=None, **kwargs):
        """ Loops over all wavelengths to apply a function/method.
            Returns the resulting values of m, sigma2 and posterior_value.
            Multiprocessing is used.
             
        Parameters
        ----------
         f     : function defined in the posterior module (such as posterior.evaluation_Delta)
                 Function that the first argument is a dictionary containing the posterior parameters value needed to evaluate it.
         cpu   : int
                 Number of available cpu
         kargs : kargs
                 kargs can be used to set function arguments.
            """
 
        processlist = list()
         
        if isinstance (f, types.MethodType):
            f = f.__name__
        for dico,k in iter_attributes(self, kwargs['delta2'], kwargs['nu'], kwargs['n'], kwargs['K'], index=True):
            processlist.append([ dico,k,f,kwargs, self.prng])
        
        processresult = map(_process_attributes, processlist)
 
        result = np.empty((self.LBDA,3))
        for k,out in processresult:
            result[k,:] = out
 
        return result

def evaluation_Delta(dic, prng, delta2 = 1., nu = 1., UnIdUn = 1., n = 0,
                     q = 1., K = None, dotKK = None):
    """ Samples model parameters (m and sigma2) and evaluates the posterior
    density value define in "Nonparametric Bayesian framework for detection
    of object configurations with large intensity dynamics in highly noisy
    hyperspectral data".
         
    Parameters
    ----------
    dic    : Dictionary
             Dictionary returns by the posterior.iter_attributes methods
    delta2 : float
             WithoutIntensityReg.delta2 parameter
    nu     : int
             WithoutIntensityReg.nu parameter
    UnIdUn : int
             WithoutIntensityReg.UnIdUn parameter
    q      : float
             WithoutIntensityReg.q parameter
    K      : float array
             L'.X'.[1, Y_1, ... , Y_K]
    dotKK  : float array
             [1, Y_1, ... , Y_K]'.X.LL'.X'.[1, Y_1, ... , Y_K]
     
    Returns
    -------
    out : 1x3 array
          1x3 array contaning the mean, the variance,
          and the energy difference evaluated at a given wavelength.
    """
    d = dic['d']
    sumH = dic['sumH']
    s2 = dic['s2']
    Gshape = dic['Gshape']
    I = dic['mtilde_0']
    J = dic['s2_0']
    HxH = dic['HxH']
    KxK = dic['KxK']
    KxF = dic['KxF']
    sizeH = dic['sizeH']
    mtilde =  dic['mtilde']
     
    if n == 0:
        mtilde = I 
        s2 = J
    else:
        if s2 <0 :
            s2 = abs(s2)
     
    tmp = prng.standard_t(nu)
    A = tmp*np.sqrt(s2) + mtilde
    B = 1./ prng.gamma(Gshape, scale = (2*delta2)*1./(nu*s2 + (A - mtilde)**2))
 
    b = -1./(2.*B)
    c = HxH - 2*A* sumH + sizeH*A**2
 
    e=(A**2)*KxK
    f = 2.*A* KxF
    a = b * ( c - ( d + e  - f ))
    res = np.empty((1,3))
 
    res[0,0] = a
    res[0,1] = A
    res[0,2] = B
    return res        
