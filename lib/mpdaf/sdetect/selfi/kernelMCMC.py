import datetime
import time
import numpy as np
import logging

class KernelMCMC(object):
    """ This class defines iterative RJMCMC algorithm used to sample the model parameters and the object configuration
        
        Parameters
        ----------
        kernel            : class KernelBayesian
                            Kernel object defining the marked point process.
        NbMCMC            : int
                            Maximum number of iteration in the RJMCMC algorithm
        ProbaBirthDeath   : float
                            Weight given to the birth and death move
        ProbaTranslation  : float
                            Weight given to the translation move
        ProbaRotation     : float
                            Weight given to the rotation move    
        ProbaPerturbation : float
                            Weight given to the perturbation move
        ProbaHyperUpdate  : float
                            Weight given to the parameters and hyperparameters sampling move 
        ExpMaxNbShapes    : int
                            Maximum number of detected objects (to limit the size of memory allocation)
        
        Attributes
        ----------
        
        pointProcess                 : class KernelBayesian
                                       Kernel object defining the marked point process.
        NbMCMC                       : int
                                       Maximum number of iteration in the RJMCMC algorithm
        ProbaBirthDeath              : float
                                       Weight given to the birth and death move
        ProbaTranslation             : float
                                       Weight given to the translation move
        ProbaRotation                : float
                                       Weight given to the rotation move    
        ProbaPerturbation            : float
                                       Weight given to the perturbation move
        ProbaHyperUpdate             : float
                                       Weight given to the parameters and hyperparameters sampling move 
        ExpMaxNbShapes               : int
                                       Maximum number of detected objects (to limit the size of memory allocation)
        nbBirthProposition           : int
                                       Counts the number of birth move proposed
        nbDeathProposition           : int
                                       Counts the number of death move proposed
        nbTranslationProposition     : int
                                       Counts the number of translation move proposed
        nbRotationProposition        : int
                                       Counts the number of rotation move proposed
        nbPerturbationProposition    : int
                                       Counts the number of perturbation move proposed 
        nbHyperParameterProposition  : int
                                       Counts the number of hyperparameter sampling move
        ratioBirthAcceptation        : int
                                       Counts the number of birth move accepted
        ratioDeathAcceptation        : int
                                       Counts the number of death move accepted
        ratioTranslationAcceptation  : int
                                       Counts the number of translation move accepted
        ratioRotationAcceptation     : int
                                       Counts the number of rotation move accepted
        ratioPerturbationAcceptation : int
                                       Counts the number of perturbation move accepted
        currentIter                  : int
                                       Iteration number
        posterior_distribution       : float
                                       Current value of the maximum a posteriori estimate of the posterior distribution      
    """
        

    def __init__(self, prng, kernel, NbMCMC, ProbaBirth, ProbaBirthDeath, ProbaTranslation, ProbaRotation, ProbaPerturbation, ProbaHyperUpdate, ExpMaxNbShapes):
        """ Creates a KernelMCMC object.
            
            Parameters
            ----------
            prng              : numpy.random.RandomState
                                instance of numpy.random.RandomState() with potential chosen seed.
            kernel            : class KernelBayesian
                                Kernel object defining the marked point process.
            NbMCMC            : int
                                Maximum number of iteration in the RJMCMC algorithm
            ProbaBirth        : float in [0,1]
                                Proportion of birth proposition in the birth and death move
            ProbaBirthDeath   : float
                                Weight given to the birth and death move
            ProbaTranslation  : float
                                Weight given to the translation move
            ProbaRotation     : float
                                Weight given to the rotation move    
            ProbaPerturbation : float
                                Weight given to the perturbation move
            ProbaHyperUpdate  : float
                                Weight given to the parameters and hyperparameters sampling move 
            ExpMaxNbShapes    : int
                                Maximum number of detected objects (to limit the size of memory allocation)
        """
        self.logger = logging.getLogger('mpdaf corelib')
        d = {'class': 'KernelMCMC', 'method': '__init__'}
        self.logger.info('SELFI - Defining the parameters of the Markov Chain Monte Carlo (MCMC) method', extra=d)
        self.logger.info('\t\t maximum number of iteration in the RJMCMC algorithm = %d'%NbMCMC, extra=d)
        self.logger.info('\t\t birth probability = %0.1f'%ProbaBirth, extra=d)
        self.logger.info('\t\t birth and death probability = %0.1f'%ProbaBirthDeath, extra=d)
        self.logger.info('\t\t translation probability = %0.1f'%ProbaTranslation, extra=d)
        self.logger.info('\t\t rotation probability = %0.1f'%ProbaRotation, extra=d)
        self.logger.info('\t\t pertubation probability = %0.1f'%ProbaPerturbation, extra=d)
        self.logger.info('\t\t sampling probability = %0.1f'%ProbaHyperUpdate, extra=d)
        self.logger.info('\t\t maximum number of detected objects = %d'%ExpMaxNbShapes, extra=d)
        self.prng = prng
        
        self.pointProcess = kernel
        self.NbMCMC = NbMCMC
        self.ProbaBirth = ProbaBirth
        self.ProbaBirthDeath = ProbaBirthDeath
        self.ProbaTranslation = ProbaTranslation
        self.ProbaRotation = ProbaRotation
        self.ProbaPerturbation = ProbaPerturbation
        self.ProbaHyperUpdate = ProbaHyperUpdate
        self.ExpMaxNbShapes = ExpMaxNbShapes

        self.nbBirthProposition = 0
        self.nbDeathProposition = 0
        self.nbTranslationProposition = 0
        self.nbRotationProposition = 0
        self.nbPerturbationProposition = 0
        self.nbHyperParameterProposition = 0
        
        self.ratioBirthAcceptation = 0
        self.ratioDeathAcceptation = 0
        self.ratioTranslationAcceptation = 0
        self.ratioRotationAcceptation = 0
        self.ratioPerturbationAcceptation = 0
        
        
        self.currentIter = 0
        
        self.posterior_distribution = - 1000000000
        
#     def addObject(self,obj):
#         """ Adds an object to the current configuration
#            
#            Parameters
#            ----------
#            obj : class 'IShape'
#                  Object to add
#         """
#         self.pointProcess.addObject(obj)
#         
#         
#     
#     def removeObject(self,obj):
#         """ Removes an object from the current configuration
#             
#             Parameters
#             ----------
#             obj : class 'IShape'
#                   Object to remove
#             """
#         self.pointProcess.removeObject()
#     
#     
#     
    def plotConfig_support(self, image):
        """ Plots the object configuration on the background image.
         
        Parameters
        ----------
        image : float array
                    Array containing the background image
        """
        self.pointProcess.plotConfig_support(image)
         
    def plotConfig_marker(self, li_x, li_y):
        """ Plots the position of the objects center on a map.
             
            Parameters
            ----------
            li_x : list of float
                   list of x coordinates
            li_y : list of float
                   list of y coordinates
            """
        self.pointProcess.plotConfig_marker(li_x,li_y)
#     
#     
#     def print_result(self, filename, list_obj_white_image = {}):
#         """ Prints the catalog of detected galaxies in a .txt file and saves corresponding spectra and images in FITS format
#         
#         Parameters
#         ----------
#         filename             : string (.txt extension)
#                                file in which the object catalog will be stored
#         list_obj_white_image : dictionary with keys = IShape.centre and values = IShape instance
#                                list of the objects detected on the white image.
#                                Used to differentiate in the catalog if the object has been detected
#                                on the white image or with an emission line.
#             """
#         self.pointProcess.print_result(filename, list_obj_white_image = list_obj_white_image)
        
        
    def initConfig(self,list_obj, m, sigma2, init_picker=True):
        """ Initializes the Markov chain with an existant configuration of objects.
            
            Parameters
            ----------
            list_obj : List of IShape objects
                       List of IShape objects of the initialisation configuration
            m        : float array
                       1xLBDA vector of mean values
            sigma2   : float array
                       1xLBDA vector of variance values
        """
        self.pointProcess.posterior.initPosterior(m, sigma2)
        for el in list_obj:
            #if init_picker:
            #    self.pointProcess.objPattern.picker.initConfig[el.centre] = el
            self.pointProcess.initBirth(el)
            self.pointProcess.posterior.posterior_update_birth(self.currentIter)
            if self.pointProcess.posterior.posterior_value[self.currentIter]  >= self.posterior_distribution:
                self.pointProcess.objPattern.backup.update(self.pointProcess.objPattern.cubeMap, self.currentIter)
                self.posterior_distribution = self.pointProcess.posterior.posterior_value[self.currentIter]
            if (int(el.centre[0]), int(el.centre[1])) in self.pointProcess.objPattern.picker.bright.keys():
                del self.pointProcess.objPattern.picker.bright[(int(el.centre[0]), int(el.centre[1]))]
            if (int(el.centre[0]), int(el.centre[1])) in self.pointProcess.objPattern.picker.intermed.keys():
                del self.pointProcess.objPattern.picker.intermed[(int(el.centre[0]), int(el.centre[1]))]
            self.currentIter = self.currentIter + 1
            
    def MCMC(self, initialIndex = -1, stop = -1, posterior_verif = False):
        """ Begins the RJMCMC algorithm. This function can be stopped by a ctrl+C and restarted by call again the function.
             
            Parameters
            ----------
            initialIndex : int
                           Iteration index to begin the RJMCMC sampling.
                           Useful for restarting the method at a given iteration.
            stop         : int
                           Iteration index to stop the RJMCMC sampling.
        """
        d = {'class': 'KernelMCMC', 'method': 'MCMC'}
        self.logger.info('SELFI - Running the RJMCMC algorithm - %s'%datetime.datetime.ctime(datetime.datetime.now()), extra=d)
        if stop == -1:
            stop = self.NbMCMC
        if initialIndex == -1:
            initialIndex = self.currentIter
        beginning = time.time()
         
        proba_cumsum = np.cumsum( [self.ProbaBirthDeath, self.ProbaTranslation,
                                   self.ProbaRotation, self.ProbaPerturbation,
                                   self.ProbaHyperUpdate] )
        u_max = np.sum( [self.ProbaBirthDeath,  self.ProbaTranslation,
                         self.ProbaRotation, self.ProbaPerturbation,
                         self.ProbaHyperUpdate] )
           
        while ( (self.currentIter < stop) and 
                (len(self.pointProcess.objPattern.cubeMap.list_obj) < self.ExpMaxNbShapes) and
                (self.currentIter - self.pointProcess.objPattern.backup.iter_stop_criterion) < 
                int(3.*len(self.pointProcess.objPattern.cubeMap.Map[0])/(self.ProbaBirthDeath*self.ProbaBirth/u_max))) :
         
            if self.currentIter%1000 ==0:
                self.logger.info('/t/t %d iterations - %s'%(self.currentIter, datetime.datetime.ctime(datetime.datetime.now())), extra=d)
                
            if len(self.pointProcess.objPattern.cubeMap.list_obj) == 0 or self.currentIter == 0 :
                u = 0
            else:
                u = self.prng.uniform(0,u_max)                        
             
            if u <= proba_cumsum[0]: # Birth or Death move
                if len(self.pointProcess.objPattern.cubeMap.list_obj) == 0:
                    v = 1
                elif len(self.pointProcess.objPattern.cubeMap.list_obj) ==  self.ExpMaxNbShapes:
                    v = 0
                else:
                    v = self.prng.uniform(0,1)
                if v >= 0.5: # Birth proposition:
                    self.nbBirthProposition = self.nbBirthProposition + 1
                    if self.pointProcess.tryBirth():
                        self.ratioBirthAcceptation = self.ratioBirthAcceptation + 1
                        self.pointProcess.posterior.posterior_update_birth(self.currentIter)
                        if self.pointProcess.posterior.posterior_value[self.currentIter] >= 1e-20:
                            self.logger.warning('Problem birth', extra=d)
                        if self.pointProcess.posterior.posterior_value\
                        [self.currentIter]  >= self.posterior_distribution:
                            self.pointProcess.objPattern.backup.update\
                            (self.pointProcess.objPattern.cubeMap,
                             self.currentIter)
                            self.posterior_distribution = self.pointProcess.\
                            posterior.posterior_value[self.currentIter]
                    else:
                        self.pointProcess.posterior.posterior_value\
                        [self.currentIter] = self.pointProcess.posterior.\
                        posterior_value[self.currentIter - 1] 
 
                else: # Death proposition
                    self.nbDeathProposition = self.nbDeathProposition + 1
                    if self.pointProcess.tryDeath():
                        self.ratioDeathAcceptation = self.ratioDeathAcceptation + 1
                        self.pointProcess.posterior.posterior_update_death(self.currentIter)
                        if self.pointProcess.posterior.posterior_value[self.currentIter] >= 1e-20:
                            self.logger.warning('Problem death', extra=d)
                        if self.pointProcess.posterior.posterior_value[self.currentIter] >= self.posterior_distribution:
                            self.pointProcess.objPattern.backup.update(self.pointProcess.objPattern.cubeMap,
                                                                       self.currentIter)
                            self.posterior_distribution = self.pointProcess.posterior.posterior_value[self.currentIter]
                    else:
                        self.pointProcess.posterior.posterior_value[self.currentIter] =\
                        self.pointProcess.posterior.posterior_value[self.currentIter-1]
                     
            elif u <= proba_cumsum[1]: # Translation move:
                self.nbTranslationProposition = self.nbTranslationProposition +1
                if self.pointProcess.tryTranslation():
                    self.ratioTranslationAcceptation =\
                    self.ratioTranslationAcceptation + 1
                    self.pointProcess.posterior.posterior_update_geometry(self.currentIter)
                    if self.pointProcess.posterior.posterior_value[self.currentIter] >= 1e20:
                        self.logger.warning('Anormal operation in translation move', extra=d)
                    if self.pointProcess.posterior.posterior_value[self.currentIter] >= self.posterior_distribution:
                        self.pointProcess.objPattern.backup.update(self.pointProcess.objPattern.cubeMap, self.currentIter)
                        self.posterior_distribution = self.pointProcess.posterior.posterior_value[self.currentIter]
                else:
                    self.pointProcess.posterior.posterior_value[self.currentIter] =\
                    self.pointProcess.posterior.posterior_value[self.currentIter-1] 
 
            elif u <= proba_cumsum[2]: # Rotation move:
                self.nbRotationProposition = self.nbRotationProposition + 1
                if self.pointProcess.tryRotation():
                    self.ratioRotationAcceptation = self.ratioRotationAcceptation + 1
                    self.pointProcess.posterior.posterior_update_geometry(self.currentIter)
                    if self.pointProcess.posterior.posterior_value[self.currentIter] >= 1e20:
                        self.logger.warning('Anormal operation in rotation move', extra=d)
                    if self.pointProcess.posterior.posterior_value[self.currentIter]  >= self.posterior_distribution:
                        self.pointProcess.objPattern.backup.update(self.pointProcess.objPattern.cubeMap, self.currentIter)
                        self.posterior_distribution = self.pointProcess.posterior.posterior_value[self.currentIter]          
                else:
                    self.pointProcess.posterior.posterior_value[self.currentIter] =\
                    self.pointProcess.posterior.posterior_value[self.currentIter-1] 
 
            elif u <= proba_cumsum[3]: # Perturbation move:
                self.nbPerturbationProposition = self.nbPerturbationProposition + 1
                if self.pointProcess.tryPerturbation():
                    self.ratioPerturbationAcceptation = self.ratioPerturbationAcceptation + 1
                    self.pointProcess.posterior.posterior_update_geometry(self.currentIter)
                    if self.pointProcess.posterior.posterior_value[self.currentIter] >= 1e20:
                        self.logger.warning('Anormal operation in perturbation move', extra=d)
                    if self.pointProcess.posterior.posterior_value[self.currentIter] >= self.posterior_distribution:
                        self.pointProcess.objPattern.backup.update(self.pointProcess.objPattern.cubeMap, self.currentIter)
                        self.posterior_distribution = self.pointProcess.posterior.posterior_value[self.currentIter]  
                else:
                    self.pointProcess.posterior.posterior_value[self.currentIter] = self.pointProcess.posterior.posterior_value[self.currentIter-1] 

            else: # Hyperparameters update
                self.nbHyperParameterProposition = self.nbHyperParameterProposition + 1
                self.pointProcess.tryHyperParameters()
                self.pointProcess.posterior.posterior_update_hyperparameters(self.currentIter,
                                                                             self.pointProcess.objPattern.cubeMap)             
                 
                if self.pointProcess.posterior.posterior_value[self.currentIter]  >= self.posterior_distribution:
                    self.pointProcess.objPattern.backup.update(self.pointProcess.objPattern.cubeMap, self.currentIter)
                    self.posterior_distribution = self.pointProcess.posterior.posterior_value[self.currentIter]
             
            self.currentIter = self.currentIter + 1
             
        end = time.time()
        t_min = int((end - beginning)/60) + 1
        self.logger.info('SELFI - End of RJMCMC algorithm (Computing time=%d min)'%t_min, extra=d)

    def get_catalog(self):
        return self.pointProcess.get_catalog()
    