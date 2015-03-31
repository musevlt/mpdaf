import numpy as np
from scipy import ndimage
from scipy.stats import norm
from numpy.random import RandomState
import random
import logging


from ...obj import Image, WCS
from ...tools.fits import add_mpdaf_method_keywords

from preprocessing import preprocessing_photometric, FSF_cube, moving_average, spectral_spread_adapted_filter, fast_processing
from ellipseSersic import EllipseSersic
from posterior import withoutIntensityReg
from kernelSersic import KernelSersic
from kernelBayesian import KernelBayesian
from kernelMCMC import KernelMCMC

class Preprocessing(object):
    """Object that contains results of the preprocessing step.
        
        Attributes
        ----------
        thresh : float
                 Threshold value that has been used for the max-test.
        nobj   : int
                 Number of detected objects.
        MaxMap : array
                 2d array that contains maxima along the wavelength axis.
        Map    : tuple of arrays
                 Contains coordinates of local maxima.
        Map2   : tuple of arrays
                 Contains coordinates of pixels above the threshold value
        ID    : dict
                Dictionary that lists labels of the local maxima.
                {ID: (x,y) of local maxima}
        ima   : :class:`mpdaf.obj.Image`
                Image containing an integer array
                where each detected object has a unique label.
        stars : list
                List of IDs tat have been classified as stars.  
    """
    def __init__(self, max_thresh, nobj, maxMap, Map, Map2, ID, ima):
        """ Object that contains results of the preprocessing step.
        
        Parameters
        ----------
        max_thresh : float
                     Threshold value that has been used for the max-test.
        nobj       : int
                     Number of detected objects.
        MaxMap     : array
                     2d array that contains maxima along the wavelength axis.
        Map        : tuple of arrays
                     Contains coordinates of local maxima.
        Map2       : tuple of arrays
                     Contains coordinates of pixels above the threshold value
        ID         : dict
                     Dictionary that lists labels of the local maxima.
                     {ID: (x,y) of local maxima}
        ima        : :class:`mpdaf.obj.Image`
                     Image containing an integer array
                     where each detected object has a unique label.  
        """
        self.thresh = max_thresh
        self.nobj = nobj
        self.maxMap = maxMap
        self.Map = Map
        self.Map2 = Map2
        self.ID = ID
        self.ima = ima
        self.stars = None
        
    def set_stars(self, IDlist):
        """Sets the list of candidates id as stars.
        """
        self.stars = IDlist

class WhiteFinder(object):
    
    def __init__(self, cube, fsf, seed=None):
        """
        Parameters
        ----------
        cube : :class:`mpdaf.obj.Cube`
               Data cube
        fsf  : float array or None       
               Array containing FSF (spatial PSF)
               If fsf = None, the FSF defines for the DryRun is used by default.
        seed : None, int, array_like
               Random seed initializing the pseudo-random number generator.
               Can be an integer, an array (or other sequence) of integers of any length,
               or None (the default). 
        
        Attributes
        ----------
        white     : :class:`mpdaf.obj.Cube`
                    White light image but in format datacube
        cube      : string
                    Cube FITS filename
        fsf       : float array    
                    Array containing FSF (spatial PSF)
        pnrg      : numpy.random.RandomState
                    Container for the pseudo-random number generator.
        algowhite : :class:`mpdaf.sdetect.selfi.KernelMCMC`
                    Link to selfi object
        """
        # logger
        self.logger = logging.getLogger('mpdaf corelib')
        d = {'class': 'WhiteFinder', 'method': '__init__'}
        self.logger.info('SELFI - Initializing white finder', extra=d)
        
        # create a white light image but in format datacube
        self.white = cube[0:1,:,:].copy()
        self.white.data[0,:,:] = np.mean(cube.data, axis = 0)
        self.cube = cube.filename
        
        #fsf
        if fsf is None:
            LBDAmin, LBDAmax = cube.wave.get_range()
            self.fsf = FSF_cube(2, LBDAmin, LBDAmax)
        else:
            self.fsf = fsf
        
        #  pseudo-random number generator.
        self.pnrg = RandomState(seed)
        random.seed(seed)
        
        # link to selfi object
        self.algowhite = None

    def preprocessing(self, max_thresh=3.72):
        """ Preprocessing phase
        
        Parameters
        ----------
        max_thresh : float
                     Threshold value for the max-test.
                     Default value: 3.72 (10**-4 PFA for Normal Distribution)
                     
        Returns
        -------
        out : :class:`mpdaf.sdetect.selfi.Preprocessing`
              Object that contains results of the preprocessing.
              
        """
        d = {'class': 'WhiteFinder', 'method': 'preprocessing'}
        
        self.logger.info('SELFI - Normalizing white Image', extra=d)
        (m, sigma) = self.white[0,:,:].background()
        whitenorm = self.white.copy()
        whitenorm.data[0,:,:] = (whitenorm.data[0,:,:] - m)/sigma

        self.logger.info('SELFI - Preprocessing', extra=d)
        max_thresh = 3.72 # 10**-4 PFA for Normal Distribution
        Map, Map2 = preprocessing_photometric(whitenorm.data[0,:,:], max_thresh)
        maxMap = whitenorm.data[0,:,:]
        nobj = len(Map[0])
        self.logger.info('SELFI - Number of candidates found: %d'%nobj, extra=d)
        
        
        im_result = np.zeros_like(maxMap)
        im_result[Map2] = 1
        
        labeled_array, num_features = ndimage.measurements.label(im_result)
        
        ID = {}
        for i in range(nobj):
            ID[labeled_array[Map[0][i], Map[1][i]]] = (Map[0][i], Map[1][i])
        
        im_result[Map] = np.arange(1, nobj+1)
        ima = Image(data=labeled_array, wcs=self.white.wcs)
        add_mpdaf_method_keywords(ima.primary_header,
                                  "selfi.WhiteFinder.preprocessing",
                                  ['cube', 'thresh'],
                                  [self.cube, max_thresh],
                                  ['cube fits',
                                   'Threshold value'])
        return Preprocessing(max_thresh, nobj, maxMap, Map, Map2, ID, ima)
    
#     def display_candidates(self):
#         """displays the candidates on the image.
#         stars show different symbol
#         """
#         pass
    
    def minimize(self, preprocessing, NbMCMC=30000,
                 pC1=1.0, MaxTranslation=0.8, MinScaleFactor=0.8,
                 MaxScaleFactor=1.2, Shape_MinRadius=0.5, Shape_MaxRadius=2.0,
                 ProbaBirth=0.5, ProbaBirthDeath=0.3, ProbaTranslation=0.1,
                 ProbaRotation=0.2, ProbaPerturbation=0.2,ProbaHyperUpdate=0.2,
                 ExpMaxNbShapes=500):
        """Detection phase on the white image
        
        Parameters
        ----------
        preprocessing     : :class:`mpdaf.sdetect.selfi.Preprocessing`
                            Object that contains results of the preprocessing phase.
        NbMCMC            : integer
                            Maximum number of iterations in the RJMCMC algorithm
                            (30000 by default)
        pC1               : float
                            Probability to sample the local maxima with respect
                            to the spaxels above the threshold.
                            (1.0 by default)
        MaxTranslation    : float
                            Maximum radius (in spaxel) for translation
                            (0.8 by default)
        MinScaleFactor    : float
                            Minimum scaling factor for FWHM
                            (0.8 by default)
        MaxScaleFactor    : float
                            Maximum scaling factor for FWHM
                            (1.2 by default)
        Shape_MinRadius   : float
                            Minimum half FWHM in spaxel (before FSF convolution)
                            (0.5 by default)
        Shape_MaxRadius   : float
                            Maximum half FWHM in spaxel (before FSF convolution)
                            (2.0 by default)
        ProbaBirth        : float in [0,1]
                            Proportion of birth proposition in the birth and death move
                            (0.5 by default)
        ProbaBirthDeath   : float in [0,1]
                            Weight given to the birth and death move
                            (0.3 by default)
        ProbaTranslation  : float in [0,1]
                            Weight given to the translation move
                            (0.1 by default)
        ProbaRotation     : float in [0,1]
                            Weight given to the rotation move    
                            (0.2 by default)
        ProbaPerturbation : float in [0,1]
                            Weight given to the perturbation move
                            (0.2 by default)
        ProbaHyperUpdate  : float in [0,1]
                            Weight given to the parameters and hyperparameters sampling move 
                            (0.2 by default)
        ExpMaxNbShapes    : int
                            Maximum number of objects
                            (500 by default)

        Returns
        -------
        out : list< :class:`mpdaf.sdetect.Source` >
              List of source objects.
        """
        d = {'class': 'WhiteFinder', 'method': 'minimize'}
        
        self.logger.info('SELFI - Minimizing on the white image', extra=d)
        #FSF light profile
        fsf_mean = np.mean(self.fsf, axis=0)
        fsf_ima = Image(data=fsf_mean, wcs = WCS())
        fwhmy, fwhmx = fsf_ima.fwhm(pix=False)
        # FSF FWHM in spaxels (must be integer)
        Shape_MaxDist2Bd = int((fwhmx+fwhmy)/2. + 0.5)
        #self.logger.info('FSF FWHM = %d spaxels'%Shape_MaxDist2Bd, extra=d)
        
        # Fraction of common energy of 2 FSF distant of 1 FWHM
        fsf1 = fsf_ima.data.data
        fsf2 = np.zeros_like(fsf_ima.data.data)
        fsf2[:-Shape_MaxDist2Bd,:-Shape_MaxDist2Bd] = fsf1[Shape_MaxDist2Bd:, Shape_MaxDist2Bd:]
        HardCore = np.linalg.norm(np.dot(fsf1,fsf2),2)/np.linalg.norm(fsf1,2)**2
        #self.logger.info('Rayleigh criterion = %0.2f spaxels'%HardCore, extra=d)
        
        # Set bright star used as FSF as an EllipseSsersic object with FSF light profile
        list_obj = []
        
        for ID in preprocessing.stars:
            el = EllipseSersic(self.pnrg, preprocessing.ID[ID], fwhmx/2., fwhmy/2., 0, convolution=None)
            el.profile_convolved = fsf_mean
            list_obj.append(el)

        # create a new class for aposteriori density function
        WithoutIntensityReg = withoutIntensityReg(self.pnrg, self.white, NbMCMC)
        # create a new class to manage Sersic objects
        KEllipse = KernelSersic(self.pnrg, self.white, Shape_MaxDist2Bd, pC1,
                                self.fsf, MaxTranslation, MinScaleFactor, 
                                MaxScaleFactor, Shape_MinRadius,
                                Shape_MaxRadius, (preprocessing.Map,
                                                  preprocessing.Map2,
                                                  preprocessing.maxMap))
        # create a new class for Bayesian model
        KBayesian = KernelBayesian(self.pnrg, KEllipse, WithoutIntensityReg,
                                   HardCore, preprocessing.thresh)
        # create a new class for MCMC iterative process
        algowhite = KernelMCMC(self.pnrg, KBayesian, NbMCMC, ProbaBirth,
                               ProbaBirthDeath, ProbaTranslation,
                               ProbaRotation, ProbaPerturbation,
                               ProbaHyperUpdate, ExpMaxNbShapes)

        # initialize config with objects (stars)
        algowhite.initConfig(list_obj,
                             algowhite.pointProcess.posterior.m[0,:],
                             algowhite.pointProcess.posterior.sigma2[0,:])

        # perform minimization
        algowhite.MCMC()
        self.algowhite = algowhite
        
        #returns list of sources
        return self.algowhite.get_catalog()
        
        
    def plot_result(self):
        """Plots results
        """
        self.algowhite.plotConfig_support(self.white[0,:,:])
        

    
class CubeFinder(object):
    
    def __init__(self, cube, fsf, seed=None):
        """
        Parameters
        ----------
        cube : :class:`mpdaf.obj.Cube`
               Data cube
        fsf  : float array or None       
               Array containing FSF (spatial PSF)
               If fsf = None, the FSF defines for the DryRun is used by default.
        seed : None, int, array_like
               Random seed initializing the pseudo-random number generator.
               Can be an integer, an array (or other sequence) of integers of any length,
               or None (the default). 
        
        Attributes
        ----------
        cube      : :class:`mpdaf.obj.Cube`
                    Data cube
        fsf       : float array    
                    Array containing FSF (spatial PSF)
        pnrg      : numpy.random.RandomState
                    Container for the pseudo-random number generator.
        algo      : :class:`mpdaf.sdetect.selfi.KernelMCMC`
                    Link to selfi object
        """
        # logger
        self.logger = logging.getLogger('mpdaf corelib')
        d = {'class': 'CubeFinder', 'method': '__init__'}
        self.logger.info('SELFI - Initializing cube finder', extra=d)
        
        # cube
        self.cube = cube
        
        # fsf
        if fsf is None:
            LBDAmin, LBDAmax = cube.wave.get_range()
            self.fsf = FSF_cube(2, LBDAmin, LBDAmax)
        else:
            self.fsf = fsf
        
        #  pseudo-random number generator.
        self.pnrg = RandomState(seed)
        random.seed(seed)
        
        # link to selfi object
        self.algo = None
        
    def preprocessing(self, max_thresh=4.43):
        """ Preprocessing phase
        
        Parameters
        ----------
        max_thresh : float
                     Threshold value for the max-test.
                     Default value: 4.43
                     
        Returns
        -------
        out : :class:`mpdaf.sdetect.selfi.Preprocessing`
              Object that contains results of the preprocessing.
        """
        d = {'class': 'CubeFinder', 'method': 'preprocessing'}
        
        LBDA, P, Q = self.cube.shape
        
        self.logger.info('SELFI - Computing SNR', extra=d)
        SNR = self.cube / np.sqrt(self.cube.var)
        
        # transform to zero mean and var=1 (whitening)
        self.logger.info('SELFI - Whitening', extra=d)
        normalized_cube = SNR.loop_ima_multiprocessing(moving_average, verbose=True)
        
        # match filter
        self.logger.info('SELFI - Computing Match Filter', extra=d)
        MF = spectral_spread_adapted_filter(normalized_cube, self.fsf)
        
        # Bootstrap on the full wavelength range to estimate new mean and variance
        self.logger.info('SELFI - Bootstrap', extra=d)
#         B = 40 # number of iteration
#         li = []
#         data = np.ravel(MF)
#         datasort = data.copy()
#         datasort.sort()
#         datasort = datasort[0:int(0.96*len(data))]
#         for b in xrange(B):
#             ind = np.random.randint(0, len(datasort), P*Q)
#             newData = datasort[ind]
#             newLi = list(newData)
#             li = li + newLi
#         (mu, std) = norm.fit(li)
#         
#         MF_Norm = (MF - mu)/std
#         self.logger.info('Normalizing values: mean %g std %g'%(mu,std), extra=d)
        
        # Bootstrap on each wavelength range to estimate new mean and variance
        B = 40 # number of iteration
        Nsamp = P*Q/100 # number of sample
        MF_Norm = MF.copy()
        muval = []
        stdval = []
        for k in range(LBDA):
            li = []
            data = np.ravel(MF[k,:,:])
            datasort = data.copy()
            datasort.sort()
            datasort = datasort[0:int(0.96*len(data))]
            for b in xrange(B):
                ind = np.random.randint(0,len(datasort), Nsamp)
                newData = datasort[ind]
                newLi = list(newData)
                li = li + newLi
            (mu, std) = norm.fit(li)
            muval.append(mu)
            stdval.append(std)
            MF_Norm[k,:,:] = (MF[k,:,:] - mu)/std
            if not k%500:
                self.logger.info('SELFI - Iter %d Normalizing values: mean %g std %g'%(k,mu,std), extra=d)
 
        

        self.logger.info('SELFI - Computing maxMap and source candidates', extra=d)
        maxMap = np.max(MF_Norm, axis=0)
        
        # threshold the maxMap
        #thresh_max = 4.43 # this gives a FPA of 1% by spaxel (use p_val_maxtest to get the correspondance)
        # Map location of local maxima
        # Map2 all spaxels greated than the treshold
        Map, Map2  = fast_processing(maxMap, P, Q, max_thresh)
        
        nobj = len(Map[0])
        self.logger.info('SELFI - Number of candidates found: %d'%nobj, extra=d)
        
        im_result = np.zeros_like(maxMap)
        im_result[Map2] = 1
        #im_result[self.Map] = np.arange(1, nobj+1)
        
        labeled_array, num_features = ndimage.measurements.label(im_result)
        
        ID = {}
        for i in range(nobj):
            ID[labeled_array[Map[0][i], Map[1][i]]] = (Map[0][i], Map[1][i])
        
        im_result[Map] = np.arange(1, nobj+1)
        ima = Image(data=labeled_array, wcs=self.cube.wcs)
        add_mpdaf_method_keywords(ima.primary_header,
                                  "selfi.CubeFinder.preprocessing",
                                  ['cube', 'thresh'],
                                  [self.cube.filename, max_thresh],
                                  ['cube fits',
                                   'Threshold value'])
        return Preprocessing(max_thresh, nobj, maxMap, Map, Map2, ID, ima)
    

    
    def minimize(self, preprocessing, white_objs=[], NbMCMC=50000, pC1=1.0,
                 MaxTranslation=0.8, MinScaleFactor=0.8, MaxScaleFactor=1.2,
                 Shape_MinRadius=0.5, Shape_MaxRadius=2.0, ProbaBirth=0.5,
                 ProbaBirthDeath=0.3, ProbaTranslation=0.1, ProbaRotation=0.2,
                 ProbaPerturbation=0.2, ProbaHyperUpdate=0.2,
                 ExpMaxNbShapes=500):
        """Performs initialization and minimization on the cube,
        at the end computes spectra, images and
        returns the list of source objects.
        
        Parameters
        ----------
        preprocessing     : :class:`mpdaf.sdetect.selfi.Preprocessing`
                            Object that contains results of the preprocessing phase.
        white_objs        : list< :class:`mpdaf.sdetect.Source` >
                            List of sources resulted of the minimization on the white image.
        NbMCMC            : integer
                            Maximum number of iterations in the RJMCMC algorithm
                            (30000 by default)
        pC1               : float
                            Probability to sample the local maxima with respect
                            to the spaxels above the threshold.
                            (1.0 by default)
        MaxTranslation    : float
                            Maximum radius (in spaxel) for translation
                            (0.8 by default)
        MinScaleFactor    : float
                            Minimum scaling factor for FWHM
                            (0.8 by default)
        MaxScaleFactor    : float
                            Maximum scaling factor for FWHM
                            (1.2 by default)
        Shape_MinRadius   : float
                            Minimum half FWHM in spaxel (before FSF convolution)
                            (0.5 by default)
        Shape_MaxRadius   : float
                            Maximum half FWHM in spaxel (before FSF convolution)
                            (2.0 by default)
        ProbaBirth        : float in [0,1]
                            Proportion of birth proposition in the birth and death move
                            (0.5 by default)
        ProbaBirthDeath   : float in [0,1]
                            Weight given to the birth and death move
                            (0.3 by default)
        ProbaTranslation  : float in [0,1]
                            Weight given to the translation move
                            (0.1 by default)
        ProbaRotation     : float in [0,1]
                            Weight given to the rotation move    
                            (0.2 by default)
        ProbaPerturbation : float in [0,1]
                            Weight given to the perturbation move
                            (0.2 by default)
        ProbaHyperUpdate  : float in [0,1]
                            Weight given to the parameters and hyperparameters sampling move 
                            (0.2 by default)
        ExpMaxNbShapes    : int
                            Maximum number of objects
                            (500 by default)

        Returns
        -------
        out : list< :class:`mpdaf.sdetect.Source` >
              List of source objects.
        """
        d = {'class': 'CubeFinder', 'method': 'minimize'}
        self.logger.info('SELFI - Minimizing on the cube', extra=d)
        
        #FSF light profile
        fsf_mean = np.mean(self.fsf, axis=0)
        fsf_ima = Image(data=fsf_mean, wcs = WCS())
        fwhmy, fwhmx = fsf_ima.fwhm(pix=False)
        # FSF FWHM in spaxels (must be integer)
        Shape_MaxDist2Bd = int((fwhmx+fwhmy)/2. + 0.5)
        #self.logger.info('FSF FWHM = %d spaxels'%Shape_MaxDist2Bd, extra=d)
        
        fsf1 = fsf_ima.data.data
        fsf2 = np.zeros_like(fsf_ima.data.data)
        fsf2[:-Shape_MaxDist2Bd,:-Shape_MaxDist2Bd] = fsf1[Shape_MaxDist2Bd:, Shape_MaxDist2Bd:]
        HardCore = np.linalg.norm(np.dot(fsf1,fsf2),2)/np.linalg.norm(fsf1,2)**2
        #self.logger.info('Rayleigh criterion = %0.2f spaxels'%HardCore, extra=d)
        
        # create a new class for aposteriori density function
        WithoutIntensityReg = withoutIntensityReg(self.pnrg, self.cube, NbMCMC)
        # create a new class to manage Sersic objects
        KEllipse = KernelSersic(self.pnrg, self.cube, Shape_MaxDist2Bd, pC1,
                                self.fsf, MaxTranslation, MinScaleFactor,
                                MaxScaleFactor, Shape_MinRadius,
                                Shape_MaxRadius, (preprocessing.Map,
                                                  preprocessing.Map2,
                                                  preprocessing.maxMap))
        # create a new class for Bayesian model
        KBayesian = KernelBayesian(self.pnrg, KEllipse, WithoutIntensityReg,
                                   HardCore, preprocessing.thresh)
        # create a new class for MCMC iterative process
        algo = KernelMCMC(self.pnrg, KBayesian, NbMCMC, ProbaBirth,
                               ProbaBirthDeath, ProbaTranslation,
                               ProbaRotation, ProbaPerturbation,
                               ProbaHyperUpdate, ExpMaxNbShapes)
        
        # initialize config with white light objects
        white_ellipses = []
        for source in white_objs:
            el = EllipseSersic.from_source(self.pnrg, source)
            white_ellipses.append(el)
        algo.initConfig(white_ellipses, algo.pointProcess.posterior.m[0,:],
                        algo.pointProcess.posterior.sigma2[0,:])

        # perform minimization
        algo.MCMC()
        self.algo = algo
        
        self.logger.info('SELFI - Number of birth proposed %d accepted %d'\
                         %(algo.nbBirthProposition,
                           algo.ratioBirthAcceptation), extra=d)
        
        return self.algo.get_catalog()
    
        
    def plot_result(self):
        """
        """
        self.algo.plotConfig_support(self.cube.sum(axis=0))
        