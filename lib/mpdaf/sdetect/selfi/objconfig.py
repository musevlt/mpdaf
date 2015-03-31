"""
Created on Tue Apr  2 10:00:37 2013
    
@author: Celine Meillier and Raphael Bacher
"""

import numpy as np
import random

from ...obj import Cube

def copyVecShapes(x,y):
    y.vecShapes=x.vecShapes.copy()
    return y
    
copyVecShapesAll = np.vectorize(copyVecShapes)



class Cell(object):
    """ This class manages the Cell object.
    It is a pixel-wise structure containing all the information about objects present on this pixel
        
    Attributes
    ----------
    vecShapes : dictionnary
                Keys are the center of the ellipses and
                the values are the profile value corresponding to these ellipses on the pixel
    """
    def __init__(self):
        """ Creates a Cell object"""
        self.vecShapes = {}

class CubeMap(object):
    """ This class manages the CubeMap object. It contains information about the object configuration.
        
        Parameters
        ----------
        LI            : int
                        Number of pixel equals to the grille number of lines
        COL           : int
                        Number of pixel equals to the grille number of columns
        cube          : class 'mpdaf.obj.Cube'
                        Cube object containing the data
        preprocessing : tupple of arrays. If no preprocessing step, preprocessing = None (default parameter).    
                        Preprocessing result (proposition map, pixels classification, wavelength index map, etc)
        
        Attributes
        ----------
        grille    : array
                    Array which has the same spatial dimensions than the data cube.
                    Each element of this array is a Cell object
        LI        : int
                    Number of pixel equals to the grille number of lines  
        COL       : int
                    Number of pixel equals to the grille number of columns
        list_obj  : list of IShape
                    List of the objects of the configuration,
                    it contains the geometrical and intensity characteristics of the objects.        
        pairs     : list
                    List of all the objects pair (i.e. with spatial interaction)
        cube      : array
                    Data cube
        Map       : tupple of two arrays
                    Coordinates of pixels (local maxima) that will be favored during the proposition step
                    for the objects centers in the RJMCMC sampling algorithm.
        Map2      : tupple of two arrays
                    Coordinates of pixels (all the pixels above the threshold value)
                    that will be favored during the proposition step for the objects centers
                    in the RJMCMC sampling algorithm.
        maxMap    : array
                    Array containing the value of the max-test applied to each spectrum of the data cube.
        index_obj : list
                    Index given to the objects of the list_obj
                    (based on the pixel coordinates of the center in the vectorized domain)
    """
    
    def __init__(self, LI, COL, cube, preprocessing = None):
        """ Creates a CubeMap object.
            
        Parameters
        ----------
        LI            : int
                        Number of pixel equals to the grille number of lines
        COL           : int
                        Number of pixel equals to the grille number of columns
        cube          : class 'mpdaf.obj.Cube'
                        Cube object containing the data
        preprocessing : tupple of arrays. If no preprocessing step, preprocessing = None (default parameter).    
                        Preprocessing result (proposition map, pixels classification, wavelength index map, etc)
        """
        self.LI= LI
        self.COL = COL
        self.grille = np.empty(shape = (self.LI,self.COL, ) , dtype = object)
        for i in xrange(LI):
            for j in xrange(COL):
                self.grille[i][j] = Cell()
        self.list_obj = {}
        self.pairs = set()
        if cube.data is not None:
            self.cube = cube.data.data
        
        if preprocessing == None:
            image = cube.data.mean(axis=0)
            thresh = np.mean(image)
            self.Map = np.where(image > thresh*np.ones((self.LI, self.COL)))
            self.Map2 = self.Map
            self.maxMap = cube.data.max(axis=0)
        else:
            self.Map,self.Map2, self.maxMap = preprocessing[0:3] #[0:4]
            
        self.index_obj = {} # contains the correspondance between the objects and their index in the configuration matrix (that induces the Cholesky decomposition computed in the posterior object.)
    
    def copy(self):
        """ Returns a new copy of the CubeMap object.
            
        Returns
        -------
        out : class 'kernelDecorated.CubeMap'
        """
        newCubeMap = CubeMap(self.LI, self.COL, Cube(), (self.Map,self.Map2, self.maxMap))
        newCubeMap.LI = self.LI
        newCubeMap.COL = self.COL
        newCubeMap.grille = copyVecShapesAll(self.grille, newCubeMap.grille)
        newCubeMap.list_obj = self.list_obj.copy()
        newCubeMap.pairs = set(self.pairs)
        newCubeMap.cube = self.cube.__copy__()
        newCubeMap.Map = self.Map
        newCubeMap.Map2 = self.Map2
        newCubeMap.index_obj = self.index_obj.copy()
        newCubeMap.maxMap = self.maxMap
    
        return newCubeMap
    
    def replace(self, newCubeMap):
        """ Replaces the CubeMap attributes by newCubeMap ones.
         
        Parameters
        ----------
        newCubeMap : class 'kernelDecorated.CubeMap'
                     CubeMap object
        """
 
        self.grille = copyVecShapesAll(newCubeMap.grille, self.grille)
         
        self.list_obj = newCubeMap.list_obj.copy()
        self.pairs = set(newCubeMap.pairs)
        self.LI = newCubeMap.LI
        self.COL = newCubeMap.COL
        self.Map = newCubeMap.Map
        self.Map2 = newCubeMap.Map2
        self.index_obj = newCubeMap.index_obj.copy()
        self.maxMap = newCubeMap.maxMap.copy()
     
    def addPairs(self,obj1,obj2):
        """ Checks and adds (obj1, obj2) if (obj1, obj2) is not already
        a pair known in the list 'pairs'. 
             
        Parameters
        ----------
        obj1 : IShape
               First object of the pair
        obj2 : IShape
               Second object of the pair
        """
        if obj1.nbPixelShared(self,obj2) > 0:
            if (obj1.centre , obj2.centre) in self.pairs \
            or (obj2.centre , obj1.centre) in self.pairs :
                pass
            else:
                self.pairs.add((obj1.centre , obj2.centre))
        else:
            pass
 
    def removePairs(self,obj1,obj2):
        """ Checks and removes (obj1, obj2) if (obj1, obj2) is a pair known in the list 'pairs'.
         
        Parameters
        ----------
        obj1 : IShape
               First object of the pair
        obj2 : IShape
               Second object of the pair
        """
        if (obj1.centre , obj2.centre) in self.pairs :
            self.pairs.remove((obj1.centre , obj2.centre))
        if (obj2.centre , obj1.centre) in self.pairs :
            self.pairs.remove((obj2.centre , obj1.centre))
         
    def addObject(self, obj1, functor):
        """ Adds an object to the list of selected objects
        with respect to the chosen functor.
         
        Parameters
        ----------
        obj1    : IShape
                  Object to be added
        functor : a class of 'functor' module  
                  Function described by one class of the 'functor' module
        """
        for centre in self.list_obj.keys():
            obj = self.list_obj[centre]
            self.addPairs(obj1, obj) # parameters = IShape (not centre)
        self.list_obj[obj1.centre] = obj1
        obj1.applyToPoints(functor, self.LI, self.COL)
         
    def removeObject(self, obj1, functor):
        """ Removes object from the list_obj attribute.
         
        Parameters
        ----------
        obj1    : IShape
                  Object to be removed
        functor : a class of 'functor' module  
                  Function described by one class of the 'functor' module
        """
        if obj1.centre in self.list_obj.keys():
            del self.list_obj[obj1.centre]
        obj1.applyToPoints(functor, self.LI, self.COL)
        for centre in self.list_obj.keys():
            obj = self.list_obj[centre]
            self.removePairs(obj1, obj)
        for i in xrange(max(0,int(obj1.centre[0]) - 10), min(int(obj1.centre[0]) +11, self.LI) ):
            for j in xrange(max(0,int(obj1.centre[1]) - 10), min(int(obj1.centre[1]) +11, self.COL) ):
                if obj1.centre in self.grille[i][j].vecShapes.keys():
                    del self.grille[i][j].vecShapes[obj1.centre]

    
class Picker(object):
    """ This class manages the Picker object.
    Its methods allows to propose one center position for birth,
    death, translation, rotation or perturbation moves
   
        :param cubeMap: CubeMap object containig the information about the object configuration
        :type cubeMap: class 'kernelDecorated.CubeMap'
        :param parameters: object containing the configuration parameters
        :type parameters: class 'ParameterFile_tmp'
        
    Parameters
    ----------
    cubeMap          : class 'kernelDecorated.CubeMap'
                       CubeMap object containig the information about the object configuration
    Shape_MaxDist2Bd : int
                       average size of the spatial PSF in pixels number
    PC1              : float
                       probability to propose a center from the class C1
    
    Attributes
    ----------
        
    cubeMap          : kernelDecorated.cubeMap
                       CubeMap object containing the information about the object configuration
    LI               : int
                       Number of lines in the spatial dimention of the data cube
    COL              : int
                       Number of columns in the spatial dimension of the data cube
    Shape_MaxDist2Bd : int
                       average size of the spatial PSF in pixels number
    map_acceptation  : array
                       array of the same size than one image of the data cube
                       wich contains, in each position (p,q), 1 if it contains an object center, 0 else.
    bright           : list
                       list of tupple (p,q) of int that correspond to the cubeMap.Map.
                       It contains 0 if the pixel (p,q) has not yet been proposed, 1 else.
    intermed         : list
                       list of tupple (p,q) of int that correspond to the cubeMap.Map2.
                       It contains 0 if the pixel (p,q) has not yet been proposed, 1 else.
    count_bright     : list
                       List of position (p,q) that has been proposed to be an object center among the positions of cubeMap.Map.
    count_intermed   : list
                       List of position (p,q) that has been proposed to be an object center among the positions of cubeMap.Map2.
    list_bright      : list
                       one element of this list is a tupple composed of :
                       - 1 if the center has not been et proposed, 0 else
                       - cubeMap.maxMap[p,q] value 
                       - p (abcissa of the considered position)
                       - q (ordinate of the considered position)
    list_intermed    : list
                       one element of this list is a tupple composed of :
                       - 1 if the center has not been et proposed, 0 else
                       - cubeMap.maxMap[p,q] value
                       - p (abcissa of the considered position)
                       - q (ordinate of the considered position)
    inConfig         : dictionary
                       Dictionary that contains the center of the objects of the configuration for the keys and values set to "1"
    initConfig       : dictionary
                       Dictionary that contains the center of the objects of the initial configuration for the keys and values = IShape objects
    pBright          : float
                       Probability of proposing in the class C1 (cubeMap.Map)
    Indice_bright    : int
                       variable used to quantify the proposition of object in the 'bright' position list
    indice_intermed  : int
                       variable used to quantify the proposition of object in the 'intermed' position list
    """
    def __init__(self,prng, cubeMap, Shape_MaxDist2Bd, pC1):
        """ Creates a Picker object.
            
            Parameters
            ----------
            prng             : numpy.random.RandomState
                               instance of numpy.random.RandomState() with potential chosen seed.
            cubeMap          : class 'kernelDecorated.CubeMap'
                               CubeMap object containig the information about the object configuration
            Shape_MaxDist2Bd : int
                               average size of the spatial PSF in pixels number
            PC1              : float
                               probability to propose a center from the class C1
        """
        self.prng = prng
        self.cubeMap = cubeMap
        self.LI= self.cubeMap.LI
        self.COL = self.cubeMap.COL
        self.Shape_MaxDist2Bd = Shape_MaxDist2Bd
        self.map_acceptation = np.zeros((self.LI,self.COL))
        
        l = len(self.cubeMap.Map[0])
        l2 = len(self.cubeMap.Map2[0])
        setMap = set()
        setMap2 = set()
        for i in np.arange(0,l):
            setMap.add((self.cubeMap.Map[0][i],self.cubeMap.Map[1][i]))
        for i in np.arange(0,l2):
            setMap2.add((self.cubeMap.Map2[0][i],self.cubeMap.Map2[1][i]))
        self.bright = {}
        self.count_bright = {}
        self.intermed = {}
        self.count_intermed = {}
        
        
        self.list_bright = []
        self.list_intermed = []
        
        for i in range(self.cubeMap.LI):
            for j in range(self.cubeMap.COL):
                if (i,j) in setMap:
                    self.bright[(i,j)] = 0
                    self.count_bright[(i,j)] = 0
                    self.list_bright.append( (1,cubeMap.maxMap[i,j], i,j))
                elif (i,j) in setMap2:
                    self.intermed[(i,j)] = 0
                    self.count_intermed[(i,j)] = 0
                    self.list_intermed.append( (1,cubeMap.maxMap[i,j], i,j))
        self.list_bright.sort(reverse = True)
        self.list_intermed.sort(reverse = True)
            
        self.inConfig = {}
        self.initConfig = {}
        self.pBright = pC1
        self.indice_bright = 0
        self.indice_intermed = 0
        
    def randomBirth(self):
        """Proposes a position for a new object, randomly selected in the 'bright' or 'intermed' lists.
        """
        if (-1,-1) in self.bright.keys():
            del self.bright[(-1,-1)]
        if (-1,-1) in self.intermed.keys():
            del self.intermed[(-1,-1)]
        choice = self.prng.uniform(0,1)
        if choice < self.pBright:
            l = len(self.bright.keys())
            u = self.prng.randint(0,l)
            p = min(self.bright.keys()[u][0] + self.prng.uniform(0,1), self.LI-1)
            q = min(self.bright.keys()[u][1] + self.prng.uniform(0,1), self.COL-1)
            n = 0
            while self.bright[(int(p),int(q))] > 0 and n <10:
                u = self.prng.randint(0,l)
                p = min(self.bright.keys()[u][0] + self.prng.uniform(0,1), self.LI-1)
                q = min(self.bright.keys()[u][1] + self.prng.uniform(0,1), self.COL-1)
                n = n+1
            if n == 10:
                p = -1
                q = -1
            if p >=0.:
                self.count_bright[(int(p),int(q))] = self.count_bright[(int(p),int(q))] + 1
        else:
            l = len(self.intermed.keys())
            u = self.prng.randint(0,l)
            p = min(self.intermed.keys()[u][0] + self.prng.uniform(0,1), self.LI-1)
            q = min(self.intermed.keys()[u][1] + self.prng.uniform(0,1), self.COL-1)
            while self.intermed[(int(p),int(q))] >0 :
                u = self.prng.randint(0,l)
                p = min(self.intermed.keys()[u][0] + self.prng.uniform(0,1), self.LI-1)
                q = min(self.intermed.keys()[u][1] + self.prng.uniform(0,1), self.COL-1)
            self.count_intermed[(int(p),int(q))] = self.count_intermed[(int(p),int(q))] + 1
        return p, q

    def objChoice(self): 
        """ Returns the centre of an existant object for death moves.
        """
        s1 = set(self.initConfig.keys())
        s2 = set(self.cubeMap.list_obj.keys())
        s3 = s2 - s1
        l3 = list(s3)
        if len(self.cubeMap.list_obj.keys()) > len(self.initConfig):
            choice = random.choice(l3)
            while (int(choice[0]), int(choice[1])) in self.initConfig.keys():
                choice = random.choice(self.cubeMap.list_obj.keys())
        else:
            choice = (-1., -1.)
        return choice
        
class Backup(object):
    """ This class manages the Backup object.
        Backup object contains the information about the maximum a posteriori estimate of the object configuration.
        
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
        
        
        Attributes
        ----------
        list_obj              : List of IShape
                                Objects list of the maximum a posteriori estimate of the object configuration
        iteration             : int
                                index of the considered iteration
        index_obj             : list
                                index given to the objects of the list_obj
                                (based on the pixel coordinates of the center in the vectorized domain)
        histoNbObj            : list
                                evolution of the objects number in the object configuration
                                corresponding to an update of the maximum a posteriori estimate of the object configuration.
        histoIter             : list
                                Iteration index corresponding to an update of
                                the maximum a posteriori estimate of the object configuration. 
        nb_obj_stop_criterion : int
                                Objects number in the 'list_obj' attribute
        iter_stop_criterion   : int
                                Index of the last iteration where the number of objects in the 'list_obj' attribute increases.
    """
    def __init__(self, cubeMap):
        """ Creates a Backup object
            
            Parameters
            ----------
            cubeMap : class 'kernelDecorated.CubeMap'
                      CubeMap object containing the information about the object configuration
        """
        self.list_obj = cubeMap.list_obj.copy()
        self.iteration = 0 
        self.index_obj = cubeMap.index_obj.copy()
        self.histoNbObj = []
        self.histoIter = []
        self.nb_obj_stop_criterion = 0
        self.iter_stop_criterion = 0
        
    def update(self,cubeMap, iteration):
        """ Updates the information about the maximum a posteriori estimate of the object configuration.
        
            Parameters
            ----------
            cubeMap    : class 'kernelDecorated.CubeMap'
                         CubeMap object containing the information about the object configuration
            iteration  : int
                         index of the considered iteration
        """
        self.list_obj = cubeMap.list_obj.copy()
        self.iteration = iteration
        self.index_obj = cubeMap.index_obj.copy()
        self.histoNbObj.append(len(self.list_obj))
        self.histoIter.append(self.iteration)
        if len(self.list_obj) != self.nb_obj_stop_criterion:
            self.nb_obj_stop_criterion = len(self.list_obj)
            self.iter_stop_criterion = iteration


    def matrix_configuration(self, P, Q):
        """ Transforms the object configuration into a matrix
        whose each column is the spatial intensity profile of an object
        of the maximum a posteriori estimate of the object configuration.
             
        Parameters
        ----------
        P : int
            Number of lines in the image representation
        Q : int
            Number of columns in the image representation
        """
        X = np.zeros((P*Q, len(self.list_obj)))
        for c in self.index_obj.keys():
            ind = self.index_obj[c]
            el = self.list_obj[c]
            X[:,ind] = np.reshape(el.profile(P,Q), P*Q)
             
        return X
# 
#     def intensity(self, cube, m ):
#         """ Computes an estimate of the spectrum of each object.
#         
#         Parameters
#         ----------
#         cube : class 'mpdaf.obj.Cube'
#                Cube object containing the data
#         m    : array
#                Vector containing the background value
#                (mean estimated at backup.iteration)
#        
#         Returns
#         -------
#         out : array
#               Matrix of size n x LBDA where n in the number of objects
#               in the maximum a posteriori configuration and LBDA the number of spectral bands in the data.
#         """
#         dimensions = cube.shape
#         LBDA = dimensions[0]
#         P= dimensions[1]
#         Q = dimensions[2]
#         w = np.zeros((len(self.list_obj), LBDA))
#         X = self.matrix_configuration(P,Q)
#         Gram = np.dot(X.T,X)
#         inverse = np.linalg.inv(Gram)
#         tmp = np.dot(inverse, X.T)
# 
#         for l in xrange(LBDA):
#             y = np.reshape(cube.data[l,:,:], P*Q) - m[l]
#             w[:,l] = np.dot(tmp, y)
# 
#         return w
# 
#     def plotConfig_support(self, image):
#         """ Plots the objects of the current configuration on the background image
#             
#             Parameters
#             ----------
#             
#             image : array
#                     Background image
#         """
#     
#         ells = [Ellipse2(xy=(obj.centre[1],obj.centre[0]), width=2*obj.axe1, height=2*obj.axe2, angle=obj.angle*180./np.pi,linewidth=2, fill=False) for obj in self.list_obj.values()]
#         fig = plt.figure()
#     
#         ax = fig.add_subplot(111) #, aspect='equal'
#         P = image.shape[0]
#         Q = image.shape[1]
#         for e in ells:
#             ax.add_artist(e)
#             e.set_clip_box(ax.bbox)
#             e.set_alpha(1)
#             e.set_edgecolor((1,0,0))
#     
#         ax.set_ylim(0, P-0.5)
#         ax.set_xlim(0, Q-.5)
#     
#         plt.imshow(image,cmap = 'binary')
        