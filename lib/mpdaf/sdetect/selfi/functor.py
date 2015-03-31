import numpy as np

class addShapeToImageSF(object):
    """ This class manages the functor addShapeToImageSF.
        
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
                  
        Attributes
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
        """
    def __init__(self,cubeMap):
        """ Creates an addShapeToImageSF object.
        
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
        """
        self.cubeMap = cubeMap
    
    def zeros(self,cubeMap):
        """ Initializes the addShapeToImageSF object.
         
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
        """
        self.cubeMap = cubeMap
     
    def applyToPoint(self,centre,position,value):
        """ Applies a processing corresponding at the pixel located at (position[0], position[1]) coordinates and registers the value at this position in the kernelDecorated.cubeMap.grille object.
             
        Parameters
        ----------
        center   : tupple of float
                   Position of the object center
        position : tupple of float
                   Position of the considered pixel (inside the object spatial support)
        value    : float
                   Value of the intensity at the position (position[0], position[1])
        """
        if centre in self.cubeMap.grille[position[0]][position[1]].vecShapes.keys():
            #There is already an object at this position
            pass
        else:
            self.cubeMap.grille[position[0]][position[1]].vecShapes[centre] = value

            
class rmvShapeToImageSF(object):
    """ This class manages the functor rmvShapeToImageSF.
        
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
                  
        Attributes
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
    """
    def __init__(self,cubeMap):
        """ Creates an rmvShapeToImageSF object.
        
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
        """
        self.cubeMap = cubeMap
    
    def zeros(self,cubeMap):
        """ Initializes the rmvShapeToImageSF object.
         
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
            """
        self.cubeMap = cubeMap
     
    def applyToPoint(self, centre,position,*value):
        """ Applies a processing corresponding at the pixel located at (position[0], position[1])
        coordinates and registers the value at this position in the kernelDecorated.cubeMap.grille object.
             
        Parameters
        ----------
        center   : tupple of float
                   Position of the object center
        position : tupple of float
                   Position of the considered pixel (inside the object spatial support)
        value    : float
                   Possible value of the intensity at the position (position[0], position[1]) not always necessary.
            """
        if centre in self.cubeMap.grille[position[0]][position[1]].vecShapes.keys():
            del self.cubeMap.grille[position[0]][position[1]].vecShapes[centre]


  

class Interactions(object):
    """ This class manages the Interactions object.
    
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration     
        
        Attributes
        ----------
        cubeMap      : class 'kernelDecorated.CubeMap'
                       CubeMap object containing the information about the object configuration
        interactions : dictionary
                       keys are the index (vectorized coordinate of the center) of objects
                       which interact with the considered object (in the applyToPoint method)
                       and values correspond to the scalar product between the intensity of the considered object
                       and the intensity of the objects of the dictionary.
        index        : dictionary
                       Dictionary containing the correspondance between the index obtained
                       by vectorizing the coordinates of the object center and the index given
                       in the CubeMap.list_index object (which correspond to the object apparition order)
    """

    def __init__(self,cubeMap):
        """ Creates an Interactions object.
        
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
            """
        self.cubeMap = cubeMap.copy()
        self.interactions = {}
        self.index = {}
    
    def zeros(self,cubeMap):
        """ Initializes the Interactions object.
        
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
            """
        self.cubeMap = cubeMap
        self.interactions.clear()
        self.index.clear()
    
    
    def applyToPoint(self, centre,position,value):
        """ Computes the scalar product between the new proposed object and the other objects of the configuration.
        
        Parameters
        ----------
        center   : tupple of float
                   Position of the object center
        position : tupple of float
                   Position of the considered pixel (inside the object spatial support)
        value    : float
                   Value of the intensity at the position (position[0], position[1])
            """
        if len(self.cubeMap.grille[position[0]][position[1]].vecShapes.keys()) > 0:
            for c in self.cubeMap.grille[position[0]][position[1]].vecShapes.keys():
                
                index_centre = int(c[1]*self.cubeMap.LI+c[0])
                self.index[index_centre] = self.cubeMap.index_obj[c]
                if index_centre in self.interactions.keys():
                    self.interactions[index_centre] = self.interactions[index_centre] + self.cubeMap.grille[position[0]][position[1]].vecShapes[c]*value
                else:
                    self.interactions[index_centre] = self.cubeMap.grille[position[0]][position[1]].vecShapes[c]*value
    
    
        index_centre = int(centre[1]*self.cubeMap.LI+centre[0])
        n = len(self.cubeMap.list_obj.keys())
        self.index[index_centre] = n
        if index_centre in self.interactions.keys():
            self.interactions[index_centre] = self.interactions[index_centre] + value*value
        else:
            self.interactions[index_centre] = value*value

    def tab_return(self,centre):
        """ Returns the different interactions in an 1D array that respects the objects index of the 'cubeMap.list_index' list.
            
        Parameters
        ----------
        center   : tupple of float
                   Position of the object center
        """
        
        n = len(self.cubeMap.list_obj.keys())
        tab = np.zeros(n+1)
        
        for ind in self.interactions.keys():
            index = self.index[ind]
            tab[int(index)] = self.interactions[ind]
        
        return tab

class CubeInteractions(object):
    """ This class manages the CubeInteractions object.
    It computes the scalar product between the new proposed object
    and the matrix composed of the unit vector and the data at each wavelength.
    
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
        
        Attributes
        ----------
        cubeMap              : class 'kernelDecorated.CubeMap'
                               CubeMap object containing the information about the object configuration
        cubeInteractions_tmp : array
                               ...
        cubeInteractions     : list
                               ...
        """


    def __init__(self,cubeMap):
        """ Creates an CubeInteractions object.
        
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
            """
        self.cubeMap = cubeMap
        self.cubeInteractions_tmp = np.zeros((self.cubeMap.cube.shape[0]+1))
        self.cubeInteractions = []
    
    def zeros(self,cubeMap):
        """ Initializes the CubeInteractions object.
            
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
        """
        self.cubeMap = cubeMap
        self.cubeInteractions_tmp[:] = 0.
        del(self.cubeInteractions[:])
        
    def applyToPoint(self, centre,position,value):
        """ Computes the scalar product between the new proposed object and the matrix composed of the unit vector and the data at each wavelength.
        
        Parameters
        ----------
        center   : tupple of float
                   Position of the object center
        position : tupple of float
                   Position of the considered pixel (inside the object spatial support)
        value    : float
                   Value of the intensity at the position (position[0], position[1])
            """
        self.cubeInteractions_tmp[0]= self.cubeInteractions_tmp[0] + value
        self.cubeInteractions_tmp[1:self.cubeMap.cube.shape[0]+1] = self.cubeInteractions_tmp[1:self.cubeMap.cube.shape[0]+1] + self.cubeMap.cube[0:self.cubeMap.cube.shape[0],position[0],position[1]]*value
    

            
    def tab_return(self):
        """Returns the cubeInteractions_tmp array"""
        return self.cubeInteractions_tmp

        
        
class Ellipse_scalar_product(object):
    """ This class manages the Ellipse_scalar_product object.
    It computes the ratio of shared energy with other objects and returns the list of these objects.
         
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
         
        Attributes
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
        dict    : dictionary
                  Keys are the objects center coordinates which interact with the considered object and values are the shared energy.
    """
    def __init__(self,cubeMap):
        """Creates an Ellipse_scalar_product functor object
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
        """
        self.dict = {}
        self.cubeMap = cubeMap
     
    def zeros(self,cubeMap):
        """Initializes an Ellipse_scalar_product functor object
        Parameters
        ----------
        cubeMap : class 'kernelDecorated.CubeMap'
                  CubeMap object containing the information about the object configuration
            """
        self.dict.clear()
        self.cubeMap = cubeMap
     
    def applyToPoint(self, centre, position, value):
        """ Computes the scalar product of the considered object with the other objects of the configuration.
         
        Parameters
        ----------
        center   : tupple of float
                   Position of the object center
        position : tupple of float
                   Position of the considered pixel (inside the object spatial support)
        value    : float
                   Value of the intensity at the position (position[0], position[1])
            """
        if len(self.cubeMap.grille[position[0]][position[1]].vecShapes.keys()) != 0 :
            for c in self.cubeMap.grille[position[0]][position[1]].vecShapes.keys():               
                if c in self.dict.keys():
                    self.dict[c] = self.dict[c] +  value*self.cubeMap.grille[position[0]][position[1]].vecShapes[c]
                else:
                    self.dict[c] =   value*self.cubeMap.grille[position[0]][position[1]].vecShapes[c]

            