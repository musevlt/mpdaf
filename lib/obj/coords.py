""" coords.py Manages coordinates"""
import numpy as np
import pyfits
import pywcs
import astropysics.coords

def deg2sexa(x):
    """Transforms the values of n coordinates from degrees to sexagesimal.
    Returns an (n,2) array of x- and y- coordinates in sexagesimal (string)

    Parameters
    ----------
    x : float array
    An (n,2) array of x- and y- coordinates in degrees
    """
    x = np.array(x)
    if len(np.shape(x))==1 and np.shape(x)[0]==2:
        ra = deg2hms(x[0])
        dec = deg2dms(x[1])
        return np.array([ra,dec])
    elif len(np.shape(x))==2 and np.shape(x)[1]==2:
        result = []
        for i in range(np.shape(x)[0]):
            ra = deg2hms(x[i][0])
            dec = deg2dms(x[i][1])
            result.append(np.array([ra,dec]))
        return np.array(result)
    else:
        raise ValueError, 'Operation forbidden'


def sexa2deg(x):
    """Transforms the values of n coordinates from sexagesimal to degrees.
    Returns an (n,2) array of x- and y- coordinates in degrees.

    Parameters
    ----------
    x : string array
    An (n,2) array of x- and y- coordinates in sexagesimal
    """
    x = np.array(x)
    if len(np.shape(x))==1 and np.shape(x)[0]==2:
        ra = hms2deg(x[0])
        dec = dms2deg(x[1])
        return np.array([ra,dec])
    elif len(np.shape(x))==2 and np.shape(x)[1]==2:
        result = []
        for i in range(np.shape(x)[0]):
            ra = hms2deg(x[i][0])
            dec = dms2deg(x[i][1])
            result.append(np.array([ra,dec]))
        return np.array(result)
    else:
        raise ValueError, 'Operation forbidden'

def deg2hms(x):
    """Transforms a degree value to a string representation of the coordinate as hours:minutes:seconds
    """
    ac = astropysics.coords.AngularCoordinate(x)
    hms = ac.getHmsStr(canonical=True)
    return hms

def hms2deg(x):
    """Transforms a string representation of the coordinate as hours:minutes:seconds to a float degree value
    """
    ac = astropysics.coords.AngularCoordinate(x,sghms=True)
    deg = ac.d
    return deg

def deg2dms(x):
    """Transforms a degree value to a string representation of the coordinate as degrees:arcminutes:arcseconds
    """
    ac = astropysics.coords.AngularCoordinate(x)
    dms = ac.getDmsStr(canonical=True)
    return dms

def dms2deg(x):
    """Transforms a string representation of the coordinate as degrees:arcminutes:arcseconds to a float degree value
    """
    ac = astropysics.coords.AngularCoordinate(x)
    deg = ac.d
    return deg


class WCS(object):
    """WCS class manages world coordinates

    Attributes
    ----------
    wcs : pywcs.WCS
    World coordinates

    Public methods
    --------------
    Creation: init, copy

    Coordinate transformation: sky2pix, pix2sky

    Info: info, isEqual
    """
    def __init__(self,hdr=None,crpix=None,dim=None,crval=(0.0,0.0),cdelt=(1.0,1.0),deg=False,rot=0):
        """creates a WCS object

        Parameters
        ----------
        crpix : float or (float,float)
        Reference pixel coordinates.
        If crpix=None, crpix = (dim+1)/2.0 and the reference point is the center of the image.
        If crpix=None and dim=None, crpix = 1.5 and the reference point is the first pixel in the image.
        Note that for crpix definition, the first pixel in the image has pixel coordinates (1.0,1.0).

        dim : integer or (integer,integer)
        Lengths of the image in X and Y.
        Note that dim=(nx,ny) is not equal to the numpy data shape: np.shape(data)=(ny,nx)

        crval : float or (float,float)
        Coordinates of the reference pixel.
        (0.0,0.0) by default.

        cdelt : float or (float,float)
        Sizes of one pixel along the axis.
        (1.0,1.0) by default.

        deg : boolean
        If True, world coordinates are in decimal degrees

        rot: float
        Rotation angle in degree

        hdr : pyfits.CardList
        A FITS header.
        If hdr is not equal to None, WCS object is created from data header and other parameters are not used.

        Examples
        --------
        WCS(hdr) creates a WCS object from data header
        WCS(): the reference point is the first pixel in the image
        WCS(crval=0,cdelt=0.2,dim=300): the reference point is the center of the image.
        WCS(crval=0,cdelt=0.2,crpix=150.5): the reference point is the center of the image.
        WCS(crval=(1.46E+02,-3.11E+01),cdelt=(4E-04,5E-04),dim=500, deg=True, rot = 20): the reference point is in decimal degree
        """
        if hdr!=None:
            self.wcs = pywcs.WCS(hdr)  # WCS object from data header
        else:
            #check attribute dimensions
            if len(crval) == 1:
                crval = (crval,crval)
            elif len(crval) == 2:
                pass
            else:
                raise ValueError, 'crval with dimension > 2'
            if len(cdelt) == 1:
                cdelt = (cdelt,cdelt)
            elif len(cdelt) == 2:
                pass
            else:
                raise ValueError, 'cdelt with dimension > 2'
            if crpix!=None:
                if len(crpix) == 1:
                    crpix = (crpix,crpix)
                elif len(crpix) == 2:
                    pass
                else:
                    raise ValueError, 'crpix with dimension > 2'
            if dim!=None:
                if len(dim) == 1:
                    dim = (dim,dim)
                elif len(dim) == 2:
                    pass
                else:
                    raise ValueError, 'dim with dimension > 2'
            #create pywcs object
            self.wcs = pywcs.WCS(naxis=2)
            #reference pixel
            if crpix!=None:
                self.wcs.wcs.crpix = np.array(crpix)
            elif dim!=None:
                crpix1 = (dim[0] + 1) / 2.0
                crpix2 = (dim[1] + 1) / 2.0
                self.wcs.wcs.crpix = np.array([crpix1,crpix2])
            else:
                self.wcs.wcs.crpix = np.array([1.0,1.0])
            #value of reference pixel
            self.wcs.wcs.crval = np.array(crval)
            if deg: #in decimal degree
                self.wcs.wcs.ctype = ['RA---TAN','DEC---TAN']
                self.wcs.wcs.cunit = ['deg','deg']
                self.wcs.wcs.cd = np.array([[-cdelt[0], 0], [0, cdelt[1]]])
            else:   #in pixel or arcsec
                self.wcs.wcs.ctype = ['PIXEL','PIXEL']
                self.wcs.wcs.cunit = ['UNITLESS','UNITLESS']
                self.wcs.wcs.cd = np.array([[cdelt[0], 0], [0, cdelt[1]]])
            # rotation
            self.wcs.rotateCD(rot)
            # dimension
            if dim!=None :
                self.wcs.naxis1 = dim[0]
                self.wcs.naxis2 = dim[1]

    def copy(self):
        """copies WCS object in a new one and returns it
        """
        out = WCS()
        out.wcs = self.wcs.__copy__()
        return out

    def info(self):
        """prints information
        """
        self.wcs.printwcs()
        print 'CUNIT \t : %s %s'%(self.wcs.wcs.cunit[0],self.wcs.wcs.cunit[1])

    def sky2pix(self,x):
        """converts world coordinates to pixel coordinates
        Returns an (n,2) array of x- and y- pixel coordinates

        Parameters
        ----------
        x : array
        An (n,2) array of x- and y- world coordinates
        """
        x = np.array(x,np.float_)
        if len(np.shape(x))==1 and np.shape(x)[0]==2:
            pixsky = np.array([[x[0],x[1]]])
        elif len(np.shape(x))==2 and np.shape(x)[1]==2:
            pixsky = x
        pixcrd = self.wcs.wcs_sky2pix(pixsky,0)
        return pixcrd

    def pix2sky(self,x):
        """converts pixel coordinates to world coordinates
        Returns an (n,2) array of x- and y- world coordinates

        Parameters
        ----------
        x : array
        An (n,2) array of x- and y- pixel coordinates
        """
        x = np.array(x,np.float_)
        if len(np.shape(x))==1 and np.shape(x)[0]==2:
            pixcrd = np.array([[x[0],x[1]]])
        elif len(np.shape(x))==2 and np.shape(x)[1]==2:
            pixcrd = x
        pixsky = self.wcs.wcs_pix2sky(pixcrd,0)
        return pixsky

    def isEqual(self,other):
        """returns True if other and self have the same attributes
        """
        if isinstance(other,WCS):
            if self.wcs.naxis1 == other.wcs.naxis1 and self.wcs.naxis2 == other.wcs.naxis2 and \
               (self.wcs.wcs.crpix == other.wcs.wcs.crpix).all() and (self.wcs.wcs.crval == other.wcs.wcs.crval).all() and \
               (self.wcs.wcs.cd == other.wcs.wcs.cd).all() :
                return True
            else:
                return False
        else:
            return False

class WaveCoord(object):
    """WaveCoord class manages coordinates of spectrum

    Attributes
    ----------
    dim : integer
    Size of spectrum

    crpix : float
    Reference pixel coordinates.
    Note that for crpix definition, the first pixel in the spectrum has pixel coordinates 1.0.
    1.0 by default.

    crval : float
    Coordinates of the reference pixel (0.0 by default).

    cdelt : float
    Step in wavelength (1.0 by default).

    cunit : string
    Wavelength unit (Angstrom by default).

    Public methods
    --------------
    Creation: init, copy

    Coordinate transformation: [], pixel

    Info: info, isEqual
    """

    def __init__(self, dim=101, crpix=1.0, cdelt=1.0, crval=0.0, cunit = 'Angstrom'):
        """creates a WaveCoord object

        Parameters
        ----------
        dim : integer
        Size of spectrum

        crpix : float
        Reference pixel coordinates.
        Note that for crpix definition, the first pixel in the spectrum has pixel coordinates 1.0.
        1.0 by default.

        crval : float
        Coordinates of the reference pixel (0.0 by default).

        cdelt : float
        Step in wavelength (1.0 by default).

        cunit : string
        Wavelength unit (Angstrom by default).
        """
        self.dim = dim
        self.crpix = crpix
        self.cdelt = cdelt
        self.crval = crval
        self.cunit = cunit

    def copy(self):
        """copies WaveCoord object in a new one and returns it
        """
        out = WaveCoord()
        out.dim = self.dim
        out.crpix = self.crpix
        out.cdelt = self.cdelt
        out.crval = self.crval
        out.cunit = self.cunit
        return out

    def info(self):
        """prints information
        """
        print 'Wavelength coordinates'
        print 'N:\t %i'%self.dim
        print 'CRPIX:\t %d'%self.crpix
        print 'CDELT:\t %d'%self.cdelt #CDELT3 our CD3_3
        print 'CRVAL:\t %d'%self.crval
        print 'CUNIT:\t %s'%self.cunit

    def isEqual(self,other):
        '''returns True if other and self have the same attributes
        '''
        if isinstance(other,WaveCoord):
            if self.crpix == other.crpix and self.cdelt == other.cdelt and \
               self.crval == other.crval and self.cunit == other.cunit and \
               self.dim == other.dim :
                return True
            else:
                return False
        else:
            return False

    def __getitem__(self, pixel):
        """ returns the coordinate(s) corresponding to pixel(s)
        If [:] the full coordinate array is returned
        """
        pix = np.arange(self.dim,dtype=np.float)
        lbda = (pix - self.crpix + 1) * self.cdelt + self.crval
        return lbda[pixel]

    def pixel(self, lbda, nearest=False):
        """ Returns the decimal pixel corresponding to the wavelength lbda
        If nearest=True; returns the nearest integer pixel
        """
        pix = (lbda - self.crval)/self.cdelt + self.crpix - 1
        if nearest:
            return(int(pix+0.5))
        return pix



