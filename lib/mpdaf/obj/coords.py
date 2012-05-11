""" coords.py Manages coordinates"""
import numpy as np
import pyfits
import pywcs
from astropysics_coords import AstropysicsAngularCoordinate

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
    ac = AstropysicsAngularCoordinate(x)
    hms = ac.getHmsStr(canonical=True)
    return hms

def hms2deg(x):
    """Transforms a string representation of the coordinate as hours:minutes:seconds to a float degree value
    """
    ac = AstropysicsAngularCoordinate(x,sghms=True)
    deg = ac.d
    return deg

def deg2dms(x):
    """Transforms a degree value to a string representation of the coordinate as degrees:arcminutes:arcseconds
    """
    ac = AstropysicsAngularCoordinate(x)
    dms = ac.getDmsStr(canonical=True)
    return dms

def dms2deg(x):
    """Transforms a string representation of the coordinate as degrees:arcminutes:arcseconds to a float degree value
    """
    ac = AstropysicsAngularCoordinate(x)
    deg = ac.d
    return deg

def deg2rad(deg):
    return (deg * np.pi / 180.)
  
def rad2deg(rad):
    return (rad * 180. / np.pi)


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
    def __init__(self,hdr=None,crpix=None,crval=(0.0,0.0),cdelt=(1.0,1.0),deg=False,rot=0, shape = None):
        """creates a WCS object

        Parameters
        ----------
        crpix : float or (float,float)
        Reference pixel coordinates.
        If crpix is None and shape is None crpix = 1.0 and the reference point is the first pixel of the image.
        If crpix is None and shape is not None crpix = (shape + 1.0)/2.0 and the reference point is the center of the image.
        Note that for crpix definition, the first pixel in the image has pixel coordinates (1.0,1.0).

        crval : float or (float,float)
        Coordinates of the reference pixel.
        (0.0,0.0) by default.

        cdelt : float or (float,float)
        Sizes of one pixel along the axis.
        (1.0,1.0) by default.

        deg : boolean
        If True, world coordinates are in decimal degrees (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg)
        If False (by default), world coordinates are linear (CTYPE1=CTYPE2='LINEAR')

        rot: float
        Rotation angle in degree

        hdr : pyfits.CardList
        A FITS header.
        If hdr is not equal to None, WCS object is created from data header and other parameters are not used.
        
        shape : array of 2 integers
        Lengths of data in Y and X (python notation: (ny,nx))

        Examples
        --------
        WCS(hdr) creates a WCS object from data header
        WCS(): the reference point is the first pixel in the image
        WCS(crval=0,cdelt=0.2,crpix=150.5): the reference point is the center of the image.
        WCS(crval=(1.46E+02,-3.11E+01),cdelt=(4E-04,5E-04), deg=True, rot = 20): the reference point is in decimal degree
        """
        if hdr!=None:
            self.wcs = pywcs.WCS(hdr,naxis=2)  # WCS object from data header
            try:
                dx = np.sqrt(self.wcs.wcs.cd[0,0]*self.wcs.wcs.cd[0,0] + self.wcs.wcs.cd[0,1]*self.wcs.wcs.cd[0][1])
                dy = np.sqrt(self.wcs.wcs.cd[1,0]*self.wcs.wcs.cd[1,0] + self.wcs.wcs.cd[1,1]*self.wcs.wcs.cd[1][1])
                self.cdelt = np.array([dx,dy])
                self.rot = deg2rad( np.arctan2(self.wcs.wcs.cd[1,0],self.wcs.wcs.cd[1,1]) )
            except:
                try:
                    dx = self.wcs.wcs.cdelt[0]*np.sqrt(self.wcs.wcs.pc[0,0]*self.wcs.wcs.pc[0,0] + self.wcs.wcs.pc[0,1]*self.wcs.wcs.pc[0][1])
                    dy = self.wcs.wcs.cdelt[1]*np.sqrt(self.wcs.wcs.pc[1,0]*self.wcs.wcs.pc[1,0] + self.wcs.wcs.pc[1,1]*self.wcs.wcs.pc[1][1])
                    self.cdelt = np.array([dx,dy])
                    self.rot = deg2rad( np.arctan2(self.wcs.wcs.pc[1,0],self.wcs.wcs.pc[1,1]) )
                except:
                    self.cdelt = None
                    self.rot = None
            # bug http://mail.scipy.org/pipermail/astropy/2011-April/001242.html if naxis=3
        else:
            #check attribute dimensions
            if isinstance(crval,int) or isinstance(crval,float):
                crval = (crval,crval)
            elif len(crval) == 2:
                pass
            else:
                raise ValueError, 'crval with dimension > 2'
            if isinstance(cdelt,int) or isinstance(cdelt,float):
                cdelt = (cdelt,cdelt)
            elif len(cdelt) == 2:
                pass
            else:
                raise ValueError, 'cdelt with dimension > 2'
            if crpix is not None:
                if isinstance(crpix,int) or isinstance(crpix,float):
                    crpix = (crpix,crpix)
                elif len(crpix) == 2:
                    pass
                else:
                    raise ValueError, 'crpix with dimension > 2'
            if shape is not None:
                if isinstance(shape,int):
                    shape = (shape,shape)
                elif len(shape) == 2:
                    pass
                else:
                    raise ValueError, 'shape with dimension > 2'
            #create pywcs object
            self.wcs = pywcs.WCS(naxis=2)
            self.cdelt = np.array(cdelt)
            #reference pixel
            if crpix is not None:
                self.wcs.wcs.crpix = np.array(crpix)
            else:
                if shape is None:
                    self.wcs.wcs.crpix = np.array([1.0,1.0])
                else:
                    self.wcs.wcs.crpix = (np.array(shape)+1)/2.
            #value of reference pixel
            self.wcs.wcs.crval = np.array(crval)
            if deg: #in decimal degree
                self.wcs.wcs.ctype = ['RA___TAN','DEC___TAN']
                self.wcs.wcs.cunit = ['deg','deg']
                self.wcs.wcs.cd = np.array([[-cdelt[0], 0], [0, cdelt[1]]])
            else:   #in pixel or arcsec
                self.wcs.wcs.ctype = ['LINEAR','LINEAR']
                self.wcs.wcs.cunit = ['UNITLESS','UNITLESS']
                self.wcs.wcs.cd = np.array([[cdelt[0], 0], [0, cdelt[1]]])
            # rotation
            self.wcs.rotateCD(-rot)
            self.rot = rot
            # dimensions
            if shape!=None:
                self.wcs.naxis1 = shape[1]
                self.wcs.naxis2 = shape[0]


    def copy(self):
        """copies WCS object in a new one and returns it
        """
        out = WCS()
        out.wcs = self.wcs.deepcopy()
        out.cdelt = self.cdelt
        out.rot = self.rot
        return out

    def info(self):
        """prints information
        """
        #self.wcs.printwcs()
        if self.wcs.wcs.ctype[0] == 'LINEAR':
            pixcrd = [[0,0],[self.wcs.naxis1 -1,self.wcs.naxis2 -1]]
            pixsky = self.pix2sky(pixcrd)
            print 'spatial coord: min:(%0.1f,%0.1f) max:(%0.1f,%0.1f) step:(%0.1f,%0.1f) rot:%0.1f' %(pixsky[0,0],pixsky[0,1],pixsky[1,0],pixsky[1,1],self.cdelt[0],self.cdelt[1],self.rot)
        else:
            # center in sexadecimal
            xc = (self.wcs.naxis1 -1) / 2.
            yc = (self.wcs.naxis2 -1) / 2.
            pixsky = self.pix2sky([xc,yc])
            sexa = deg2sexa(pixsky)
            ra = sexa[0][0]
            dec = sexa[0][1]
            # step in arcsec
            dx = self.cdelt[0]  * 3600
            dy = self.cdelt[1]  * 3600
            sizex = self.wcs.naxis1 * dx
            sizey = self.wcs.naxis2 * dy
            print 'center:(%s,%s) size in arcsec:(%0.3f,%0.3f) step in arcsec:(%0.3f,%0.3f) rot:%0.1f' %(ra,dec,sizex,sizey,dx,dy,self.rot)

    def to_header(self):
        return self.wcs.to_header()

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
               (self.cdelt == other.cdelt).all() and self.rot == other.rot:
                return True
            else:
                return False
        else:
            return False

    def __getitem__(self, item):
        """ returns the corresponding WCS
        """
        if isinstance(item,tuple) and len(item)==2:
            try:
                if item[0].start is None:
                    imin = 0
                else:
                    imin = item[0].start
                    if imin < 0:
                        imin = self.wcs.naxis1 + imin
                    if imin > self.wcs.naxis1 :
                        imin = self.wcs.naxis1
                if item[0].stop is None:
                    imax = self.wcs.naxis1
                else:
                    imax = item[0].stop
                    if imax < 0:
                        imax = self.wcs.naxis1 + imax
                    if imax > self.wcs.naxis1 :
                        imax = self.wcs.naxis1
            except:
                imin = item[0]
                imax = item[0] +1
            try:
                if item[1].start is None:
                    jmin = 0
                else:
                    jmin = item[1].start
                    if jmin < 0:
                        jmin = self.wcs.naxis2 + jmin
                    if jmin > self.wcs.naxis2 :
                        jmin = self.wcs.naxis2
                if item[1].stop is None:
                    jmax = self.wcs.naxis2
                else:
                    jmax = item[1].stop
                    if jmax < 0:
                        jmax = self.wcs.naxis2 + jmax
                        if jmax > self.wcs.naxis2 :
                            jmax = self.wcs.naxis2
            except:
                jmin = item[1]
                jmax = item[1]+1
            crpix = (self.wcs.wcs.crpix[0]-imin,self.wcs.wcs.crpix[1]-jmin)
            
            res = self.copy()            
            res.wcs.wcs.crpix = np.array(crpix)
            res.wcs.naxis1 = int(imax-imin)       
            res.wcs.naxis2 = int(jmax-jmin)
            return res
        else:
            raise ValueError, 'Operation forbidden'
        
    def get_range(self):
        """returns [ [ra_min,dec_min], [ra_max,dec_max] ]
        """
        pixcrd = [ [0,0], [self.wcs.naxis1 -1,0], [0,self.wcs.naxis2 -1], [self.wcs.naxis1 -1,self.wcs.naxis2 -1]]
        pixsky = self.pix2sky(pixcrd)
        ra_min = np.min(pixsky[:,0])
        dec_min = np.min(pixsky[:,1])
        ra_max = np.max(pixsky[:,0])
        dec_max = np.max(pixsky[:,1])
        return [ [ra_min,dec_min], [ra_max,dec_max] ]
    
    def get_start(self):
        """returns [ra,dec] corresponding to pixel (0,0)
        """
        pixcrd = [[0,0]]
        pixsky = self.pix2sky(pixcrd)
        ra = pixsky[0,0]
        dec = pixsky[0,1]
        return [ra,dec]
    
    def rotate(self, theta):
        """rotates WCS coordinates to new orientation given by theta
        
        Parameter
        ---------
        
        theta : float
        Rotation in degree.
        """
        _theta = deg2rad(theta)
        _mrot = np.zeros(shape=(2,2),dtype=np.double)
        _mrot[0] = (np.cos(_theta),-np.sin(_theta))
        _mrot[1] = (np.sin(_theta),np.cos(_theta))
        try:
            new_cd = np.dot(self.wcs.wcs.cd, _mrot)
            self.wcs.wcs.cd = new_cd
            self.rot = rad2deg( np.arctan2(self.wcs.wcs.cd[1,0],self.wcs.wcs.cd[1,1]) )
        except:
            try:
                new_pc = np.dot(self.wcs.wcs.pc, _mrot)
                self.wcs.wcs.pc = new_pc
                self.rot = rad2deg( np.arctan2(self.wcs.wcs.pc[1,0],self.wcs.wcs.pc[1,1]) )
            except:
                print "problem with rotation"
                self.cdelt = None
                self.rot = None
                
    def rebin(self, step, start):
        """rebins to a new coordinate system.
        """
        if self.wcs.wcs.ctype[0] == 'LINEAR':
            deg = False
        else:
            deg = True
        if start == None:
            xc = 0
            yc = 0
            pixsky = self.pix2sky([xc,yc])
            start = (pixsky[0][0] -0.5*self.cdelt[0] + 0.5*step[0],pixsky[0][1] -0.5*self.cdelt[1] + 0.5*step[1])
        
        res = WCS(crpix=1.0,crval=start,cdelt=step,deg=deg,rot=self.rot)
        return res
    
    def is_deg(self):
        """Returns True if world coordinates are in decimal degrees (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg)
        """
        if self.wcs.wcs.ctype[0] == 'LINEAR':
            return False
        else:
            return True

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

    def __init__(self, crpix=1.0, cdelt=1.0, crval=0.0, cunit = 'Angstrom', shape = None):
        """creates a WaveCoord object

        Parameters
        ----------
    
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
        
        shape : integer
        Size of spectrum.
        """
        self.shape = shape
        self.crpix = crpix
        self.cdelt = cdelt
        self.crval = crval
        self.cunit = cunit


    def copy(self):
        """copies WaveCoord object in a new one and returns it
        """
        out = WaveCoord()
        out.shape = self.shape
        out.crpix = self.crpix
        out.cdelt = self.cdelt
        out.crval = self.crval
        out.cunit = self.cunit
        return out


    def info(self):
        """prints information
        """
        if self.shape is None:
            print 'wavelength: min:%0.2f step:%0.2f %s' %(self.__getitem__(0),self.cdelt,self.cunit)
        else:
            print 'wavelength: min:%0.2f max:%0.2f step:%0.2f %s' %(self.__getitem__(0),self.__getitem__(self.shape-1),self.cdelt,self.cunit)


    def isEqual(self,other):
        '''returns True if other and self have the same attributes
        '''
        if isinstance(other,WaveCoord):
            if self.crpix == other.crpix and self.cdelt == other.cdelt and \
               self.crval == other.crval and self.cunit == other.cunit and \
               self.shape == other.shape :
                return True
            else:
                return False
        else:
            return False


    def coord(self, pixel=None):
        """ returns the coordinate corresponding to pixel
        if pixel is None, the full coordinate array is returned
        """
        if pixel is None:
            if self.shape is None:
                print "error : wavelength coordinates without dimension"
            else:
                pix = np.arange(self.shape,dtype=np.float)
                lbda = (pix - self.crpix + 1) * self.cdelt + self.crval
                return lbda
        else:
            pixel = np.array(pixel)
            return (pixel - self.crpix + 1) * self.cdelt + self.crval

    def pixel(self, lbda, nearest=False):
        """ Returns the decimal pixel corresponding to the wavelength lbda
        If nearest=True; returns the nearest integer pixel
        """
        lbda = np.array(lbda)
        pix = (lbda - self.crval)/self.cdelt + self.crpix - 1
        if nearest:
            if self.shape is None:
                pix = max( int(pix+0.5), 0)
            else:
                try:
                    pix = min( max( int(pix+0.5), 0), self.shape-1)
                except:
                    for i in range(len(pix)):
                        pix[i] = min( max( int(pix[i]+0.5), 0), self.shape-1)
        return pix

    def __getitem__(self, item):
        """ returns the coordinate corresponding to pixel if item is an integer
        returns the corresponding WaveCoord object if item is a slice
        """
        if self.shape is None:
            print "error : wavelength coordinates without dimension"
        else:
            lbda = (np.arange(self.shape,dtype=np.float) - self.crpix + 1) * self.cdelt + self.crval
        if isinstance(item, int):
            return lbda[item]
        elif isinstance(item, slice):
            newlbda = lbda[item]
            dim = newlbda.shape[0]
            if dim < 2:
                raise ValueError, 'Spectrum with dim < 2'
            cdelt = newlbda[1] - newlbda[0]
            res = WaveCoord(crpix=1.0, cdelt=cdelt, crval=newlbda[0], cunit = self.cunit, shape = dim)
            return res
        else:
            raise ValueError, 'Operation forbidden'

    def rebin(self,step,start):
        # vector of pixel edges
        pix = np.arange(self.shape,dtype=np.float)
        lbda = (pix - self.crpix + 1) * self.cdelt + self.crval - 0.5 * self.cdelt
        # vector of new pixel positions
        if start == None:
            start = lbda[0] + step*0.5
        # pixel number necessary to cover old range
        dim = np.ceil((lbda[-1] + self.cdelt - (start-step*0.5)) / step)
        res = WaveCoord(crpix=1.0, cdelt=step, crval=start, cunit = self.cunit, shape = int(dim))
        return res
