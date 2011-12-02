""" image.py Manages image object"""
import numpy as np
import pyfits
import pywcs

class Image(object):
    """Image class

    Attributes
    ----------
    filename : string
    Possible FITS filename

    unit : string
    Possible data unit type

    cards : pyfits.CardList
    Possible FITS header instance

    data : array or masked array
    Array containing the pixel values of the image

    nx,ny : integers
    Lengths of data in X and Y

    noise : array
    Array containing the variance

    fscale : float
    Flux scaling factor (1 by default)

    maskinfo : string
    Information about the masked array (if relevant)

    wcs : pywcs.WCS
    World coordinates

    Public methods
    --------------
    Creation: init, copy

    Arithmetic: + - * / pow

    Info: info, []
    """

    def __init__(self, dim=(101,101), start=(0.,0.), step=(1.0,1.0), unit=None, data=None, var=None, noise=False, empty=False, fscale=1):
        """creates a Image object

        Parameters
        ----------
        dim : (integer,integer)
        Lengths of data in X and Y. (101,101) by default.

        start : (float,float)
        Physical positions of the pixel [0,0].
        (0,0) by default.

        step : (float,float)
        sizes of one pixel along the axis.
        (1.0,1.0) by default.

        unit : string
        Possible data unit type. None by default.

        data : array
        Array containing the pixel values of the image. None by default.

        var : array
        Array containing the variance. None by default.

        noise : bool
        Use noise=False to create image without variance extension

        empty : bool
        If empty is True, the data and noise array are set to None

        fscale : float
        Flux scaling factor (1 by default)
        """
        #possible FITS filename
        self.filename = None
        #possible data unit type
        self.unit = unit
        # possible FITS header instance
        self.cards = pyfits.CardList()
        #data
        if data is None:
            if empty:
                self.data = None
            else:
                self.data = np.zeros(dim, dtype = float)
            self.nx, self.ny = dim
            if not noise or empty:
                self.noise = None
            else:
                self.noise = numpy.zeros(dim, dtype = float)
        else:
            self.data = data
            self.noise = var
            self.nx, self.ny = data.shape
        self.fscale = fscale
        self.maskinfo = ""
        #coordinates
        self.wcs = pywcs.WCS(naxis=2)
        crpix1 = (self.nx + 1) / 2.0
        crpix2 = (self.ny + 1) / 2.0
        self.wcs.wcs.crpix = np.array([crpix1,crpix2])
        self.wcs.wcs.cdelt = np.array(step)
        crval1 = start[0] + (crpix1 - 1) * step[0]
        crval2 = start[1] + (crpix2 - 1) * step[1]
        self.wcs.wcs.crval = np.array([crval1,crval2])
        self.wcs.wcs.pc = np.array([[1, 0], [0, 1]])
        self.wcs.wcs.ctype = ['PIXEL','PIXEL']
        self.wcs.wcs.cunit = ['UNITLESS','UNITLESS']

    def copy(self):
        """copies Image object in a new one and returns it
        """
        ima = Image(empty=True)
        ima.filename = self.filename
        ima.unit = self.unit
        ima.cards = pyfits.CardList(self.cards)
        ima.nx = self.nx
        ima.ny = self.ny
        try:
            ima.data = self.data.__copy__()
        except:
            ima.data = None
        try:
            ima.noise = self.noise.__copy__()
        except:
            ima.noise = None
        ima.fscale = self.fscale
        ima.maskinfo = self.maskinfo
        ima.wcs = self.wcs.__copy__()
        return ima

    def info(self):
        """prints information
        """
        if self.filename != None:
            print self.filename
        else:
            print 'no name'
        if isinstance(self.data,np.ma.core.MaskedArray):
            print 'masked array:\t(%i,%i) %s'% (self.nx,self.ny,self.maskinfo)
        elif self.data != None:
            print 'image data:\t(%i,%i)'% (self.nx,self.ny)
        else:
            print 'no data'
        print 'fscale:\t %d'%self.fscale
        if self.unit != None:
            print 'data unit:\t %s'%self.unit
        else:
            print 'no data unit'
        if self.noise != None:
            print 'noise variance:\t(%i,%i)'% (self.nx,self.ny)
        else:
            print 'no noise'
        self.wcs.printwcs()

    def __le__ (self, item):
        """masks data array where greater than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        result.data = np.ma.masked_greater(result.data, item/result.fscale)
        result.maskinfo += " <= %d"%item
        return result

    def __lt__ (self, item):
        """masks data array where greater or equal than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        result.data = np.ma.masked_greater_equal(result.data, item/result.fscale)
        result.maskinfo += " < %d"%item
        return result

    def __ge__ (self, item):
        """masks data array where less than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        result.data = np.ma.masked_less(result.data, item/result.fscale)
        result.maskinfo += " >= %d"%item
        return result

    def __gt__ (self, item):
        """masks data array where less or equal than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        result.data = np.ma.masked_less_equal(result.data, item/result.fscale)
        result.maskinfo += " >%d"%item
        return result

    def _isData (self):
        """returns False if Data is None
        """
        try :
            if self.data == None:
                return False
            else:
                return True
        except:
            return True

    def __add__(self, other):
        """ adds either a number or an image (pixel per pixel)
        """
        if not self._isData():
            raise ValueError, 'empty data array'
        res = self.copy()
        if type(other) is float or type(other) is int:
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__add__(self.data,other/np.double(self.fscale))
            else:
                res.data = np.ndarray.__add__(self.data,other/np.double(self.fscale))
        elif isinstance(other, Image):
            if self.nx != other.nx or self.ny != other.ny or not other.isData():
                raise ValueError, 'Operation forbidden for images with different sizes'
            else:
                if isinstance(self.data,np.ma.core.MaskedArray):
                    res.data = np.ma.MaskedArray.__add__(self.data,other.data*np.double(other.fscale/self.fscale))
                elif isinstance(other.data,np.ma.core.MaskedArray):
                    res.data = np.ma.MaskedArray.__add__(other.data*np.double(other.fscale/self.fscale),self.data)
                else:
                    res.data = np.ndarray.__add__(self.data,other.data*np.double(other.fscale/self.fscale))
                res.noise = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

    def __radd__(self, other):
        if type(other) is float or type(other) is int:
            return self.__add__(other)
        elif isinstance(other, Image):
            return other.__add__(self)
        else:
            raise ValueError, 'Operation forbidden'

    def __sub__(self, other):
        """ subtracts either a number or an image (pixel par pixel)
        """
        if not self._isData():
            raise ValueError, 'empty data array'
        res = self.copy()
        if type(other) is float or type(other) is int:
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__sub__(self.data,other/np.double(self.fscale))
            else:
                res.data = np.ndarray.__sub__(self.data,other/np.double(self.fscale))
        elif isinstance(other, Image):
            if self.nx != other.nx or self.ny != other.ny or not other.isData():
                raise ValueError, 'Operation forbidden for images with different sizes'
            else:
                if isinstance(self.data,np.ma.core.MaskedArray):
                    res.data = np.ma.MaskedArray.__sub__(self.data,other.data*np.double(other.fscale/self.fscale))
                elif isinstance(other.data,np.ma.core.MaskedArray):
                    res.data = np.ma.MaskedArray.__sub__(other.data*np.double(other.fscale/self.fscale),self.data)
                else:
                    res.data = np.ndarray.__sub__(self.data,other.data*np.double(other.fscale/self.fscale))
                res.noise = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

    def __rsub__(self, other):
        if type(other) is float or type(other) is int:
            return self.__sub__(other)
        elif isinstance(other, Image):
            return other.__sub__(self)
        else:
            raise ValueError, 'Operation forbidden'

    def __mul__(self, other):
        """multiplies either a number or an image (pixel par pixel)
        """
        if not self._isData():
            raise ValueError, 'empty data array'
        res = self.copy()
        if type(other) is float or type(other) is int:
            res.fscale *= other
            if res.noise != None:
                res.noise *= other*other
        elif isinstance(other, Image):
            if self.nx != other.nx or self.ny != other.ny or not other.isData():
                raise ValueError, 'Operation forbidden for images with different sizes'
            else:
                if isinstance(self.data,np.ma.core.MaskedArray):
                    res.data = np.ma.MaskedArray.__mul__(self.data,other.data)
                elif isinstance(other.data,np.ma.core.MaskedArray):
                    res.data = np.ma.MaskedArray.__sub__(other.data,self.data)
                else:
                    res.data = np.ndarray.__sub__(self.data,other.data)
                res.fscale *= other.fscale
                res.noise = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

    def __rmul__(self, other):
        if type(other) is float or type(other) is int:
            return self.__mul__(other)
        elif isinstance(other, Image):
            return other.__mul__(self)
        else:
            raise ValueError, 'Operation forbidden'

    def __div__(self, other):
        """divides either a number or an image (pixel par pixel)
        """
        if not self._isData():
            raise ValueError, 'empty data array'
        res = self.copy()
        if type(other) is float or type(other) is int:
            res.fscale /= other
            if res.noise != None:
                res.noise /= other*other
        elif isinstance(other, Image):
            if self.nx != other.nx or self.ny != other.ny or not other.isData():
                raise ValueError, 'Operation forbidden for images with different sizes'
            else:
                if isinstance(self.data,np.ma.core.MaskedArray):
                    res.data = np.ma.MaskedArray.__div__(self.data,other.data)
                elif isinstance(other.data,np.ma.core.MaskedArray):
                    res.data = np.ma.MaskedArray.__div__(other.data,self.data)
                else:
                    res.data = np.ndarray.__div__(self.data,other.data)
                res.fscale /= other.fscale
                res.noise = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

    def __rdiv__(self, other):
        if type(other) is float or type(other) is int:
            return self.__div__(other)
        elif isinstance(other, Image):
            return other.__div__(self)
        else:
            raise ValueError, 'Operation forbidden'

    def __pow__(self, other):
        """computes the power exponent"""
        if not self._isData():
            raise ValueError, 'empty data array'
        res = self.copy()
        if type(other) is float or type(other) is int:
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__pow__(self.data,other)
            else:
                res.data = np.ndarray.__pow__(self.data,other)
            res.fscale = res.fscale**other
            res.noise = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

    def __getitem__(self, item):
        #tuple case
        if isinstance(item, tuple):
            if len(item)==2:
                if isinstance(item[0], int) and isinstance(item[1], int):
                    return self.data[item]
                elif isinstance(item[0], float) and isinstance(item[1], float):
                    pixsky = np.array([[item[0],item[1]]], np.float_)
                    # Convert world coordinates to pixel coordinates
                    pixcrd = self.wcs.wcs_sky2pix(pixsky, 0)
                    return self.data[pixcrd[0][0],pixcrd[0][1]]
                else:
                    raise ValueError, 'Operation forbidden'


def imageFromFITS(filename, getnoise=False, ext=None, verbose = False):
    """ read image from fits file
    filename: name of fits file
    getnoise: if True the Noise Variance spectrum is read (if it exist)
    ext: extension number to read in case of a multifits file (if not set, the DATA extension is searched)
    """
    f = pyfits.open(filename)
    # create image object
    ima = Image(empty=True)
    ima.filename = filename
    # we read the primary header
    hdr = f[0].header
    if len(f) == 1:
        # if the number of extension is 1, we just read the data from the primary header
        # test if image
        if hdr['NAXIS'] != 2:
            raise IOError, '  not an image'
        ima.unit = hdr.get('UNIT', None)
        ima.cards = hdr.ascardlist()
        ima.nx = hdr['NAXIS1']
        ima.ny = hdr['NAXIS2']
        ima.data = f[0].data
        ima.noise = None
        ima.fscale = hdr.get('FSCALE', 1.0)
        ima.wcs = pywcs.WCS(hdr)  # WCS object from data header
    else:
        if ext is None:
            h = f['DATA'].header
            d = f['DATA'].data
        else:
            h = f[ext].header
            d = f[ext].data
        if h['NAXIS'] != 2:
            raise IOError, 'Wrong dimension number in DATA extension'
        ima.unit = hdr.get('UNIT', None)
        ima.cards = h.ascardlist()
        ima.nx = h['NAXIS1']
        ima.ny = h['NAXIS2']
        ima.data = d
        ima.fscale = h.get('FSCALE', 1.0)
        ima.wcs = pywcs.WCS(h)  # WCS object from data header
        if getnoise:
            if f['VARIANCE'].header['NAXIS'] != 2:
                raise IOError, 'Wrong dimension number in VARIANCE extension'
            if f['VARIANCE'].header['NAXIS1'] != ima.dim[0] and f['VARIANCE'].header['NAXIS2'] != ima.dim[1]:
                raise IOError, 'Number of points in VARIANCE not equal to DATA'
            ima.noise = f['VARIANCE'].data
        else:
            ima.noise = None
    f.close()
    return ima
