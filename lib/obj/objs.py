""" obj.py Manages cube, image and spectrum objects"""
import numpy as np
import pyfits
from coords import WCS
from coords import WaveCoord

class Spectrum(object):
    """Spectrum class

    Attributes
    ----------
    filename : string
    Possible FITS filename

    unit : string
    Possible data unit type

    cards : pyfits.CardList
    Possible FITS header instance

    data : array or masked array
    Array containing the pixel values of the spectrum

    dim : integer
    Size of spectrum

    var : array
    Array containing the variance

    fscale : float
    Flux scaling factor (1 by default)

    maskinfo : string
    Information about the masked array (if relevant)

    wave : WaveCoord
    Wavelength coordinates

    Public methods
    --------------
    Creation: init, copy

    Selection: <, >, <=, >=

    Arithmetic: + - * / pow

    Info: info, []
    """

    def __init__(self, filename=None, ext = None, getnoise=False, dim=101, wave = None, unit=None, data=None, var=None,fscale=1.0,empty=False):
        """creates a Spectrum object

        Parameters
        ----------
        filename : string
        Possible FITS filename

        ext : integer
        Number of the corresponding extension in the file

        getnoise: boolean
        True if the noise Variance spectrum is read (if it exists)
        Use getnoise=False to create spectrum without variance extension

        dim : integer
        size of the spectrum. 101 by default.

        wave : WaveCoord
        Wavelength coordinates

        unit : string
        Possible data unit type. None by default.

        data : array
        Array containing the pixel values of the spectrum. None by default.

        var : array
        Array containing the variance. None by default.

        empty : bool
        If empty is True, the data and variance array are set to None

        fscale : float
        Flux scaling factor (1 by default)

        Examples
        --------
        Spectrum(filename="toto.fits",ext=1,getnoise=False): spectrum from file (extension number is 1).

        wave = WaveCoord(dim=4000, cdelt=1.25, crval=4000.0, cunit = 'Angstrom')
        Spectrum(dim=4000, wave=wave) : spectrum filled with zeros
        Spectrum(wave=wave, data = MyData) : spectrum filled with MyData
        """

        self.spectrum = True
        #possible FITS filename
        self.filename = filename
        self.maskinfo = ""
        if filename is not None:
            f = pyfits.open(filename)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                # if the number of extension is 1, we just read the data from the primary header
                # test if spectrum
                if hdr['NAXIS'] != 1:
                    raise IOError, 'Wrong dimension number: not a spectrum'
                self.unit = hdr.get('UNIT', None)
                self.cards = hdr.ascardlist()
                self.dim =hdr['NAXIS1']
                self.data = f[0].data
                self.var = None
                self.fscale = hdr.get('FSCALE', 1.0)
                if hdr.has_key('CDELT1'):
                    cdelt = hdr.get('CDELT1')
                elif hdr.has_key('CD1_1'):
                    cdelt = hdr.get('CD1_1')
                else:
                    cdelt = 1.0
                crpix = hdr.get('CRPIX1')
                crval = hdr.get('CRVAL1')
                cunit = hdr.get('CUNIT1')
                self.wave = WaveCoord(self.dim, crpix, cdelt, crval, cunit)
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = f['DATA'].data
                else:
                    h = f[ext].header
                    d = f[ext].data
                if h['NAXIS'] != 1:
                    raise IOError, 'Wrong dimension number: not a spectrum'
                self.unit = h.get('UNIT', None)
                self.cards = h.ascardlist()
                self.dim = h['NAXIS1']
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                if h.has_key('CDELT1'):
                    cdelt = h.get('CDELT1')
                elif h.has_key('CD1_1'):
                    cdelt = h.get('CD1_1')
                else:
                    cdelt = 1.0
                crpix = h.get('CRPIX1')
                crval = h.get('CRVAL1')
                cunit = h.get('CUNIT1')
                self.wave = WaveCoord(self.dim, crpix, cdelt, crval, cunit)
                if getnoise:
                    if f['VARIANCE'].header['NAXIS'] != 1:
                        raise IOError, 'Wrong dimension number in VARIANCE extension'
                    if f['VARIANCE'].header['NAXIS1'] != dim:
                        raise IOError, 'Number of points in VARIANCE not equal to DATA'
                    self.var = f['VARIANCE'].data
                else:
                    self.var = None
            f.close()
        else:
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
                self.dim = dim
            else:
                self.data = np.array(data, dtype = float)
                self.dim = data.shape[0]

            if not getnoise or empty:
                self.var = None
            elif var is None:
                self.var = numpy.zeros(dim, dtype = float)
            else:
                self.var = np.array(var, dtype = float)
            self.fscale = np.float(fscale)
            try:
                if wave.dim == self.dim:
                    self.wave = wave
                else:
                    print 'Dimensions of WaveCoord object and DATA are not equal'
            except :
                self.wave = None

    def copy(self):
        """copies spectrum object in a new one and returns it
        """
        spe = Spectrum(empty=True)
        spe.filename = self.filename
        spe.unit = self.unit
        spe.cards = pyfits.CardList(self.cards)
        spe.dim = self.dim
        try:
            spe.data = self.data.__copy__()
        except:
            spe.data = None
        try:
            spe.var = self.var.__copy__()
        except:
            spe.var = None
        spe.fscale = self.fscale
        spe.maskinfo = self.maskinfo
        try:
            spe.wave = self.wave.copy()
        except:
            spe.wave = None
        return spe

    def info(self):
        """prints information
        """
        if self.filename != None:
            print self.filename
        else:
            print 'no name'
        if isinstance(self.data,np.ma.core.MaskedArray):
            print 'masked array:\t(%i,) %s'% (self.dim,self.maskinfo)
        elif self.data is None:
            print 'no data'
        else:
            print 'spectrum data:\t(%i,)'% self.dim
        print 'fscale:\t %d'%self.fscale
        if self.unit is None:
            print 'no data unit'
        else:
            print 'data unit:\t %s'%self.unit
        if self.var is None:
            print 'no noise'
        else:
            print 'noise variance:\t(%i,)'% self.dim
        if self.wave is None:
            print 'no coordinates'
        else:
            self.wave.info()

    def __le__ (self, item):
        """masks data array where greater than a given value.
        Returns a Spectrum object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " <= %d"%item
        return result

    def __lt__ (self, item):
        """masks data array where greater or equal than a given value.
        Returns a Spectrum object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " < %d"%item
        return result

    def __ge__ (self, item):
        """masks data array where less than a given value.
        Returns a Spectrum object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " >= %d"%item
        return result

    def __gt__ (self, item):
        """masks data array where less or equal than a given value.
        Returns a Spectrum object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " >%d"%item
        return result

    def __add__(self, other):
        """ adds other

        spectrum1 + number = spectrum2 (spectrum2[k]=spectrum1[k]+number)

        spectrum1 + spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]+spectrum2[k])
        Dimension must be the same.
        If not equal to None, world coordinates must be the same.

        spectrum + cube1 = cube2 (cube2[k,j,i]=cube1[k,j,i]+spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #spectrum1 + number = spectrum2 (spectrum2[k]=spectrum1[k]+number)
            res = self.copy()
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__add__(self.data,other/np.double(self.fscale))
            else:
                res.data = np.ndarray.__add__(self.data,other/np.double(self.fscale))
            return res
        try:
            #spectrum1 + spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]+spectrum2[k])
            #Dimension must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.spectrum:
                if other.data is None or self.dim != other.dim:
                    print 'Operation forbidden for spectra with different sizes'
                    return None
                else:
                    res = Spectrum(empty=True,dim=self.dim,fscale=self.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for spectra with different world coordinates'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__add__(self.data,other.data*np.double(other.fscale/self.fscale))
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__add__(other.data*np.double(other.fscale/self.fscale),self.data)
                    else:
                        res.data = np.ndarray.__add__(self.data,other.data*np.double(other.fscale/self.fscale))
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                #spectrum + cube1 = cube2 (cube2[k,j,i]=cube1[k,j,i]+spectrum[k])
                #The last dimension of cube1 must be equal to the spectrum dimension.
                #If not equal to None, world coordinates in spectral direction must be the same.
                if other.cube:
                    res = other.__add__(self)
                    return res
            except:
                print 'Operation forbidden'
                return None


    def __sub__(self, other):
        """ subtracts other

        spectrum1 - number = spectrum2 (spectrum2[k]=spectrum1[k]-number)

        spectrum1 + spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]-spectrum2[k])
        Dimension must be the same.
        If not equal to None, world coordinates must be the same.

        spectrum - cube1 = cube2 (cube2[k,j,i]=spectrum[k]-cube1[k,j,i])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #spectrum1 - number = spectrum2 (spectrum2[k]=spectrum1[k]-number)
            res = self.copy()
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__sub__(self.data,other/np.double(self.fscale))
            else:
                res.data = np.ndarray.__sub__(self.data,other/np.double(self.fscale))
            return res
        try:
            #spectrum1 + spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]-spectrum2[k])
            #Dimension must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.spectrum:
                if other.data is None or self.dim != other.dim:
                    print 'Operation forbidden for spectra with different sizes'
                    return None
                else:
                    res = Spectrum(empty=True,dim=self.dim,fscale=self.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for spectra with different world coordinates'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__sub__(self.data,other.data*np.double(other.fscale/self.fscale))
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = -np.ma.MaskedArray.__sub__(other.data*np.double(other.fscale/self.fscale),self.data)
                    else:
                        res.data = np.ndarray.__sub__(self.data,other.data*np.double(other.fscale/self.fscale))
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                #spectrum - cube1 = cube2 (cube2[k,j,i]=spectrum[k]-cube1[k,j,i])
                #The last dimension of cube1 must be equal to the spectrum dimension.
                #If not equal to None, world coordinates in spectral direction must be the same.
                if other.cube:
                    if other.data is None or self.dim != other.dim[2]:
                        print 'Operation forbidden for objects with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,dim=other.dim , wcs= other.wcs, fscale=self.fscale)
                        if self.wave is None or other.wave is None:
                            res.wave = None
                        elif self.wave.isEqual(other.wave):
                            res.wave = self.wave
                        else:
                            print 'Operation forbidden for spectra with different world coordinates'
                            return None
                        if isinstance(self.data,np.ma.core.MaskedArray):
                            res.data = np.ma.MaskedArray.__sub__(self.data[:,np.newaxis,np.newaxis],other.data*np.double(other.fscale/self.fscale))
                        elif isinstance(other.data,np.ma.core.MaskedArray):
                            res.data = -np.ma.MaskedArray.__sub__(other.data*np.double(other.fscale/self.fscale),self.data[:,np.newaxis,np.newaxis])
                        else:
                            res.data = np.ndarray.__sub__(self.data[:,np.newaxis,np.newaxis],other.data*np.double(other.fscale/self.fscale))
                        if self.unit == other.unit:
                            res.unit = self.unit
                        res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                        return res
            except:
                print 'Operation forbidden'
                return None


    def __mul__(self, other):
        """ multiplies by other

        spectrum1 * number = spectrum2 (spectrum2[k]=spectrum1[k]*number)

        spectrum1 * spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]*spectrum2[k])
        Dimension must be the same.
        If not equal to None, world coordinates must be the same.

        spectrum * cube1 = cube2 (cube2[k,j,i]=spectrum[k]*cube1[k,j,i])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.

        spectrum * image = cube (cube[k,j,i]=image[j,i]*spectrum[k]
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #spectrum1 * number = spectrum2 (spectrum2[k]=spectrum1[k]*number)
            res = self.copy()
            res.fscale *= other
            if res.var is not None:
                res.var *= other*other
            return res
        try:
            #spectrum1 * spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]*spectrum2[k])
            #Dimension must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.spectrum:
                if other.data is None or self.dim != other.dim:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Spectrum(empty=True,dim=self.dim,fscale=self.fscale*other.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for spectra with different world coordinates'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__mul__(self.data,other.data)
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__mul__(other.data,self.data)
                    else:
                        res.data = np.ndarray.__mul__(self.data,other.data)
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                res = other.__mul__(self)
                return res
            except:
                print 'Operation forbidden'
                return None


    def __div__(self, other):
        """ divides by other

        spectrum1 / number = spectrum2 (spectrum2[k]=spectrum1[k]/number)

        spectrum1 / spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]/spectrum2[k])
        Dimension must be the same.
        If not equal to None, world coordinates must be the same.

        spectrum / cube1 = cube2 (cube2[k,j,i]=spectrum[k]/cube1[k,j,i])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #spectrum1 / number = spectrum2 (spectrum2[k]=spectrum1[k]/number)
            res = self.copy()
            res.fscale /= other
            if res.var is not None:
                res.var /= other*other
            return res
        try:
            #spectrum1 / spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]/spectrum2[k])
            #Dimension must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.spectrum:
                if other.data is None or self.dim != other.dim:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Spectrum(empty=True,dim=self.dim,fscale=self.fscale/other.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for spectra with different world coordinates'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__div__(self.data,other.data)
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = 1.0 / np.ma.MaskedArray.__div__(other.data,self.data)
                    else:
                        res.data = np.ndarray.__div__(self.data,other.data)
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                #spectrum / cube1 = cube2 (cube2[k,j,i]=spectrum[k]/cube1[k,j,i])
                #The last dimension of cube1 must be equal to the spectrum dimension.
                #If not equal to None, world coordinates in spectral direction must be the same.
                if other.cube:
                    if other.data is None or self.dim != other.dim[2]:
                        print 'Operation forbidden for objects with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,dim=other.dim , wcs= other.wcs, fscale=self.fscale/other.fscale)
                        if self.wave is None or other.wave is None:
                            res.wave = None
                        elif self.wave.isEqual(other.wave):
                            res.wave = self.wave
                        else:
                            print 'Operation forbidden for spectra with different world coordinates'
                            return None
                        if isinstance(self.data,np.ma.core.MaskedArray):
                            res.data = np.ma.MaskedArray.__div__(self.data[:,np.newaxis,np.newaxis],other.data)
                        elif isinstance(other.data,np.ma.core.MaskedArray):
                            res.data = 1.0 / np.ma.MaskedArray.__div__(other.data,self.data[:,np.newaxis,np.newaxis])
                        else:
                            res.data = np.ndarray.__div__(self.data[:,np.newaxis,np.newaxis],other.data)
                        if self.unit == other.unit:
                            res.unit = self.unit
                        res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                        return res
            except:
                print 'Operation forbidden'
                return None

    def __pow__(self, other):
        """computes the power exponent"""
        if self.data is None:
            raise ValueError, 'empty data array'
        res = self.copy()
        if type(other) is float or type(other) is int:
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__pow__(self.data,other)
            else:
                res.data = np.ndarray.__pow__(self.data,other)
            res.fscale = res.fscale**other
            res.var = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

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

    dim : array of 2 integers
    Lengths of data in X and Y

    var : array
    Array containing the variance

    fscale : float
    Flux scaling factor (1 by default)

    maskinfo : string
    Information about the masked array (if relevant)

    wcs : WCS
    World coordinates

    Public methods
    --------------
    Creation: init, copy

    Arithmetic: + - * / pow

    Selection: <, >, <=, >=

    Info: info, []
    """

    def __init__(self, filename=None, ext = None, getnoise=False, dim=(101,101), wcs = None, unit=None, data=None, var=None,fscale=1.0,empty=False):
        """creates a Image object

        Parameters
        ----------
        filename : string
        Possible FITS filename

        ext : integer
        Number of the corresponding extension in the file

        getnoise: boolean
        True if the noise Variance image is read (if it exists)
        Use getnoise=False to create image without variance extension

        dim : integer or (integer,integer)
        Lengths of data in X and Y. (101,101) by default.

        wcs : WCS
        World coordinates

        unit : string
        Possible data unit type. None by default.

        data : array
        Array containing the pixel values of the image. None by default.

        var : array
        Array containing the variance. None by default.

        empty : bool
        If empty is True, the data and variance array are set to None

        fscale : float
        Flux scaling factor (1 by default)

        Examples
        --------
        Image(filename="toto.fits",ext=1,getnoise=False): image from file (extension number is 1).

        wcs = WCS(crval=0,cdelt=0.2,dim=300)
        Image(dim=300, wcs=wcs) : image 300x300 filled with zeros
        Image(wcs=wcs, data = MyData) : image 300x300 filled with MyData
        """

        self.image = True
        #possible FITS filename
        self.filename = filename
        self.maskinfo = ""
        if filename is not None:
            f = pyfits.open(filename)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                # if the number of extension is 1, we just read the data from the primary header
                # test if image
                if hdr['NAXIS'] != 2:
                    raise IOError, '  not an image'
                self.unit = hdr.get('UNIT', None)
                self.cards = hdr.ascardlist()
                self.dim = np.array([hdr['NAXIS1'],hdr['NAXIS2']])
                self.data = f[0].data
                self.var = None
                self.fscale = hdr.get('FSCALE', 1.0)
                self.wcs = WCS(hdr)  # WCS object from data header
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = f['DATA'].data
                else:
                    h = f[ext].header
                    d = f[ext].data
                if h['NAXIS'] != 2:
                    raise IOError, 'Wrong dimension number in DATA extension'
                self.unit = h.get('UNIT', None)
                self.cards = h.ascardlist()
                self.dim = np.array([h['NAXIS1'],h['NAXIS2']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                self.wcs = WCS(h)  # WCS object from data header
                if getnoise:
                    if f['VARIANCE'].header['NAXIS'] != 2:
                        raise IOError, 'Wrong dimension number in VARIANCE extension'
                    if f['VARIANCE'].header['NAXIS1'] != ima.dim[0] and f['VARIANCE'].header['NAXIS2'] != ima.dim[1]:
                        raise IOError, 'Number of points in VARIANCE not equal to DATA'
                    self.var = f['VARIANCE'].data
                else:
                    self.var = None
            f.close()
        else:
            #possible data unit type
            self.unit = unit
            # possible FITS header instance
            self.cards = pyfits.CardList()
            #data
            if len(dim) == 1:
                dim = (dim,dim)
            elif len(dim) == 2:
                pass
            else:
                raise ValueError, 'dim with dimension > 2'
            if data is None:
                if empty:
                    self.data = None
                else:
                    self.data = np.zeros(shape=(dim[1],dim[0]), dtype = float)
                self.dim = np.array(dim)
            else:
                self.data = np.array(data, dtype = float)
                self.dim = np.array((data.shape[1],data.shape[0]))

            if not getnoise or empty:
                self.var = None
            elif var is None:
                self.var = numpy.zeros(shape=(dim[1],dim[0]), dtype = float)
            else:
                self.var = np.array(var, dtype = float)
            self.fscale = np.float(fscale)
            try:
                if wcs.wcs.naxis1 == self.dim[0] and wcs.wcs.naxis2 == self.dim[1]:
                    self.wcs = wcs
                else:
                    print 'Dimensions of WCS object and DATA are not equal'
            except :
                self.wcs = None

    def copy(self):
        """copies Image object in a new one and returns it
        """
        ima = Image(empty=True)
        ima.filename = self.filename
        ima.unit = self.unit
        ima.cards = pyfits.CardList(self.cards)
        ima.dim = self.dim.__copy__()
        try:
            ima.data = self.data.__copy__()
        except:
            ima.data = None
        try:
            ima.var = self.var.__copy__()
        except:
            ima.var = None
        ima.fscale = self.fscale
        ima.maskinfo = self.maskinfo
        try:
            ima.wcs = self.wcs.copy()
        except:
            ima.wcs = None
        return ima

    def info(self):
        """prints information
        """
        if self.filename is None:
            print 'no name'
        else:
            print self.filename
        if self.data is None:
            print 'no data'
        elif isinstance(self.data,np.ma.core.MaskedArray):
            print 'masked array:\t(%i,%i) %s'% (self.dim[0],self.dim[1],self.maskinfo)
        else:
            print 'image data:\t(%i,%i)'% (self.dim[0],self.dim[1])
        print 'fscale:\t %d'%self.fscale
        if self.unit is None:
            print 'no data unit'
        else:
            print 'data unit:\t %s'%self.unit
        if self.var is None:
            print 'no noise'
        else:
            print 'noise variance:\t(%i,%i)'% (self.dim[0],self.dim[1])
        if self.wcs is None:
            print 'no world coordinates'
        else:
            self.wcs.info()

    def __le__ (self, item):
        """masks data array where greater than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " <= %d"%item
        return result

    def __lt__ (self, item):
        """masks data array where greater or equal than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " < %d"%item
        return result

    def __ge__ (self, item):
        """masks data array where less than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " >= %d"%item
        return result

    def __gt__ (self, item):
        """masks data array where less or equal than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " >%d"%item
        return result

    def __add__(self, other):
        """ adds other

        image1 + number = image2 (image2[j,i]=image1[j,i]+number)

        image1 + image2 = image3 (image3[j,i]=image1[j,i]+image2[j,i])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        image + cube1 = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #image1 + number = image2 (image2[j,i]=image1[j,i]+number)
            res = self.copy()
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__add__(self.data,other/np.double(self.fscale))
            else:
                res.data = np.ndarray.__add__(self.data,other/np.double(self.fscale))
            return res
        try:
            #image1 + image2 = image3 (image3[j,i]=image1[j,i]+image2[j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.image:
                if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Image(empty=True,dim=self.dim,fscale=self.fscale)
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for images with different world coordinates'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__add__(self.data,other.data*np.double(other.fscale/self.fscale))
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__add__(other.data*np.double(other.fscale/self.fscale),self.data)
                    else:
                        res.data = np.ndarray.__add__(self.data,other.data*np.double(other.fscale/self.fscale))
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                #image + cube1 = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
                #The first two dimensions of cube1 must be equal to the image dimensions.
                #If not equal to None, world coordinates in spatial directions must be the same.
                if other.cube:
                    res = other.__add__(self)
                    return res
            except:
                print 'Operation forbidden'
                return None


    def __sub__(self, other):
        """ subtracts other

        image1 - number = image2 (image2[j,i]=image1[j,i]-number)

        image1 - image2 = image3 (image3[j,i]=image1[j,i]-image2[j,i])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        image - cube1 = cube2 (cube2[k,j,i]=image[j,i] - cube1[k,j,i])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #image1 - number = image2 (image2[j,i]=image1[j,i]-number)
            res = self.copy()
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__sub__(self.data,other/np.double(self.fscale))
            else:
                res.data = np.ndarray.__sub__(self.data,other/np.double(self.fscale))
            return res
        try:
            #image1 - image2 = image3 (image3[j,i]=image1[j,i]-image2[j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.image:
                if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Image(empty=True,dim=self.dim,fscale=self.fscale)
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for images with different world coordinates'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__sub__(self.data,other.data*np.double(other.fscale/self.fscale))
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = -np.ma.MaskedArray.__sub__(other.data*np.double(other.fscale/self.fscale),self.data)
                    else:
                        res.data = np.ndarray.__sub__(self.data,other.data*np.double(other.fscale/self.fscale))
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                #image - cube1 = cube2 (cube2[k,j,i]=image[j,i] - cube1[k,j,i])
                #The first two dimensions of cube1 must be equal to the image dimensions.
                #If not equal to None, world coordinates in spatial directions must be the same.
                if other.cube:
                    if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,dim=other.dim , wave= other.wave, fscale=self.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        if isinstance(self.data,np.ma.core.MaskedArray):
                            res.data = np.ma.MaskedArray.__sub__(self.data[np.newaxis,:,:],other.data*np.double(other.fscale/self.fscale))
                        elif isinstance(other.data,np.ma.core.MaskedArray):
                            res.data = -np.ma.MaskedArray.__sub__(other.data*np.double(other.fscale/self.fscale),self.data[np.newaxis,:,:])
                        else:
                            res.data = np.ndarray.__sub__(self.data[np.newaxis,:,:],other.data*np.double(other.fscale/self.fscale))
                        if self.unit == other.unit:
                            res.unit = self.unit
                        res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                        return res
            except:
                print 'Operation forbidden'
                return None


    def __mul__(self, other):
        """ multiplies by other

        image1 * number = image2 (image2[j,i]=image1[j,i]*number)

        image1 * image2 = image3 (image3[j,i]=image1[j,i]*image2[j,i])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        image * cube1 = cube2 (cube2[k,j,i]=image[j,i] * cube1[k,j,i])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.

        image * spectrum = cube (cube[k,j,i]=image[j,i]*spectrum[k]
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #image1 * number = image2 (image2[j,i]=image1[j,i]*number)
            res = self.copy()
            res.fscale *= other
            if res.var is not None:
                res.var *= other*other
            return res
        try:
            #image1 * image2 = image3 (image3[j,i]=image1[j,i]*image2[j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.image:
                if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Image(empty=True,dim=self.dim,fscale=self.fscale * other.fscale)
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for images with different world coordinates'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__mul__(self.data,other.data)
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__mul__(other.data,self.data)
                    else:
                        res.data = np.ndarray.__mul__(self.data,other.data)
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                #image * cube1 = cube2 (cube2[k,j,i]=image[j,i] * cube1[k,j,i])
                #The first two dimensions of cube1 must be equal to the image dimensions.
                #If not equal to None, world coordinates in spatial directions must be the same.
                if other.cube:
                    res = other.__mul__(self)
                    return res
            except:
                try:
                    #image * spectrum = cube (cube[k,j,i]=image[j,i]*spectrum[k]
                    if other.spectrum:
                        if other.data is None :
                            print 'Operation forbidden for empty data'
                            return None
                        else:
                            dim = (self.dim[0],self.dim[1],other.dim)
                            res = Cube(empty=True ,dim=dim , wave= other.wave, wcs = self.wcs, fscale=self.fscale * other.fscale)
                            if isinstance(self.data,np.ma.core.MaskedArray):
                                res.data = np.ma.MaskedArray.__mul__(self.data[np.newaxis,:,:],other.data[:,np.newaxis,np.newaxis])
                            elif isinstance(other.data,np.ma.core.MaskedArray):
                                res.data = np.ma.MaskedArray.__mul__(other.data[:,np.newaxis,np.newaxis],self.data[np.newaxis,:,:])
                            else:
                                res.data = np.ndarray.__mul__(self.data[np.newaxis,:,:],other.data[:,np.newaxis,np.newaxis])
                            if self.unit == other.unit:
                                res.unit = self.unit
                            res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                            return res
                except:
                    print 'Operation forbidden'
                    return None


    def __div__(self, other):
        """ divides by other

        image1 / number = image2 (image2[j,i]=image1[j,i]/number)

        image1 / image2 = image3 (image3[j,i]=image1[j,i]/image2[j,i])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        image / cube1 = cube2 (cube2[k,j,i]=image[j,i] / cube1[k,j,i])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #image1 / number = image2 (image2[j,i]=image1[j,i]/number
            res = self.copy()
            res.fscale /= other
            if res.var is not None:
                res.var /= other*other
            return res
        try:
            #image1 / image2 = image3 (image3[j,i]=image1[j,i]/image2[j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.image:
                if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Image(empty=True,dim=self.dim,fscale=self.fscale / other.fscale)
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for images with different world coordinates'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__div__(self.data,other.data)
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = 1.0/ np.ma.MaskedArray.__div__(other.data,self.data)
                    else:
                        res.data = np.ndarray.__div__(self.data,other.data)
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                #image / cube1 = cube2 (cube2[k,j,i]=image[j,i] / cube1[k,j,i])
                #The first two dimensions of cube1 must be equal to the image dimensions.
                #If not equal to None, world coordinates in spatial directions must be the same.
                if other.cube:
                    if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1]:
                        raise ValueError, 'Operation forbidden for images with different sizes'
                    else:
                        res = Cube(empty=True ,dim=other.dim , wave= other.wave, fscale=self.fscale / other.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            raise ValueError, 'Operation forbidden for objects with different world coordinates'
                        if isinstance(self.data,np.ma.core.MaskedArray):
                            res.data = np.ma.MaskedArray.__div__(self.data[np.newaxis,:,:],other.data)
                        elif isinstance(other.data,np.ma.core.MaskedArray):
                            res.data = 1.0/ np.ma.MaskedArray.__div__(other.data,self.data[np.newaxis,:,:])
                        else:
                            res.data = np.ndarray.__div__(self.data[np.newaxis,:,:],other.data)
                        if self.unit == other.unit:
                            res.unit = self.unit
                        res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                        return res
            except:
                print 'Operation forbidden'
                return None


    def __pow__(self, other):
        """computes the power exponent"""
        if self.data is None:
            raise ValueError, 'empty data array'
        res = self.copy()
        if type(other) is float or type(other) is int:
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__pow__(self.data,other)
            else:
                res.data = np.ndarray.__pow__(self.data,other)
            res.fscale = res.fscale**other
            res.var = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

class Cube(object):
    """cube class

    Attributes
    ----------
    filename : string
    Possible FITS filename

    unit : string
    Possible data unit type

    cards : pyfits.CardList
    Possible FITS header instance

    data : array or masked array
    Array containing the pixel values of the cube

    dim : array of 3 integers
    Lengths of data in X and Y and Z

    var : array
    Array containing the variance

    fscale : float
    Flux scaling factor (1 by default)

    maskinfo : string
    Information about the masked array (if relevant)

    wcs : WCS
    World coordinates

    wave : WaveCoord
    Wavelength coordinates

    Public methods
    --------------
    Creation: init, copy

    Arithmetic: + - * / pow

    Selection: <, >, <=, >=

    Info: info, []
    """

    def __init__(self, filename=None, ext = None, getnoise=False, dim=(101,101,101), wcs = None, wave = None, unit=None, data=None, var=None,fscale=1.0,empty=False):
        """creates a Cube object

        Parameters
        ----------
        filename : string
        Possible FITS filename

        ext : integer
        Number of the corresponding extension in the file

        getnoise: boolean
        True if the noise Variance image is read (if it exists)
        Use getnoise=False to create image without variance extension

        dim : integer or (integer,integer)
        Lengths of data in X and Y. (101,101) by default.

        wcs : WCS
        World coordinates

        wave : WaveCoord
        Wavelength coordinates

        unit : string
        Possible data unit type. None by default.

        data : array
        Array containing the pixel values of the image. None by default.

        var : array
        Array containing the variance. None by default.

        empty : bool
        If empty is True, the data and variance array are set to None

        fscale : float
        Flux scaling factor (1 by default)

        Examples
        --------

        """

        #possible FITS filename
        self.cube = True
        self.filename = filename
        self.maskinfo = ""
        if filename is not None:
            f = pyfits.open(filename)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                # if the number of extension is 1, we just read the data from the primary header
                # test if image
                if hdr['NAXIS'] != 3:
                    raise IOError, 'Wrong dimension number: not a cube'
                self.unit = hdr.get('UNIT', None)
                self.cards = hdr.ascardlist()
                self.dim = np.array([hdr['NAXIS1'],hdr['NAXIS2'],hdr['NAXIS3']])
                self.data = f[0].data
                self.var = None
                self.fscale = hdr.get('FSCALE', 1.0)
                # WCS object from data header
                self.wcs = WCS(hdr)
                #Wavelength coordinates
                if hdr.has_key('CDELT3'):
                    cdelt = hdr.get('CDELT3')
                elif hdr.has_key('CD3_3'):
                    cdelt = hdr.get('CD3_3')
                else:
                    cdelt = 1.0
                crpix = hdr.get('CRPIX3')
                crval = hdr.get('CRVAL3')
                cunit = hdr.get('CUNIT3')
                self.wave = WaveCoord(self.dim[2], crpix, cdelt, crval, cunit)
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = f['DATA'].data
                else:
                    h = f[ext].header
                    d = f[ext].data
                if h['NAXIS'] != 3:
                    raise IOError, 'Wrong dimension number in DATA extension'
                self.unit = h.get('UNIT', None)
                self.cards = h.ascardlist()
                self.dim = np.array([h['NAXIS1'],h['NAXIS2'],h['NAXIS3']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                # WCS object from data header
                self.wcs = WCS(h)
                #Wavelength coordinates
                if hdr.has_key('CDELT3'):
                    cdelt = hdr.get('CDELT3')
                elif hdr.has_key('CD3_3'):
                    cdelt = hdr.get('CD3_3')
                else:
                    cdelt = 1.0
                crpix = hdr.get('CRPIX3')
                crval = hdr.get('CRVAL3')
                cunit = hdr.get('CUNIT3')
                self.wave = WaveCoord(self.dim[2], crpix, cdelt, crval, cunit)
                if getnoise:
                    if f['VARIANCE'].header['NAXIS'] != 3:
                        raise IOError, 'Wrong dimension number in VARIANCE extension'
                    if f['VARIANCE'].header['NAXIS1'] != ima.dim[0] and f['VARIANCE'].header['NAXIS2'] != ima.dim[1] and f['VARIANCE'].header['NAXIS3'] != ima.dim[2]:
                        raise IOError, 'Number of points in VARIANCE not equal to DATA'
                    self.var = f['VARIANCE'].data
                else:
                    self.var = None
            f.close()
        else:
            #possible data unit type
            self.unit = unit
            # possible FITS header instance
            self.cards = pyfits.CardList()
            #data
            if len(dim) == 1:
                dim = (dim,dim,dim)
            elif len(dim) == 2:
                dim = (dim[0],dim[0],dim[1])
            elif len(dim) == 3:
                pass
            else:
                raise ValueError, 'dim with dimension > 3'
            if data is None:
                if empty:
                    self.data = None
                else:
                    self.data = np.zeros(shape=(dim[2],dim[1],dim[0]), dtype = float)
                self.dim = np.array(dim)
            else:
                self.data = np.array(data, dtype = float)
                self.dim = np.array((data.shape[2],data.shape[1],data.shape[0]))

            if not getnoise or empty:
                self.var = None
            elif var is None:
                self.var = numpy.zeros(shape=(dim[2],dim[1],dim[0]), dtype = float)
            else:
                self.var = np.array(var, dtype = float)
            self.fscale = np.float(fscale)
            try:
                if wcs.wcs.naxis1 == self.dim[0] and wcs.wcs.naxis2 == self.dim[1]:
                    self.wcs = wcs
                else:
                    print 'Dimensions of WCS object and DATA are not equal'
            except :
                self.wcs = None
            try:
                if wave.dim == self.dim[2]:
                    self.wave = wave
                else:
                    print 'Dimensions of WaveCoord object and DATA are not equal'
            except :
                self.wave = None

    def copy(self):
        """copies Cube object in a new one and returns it
        """
        cub = Cube(empty=True)
        cub.filename = self.filename
        cub.unit = self.unit
        cub.cards = pyfits.CardList(self.cards)
        cub.dim = self.dim.__copy__()
        try:
            cub.data = self.data.__copy__()
        except:
            cub.data = None
        try:
            cub.var = self.var.__copy__()
        except:
            cub.var = None
        cub.fscale = self.fscale
        cub.maskinfo = self.maskinfo
        try:
            cub.wcs = self.wcs.copy()
        except:
            cub.wcs = None
        try:
            cub.wave = self.wave.copy()
        except:
            cub.wave = None
        return cub

    def info(self):
        """prints information
        """
        if self.filename is None:
            print 'no name'
        else:
            print self.filename
        if self.data is None:
            print 'no data'
        elif isinstance(self.data,np.ma.core.MaskedArray):
            print 'masked array:\t(%i,%i) %s'% (self.dim[0],self.dim[1],self.maskinfo)
        else:
            print 'image data:\t(%i,%i)'% (self.dim[0],self.dim[1])
        print 'fscale:\t %d'%self.fscale
        if self.unit is None:
            print 'no data unit'
        else:
            print 'data unit:\t %s'%self.unit
        if self.var is None:
            print 'no noise'
        else:
            print 'noise variance:\t(%i,%i)'% (self.dim[0],self.dim[1])
        if self.wcs is None:
            print 'no world coordinates for spatial directions'
        else:
            self.wcs.info()
        if self.wave is None:
            print 'no world coordinates for spectral direction'
        else:
            self.wave.info()

    def __le__ (self, item):
        """masks data array where greater than a given value.
        Returns a cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " <= %d"%item
        return result

    def __lt__ (self, item):
        """masks data array where greater or equal than a given value.
        Returns a cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " < %d"%item
        return result

    def __ge__ (self, item):
        """masks data array where less than a given value.
        Returns a Cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " >= %d"%item
        return result

    def __gt__ (self, item):
        """masks data array where less or equal than a given value.
        Returns a Cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data, item/self.fscale)
        if self.var is not None:
            result.var = np.ma.MaskedArray(self.var, mask=result.data.mask, copy=True)
        result.maskinfo += " >%d"%item
        return result

    def __add__(self, other):
        """ adds other

        cube1 + number = cube2 (cube2[k,j,i]=cube1[k,j,i]+number)

        cube1 + cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]+cube2[k,j,i])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 + image = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.

        cube1 + spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]+spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            # cube + number = cube (add pixel per pixel)
            res = self.copy()
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__add__(self.data,other/np.double(self.fscale))
            else:
                res.data = np.ndarray.__add__(self.data,other/np.double(self.fscale))
            return res
        try:
            # cube1 + cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]+cube2[k,j,i])
            # dimensions must be the same
            # if not equal to None, world coordinates must be the same
            if other.cube:
                if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1] \
                   or self.dim[2] != other.dim[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(empty=True ,dim=self.dim , fscale=self.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spectral direction'
                        return None
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spatial directions'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__add__(self.data,other.data*np.double(other.fscale/self.fscale))
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__add__(other.data*np.double(other.fscale/self.fscale),self.data)
                    else:
                        res.data = np.ndarray.__add__(self.data,other.data*np.double(other.fscale/self.fscale))
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                # cube1 + image = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
                # the 2 first dimensions of cube1 must be equal to the image dimensions
                # if not equal to None, world coordinates in spatial directions must be the same
                if other.image:
                    if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1]:
                        print 'Operation forbidden for objects with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,dim=self.dim , wave= self.wave, fscale=self.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        if isinstance(self.data,np.ma.core.MaskedArray):
                            res.data = np.ma.MaskedArray.__add__(self.data,other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale))
                        elif isinstance(other.data,np.ma.core.MaskedArray):
                            res.data = np.ma.MaskedArray.__add__(other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale),self.data)
                        else:
                            res.data = np.ndarray.__add__(self.data,other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale))
                        if self.unit == other.unit:
                            res.unit = self.unit
                        res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                        return res
            except:
                try:
                    # cube1 + spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]+spectrum[k])
                    # the last dimension of cube1 must be equal to the spectrum dimension
                    # if not equal to None, world coordinates in spectral direction must be the same
                    if other.spectrum:
                        if other.data is None or other.dim != self.dim[2]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(empty=True ,dim=self.dim , wcs= self.wcs, fscale=self.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            if isinstance(self.data,np.ma.core.MaskedArray):
                                res.data = np.ma.MaskedArray.__add__(self.data,other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale))
                            elif isinstance(other.data,np.ma.core.MaskedArray):
                                res.data = np.ma.MaskedArray.__add__(other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale),self.data)
                            else:
                                res.data = np.ndarray.__add__(self.data,other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale))
                            if self.unit == other.unit:
                                res.unit = self.unit
                            res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                            return res
                except:
                    print 'Operation forbidden'
                    return None


    def __sub__(self, other):
        """  subtracts other

        cube1 - number = cube2 (cube2[k,j,i]=cube1[k,j,i]-number)

        cube1 - cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]-cube2[k,j,i])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 - image = cube2 (cube2[k,j,i]=cube1[k,j,i]-image[j,i])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.

        cube1 - spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]-spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #cube1 - number = cube2 (cube2[k,j,i]=cube1[k,j,i]-number)
            res = self.copy()
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__sub__(self.data,other/np.double(self.fscale))
            else:
                res.data = np.ndarray.__sub__(self.data,other/np.double(self.fscale))
            return res
        try:
            #cube1 - cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]-cube2[k,j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.cube:
                if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1] \
                   or self.dim[2] != other.dim[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(empty=True ,dim=self.dim , fscale=self.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spectral direction'
                        return None
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spatial directions'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__sub__(self.data,other.data*np.double(other.fscale/self.fscale))
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = -np.ma.MaskedArray.__sub__(other.data*np.double(other.fscale/self.fscale),self.data)
                    else:
                        res.data = np.ndarray.__sub__(self.data,other.data*np.double(other.fscale/self.fscale))
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                #cube1 - image = cube2 (cube2[k,j,i]=cube1[k,j,i]-image[j,i])
                #The first two dimensions of cube1 must be equal to the image dimensions.
                #If not equal to None, world coordinates in spatial directions must be the same.
                if other.image:
                    if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,dim=self.dim , wave= self.wave, fscale=self.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        if isinstance(self.data,np.ma.core.MaskedArray):
                            res.data = np.ma.MaskedArray.__sub__(self.data,other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale))
                        elif isinstance(other.data,np.ma.core.MaskedArray):
                            res.data = -np.ma.MaskedArray.__sub__(other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale),self.data)
                        else:
                            res.data = np.ndarray.__sub__(self.data,other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale))
                        if self.unit == other.unit:
                            res.unit = self.unit
                        res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                        return res
            except:
                try:
                    #cube1 - spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]-spectrum[k])
                    #The last dimension of cube1 must be equal to the spectrum dimension.
                    #If not equal to None, world coordinates in spectral direction must be the same.
                    if other.spectrum:
                        if other.data is None or other.dim != self.dim[2]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(empty=True ,dim=self.dim , wcs= self.wcs, fscale=self.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            if isinstance(self.data,np.ma.core.MaskedArray):
                                res.data = np.ma.MaskedArray.__sub__(self.data,other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale))
                            elif isinstance(other.data,np.ma.core.MaskedArray):
                                res.data = -np.ma.MaskedArray.__sub__(other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale),self.data)
                            else:
                                res.data = np.ndarray.__sub__(self.data,other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale))
                            if self.unit == other.unit:
                                res.unit = self.unit
                            res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                            return res
                except:
                    print 'Operation forbidden'
                    return None


    def __mul__(self, other):
        """  multiplies by other

        cube1 * number = cube2 (cube2[k,j,i]=cube1[k,j,i]*number)

        cube1 * cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]*cube2[k,j,i])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 * image = cube2 (cube2[k,j,i]=cube1[k,j,i]*image[j,i])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.

        cube1 * spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]*spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #cube1 * number = cube2 (cube2[k,j,i]=cube1[k,j,i]*number)
            res = self.copy()
            res.fscale *= other
            if res.var is not None:
                res.var *= other*other
            return res
        try:
            #cube1 * cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]*cube2[k,j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.cube:
                if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1] \
                   or self.dim[2] != other.dim[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(empty=True ,dim=self.dim , fscale=self.fscale*other.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spectral direction'
                        return None
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spatial directions'
                        return None
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__mul__(self.data,other.data)
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__mul__(other.data,self.data)
                    else:
                        res.data = np.ndarray.__mul__(self.data,other.data)
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                #cube1 * image = cube2 (cube2[k,j,i]=cube1[k,j,i]*image[j,i])
                #The first two dimensions of cube1 must be equal to the image dimensions.
                #If not equal to None, world coordinates in spatial directions must be the same.
                if other.image:
                    if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,dim=self.dim , wave= self.wave, fscale=self.fscale * other.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        if isinstance(self.data,np.ma.core.MaskedArray):
                            res.data = np.ma.MaskedArray.__mul__(self.data,other.data[np.newaxis,:,:])
                        elif isinstance(other.data,np.ma.core.MaskedArray):
                            res.data = np.ma.MaskedArray.__mul__(other.data[np.newaxis,:,:],self.data)
                        else:
                            res.data = np.ndarray.__mul__(self.data,other.data[np.newaxis,:,:])
                        if self.unit == other.unit:
                            res.unit = self.unit
                        res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                        return res
            except:
                try:
                    #cube1 * spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]*spectrum[k])
                    #The last dimension of cube1 must be equal to the spectrum dimension.
                    #If not equal to None, world coordinates in spectral direction must be the same.
                    if other.spectrum:
                        if other.data is None or other.dim != self.dim[2]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(empty=True ,dim=self.dim , wcs= self.wcs, fscale=self.fscale*other.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            if isinstance(self.data,np.ma.core.MaskedArray):
                                res.data = np.ma.MaskedArray.__mul__(self.data,other.data[:,np.newaxis,np.newaxis])
                            elif isinstance(other.data,np.ma.core.MaskedArray):
                                res.data = np.ma.MaskedArray.__mul__(other.data[:,np.newaxis,np.newaxis],self.data)
                            else:
                                res.data = np.ndarray.__mul__(self.data,other.data[:,np.newaxis,np.newaxis])
                            if self.unit == other.unit:
                                res.unit = self.unit
                            res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                            return res
                except:
                    print 'Operation forbidden'
                    return None


    def __div__(self, other):
        """  divides by other

        cube1 / number = cube2 (cube2[k,j,i]=cube1[k,j,i]/number)

        cube1 / cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]/cube2[k,j,i])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 / image = cube2 (cube2[k,j,i]=cube1[k,j,i]/image[j,i])
        The first two dimensions of cube1 must be equal to the image dimensions.
        If not equal to None, world coordinates in spatial directions must be the same.

        cube1 / spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]/spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #cube1 / number = cube2 (cube2[k,j,i]=cube1[k,j,i]/number)
            res = self.copy()
            res.fscale /= other
            if res.var is not None:
                res.var /= other*other
            return res
        try:
            #cube1 / cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]/cube2[k,j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.cube:
                if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1] \
                   or self.dim[2] != other.dim[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(empty=True ,dim=self.dim , fscale=self.fscale/other.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for cubes with different world coordinates in spectral direction'
                        return None
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        raise ValueError, 'Operation forbidden for cubes with different world coordinates in spatial directions'
                    if isinstance(self.data,np.ma.core.MaskedArray):
                        res.data = np.ma.MaskedArray.__div__(self.data,other.data)
                    elif isinstance(other.data,np.ma.core.MaskedArray):
                        res.data = 1.0/ np.ma.MaskedArray.__div__(other.data,self.data)
                    else:
                        res.data = np.ndarray.__div__(self.data,other.data)
                    if self.unit == other.unit:
                        res.unit = self.unit
                    res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                    return res
        except:
            try:
                #cube1 / image = cube2 (cube2[k,j,i]=cube1[k,j,i]/image[j,i])
                #The first two dimensions of cube1 must be equal to the image dimensions.
                #If not equal to None, world coordinates in spatial directions must be the same.
                if other.image:
                    if other.data is None or self.dim[0] != other.dim[0] or self.dim[1] != other.dim[1]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,dim=self.dim , wave= self.wave, fscale=self.fscale / other.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        if isinstance(self.data,np.ma.core.MaskedArray):
                            res.data = np.ma.MaskedArray.__div__(self.data,other.data[np.newaxis,:,:])
                        elif isinstance(other.data,np.ma.core.MaskedArray):
                            res.data = 1.0/ np.ma.MaskedArray.__div__(other.data[np.newaxis,:,:],self.data)
                        else:
                            res.data = np.ndarray.__div__(self.data,other.data[np.newaxis,:,:])
                        if self.unit == other.unit:
                            res.unit = self.unit
                        res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                        return res
            except:
                try:
                    #cube1 / spectrum = cube2 (cube2[k,j,i]=cube1[k,j,i]/spectrum[k])
                    #The last dimension of cube1 must be equal to the spectrum dimension.
                    #If not equal to None, world coordinates in spectral direction must be the same.
                    if other.spectrum:
                        if other.data is None or other.dim != self.dim[2]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(empty=True ,dim=self.dim , wcs= self.wcs, fscale=self.fscale/other.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            if isinstance(self.data,np.ma.core.MaskedArray):
                                res.data = np.ma.MaskedArray.__div__(self.data,other.data[:,np.newaxis,np.newaxis])
                            elif isinstance(other.data,np.ma.core.MaskedArray):
                                res.data = 1.0 / np.ma.MaskedArray.__div__(other.data[:,np.newaxis,np.newaxis],self.data)
                            else:
                                res.data = np.ndarray.__div__(self.data,other.data[:,np.newaxis,np.newaxis])
                            if self.unit == other.unit:
                                res.unit = self.unit
                            res.maskinfo = " [%s] + [%s]"%(self.maskinfo,other.maskinfo)
                            return res
                except:
                    print 'Operation forbidden'
                    return None

    def __pow__(self, other):
        """computes the power exponent"""
        if self.data is None:
            raise ValueError, 'empty data array'
        res = self.copy()
        if type(other) is float or type(other) is int:
            if isinstance(self.data,np.ma.core.MaskedArray):
                res.data = np.ma.MaskedArray.__pow__(self.data,other)
            else:
                res.data = np.ndarray.__pow__(self.data,other)
            res.fscale = res.fscale**other
            res.var = None
        else:
            raise ValueError, 'Operation forbidden'
        return res



