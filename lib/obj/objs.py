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

    shape : integer
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

    Selection: <, >, <=, >=, removeMask

    Arithmetic: + - * / pow

    Info: info, []
    """

    def __init__(self, filename=None, ext = None, getnoise=False, shape=101, wave = None, unit=None, data=None, var=None,fscale=1.0,empty=False):
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

        shape : integer
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

        wave = WaveCoord(shape=4000, cdelt=1.25, crval=4000.0, cunit = 'Angstrom')
        Spectrum(shape=4000, wave=wave) : spectrum filled with zeros
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
                self.shape =hdr['NAXIS1']
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
                self.wave = WaveCoord(self.shape, crpix, cdelt, crval, cunit)
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
                self.shape = h['NAXIS1']
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
                self.wave = WaveCoord(self.shape, crpix, cdelt, crval, cunit)
                if getnoise:
                    if f['VARIANCE'].header['NAXIS'] != 1:
                        raise IOError, 'Wrong dimension number in VARIANCE extension'
                    if f['VARIANCE'].header['NAXIS1'] != shape:
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
                    self.data = np.zeros(shape, dtype = float)
                self.shape = shape
            else:
                self.data = np.array(data, dtype = float)
                self.shape = data.shape[0]

            if not getnoise or empty:
                self.var = None
            elif var is None:
                self.var = numpy.zeros(shape, dtype = float)
            else:
                self.var = np.array(var, dtype = float)
            self.fscale = np.float(fscale)
            try:
                if wave.dim == self.shape:
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
        spe.shape = self.shape
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
            print 'masked array:\t(%i,) %s'% (self.shape,self.maskinfo)
        elif self.data is None:
            print 'no data'
        else:
            print 'spectrum data:\t(%i,)'% self.shape
        print 'fscale:\t %d'%self.fscale
        if self.unit is None:
            print 'no data unit'
        else:
            print 'data unit:\t %s'%self.unit
        if self.var is None:
            print 'no noise'
        else:
            print 'noise variance:\t(%i,)'% self.shape
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

    def removeMask(self):
        """removes the mask on the array
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            self.data = self.data.data
            self.maskinfo = ""


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
            res.data = self.data + (other/np.double(self.fscale))
            return res
        try:
            #spectrum1 + spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]+spectrum2[k])
            #Dimension must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    print 'Operation forbidden for spectra with different sizes'
                    return None
                else:
                    res = Spectrum(empty=True,shape=self.shape,fscale=self.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for spectra with different world coordinates'
                        return None
                    res.data = self.data + other.data*np.double(other.fscale/self.fscale)
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
            res.data = self.data - (other/np.double(self.fscale))
            return res
        try:
            #spectrum1 + spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]-spectrum2[k])
            #Dimension must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    print 'Operation forbidden for spectra with different sizes'
                    return None
                else:
                    res = Spectrum(empty=True,shape=self.shape,fscale=self.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for spectra with different world coordinates'
                        return None
                    res.data = self.data - (other.data*np.double(other.fscale/self.fscale))
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
                    if other.data is None or self.shape != other.shape[0]:
                        print 'Operation forbidden for objects with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,shape=other.shape , wcs= other.wcs, fscale=self.fscale)
                        if self.wave is None or other.wave is None:
                            res.wave = None
                        elif self.wave.isEqual(other.wave):
                            res.wave = self.wave
                        else:
                            print 'Operation forbidden for spectra with different world coordinates'
                            return None
                        res.data = self.data[:,np.newaxis,np.newaxis] - (other.data*np.double(other.fscale/self.fscale))
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
                if other.data is None or self.shape != other.shape:
                    print 'Operation forbidden for spectra with different sizes'
                    return None
                else:
                    res = Spectrum(empty=True,shape=self.shape,fscale=self.fscale*other.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for spectra with different world coordinates'
                        return None
                    res.data = self.data * other.data
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

        Note : divide functions that have a validity domain returns the masked constant whenever the input is masked or falls outside the validity domain.
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
                if other.data is None or self.shape != other.shape:
                    print 'Operation forbidden for spectra with different sizes'
                    return None
                else:
                    res = Spectrum(empty=True,shape=self.shape,fscale=self.fscale/other.fscale)
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        print 'Operation forbidden for spectra with different world coordinates'
                        return None
                    res.data = self.data / other.data
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
                    if other.data is None or self.shape != other.shape[0]:
                        print 'Operation forbidden for objects with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,shape=other.shape , wcs= other.wcs, fscale=self.fscale/other.fscale)
                        if self.wave is None or other.wave is None:
                            res.wave = None
                        elif self.wave.isEqual(other.wave):
                            res.wave = self.wave
                        else:
                            print 'Operation forbidden for spectra with different world coordinates'
                            return None
                        res.data = self.data[:,np.newaxis,np.newaxis] / other.data
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
            res.data = self.data**other
            res.fscale = res.fscale**other
            res.var = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

    def __getitem__(self,item):
        """ returns the corresponding value or sub-spectrum
        """
        if isinstance(item, int):
            return self.data[item]
        elif isinstance(item, slice):
            data = self.data[item]
            shape = data.shape[0]
            getnoise = False
            var = None
            if self.var is not None:
                getnoise = True
                var = self.var[item]
            try:
                wave = self.wave[item]
            except:
                wave = None
            res = Spectrum(getnoise=getnoise, shape=shape, wave = wave, unit=self.unit, data=data, var=var,fscale=self.fscale)
            return res
        else:
            raise ValueError, 'Operation forbidden'

    def __setitem__(self,key,value):
        """ sets the corresponding part of data
        """
        self.data[key] = value

    def get_lambda(self,lbda_min,lbda_max):
        """ returns the corresponding value or sub-spectrum

        Parameters
        ----------
        lbda_min : float
        minimum wavelength

        lbda_max : float
        maximum wavelength
        """
        if self.wave is None:
            raise ValueError, 'Operation forbidden without world coordinates along the spectral direction'
        else:
            pix_min = int(self.wave.pixel(lbda_min))
            pix_max = int(self.wave.pixel(lbda_max)) + 1
            if pix_min==pix_max:
                return self.data[pix_min]
            else:
                return self[pix_min:pix_max]

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

    shape : array of 2 integers
    Lengths of data in X and Y (python notation: (ny,nx))

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

    def __init__(self, filename=None, ext = None, getnoise=False, shape=(101,101), wcs = None, unit=None, data=None, var=None,fscale=1.0,empty=False):
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

        shape : integer or (integer,integer)
        Lengths of data in X and Y. (101,101) by default.
        python notation: (ny,nx)

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

        wcs = WCS(crval=0,cdelt=0.2,shape=300)
        Image(shape=300, wcs=wcs) : image 300x300 filled with zeros
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
                self.shape = np.array([hdr['NAXIS2'],hdr['NAXIS1']])
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
                self.shape = np.array([h['NAXIS2'],h['NAXIS1']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                self.wcs = WCS(h)  # WCS object from data header
                if getnoise:
                    if f['VARIANCE'].header['NAXIS'] != 2:
                        raise IOError, 'Wrong dimension number in VARIANCE extension'
                    if f['VARIANCE'].header['NAXIS1'] != ima.shape[1] and f['VARIANCE'].header['NAXIS2'] != ima.shape[0]:
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
            if len(shape) == 1:
                shape = (shape,shape)
            elif len(shape) == 2:
                pass
            else:
                raise ValueError, 'dim with dimension > 2'
            if data is None:
                if empty:
                    self.data = None
                else:
                    self.data = np.zeros(shape=shape, dtype = float)
                self.shape = np.array(shape)
            else:
                self.data = np.array(data, dtype = float)
                try:
                    self.shape = np.array(data.shape)
                except:
                    self.shape = np.array(shape)

            if not getnoise or empty:
                self.var = None
            elif var is None:
                self.var = numpy.zeros(shape=shape, dtype = float)
            else:
                self.var = np.array(var, dtype = float)
            self.fscale = np.float(fscale)
            try:
                if wcs.wcs.naxis1 == self.shape[1] and wcs.wcs.naxis2 == self.shape[0]:
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
        ima.shape = self.shape.__copy__()
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
            print 'masked array:\t(%i,%i) %s'% (self.shape[1],self.shape[0],self.maskinfo)
        else:
            print 'image data:\t(%i,%i)'% (self.shape[1],self.shape[0])
        print 'fscale:\t %d'%self.fscale
        if self.unit is None:
            print 'no data unit'
        else:
            print 'data unit:\t %s'%self.unit
        if self.var is None:
            print 'no noise'
        else:
            print 'noise variance:\t(%i,%i)'% (self.shape[1],self.shape[0])
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

    def removeMask(self):
        """removes the mask on the array
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            self.data = self.data.data
            self.maskinfo = ""

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
            res.data = self.data + (other/np.double(self.fscale))
            return res
        try:
            #image1 + image2 = image3 (image3[j,i]=image1[j,i]+image2[j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.image:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Image(empty=True,shape=self.shape,fscale=self.fscale)
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for images with different world coordinates'
                        return None
                    res.data = self.data + (other.data*np.double(other.fscale/self.fscale))
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
            res.data = self.data - (other/np.double(self.fscale))
            return res
        try:
            #image1 - image2 = image3 (image3[j,i]=image1[j,i]-image2[j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.image:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Image(empty=True,shape=self.shape,fscale=self.fscale)
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for images with different world coordinates'
                        return None
                    res.data = self.data - (other.data*np.double(other.fscale/self.fscale))
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
                    if other.data is None or self.shape[0] != other.shape[1] or self.shape[1] != other.shape[2]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,shape=other.shape , wave= other.wave, fscale=self.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        res.data = self.data[np.newaxis,:,:] - (other.data*np.double(other.fscale/self.fscale))
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
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Image(empty=True,shape=self.shape,fscale=self.fscale * other.fscale)
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for images with different world coordinates'
                        return None
                    res.data = self.data * other.data
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
                            shape = (other.shape,self.shape[0],self.shape[1])
                            res = Cube(empty=True ,shape=shape , wave= other.wave, wcs = self.wcs, fscale=self.fscale * other.fscale)
                            res.data = self.data[np.newaxis,:,:] * other.data[:,np.newaxis,np.newaxis]
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
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Image(empty=True,shape=self.shape,fscale=self.fscale / other.fscale)
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        print 'Operation forbidden for images with different world coordinates'
                        return None
                    res.data = self.data / other.data
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
                    if other.data is None or self.shape[0] != other.shape[1] or self.shape[1] != other.shape[2]:
                        raise ValueError, 'Operation forbidden for images with different sizes'
                    else:
                        res = Cube(empty=True ,shape=other.shape , wave= other.wave, fscale=self.fscale / other.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            raise ValueError, 'Operation forbidden for objects with different world coordinates'
                        res.data = self.data[np.newaxis,:,:] / other.data
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
            res.data = self.data**other
            res.fscale = res.fscale**other
            res.var = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

    def __getitem__(self,item):
        """ returns the corresponding value or sub-image
        """
        if isinstance(item, tuple) and len(item)==2:
                data = self.data[item]
                if isinstance(item[0],int):
                    shape = (1,data.shape[0])
                elif isinstance(item[1],int):
                    shape = (data.shape[0],1)
                else:
                    shape = (data.shape[0],data.shape[1])
                getnoise = False
                var = None
                if self.var is not None:
                    getnoise = True
                    var = self.var[item]
                try:
                    wcs = self.wcs[item[1],item[0]] #data[y,x], image[y,x] but wcs[x,y]
                except:
                    wcs = None
                res = Image(getnoise=getnoise, shape=shape, wcs = wcs, unit=self.unit, data=data, var=var,fscale=self.fscale)
                return res
        else:
            raise ValueError, 'Operation forbidden'

    def __setitem__(self,key,value):
        """ sets the corresponding part of data
        """
        self.data[key] = value

    def get_deg(self,ra_min,ra_max,dec_min,dec_max):
        """ returns the corresponding sub-image

        Parameters
        ----------
        ra_min : float
        minimum right ascension in degrees

        ra_max : float
        maximum right ascension in degrees

        dec_min : float
        minimum declination in degrees

        dec_max : float
        maximum declination in degrees
        """
        skycrd = [[ra_min,dec_min],[ra_min,dec_max],[ra_max,dec_min],[ra_max,dec_max]]
        pixcrd = self.wcs.sky2pix(skycrd)
        print pixcrd
        imin = int(np.min(pixcrd[:,0]))
        if imin<0:
            imin = 0
        imax = int(np.max(pixcrd[:,0]))+1
        jmin = int(np.min(pixcrd[:,1]))
        if jmin<0:
            jmin = 0
        jmax = int(np.max(pixcrd[:,1]))+1
        print 'imin',imin
        print 'imax',imax
        print 'jmin',jmin
        print 'jmax',jmax
        res = self[jmin:jmax,imin:imax]
        #mask outside pixels
        mask = np.ma.make_mask_none(res.data.shape)
        for j in range(res.shape[0]):
            print j
            pixcrd = np.array([np.arange(res.shape[1]),np.ones(res.shape[1])*j]).T
            print pixcrd
            skycrd = self.wcs.pix2sky(pixcrd)
            print skycrd
            test_ra_min = np.array(skycrd[:,0]) < ra_min
            test_ra_max = np.array(skycrd[:,0]) > ra_max
            test_dec_min = np.array(skycrd[:,1]) < dec_min
            test_dec_max = np.array(skycrd[:,1]) > dec_max
            mask[j,:] = test_ra_min + test_ra_max + test_dec_min + test_dec_max
            print mask[j,:]
        try:
            mask = np.ma.mask_or(mask,np.ma.getmask(res.data))
        except:
            pass
        res.data = np.ma.MaskedArray(res.data, mask=mask)
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

    shape : array of 3 integers
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

    def __init__(self, filename=None, ext = None, getnoise=False, shape=(101,101,101), wcs = None, wave = None, unit=None, data=None, var=None,fscale=1.0,empty=False):
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

        shape : integer or (integer,integer)
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
                self.shape = np.array([hdr['NAXIS3'],hdr['NAXIS2'],hdr['NAXIS1']])
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
                self.wave = WaveCoord(self.shape[0], crpix, cdelt, crval, cunit)
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
                self.shape= np.array([h['NAXIS3'],h['NAXIS2'],h['NAXIS1']])
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
                self.wave = WaveCoord(self.shape[0], crpix, cdelt, crval, cunit)
                if getnoise:
                    if f['VARIANCE'].header['NAXIS'] != 3:
                        raise IOError, 'Wrong dimension number in VARIANCE extension'
                    if f['VARIANCE'].header['NAXIS1'] != ima.shape[2] and f['VARIANCE'].header['NAXIS2'] != ima.shape[1] and f['VARIANCE'].header['NAXIS3'] != ima.shape[0]:
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
            if len(shape) == 1:
                shape = (shape,shape,shape)
            elif len(shape) == 2:
                shape = (shape[0],shape[1],shape[1])
            elif len(shape) == 3:
                pass
            else:
                raise ValueError, 'dim with dimension > 3'
            if data is None:
                if empty:
                    self.data = None
                else:
                    self.data = np.zeros(shape=shape, dtype = float)
                self.shape = np.array(shape)
            else:
                self.data = np.array(data, dtype = float)
                try:
                    self.shape = np.array(data.shape)
                except:
                    self.shape = np.array(shape)

            if not getnoise or empty:
                self.var = None
            elif var is None:
                self.var = numpy.zeros(shape=shape, dtype = float)
            else:
                self.var = np.array(var, dtype = float)
            self.fscale = np.float(fscale)
            try:
                if wcs.wcs.naxis1 == self.shape[2] and wcs.wcs.naxis2 == self.shape[1]:
                    self.wcs = wcs
                else:
                    print 'Dimensions of WCS object and DATA are not equal'
            except :
                self.wcs = None
            try:
                if wave.dim == self.shape[0]:
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
        cub.shape = self.shape.__copy__()
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
            print 'masked array:\t(%i,%i,%i) %s'% (self.shape[2],self.shape[1],self.shape[1],self.maskinfo)
        else:
            print 'cube data:\t(%i,%i,%i)'% (self.shape[2],self.shape[1],self.shape[0])
        print 'fscale:\t %d'%self.fscale
        if self.unit is None:
            print 'no data unit'
        else:
            print 'data unit:\t %s'%self.unit
        if self.var is None:
            print 'no noise'
        else:
            print 'noise variance:\t(%i,%i,%i)'% (self.shape[2],self.shape[1],self.shape[0])
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

    def removeMask(self):
        """removes the mask on the array
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            self.data = self.data.data
            self.maskinfo = ""

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
            res.data = self.data + (other/np.double(self.fscale))
            return res
        try:
            # cube1 + cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]+cube2[k,j,i])
            # dimensions must be the same
            # if not equal to None, world coordinates must be the same
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(empty=True ,shape=self.shape , fscale=self.fscale)
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
                    res.data = self.data + (other.data*np.double(other.fscale/self.fscale))
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
                    if other.data is None or self.shape[2] != other.shape[1] or self.shape[1] != other.shape[0]:
                        print 'Operation forbidden for objects with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,shape=self.shape , wave= self.wave, fscale=self.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        res.data = self.data + (other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale))
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
                        if other.data is None or other.shape != self.shape[0]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(empty=True ,shape=self.shape , wcs= self.wcs, fscale=self.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            res.data = self.data + (other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale))
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
            res.data = self.data - (other/np.double(self.fscale))
            return res
        try:
            #cube1 - cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]-cube2[k,j,i])
            #Dimensions must be the same.
            #If not equal to None, world coordinates must be the same.
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(empty=True ,shape=self.shape , fscale=self.fscale)
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
                    res.data = self.data - (other.data*np.double(other.fscale/self.fscale))
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
                    if other.data is None or self.shape[2] != other.shape[1] or self.shape[1] != other.shape[0]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,shape=self.shape , wave= self.wave, fscale=self.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        res.data = self.data - (other.data[np.newaxis,:,:]*np.double(other.fscale/self.fscale))
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
                        if other.data is None or other.shape != self.shape[0]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(empty=True ,shape=self.shape , wcs= self.wcs, fscale=self.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            res.data = self.data - (other.data[:,np.newaxis,np.newaxis]*np.double(other.fscale/self.fscale))
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
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(empty=True ,shape=self.shape , fscale=self.fscale*other.fscale)
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
                    res.data = self.data * other.data
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
                    if other.data is None or self.shape[2] != other.shape[1] or self.shape[1] != other.shape[0]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,shape=self.shape , wave= self.wave, fscale=self.fscale * other.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        res.data = self.data * other.data[np.newaxis,:,:]
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
                        if other.data is None or other.shape != self.shape[0]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(empty=True ,shape=self.shape , wcs= self.wcs, fscale=self.fscale*other.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            res.data = self.data * other.data[:,np.newaxis,np.newaxis]
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
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = Cube(empty=True ,shape=self.shape , fscale=self.fscale/other.fscale)
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
                    res.data = self.data / other.data
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
                    if other.data is None or self.shape[2] != other.shape[1] or self.shape[1] != other.shape[0]:
                        print 'Operation forbidden for images with different sizes'
                        return None
                    else:
                        res = Cube(empty=True ,shape=self.shape , wave= self.wave, fscale=self.fscale / other.fscale)
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            print 'Operation forbidden for objects with different world coordinates'
                            return None
                        res.data = self.data / other.data[np.newaxis,:,:]
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
                        if other.data is None or other.shape != self.shape[0]:
                            print 'Operation forbidden for objects with different sizes'
                            return None
                        else:
                            res = Cube(empty=True ,shape=self.shape , wcs= self.wcs, fscale=self.fscale/other.fscale)
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                print 'Operation forbidden for spectra with different world coordinates'
                                return None
                            res.data = self.data / other.data[:,np.newaxis,np.newaxis]
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
            res.data = self.data**other
            res.fscale = res.fscale**other
            res.var = None
        else:
            raise ValueError, 'Operation forbidden'
        return res

    def __getitem__(self,item):
        """returns the corresponding object:
        cube[k,j,i] = value
        cube[k,:,:] = spectrum
        cube[:,j,i] = image
        cube[:,:,:] = sub-cube
        """
        if isinstance(item, tuple) and len(item)==3:
            data = self.data[item]
            if isinstance(item[0],int):
                if isinstance(item[1],int) and isinstance(item[2],int):
                    #return a float
                    return data
                else:
                    #return an image
                    if isinstance(item[1],int):
                        shape = (1,data.shape[0])
                    elif isinstance(item[2],int):
                        shape = (data.shape[0],1)
                    else:
                        shape = data.shape
                    getnoise = False
                    var = None
                    if self.var is not None:
                        getnoise = True
                        var = self.var[item]
                    try:
                        wcs = self.wcs[item[2],item[1]] #data[y,x], image[y,x] byt wcs[x,y]
                    except:
                        wcs = None
                    res = Image(getnoise=getnoise, shape=shape, wcs = wcs, unit=self.unit, data=data, var=var,fscale=self.fscale)
                    return res
            elif isinstance(item[1],int) and isinstance(item[2],int):
                #return a spectrum
                shape = data.shape[0]
                getnoise = False
                var = None
                if self.var is not None:
                    getnoise = True
                    var = self.var[item]
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Spectrum(getnoise=getnoise, shape=shape, wave = wave, unit=self.unit, data=data, var=var,fscale=self.fscale)
                return res
            else:
                #return a cube
                if isinstance(item[1],int):
                    shape = (data.shape[0],1,data.shape[1])
                elif isinstance(item[2],int):
                    shape = (data.shape[0],data.shape[1],1)
                else:
                    shape = data.shape
                getnoise = False
                var = None
                if self.var is not None:
                    getnoise = True
                    var = self.var[item]
                try:
                    wcs = self.wcs[item[2],item[1]] #data[y,x], image[y,x] but wcs[x,y]
                except:
                    wcs = None
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Cube(getnoise=getnoise, shape=shape, wcs = wcs, wave = wave, unit=self.unit, data=data, var=var,fscale=self.fscale)
                return res
        else:
            raise ValueError, 'Operation forbidden'

    def __setitem__(self,key,value):
        """ sets the corresponding part of data
        """
        self.data[key] = value

    def get_lambda(self,lbda_min,lbda_max):
        """ returns the corresponding sub-cube

        Parameters
        ----------
        lbda_min : float
        minimum wavelength

        lbda_max : float
        maximum wavelength
        """
        if self.wave is None:
            raise ValueError, 'Operation forbidden without world coordinates along the spectral direction'
        else:
            pix_min = int(self.wave.pixel(lbda_min))
            pix_max = int(self.wave.pixel(lbda_max)) + 1
            if pix_min==pix_max:
                return self.data[pix_min,:,:]
            else:
                return self[pix_min:pix_max,:,:]
