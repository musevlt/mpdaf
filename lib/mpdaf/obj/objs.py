""" obj.py Manages cube, image and spectrum objects"""
import numpy as np
import pyfits
import datetime
from coords import WCS
from coords import WaveCoord

from scipy import integrate

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

    Selection: <, >, <=, >=, remove_mask

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

        wave = WaveCoord(cdelt=1.25, crval=4000.0, cunit = 'Angstrom')
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
                self.wave = WaveCoord(crpix, cdelt, crval, cunit)
                self.wave.dim = self.shape
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
                self.wave = WaveCoord(crpix, cdelt, crval, cunit)
                self.wave.dim = self.shape
                if getnoise:
                    if f['STAT'].header['NAXIS'] != 1:
                        raise IOError, 'Wrong dimension number in STAT extension'
                    if f['STAT'].header['NAXIS1'] != shape:
                        raise IOError, 'Number of points in STAT not equal to DATA'
                    self.var = f['STAT'].data
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
                self.wave = wave
                self.wave.dim = self.shape
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

    def write(self,filename):
        """ saves the object in a FITS file
        Parameters
        ----------
        filename : string
        The FITS filename
        """
        # create primary header
        prihdu = pyfits.PrimaryHDU()
        if self.cards is not None:
            for card in self.cards:
                try:
                    prihdu.header.update(card.key, card.value, card.comment)
                except:
                    pass
        prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
        prihdu.header.update('author', 'MPDAF', 'origin of the file')

        if self.var is None: # write simple fits file without extension
            prihdu.data = self.data
            # add world coordinate
            prihdu.header.update('CRVAL1', self.wave.crval, 'Start in world coordinate')
            prihdu.header.update('CRPIX1', self.wave.crpix, 'Start in pixel')
            prihdu.header.update('CDELT1', self.wave.cdelt, 'Step in world coordinate')
            prihdu.header.update('CTYPE1', 'LINEAR', 'world coordinate type')
            prihdu.header.update('CUNIT1', self.wave.cunit, 'world coordinate units')
            if self.unit is not None:
                prihdu.header.update('UNIT', self.unit, 'data unit type')
            prihdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist = [prihdu]
        else: # write fits file with primary header and two extensions
            hdulist = [prihdu]
            # create spectrum DATA in first extension
            tbhdu = pyfits.ImageHDU(name='DATA', data=self.data)
            # add world coordinate
            tbhdu.header.update('CRVAL1', self.wave.crval, 'Start in world coordinate')
            tbhdu.header.update('CRPIX1', self.wave.crpix, 'Start in pixel')
            tbhdu.header.update('CDELT1', self.wave.cdelt, 'Step in world coordinate')
            tbhdu.header.update('CTYPE1', 'LINEAR', 'world coordinate type')
            tbhdu.header.update('CUNIT1', self.wave.cunit, 'world coordinate units')
            if self.unit is not None:
                tbhdu.header.update('UNIT', self.unit, 'data unit type')
            tbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist.append(tbhdu)
            # create spectrum STAT in second extension
            nbhdu = pyfits.ImageHDU(name='STAT', data=self.var)
            # add world coordinate
            nbhdu.header.update('CRVAL1', self.wave.crval, 'Start in world coordinate')
            nbhdu.header.update('CRPIX1', self.wave.crpix, 'Start in pixel')
            nbhdu.header.update('CDELT1', self.wave.cdelt, 'Step in world coordinate')
            nbhdu.header.update('CUNIT1', self.wave.cunit, 'world coordinate units')
#            if self.unit is not None:
#                nbhdu.header.update('UNIT', self.unit, 'data unit type')
#            nbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist.append(nbhdu)
        # save to disk
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True)

        self.filename = filename

    def info(self):
        """prints information
        """
        if self.filename != None:
            print 'spectrum of %i elements (%s)' %(self.shape,self.filename)
        else:
            print 'spectrum of %i elements (no name)' %self.shape
        data = '.data'
        if self.data is None:
            data = 'no data'
        noise = '.var'
        if self.var is None:
            noise = 'no noise'
        if self.unit is None:
            unit = 'no unit'
        else:
            unit = self.unit
        print '%s (%s) fscale=%0.2f, %s' %(data,unit,self.fscale,noise)
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
        result.maskinfo += " <= %f"%item
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
        result.maskinfo += " < %f"%item
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
        result.maskinfo += " >= %f"%item
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
        result.maskinfo += " >%f"%item
        return result

    def remove_mask(self):
        """removes the mask on the array
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            self.data = self.data.data
            self.maskinfo = ""

    def resize(self):
        """resize the spectrum to have a minimum number of masked values
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            ksel = np.where(self.data.mask==False)
            item = slice (ksel[0][0],ksel[0][-1]+1,None)
            self.data = self.data[item]
            self.shape = self.data.shape[0]
            if self.var is not None:
                self.var = self.var[item]
            try:
                self.wave = self.wave[item]
            except:
                self.wave = None

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

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """ subtracts other

        spectrum1 - number = spectrum2 (spectrum2[k]=spectrum1[k]-number)

        spectrum1 - spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]-spectrum2[k])
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

    def __rsub__(self, other):
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            res = self.copy()
            res.data = (other/np.double(self.fscale)) - self.data
            return res
        try:
            if other.spectrum:
                return other.__sub__(self)
        except:
            try:
                if other.cube:
                    return other.__sub__(self)
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

    def __rmul__(self, other):
        return self.__mul__(other)

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

    def __rdiv__(self, other):
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            res = self.copy()
            res.fscale = other / res.fscale
            if res.var is not None:
                res.var = other*other /(res.var*res.var)
            return res
        try:
            if other.spectrum:
                return other.__div__(self)
        except:
            try:
                if other.cube:
                    return other.__div__(self)
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

    def sqrt(self):
        """computes the power exponent"""
        if self.data is None:
            raise ValueError, 'empty data array'
        res = self.copy()
        res.data = np.sqrt(self.data)
        res.fscale = np.sqrt(self.fscale)
        res.var = None
        return res

    def abs(self):
        """computes the absolute value"""
        if self.data is None:
            raise ValueError, 'empty data array'
        res = self.copy()
        res.data = np.abs(self.data)
        res.fscale = np.abs(self.fscale)
        res.var = None
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

    def get_lambda(self,lbda_min,lbda_max=None):
        """ returns the corresponding value or sub-spectrum

        Parameters
        ----------
        lbda_min : float
        minimum wavelength

        lbda_max : float
        maximum wavelength
        """
        if lbda_max is None:
            lbda_max = lbda_min
        if self.wave is None:
            raise ValueError, 'Operation forbidden without world coordinates along the spectral direction'
        else:
            pix_min = int(self.wave.pixel(lbda_min))
            pix_max = int(self.wave.pixel(lbda_max)) + 1
            if pix_min==pix_max:
                return self.data[pix_min]
            else:
                return self[pix_min:pix_max]

    def set_wcs(self, wave):
        """sets the world coordinates

        Parameters
        ----------
        wave : WaveCoord
        Wavelength coordinates
        """
        self.wave = wave
        self.wave.dim = self.shape
        
    def rebin_factor(self, factor):
        '''rebins an array to a new shape.
        The new shape must be a factor of self.shape.
        
        Parameters
        ----------
        factor : int
        Factor
        '''
        assert not np.sometrue(np.mod( self.shape, factor ))
        newshape = self.shape/factor
        data = self.data.reshape(newshape,factor).sum(1)
        getnoise = False
        var = None
        if self.var is not None:
            getnoise = True
            var = self.var.reshape(newshape,factor).sum(1)
        try:
            print 'factor ', factor
            print self.wave.coord()
            crval = self.wave.coord()[slice(0,factor,1)].sum()/factor
            print 'crval', crval
            wave = WaveCoord(1, self.wave.cdelt*factor, crval, self.wave.cunit)
        except:
            wave = None
        res = Spectrum(getnoise=getnoise, shape=newshape, wave = wave, unit=self.unit, data=data, var=var,fscale=self.fscale)
        return res
    
    def rebin(self,step,start=None):
        """returns a spectrum with data rebinned to different wavelength step size.
        
        Parameters
        ----------
        step: float
        New pixel size in spectral direction
        
        start: float
        Spectral position of the first new pixel.
        It can be set or kept at the edge of the old first one.       
        """
#        flux = self.data.sum()*self.wave.cdelt
#        print "init flux", flux
        f = lambda x: self.data[int(self.wave.pixel(x)+0.5)]
        
        newwave = self.wave.rebin(step,start)
        newshape = newwave.dim   
            
        newdata = np.zeros(newshape)        
        pix = np.arange(newshape,dtype=np.float)
        x1 = (pix - newwave.crpix + 1) * newwave.cdelt + newwave.crval - 0.5 * newwave.cdelt
        x2 = (pix - newwave.crpix + 1) * newwave.cdelt + newwave.crval + 0.5 * newwave.cdelt
        lbdamax = (self.shape - self.wave.crpix ) * self.wave.cdelt + self.wave.crval + 0.5 * self.wave.cdelt
        if x2[-1]> lbdamax:
            x2[-1] = lbdamax
        
        for i in range(newshape):
            newdata[i] = integrate.quad(f,x1[i],x2[i])[0]/newwave.cdelt
            
#        newflux = newdata.sum()*newwave.cdelt
#        print "new flux", newflux
        
        res = Spectrum(getnoise=False, shape=newshape, wave = newwave, unit=self.unit, data=newdata,fscale=self.fscale)
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

        wcs = WCS(crval=0,cdelt=0.2)
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
                    if f['STAT'].header['NAXIS'] != 2:
                        raise IOError, 'Wrong dimension number in STAT extension'
                    if f['STAT'].header['NAXIS1'] != ima.shape[1] and f['STAT'].header['NAXIS2'] != ima.shape[0]:
                        raise IOError, 'Number of points in STAT not equal to DATA'
                    self.var = f['STAT'].data
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
                self.wcs = wcs
                self.wcs.wcs.naxis1 = self.shape[1]
                self.wcs.wcs.naxis2 = self.shape[0]
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

    def write(self,filename):
        """ saves the object in a FITS file
        Parameters
        ----------
        filename : string
        The FITS filename
        """
        # create primary header
        prihdu = pyfits.PrimaryHDU()
        if self.cards is not None:
            for card in self.cards:
                try:
                    prihdu.header.update(card.key, card.value, card.comment)
                except:
                    pass
        prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
        prihdu.header.update('author', 'MPDAF', 'origin of the file')

        #world coordinates
        wcs_cards = self.wcs.to_header().ascardlist()

        if self.var is None: # write simple fits file without extension
            prihdu.data = self.data
            for card in wcs_cards:
                prihdu.header.update(card.key, card.value, card.comment)
            if self.unit is not None:
                prihdu.header.update('UNIT', self.unit, 'data unit type')
            prihdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist = [prihdu]
        else: # write fits file with primary header and two extensions
            hdulist = [prihdu]
            # create spectrum DATA in first extension
            tbhdu = pyfits.ImageHDU(name='DATA', data=self.data)
            for card in wcs_cards:
                tbhdu.header.update(card.key, card.value, card.comment)
            if self.unit is not None:
                tbhdu.header.update('UNIT', self.unit, 'data unit type')
            tbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist.append(tbhdu)
            # create spectrum STAT in second extension
            nbhdu = pyfits.ImageHDU(name='STAT', data=self.var)
            for card in wcs_cards:
                nbhdu.header.update(card.key, card.value, card.comment)
#            if self.unit is not None:
#                nbhdu.header.update('UNIT', self.unit, 'data unit type')
#            nbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist.append(nbhdu)
        # save to disk
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True)

        self.filename = filename

    def info(self):
        """prints information
        """
        if self.filename is None:
            print '%i X %i image (no name)' %(self.shape[1],self.shape[0])
        else:
            print '%i X %i image (%s)' %(self.shape[1],self.shape[0],self.filename)
        data = '.data(%i,%i)' %(self.shape[0],self.shape[1])
        if self.data is None:
            data = 'no data'
        noise = '.var(%i,%i)' %(self.shape[0],self.shape[1])
        if self.var is None:
            noise = 'no noise'
        if self.unit is None:
            unit = 'no unit'
        else:
            unit = self.unit
        print '%s (%s) fscale=%0.2f, %s' %(data,unit,self.fscale,noise)
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
        result.maskinfo += " <= %f"%item
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
        result.maskinfo += " < %f"%item
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
        result.maskinfo += " >= %f"%item
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
        result.maskinfo += " >%f"%item
        return result

    def remove_mask(self):
        """removes the mask on the array
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            self.data = self.data.data
            self.maskinfo = ""

    def resize(self):
        """resize the image to have a minimum number of masked values
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            ksel = np.where(self.data.mask==False)
            item = (slice(ksel[0][0], ksel[0][-1]+1, None), slice(ksel[1][0], ksel[1][-1]+1, None))
            self.data = self.data[item]
            if isinstance(item[0],int):
                self.shape = (1,self.data.shape[0])
            elif isinstance(item[1],int):
                self.shape = (self.data.shape[0],1)
            else:
                self.shape = (self.data.shape[0],self.data.shape[1])
            if self.var is not None:
                self.var = self.var[item]
            try:
                self.wcs = self.wcs[item[1],item[0]] #data[y,x], image[y,x] but wcs[x,y]
            except:
                self.wcs = None

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

    def __radd__(self, other):
        return self.__add__(other)

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

    def __rsub__(self, other):
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            res = self.copy()
            res.data = (other/np.double(self.fscale)) - self.data
            return res
        try:
            if other.image:
                return other.__sub__(self)
        except:
            try:
                if other.cube:
                   return other.__sub__(self)
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

    def __rmul__(self, other):
        return self.__mul__(other)

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

    def __rdiv__(self, other):
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #image1 / number = image2 (image2[j,i]=image1[j,i]/number
            res = self.copy()
            res.fscale = other / res.fscale
            if res.var is not None:
                res.var = other*other / (res.var*res.var)
            return res
        try:
            if other.image:
                return other.__sub__(self)
        except:
            try:
                if other.cube:
                    return other.__sub__(self)
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

    def sqrt(self):
        """computes the power exponent"""
        if self.data is None:
            raise ValueError, 'empty data array'
        res = self.copy()
        res.data = np.sqrt(self.data)
        res.fscale = np.sqrt(self.fscale)
        res.var = None
        return res

    def abs(self):
        """computes the absolute value"""
        if self.data is None:
            raise ValueError, 'empty data array'
        res = self.copy()
        res.data = np.abs(self.data)
        res.fscale = np.abs(self.fscale)
        res.var = None
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

    def set_wcs(self, wcs):
        """sets the world coordinates

        Parameters
        ----------
        wcs : WCS
        World coordinates
        """
        self.wcs = wcs
        self.wcs.wcs.naxis1 = self.shape[1]
        self.wcs.wcs.naxis2 = self.shape[0]

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
                self.wave = WaveCoord(crpix, cdelt, crval, cunit)
                self.wave.dim = self.shape[0]
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
                if h.has_key('CDELT3'):
                    cdelt = h.get('CDELT3')
                elif h.has_key('CD3_3'):
                    cdelt = h.get('CD3_3')
                else:
                    cdelt = 1.0
                crpix = h.get('CRPIX3')
                crval = h.get('CRVAL3')
                cunit = h.get('CUNIT3')
                self.wave = WaveCoord(crpix, cdelt, crval, cunit)
                self.wave.dim = self.shape[0]
                if getnoise:
                    if f['STAT'].header['NAXIS'] != 3:
                        raise IOError, 'Wrong dimension number in STAT extension'
                    if f['STAT'].header['NAXIS1'] != ima.shape[2] and f['STAT'].header['NAXIS2'] != ima.shape[1] and f['STAT'].header['NAXIS3'] != ima.shape[0]:
                        raise IOError, 'Number of points in STAT not equal to DATA'
                    self.var = f['STAT'].data
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
                self.wcs = wcs
                self.wcs.wcs.naxis1 = self.shape[2]
                self.wcs.wcs.naxis2 = self.shape[1]
            except :
                self.wcs = None
            try:
                self.wave = wave
                self.wave.dim = self.shape[0]
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

    def write(self,filename):
        """ saves the object in a FITS file
        Parameters
        ----------
        filename : string
        The FITS filename
        """
        # create primary header
        prihdu = pyfits.PrimaryHDU()
        if self.cards is not None:
            for card in self.cards:
                try:
                    prihdu.header.update(card.key, card.value, card.comment)
                except:
                    pass
        prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
        prihdu.header.update('author', 'MPDAF', 'origin of the file')

        #world coordinates
        wcs_cards = self.wcs.to_header().ascardlist()

        if self.var is None: # write simple fits file without extension
            prihdu.data = self.data
            # add world coordinate
            for card in wcs_cards:
                prihdu.header.update(card.key, card.value, card.comment)
            prihdu.header.update('CRVAL3', self.wave.crval, 'Start in world coordinate')
            prihdu.header.update('CRPIX3', self.wave.crpix, 'Start in pixel')
            prihdu.header.update('CDELT3', self.wave.cdelt, 'Step in world coordinate')
            prihdu.header.update('CTYPE3', 'LINEAR', 'world coordinate type')
            prihdu.header.update('CUNIT3', self.wave.cunit, 'world coordinate units')
            if self.unit is not None:
                prihdu.header.update('UNIT', self.unit, 'data unit type')
            prihdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist = [prihdu]
        else: # write fits file with primary header and two extensions
            hdulist = [prihdu]
            # create spectrum DATA in first extension
            tbhdu = pyfits.ImageHDU(name='DATA', data=self.data)
            # add world coordinate
            for card in wcs_cards:
                tbhdu.header.update(card.key, card.value, card.comment)
            tbhdu.header.update('CRVAL3', self.wave.crval, 'Start in world coordinate')
            tbhdu.header.update('CRPIX3', self.wave.crpix, 'Start in pixel')
            tbhdu.header.update('CDELT3', self.wave.cdelt, 'Step in world coordinate')
            tbhdu.header.update('CTYPE3', 'LINEAR', 'world coordinate type')
            tbhdu.header.update('CUNIT3', self.wave.cunit, 'world coordinate units')
            if self.unit is not None:
                tbhdu.header.update('UNIT', self.unit, 'data unit type')
            tbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist.append(tbhdu)
            # create spectrum STAT in second extension
            nbhdu = pyfits.ImageHDU(name='STAT', data=self.var)
            # add world coordinate
            for card in wcs_cards:
                nbhdu.header.update(card.key, card.value, card.comment)
            nbhdu.header.update('CRVAL3', self.wave.crval, 'Start in world coordinate')
            nbhdu.header.update('CRPIX3', self.wave.crpix, 'Start in pixel')
            nbhdu.header.update('CDELT3', self.wave.cdelt, 'Step in world coordinate')
            nbhdu.header.update('CUNIT3', self.wave.cunit, 'world coordinate units')
#            if self.unit is not None:
#                nbhdu.header.update('UNIT', self.unit, 'data unit type')
#            nbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist.append(nbhdu)
        # save to disk
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True)

        self.filename = filename

    def info(self):
        """prints information
        """
        if self.filename is None:
            print '%i X %i X %i cube (no name)' %(self.shape[2],self.shape[1],self.shape[0])
        else:
            print '%i X %i X %i cube (%s)' %(self.shape[2],self.shape[1],self.shape[0],self.filename)
        data = '.data(%i,%i,%i)' %(self.shape[0],self.shape[1],self.shape[2])
        if self.data is None:
            data = 'no data'
        noise = '.var(%i,%i,%i)' %(self.shape[0],self.shape[1],self.shape[2])
        if self.var is None:
            noise = 'no noise'
        if self.unit is None:
            unit = 'no unit'
        else:
            unit = self.unit
        print '%s (%s) fscale=%0.2f, %s' %(data,unit,self.fscale,noise)
        if self.wcs is None:
            print 'no world coordinates for spectral direction'
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
        result.maskinfo += " <= %f"%item
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
        result.maskinfo += " < %f"%item
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
        result.maskinfo += " >= %f"%item
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
        result.maskinfo += " >%f"%item
        return result

    def remove_mask(self):
        """removes the mask on the array
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            self.data = self.data.data
            self.maskinfo = ""

    def resize(self):
        """resize the cube to have a minimum number of masked values
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            ksel = np.where(self.data.mask==False)
            item = (slice(ksel[0][0], ksel[0][-1]+1, None), slice(ksel[1][0], ksel[1][-1]+1, None),slice(ksel[2][0], ksel[2][-1]+1, None))
            self.data = self.data[item]
            if isinstance(item[0],int):
                if isinstance(item[1],int):
                    self.shape = (1,1,self.data.shape[0])
                elif isinstance(item[2],int):
                    self.shape = (1,self.data.shape[0],1)
                else:
                    self.shape = (1,self.data.shape[0],self.data.shape[1])
            elif isinstance(item[1],int):
                if isinstance(item[2],int):
                    self.shape = (self.data.shape[0],1,1)
                else:
                    self.shape = (self.data.shape[0],1,self.data.shape[1])
            elif isinstance(item[2],int):
                    self.shape = (self.data.shape[0],self.data.shape[1],1)
            else:
                self.shape = self.data.shape
            if self.var is not None:
                self.var = self.var[item]
            try:
                self.wcs = self.wcs[item[2],item[1]] #data[y,x], image[y,x] but wcs[x,y]
            except:
                self.wcs = None
            try:
                self.wave = self.wave[item[0]]
            except:
                self.wave = None

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

    def __radd__(self, other):
        return self.__add__(other)

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

    def __rsub__(self, other):
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            res = self.copy()
            res.data = (other/np.double(self.fscale)) - self.data
            return res
        try:
            if other.cube:
                return other.__sub__(self)
        except:
            try:
                if other.image:
                    return other.__sub__(self)
            except:
                try:
                    if other.spectrum:
                        return other.__sub__(self)
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

    def __rmul__(self, other):
        return self.__mul__(other)

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

    def __rdiv__(self, other):
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is float or type(other) is int:
            #cube1 / number = cube2 (cube2[k,j,i]=cube1[k,j,i]/number)
            res = self.copy()
            res.fscale = other / res.fscale
            if res.var is not None:
                res.var = other*other / (res.var*res.var)
            return res
        try:
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    return other.__div__(self)
        except:
            try:
                if other.image:
                    return other.__div__(self)
            except:
                try:
                    if other.spectrum:
                       return other.__div__(self)
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

    def sqrt(self):
        """computes the power exponent"""
        if self.data is None:
            raise ValueError, 'empty data array'
        res = self.copy()
        res.data = np.sqrt(self.data)
        res.fscale = np.sqrt(self.fscale)
        res.var = None
        return res

    def abs(self):
        """computes the absolute value"""
        if self.data is None:
            raise ValueError, 'empty data array'
        res = self.copy()
        res.data = np.abs(self.data)
        res.fscale = np.abs(self.fscale)
        res.var = None
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

    def get_lambda(self,lbda_min,lbda_max=None):
        """ returns the corresponding sub-cube

        Parameters
        ----------
        lbda_min : float
        minimum wavelength

        lbda_max : float
        maximum wavelength
        """
        if lbda_max is None:
            lbda_max = lbda_min
        if self.wave is None:
            raise ValueError, 'Operation forbidden without world coordinates along the spectral direction'
        else:
            pix_min = int(self.wave.pixel(lbda_min))
            pix_max = int(self.wave.pixel(lbda_max)) + 1
            if pix_min==pix_max:
                return self.data[pix_min,:,:]
            else:
                return self[pix_min:pix_max,:,:]

    def set_wcs(self, wcs, wave):
        """sets the world coordinates

        Parameters
        ----------
        wcs : WCS
        World coordinates

        wave : WaveCoord
        Wavelength coordinates
        """
        self.wcs = wcs
        self.wcs.wcs.naxis1 = self.shape[2]
        self.wcs.wcs.naxis2 = self.shape[1]
        self.wave = wave
        self.wave.dim = self.shape[0]
