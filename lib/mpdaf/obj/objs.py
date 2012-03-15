""" obj.py Manages cube, image and spectrum objects"""
import numpy as np
import pyfits
import datetime
from coords import WCS
from coords import WaveCoord

from scipy import integrate
from scipy import interpolate
from scipy.optimize import leastsq

import matplotlib.pyplot as plt

import ABmag_filters

def flux2mag(flux, wave):
    """ convert flux from erg.s-1.cm-2.A-1 to AB mag
    wave is the wavelength in A
    """
    c = 2.998e18 # speed of light in A/s
    mag = -48.60 - 2.5*np.log10(wave**2*flux/c)
    return mag

def mag2flux(mag, wave):
    """ convert flux from AB mag to erg.s-1.cm-2.A-1
    wave is the wavelength in A
    """
    c = 2.998e18 # speed of light in A/s
    flux = 10**(-0.4*(mag + 48.60))*c/wave**2
    return flux

class SpectrumClicks:
    "Spec Cursor"
    def __init__(self, binding_id, filename=None):
        self.filename = filename # name of the table fits file where are saved the clicks vlaues.
        self.binding_id = binding_id # connection id
        self.xc = [] # cursor position in spectrum (world coord)
        self.yc = [] # cursor position in spectrum (world coord)
        self.i = [] # nearest pixel in spectrum
        self.x = [] # corresponding nearest position in spectrum (world coord)
        self.data = [] # corresponding spectrum data values
        self.id_lines = []  # list of plots (points for cursor positions)

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

        dq : array
        Array containing bad pixel

        empty : boolean
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
        self._clicks = None
        self.spectrum = True
        #possible FITS filename
        self.filename = filename
        if filename is not None:
            f = pyfits.open(filename)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                # if the number of extension is 1, we just read the data from the primary header
                # test if spectrum
                if hdr['NAXIS'] != 1:
                    raise IOError, 'Wrong dimension number: not a spectrum'
                self.unit = hdr.get('BUNIT', None)
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
                self.wave = WaveCoord(crpix, cdelt, crval, cunit, self.shape)
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = f['DATA'].data
                else:
                    h = f[ext].header
                    d = f[ext].data
                if h['NAXIS'] != 1:
                    raise IOError, 'Wrong dimension number: not a spectrum'
                self.unit = h.get('BUNIT', None)
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
                self.wave = WaveCoord(crpix, cdelt, crval, cunit, self.shape)
                # STAT extension
                self.var = None
                if getnoise:
                    if f['STAT'].header['NAXIS'] != 1:
                        raise IOError, 'Wrong dimension number in STAT extension'
                    if f['STAT'].header['NAXIS1'] != self.shape:
                        raise IOError, 'Number of points in STAT not equal to DATA'
                    self.var = f['STAT'].data
                # DQ extension
                try:
                    mask = np.ma.make_mask(f['DQ'].data)
                    self.data = np.ma.array(self.data, mask=mask)
                except:
                    pass
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
                if wave.shape is not None and wave.shape != self.shape:
                    print "warning: wavelength coordinates and data have not the same dimensions."
                    self.wave = None
                self.wave.shape = self.shape
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
        assert self.data is not None
        prihdu = pyfits.PrimaryHDU()
        if self.cards is not None:
            for card in self.cards:
                try:
                    prihdu.header.update(card.key, card.value, card.comment)
                except:
                    pass
        prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
        prihdu.header.update('author', 'MPDAF', 'origin of the file')

        if isinstance(self.data,np.ma.core.MaskedArray):
            hdulist = [prihdu]
            # create spectrum DATA in first extension
            tbhdu = pyfits.ImageHDU(name='DATA', data=self.data.data)
            tbhdu.header.update('CRVAL1', self.wave.crval, 'Start in world coordinate')
            tbhdu.header.update('CRPIX1', self.wave.crpix, 'Start in pixel')
            tbhdu.header.update('CDELT1', self.wave.cdelt, 'Step in world coordinate')
            tbhdu.header.update('CTYPE1', 'LINEAR', 'world coordinate type')
            tbhdu.header.update('CUNIT1', self.wave.cunit, 'world coordinate units')
            if self.unit is not None:
                tbhdu.header.update('BUNIT', self.unit, 'data unit type')
            tbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist.append(tbhdu)
            if self.var is not None:
                # create spectrum STAT in second extension
                nbhdu = pyfits.ImageHDU(name='STAT', data=self.var)
                nbhdu.header.update('CRVAL1', self.wave.crval, 'Start in world coordinate')
                nbhdu.header.update('CRPIX1', self.wave.crpix, 'Start in pixel')
                nbhdu.header.update('CDELT1', self.wave.cdelt, 'Step in world coordinate')
                nbhdu.header.update('CUNIT1', self.wave.cunit, 'world coordinate units')
    #            if self.unit is not None:
    #                nbhdu.header.update('UNIT', self.unit, 'data unit type')
    #            nbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
                hdulist.append(nbhdu)
            # DQ extension
            dqhdu = pyfits.ImageHDU(name='DQ', data=np.uint8(self.data.mask))
            dqhdu.header.update('CRVAL1', self.wave.crval, 'Start in world coordinate')
            dqhdu.header.update('CRPIX1', self.wave.crpix, 'Start in pixel')
            dqhdu.header.update('CDELT1', self.wave.cdelt, 'Step in world coordinate')
            dqhdu.header.update('CUNIT1', self.wave.cunit, 'world coordinate units')
            hdulist.append(dqhdu)
        else:
            if self.var is None: # write simple fits file without extension
                prihdu.data = self.data
                prihdu.header.update('CRVAL1', self.wave.crval, 'Start in world coordinate')
                prihdu.header.update('CRPIX1', self.wave.crpix, 'Start in pixel')
                prihdu.header.update('CDELT1', self.wave.cdelt, 'Step in world coordinate')
                prihdu.header.update('CTYPE1', 'LINEAR', 'world coordinate type')
                prihdu.header.update('CUNIT1', self.wave.cunit, 'world coordinate units')
                if self.unit is not None:
                    prihdu.header.update('BUNIT', self.unit, 'data unit type')
                prihdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
                hdulist = [prihdu]
            else: # write fits file with primary header and two extensions
                hdulist = [prihdu]
                # create spectrum DATA in first extension
                tbhdu = pyfits.ImageHDU(name='DATA', data=self.data)
                tbhdu.header.update('CRVAL1', self.wave.crval, 'Start in world coordinate')
                tbhdu.header.update('CRPIX1', self.wave.crpix, 'Start in pixel')
                tbhdu.header.update('CDELT1', self.wave.cdelt, 'Step in world coordinate')
                tbhdu.header.update('CTYPE1', 'LINEAR', 'world coordinate type')
                tbhdu.header.update('CUNIT1', self.wave.cunit, 'world coordinate units')
                if self.unit is not None:
                    tbhdu.header.update('BUNIT', self.unit, 'data unit type')
                tbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
                hdulist.append(tbhdu)
                # create spectrum STAT in second extension
                nbhdu = pyfits.ImageHDU(name='STAT', data=self.var)
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
        return result

    def __lt__ (self, item):
        """masks data array where greater or equal than a given value.
        Returns a Spectrum object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data, item/self.fscale)
        return result

    def __ge__ (self, item):
        """masks data array where less than a given value.
        Returns a Spectrum object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item/self.fscale)
        return result

    def __gt__ (self, item):
        """masks data array where less or equal than a given value.
        Returns a Spectrum object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data, item/self.fscale)
        return result

    def remove_mask(self):
        """removes the mask on the array
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            self.data = self.data.data

    def resize(self):
        """resizes the spectrum to have a minimum number of masked values
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            ksel = np.where(self.data.mask==False)
            try:
                item = slice (ksel[0][0],ksel[0][-1]+1,None)
                self.data = self.data[item]
                self.shape = self.data.shape[0]
                if self.var is not None:
                    self.var = self.var[item]
                    try:
                        self.wave = self.wave[item]
                    except:
                        self.wave = None
            except:
                pass

    def __add__(self, other):
        """ adds other

        spectrum1 + number = spectrum2 (spectrum2[k]=spectrum1[k]+number)

        spectrum1 + spectrum2 = spectrum3 (spectrum3[k]=spectrum1[k]+spectrum2[k])
        Dimension must be the same.
        If not equal to None, world coordinates must be the same.

        spectrum + cube1 = cube2 (cube2[k,j,i]=cube1[k,j,i]+spectrum[k])
        The first dimension of cube1 must be equal to the spectrum dimension.
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
        The first dimension of cube1 must be equal to the spectrum dimension.
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
        The first dimension of cube1 must be equal to the spectrum dimension.
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
        The first dimension of cube1 must be equal to the spectrum dimension.
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
            res = Spectrum(getnoise=getnoise, shape=shape, wave = wave, unit=self.unit, empty = True,fscale=self.fscale)
            res.data = data
            if getnoise:
                res.var = var
            return res
        else:
            raise ValueError, 'Operation forbidden'

    def __setitem__(self,key,value):
        """ sets the corresponding part of data
        """
        self.data[key] = value

    def get_lambda(self,lmin,lmax=None):
        """ returns the corresponding value or sub-spectrum

        Parameters
        ----------
        lmin : float
        minimum wavelength

        lmax : float
        maximum wavelength
        """
        if lmax is None:
            lmax = lbda_min
        if self.wave is None:
            raise ValueError, 'Operation forbidden without world coordinates along the spectral direction'
        else:
            pix_min = int(self.wave.pixel(lmin))
            pix_max = int(self.wave.pixel(lmax)) + 1
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
        if wave.shape is not None and wave.shape != self.shape:
            print "warning: wavelength coordinates and data have not the same dimensions."
            self.wave = None
        else:
            self.wave = wave
            self.wave.shape = self.shape
            
    def mask(self, lmin=None, lmax=None):
        """ mask the corresponding sub-spectrum

        Parameters
        ----------
        lmin : float
        minimum wavelength

        lmax : float
        maximum wavelength
        """
        if self.wave is None:
            raise ValueError, 'Operation forbidden without world coordinates along the spectral direction'
        else:
            if lmin is None:
                pix_min = 0
            else:
                pix_min = int(self.wave.pixel(lmin))
            if lmax is None:
                pix_max = self.shape
            else:
                pix_max = int(self.wave.pixel(lmax)) + 1
            if isinstance(self.data,np.ma.core.MaskedArray):
                self.data[pix_min:pix_max] = np.ma.masked
            else:
                mask = np.zeros(self.shape,dtype=bool)
                mask[pix_min:pix_max] = True
                print mask[pix_min-1:pix_max+1]
                self.data = np.ma.array(self.data, mask=mask)
                
            
        
    def interp(self, wavelengths, spline=False):
        """ returns the interpolated values corresponding to the wavelength array
        
        Parameters
        ----------
        wavelengths : array of float
        wavelength values
        
        spline : boolean
        False: linear interpolation, True: spline interpolation 
        """
        lbda = self.wave.coord()
        if isinstance(self.data,np.ma.core.MaskedArray):
            ksel = np.where(self.data.mask==False)            
            d = np.zeros(np.shape(ksel)[1]+2)
            d[1:-1] = self.data.data[ksel]
            w = np.zeros(np.shape(ksel)[1]+2)      
            w[1:-1] = lbda[ksel]
            if self.var is not None:    
                weight = np.zeros(np.shape(ksel)[1]+2)
                weight[1:-1] = 1./self.var[ksel]
        else:
            d = np.zeros(self.shape+2)
            d[1:-1] = self.data[:]
            w = np.zeros(self.shape+2)
            w[1:-1] = lbda[:]
            if self.var is not None:    
                weight = np.zeros(self.shape+2)
                weight[1:-1] = 1./self.var[:]
        d[0] = d[1]
        d[-1] = d[-2]
        w[0] = (-self.wave.crpix + 1) * self.wave.cdelt + self.wave.crval - 0.5 * self.wave.cdelt
        w[-1] = (self.shape - self.wave.crpix ) * self.wave.cdelt + self.wave.crval + 0.5 * self.wave.cdelt
        if self.var is not None:
            weight[0] = var[1]
            weight[-1] = var[-2]
        else:
            weight = None
        if spline:
            tck = interpolate.splrep(w,d,w=weight)
            return interpolate.splev(wavelengths,tck,der=0)
        else:
            if weight is not None:
                w *= weight / weight.sum()
            f = interpolate.interp1d(w, d)
            return f(wavelengths)

        
    def interp_data(self, spline=False):
        """ returns data array with interpolated values for masked pixels
        
        Parameter
        ----------
        spline : boolean
        False: linear interpolation, True: spline interpolation 
        """
        if isinstance(self.data,np.ma.core.MaskedArray):
            lbda = self.wave.coord()
            ksel = np.where(self.data.mask==True)
            wnew = lbda[ksel]
            data = self.data.data
            data[ksel] = self.interp(wnew,spline)
            return data
        else:
            return self.data
            
    
    def rebin_factor(self, factor):
        '''rebins an array to a new shape.
        The new shape must be a factor of self.shape.
        
        Parameter
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
            crval = self.wave.coord()[slice(0,factor,1)].sum()/factor
            wave = WaveCoord(1, self.wave.cdelt*factor, crval, self.wave.cunit)
        except:
            wave = None
        res = Spectrum(getnoise=getnoise, shape=newshape, wave = wave, unit=self.unit, fscale=self.fscale, empty=True)
        res.data = data
        if getnoise:
            res.var = var
        return res
    
    def rebin(self, step, start=None, spline = False):
        """returns a spectrum with data rebinned to different wavelength step size.
        
        Parameters
        ----------
        step: float
        New pixel size in spectral direction
        
        start: float
        Spectral position of the first new pixel.
        It can be set or kept at the edge of the old first one.     
        
        spline : boolean
        linear/spline interpolation to interpolate masked values
        """
        data = self.interp_data(spline)

        f = lambda x: data[int(self.wave.pixel(x)+0.5)]
        
        newwave = self.wave.rebin(step,start)
        newshape = newwave.shape   
            
        newdata = np.zeros(newshape)        
        pix = np.arange(newshape,dtype=np.float)
        x1 = (pix - newwave.crpix + 1) * newwave.cdelt + newwave.crval - 0.5 * newwave.cdelt
        x2 = (pix - newwave.crpix + 1) * newwave.cdelt + newwave.crval + 0.5 * newwave.cdelt
        lbdamax = (self.shape - self.wave.crpix ) * self.wave.cdelt + self.wave.crval + 0.5 * self.wave.cdelt
        if x2[-1]> lbdamax:
            x2[-1] = lbdamax
        
        for i in range(newshape):
            newdata[i] = integrate.quad(f,x1[i],x2[i])[0]/newwave.cdelt
        
        res = Spectrum(getnoise=False, shape=newshape, wave = newwave, unit=self.unit, data=newdata,fscale=self.fscale)
        return res

    def mean(self, lmin=None, lmax=None, weight=True, spline=False):
        """ computes the mean value on [lmin,lmax]. Returns mean or mean,std

        Parameters
        ----------
        lmin : float
        Minimum wavelength.

        lmax : float
        Maximum wavelength.

        weight : boolean
        If weight is True, compute the weighted average with the inverse of variance as weight.

        spline : boolean
        linear/spline interpolation to interpolate masked values
        """
        if self.var is None:
            weight = False
        if lmin is None:
            i1 = 0
        else:
            i1 = self.wave.pixel(lmin, nearest=True)
        if lmax is None:
            i2 = self.shape
        else:
            i2 = self.wave.pixel(lmax, nearest=True)

        #replace masked values by interpolated values
        data = self.interp_data(spline)

        if weight:
            flux = np.average(data[i1:i2], weights=1.0/self.var[i1:i2])*self.fscale
        else:
            flux = data[i1:i2].mean()*self.fscale
            if self.var is not None:
                err = np.sqrt(self.var[i1:i2].sum()/(self.var[i1:i2].shape[0])**2)*self.fscale
                return flux,err
        return flux

    def sum(self, lmin=None, lmax=None, weight=True, spline=False):
        """ computes the flux value on [lmin,lmax]. Returns flux or flux,std

        Parameters
        ----------
        lmin : float
        Minimum wavelength.

        lmax : float
        Maximum wavelength.

        weight : boolean
        If weight is True, compute the weighted sum with the inverse of variance as weight.

        spline : boolean
        linear/spline interpolation to interpolate masked values
        """
        if self.var is None:
            weight = False
        if lmin is None:
            i1 = 0
        else:
            i1 = self.wave.pixel(lmin, True)
        if lmax is None:
            i2 = self.shape
        else:
            i2 = self.wave.pixel(lmax, True)
            
        #replace masked values by interpolated values
        data = self.interp_data(spline)

        if weight:
            flux = (i2-i1)*np.average(data[i1:i2], weights=1.0/self.var[i1:i2])*self.fscale
        else:
            flux = data[i1:i2].sum()*self.fscale
            if self.var is not None:
                err = np.sqrt(self.var[i1:i2].sum())*self.fscale
                return flux,err
        return flux

    def polfit(self, order, maxiter=0, nsig=(-3.0,3.0), wind=None, weight=True, quiet=False):
        """ performs polynomial fit on spectrum

        Parameters
        ----------
        order : integer
        Polynomial order.

        maxiter : integer
        Maximum allowed iterations (0 by default).

        nsig : (float,float)
        Low and high rejection factor in std units. (-3.0,3.0) by default.

        wind : list of float
        wind is the list of wavelength interval to skip in the fit (None by default)

        weight : boolean
        if weight is True, the weight is computed as the inverse of variance
        """
        if self.shape <= order+1:
            raise ValueError, 'Too few points to perform polynomial fit'

        if self.var is None:
            weight = False

        if weight:
            vec_weight = 1./self.var
        else:
            vec_weight = None

        if wind is not None:
            l1 = np.array(wind)[0::2]
            l2 = np.array(wind)[1::2]
            # create mask array
            mask = np.ones(self.shape, dtype=np.bool)
            for lb1,lb2 in zip(l1,l2):
                i1 = self.wave.pixel(lb1, True)
                i2 = self.wave.pixel(lb2, True)
                mask[i1:i2] = np.zeros(i2-i1, dtype=np.bool)
            if isinstance(self.data,np.ma.core.MaskedArray):
                mask *= np.array(1 - self.data.mask,dtype=bool)
            d = self.data.compress(mask)
            w = self.wave.coord().compress(mask)
            if weight:
                vec_weight = vec_weight.compress(mask)
        else:
            if isinstance(self.data,np.ma.core.MaskedArray):
                mask = np.array(1 - self.data.mask,dtype=bool)
                d = self.data.compress(mask)
                w = self.wave.coord().compress(mask)
                if weight:
                    vec_weight = vec_weight.compress(mask)
            else:
                d = self.data
                w = self.wave.coord()

        #p = np.polyfit(w, d, order, w=vec_weight) numpy 1.5
        if weight:
            d *= vec_weight/vec_weight.sum()
        p = np.polyfit(w, d, order)
        if maxiter > 0:
            err = d - np.polyval(p, w)
            sig = np.std(err)
            n = len(d)
            for iter in range(maxiter):
                ind = np.where((err >= nsig[0]*sig) & (np.abs(err) <= nsig[1]*sig))
                if len(ind[0]) == n:
                    break
                if len(ind[0]) <= order+1:
                    raise ValueError, 'Too few points to perform polynomial fit'
                #p = np.polyfit(w[ind], d[ind], order, w=vec_weight[ind])
                p = np.polyfit(w[ind], d[ind], order)
                err = d[ind] - np.polyval(p, w[ind])
                sig = np.std(err)
                n = len(ind[0])
            if not quiet:
                print 'Number of iteration: %d Std: %10.4e Np: %d Frac: %4.2f'%(iter+1, sig, n, 100.*n/self.shape)
        return p

    def abmag_band(self, lbda, dlbda, out=1, spline=False):
        """computes AB magnitude corresponding to the wavelength band.

        Parameters
        ----------
        lbda : float
        Mean wavelength.

        dlbda : float
        width of the wavelength band.

        out : 1 or 2
        1: the magnitude is returned
        2: the magnitude, mean flux and mean lbda are returned

        spline : boolean
        linear/spline interpolation to interpolate masked values
        """
        data = self.interp_data(spline)
        vflux = data[self.wave.pixel(lbda-dlbda/2,nearest=True):self.wave.pixel(lbda+dlbda/2,nearest=True)].mean()*self.fscale
        mag = flux2mag(vflux, lbda)
        if out == 1:
            return mag
        if out == 2:
            return mag,vflux,lbda

    def abmag_filter_name(self, name, out=1, spline=False):
        """ computes AB magnitude using the filter name.

        Parameters
        ----------
        name : string
        'U', 'B', 'V', 'Rc', 'Ic', 'z', 'R-Johnson','F606W'

        out : 1 or 2
        1: the magnitude is returned
        2: the magnitude, mean flux and mean lbda are returned

        spline : boolean
        linear/spline interpolation to interpolate masked values
        """
        if name == 'U':
            return abmag_band(366.3, 65, out)
        elif name == 'B':
            return abmag_band(436.1, 89, out)
        elif name == 'V':
            return abmag_band(544.8, 84, out)
        elif name == 'Rc':
            return abmag_band(641.0, 160., out)
        elif name == 'Ic':
            return abmag_band(798.0, 150., out)
        elif name == 'z':
            return abmag_band(893.0, 147., out)
        elif name == 'R-Johnson':
            (l0,lmin,lmax,tck) = ABmag_filters.mag_RJohnson()
            return self._filter(l0, lmin, lmax, tck, out, spline)
        elif name == 'F606W':
            (l0,lmin,lmax,tck) = ABmag_filters.mag_F606W()
            return self._filter(l0, lmin, lmax, tck, out, spline)
        else:
            pass
        
    def abmag_filter(self, lbda, eff, out=1, spline=False):
        """ computes AB magnitude using array filter.

        Parameters
        ----------
        lbda : array
        Wavelength values.
        
        eff : array
        Efficiency values.

        out : 1 or 2
        1: the magnitude is returned
        2: the magnitude, mean flux and mean lbda are returned

        spline : boolean
        linear/spline interpolation to interpolate masked values
        
        if you want to use a txt file :
        lbda,eff = np.loadtxt(name, unpack=True)
        """            
        l0 = np.average(lbda, weights=eff)
        lmin = lbda[0]
        lmax = lbda[-1]
        tck = interpolate.splrep(lbda,eff)           
        return self._filter(l0, lmin, lmax, tck, out, spline)

        
    def _filter(self, l0, lmin, lmax, tck, out=1, spline=False):
        """ computes AB magnitude

        Parameters
        ----------
        l0 : float
        Mean wavelength
        
        lmin : float
        Minimum wavelength
        
        lmax : float
        Maximum wavelength
        
        tck : 3-tuple
        (t,c,k) contains the spline representation. 
        t = the knot-points, c = coefficients and  k = the order of the spline.

        out : 1 or 2
        1: the magnitude is returned
        2: the magnitude, mean flux and mean lbda are returned

        spline : boolean
        linear/spline interpolation to interpolate masked values
        """
        imin = self.wave.pixel(lmin,True)
        imax = self.wave.pixel(lmax,True)
        if imin == imax:
            if imin==0 or imin==self.shape:
                raise ValueError, 'Spectrum outside Filter band'
            else:
                raise ValueError, 'filter band smaller than spectrum step'
        lb = (numpy.arange(imin,imax) - self.wave.crpix + 1) * self.wave.cdelt + self.wave.crval
        w = interpolate.splev(lb,tck,der=0)
        data = self.interp_data(spline)
        vflux = np.average(data[imin:imax], weights=w)*self.fscale
        mag = flux2mag(vflux, l0)
#        if vflux > 0:
#            mag = flux2mag(vflux, l0*10)
#        else:
#            mag = 99
        if out == 1:
            return mag
        if out == 2:
            return mag,vflux,l0

    def truncate(self, lmin=None, lmax=None):
        """truncates a spectrum

        Parameters
        ----------
        lmin : float
        Minimum wavelength.

        lmax : float
        Maximum wavelength.
        """
        if lmin is None:
            i1 = 0
        else:
            i1 = self.wave.pixel(lmin, True)
        if lmax is None:
            i2 = self.shape
        else:
            i2 = self.wave.pixel(lmax, True)
        return self.__getitem__(slice(i1,i2,1))

    def fwhm(self, l0, cont=0):
        """ Returns the fwhm of a peak

        Parameters
        ----------
        l0 : float
        wavelength value corresponding to the peak position.

        cont : integer
        The continuum [default 0].
        """
        k0 = self.wave.pixel(l0, nearest=True)
        d = self.data - cont
        f2 = d[k0]/2
        k2 = np.argwhere(d[k0:]<f2)[0][0] + k0
        i2 = np.interp(f2, d[k2:k2-2:-1], [k2,k2-1])
        k1 = k0 - np.argwhere(d[k0::-1]<f2)[0][0]
        i1 = np.interp(f2, d[k1:k1+2], [k1,k1+1])
        fwhm = (i2 - i1)*self.wave.cdelt
        return fwhm

    def gauss_fit(self,lmin,lmax,fwhm=None,lpeak=None,fpeak=None):
        """performs polynomial fit on spectrum.
        Returns [[fwhm,lpeak,fpeak],[err_fwhm,err_lpeak,err_fpeak]]

        Parameters
        ----------
        lmin : float
        Minimum wavelength.

        lmax : float
        Maximum wavelength.

        fwhm : float
        input gaussian fwhm, if None it is estimated

        lpeak : float
        input gaussian center, if None it is estimated

        fpeak : float
        input gaussian peak value, if None it is estimated
        """
        gauss_fit = lambda p, x: p[0]*(1/np.sqrt(2*np.pi*(p[2]**2)))*np.exp(-(x-p[1])**2/(2*p[2]**2)) #1d Gaussian func
        e_gauss_fit = lambda p, x, y: (gauss_fit(p,x) -y) #1d Gaussian fit

        spec = self.truncate(lmin, lmax)

        if isinstance(spec.data,np.ma.core.MaskedArray):
            mask = np.array(1 - spec.data.mask,dtype=bool)
            l = spec.wave.coord().compress(mask)
            d = spec.data.compress(mask)*self.fscale
            x = np.arange(self.shape).compress(mask)
        else:
            l = spec.wave.coord() #!!!!!this is expected to be pixel number
            d = spec.data*self.fscale
            x = np.arange(self.shape)

        if lpeak is None:
            lpeak = l[d.argmax()]
        if fpeak is None:
            fpeak = d.max()
        if fwhm is None:
            fwhm = spec.fwhm(lpeak, 0)

        sigma = fwhm/(2.*np.sqrt(2.*np.log(2.0)))

        v0 = [fpeak,lpeak,sigma] #inital guesses for Gaussian Fit. $just do it around the peak
        print v0
        out = leastsq(e_gauss_fit, v0[:], args=(l, d), maxfev=100000, full_output=1) #Gauss Fit
        v = out[0] #fit parammeters out
        covar = out[1] #covariance matrix output

        if self.shape > len(v) and covar is not None:
            s_sq = (e_gauss_fit(v, l, d)**2).sum()/(self.shape-len(v))
            err = np.diag(covar * s_sq)
        else:
            err = np.zeros(len(v))

        lpeak = v[1]
        err_lpeak = err[1]
        sigma = v[2]
        fwhm = sigma*2*np.sqrt(2*np.log(2))
        err_sigma = err[2]
        err_fwhm = err_sigma*2*np.sqrt(2*np.log(2))
        fpeak = v[0]
        err_fpeak = err[0]

        print "lpeak", lpeak, err_lpeak
        print 'sigma',sigma, err_sigma
        print 'fwhm',fwhm, err_fwhm
        print 'fpeak',fpeak,err_fpeak


        xxx = np.arange(min(l),max(l),l[1]-l[0])
        ccc = gauss_fit(v,xxx) # this will only work if the units are pixel and not wavelength
        iii = gauss_fit(v0,xxx)

#        import matplotlib.pyplot as plt
#        plt.figure()
        plt.plot(xxx,ccc,'r--')
#        plt.show()

        return[[fwhm,lpeak,fpeak],[err_fwhm,err_lpeak,err_fpeak]]

    def add_gaussian(self,fwhm,lpeak,fpeak):
        """adds a gausian on spectrum.

        Parameters
        ----------

        fwhm : float
        gaussian fwhm

        lpeak : float
        gaussian center

        fpeak : float
        gaussian peak value
        """
        gauss = lambda p, x: p[0]*(1/np.sqrt(2*np.pi*(p[2]**2)))*np.exp(-(x-p[1])**2/(2*p[2]**2)) #1d Gaussian func

        sigma = fwhm/(2.*np.sqrt(2.*np.log(2.0)))

        lmin = lpeak - 5*sigma
        lmax = lpeak + 5*sigma
        imin = self.wave.pixel(lmin, True)
        imax = self.wave.pixel(lmax, True)
        if imin == imax:
            if imin==0 or imin==self.shape:
                raise ValueError, 'Gaussian outside spectrum wavelength range'

        wave  = self.wave.coord()[imin:imax]
        v = [fpeak,lpeak,sigma]

        res = self.copy()
        res.data[imin:imax] = self.data[imin:imax] + gauss(v,wave)

        return res
    
    def plot(self, max=None, title=None, noise=False, lmin=None, lmax=None, drawstyle='steps-mid'): 
        """ plots the spectrum.
        
        Parameters
        ----------
        
        max : boolean
        If max is True, the plot is normalized to peak at max value.
        
        title : string
        Figure tiltle (None by default).
        
        noise : boolean
        If noise is True, the +/- standard deviation is overplotted.
        
        lmin : float
        Minimum wavelength.

        lmax : float
        Maximum wavelength.
        
        drawstyle : [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
        Drawstyle of the plot. 'default' connects the points with lines. 
        The steps variants produce step-plots. 'steps' is equivalent to 'steps-pre'.
        'steps-pre' by default.        
        """
        plt.ion()
        
        res = self.truncate(lmin,lmax)
        x = res.wave.coord()
        f = res.data*res.fscale
        if max != None:
            f = f*max/f.max()
        if res.var is  None:
            noise = False
        plt.clf() #Clear the current figure
        plt.plot(x, f, drawstyle=drawstyle)
        if noise: 
            plt.fill_between(x, f + np.sqrt(res.var)*res.fscale, f -np.sqrt(res.var)*res.fscale, color='0.75', facecolor='0.75', alpha=0.5) 
#            if isinstance(self.data,np.ma.core.MaskedArray):
#                f = res.data.data*res.fscale
#                plt.fill_between(x, f + np.sqrt(res.var)*res.fscale, f -np.sqrt(res.var)*res.fscale, color='0.75', facecolor='0.75', alpha=0.5) 
#            else:        
#                plt.fill_between(x, f + np.sqrt(res.var)*res.fscale, f -np.sqrt(res.var)*res.fscale, color='0.75', facecolor='0.75', alpha=0.5) 
        if title is not None:
                plt.title(title)   
        if res.wave.cunit is not None:
            plt.xlabel(res.wave.cunit)
        if res.unit is not None:
            plt.ylabel(res.unit)
        plt.connect('motion_notify_event', self._on_move)
        
    def log_plot(self, max=None, title=None, noise=False, lmin=None, lmax=None, drawstyle='steps-mid'): 
        """ plots the spectrum with y logarithmic scale.
        
        Parameters
        ----------
        
        max : boolean
        If max is True, the plot is normalized to peak at max value.
        
        title : string
        Figure tiltle (None by default).
        
        noise : boolean
        If noise is True, the +/- standard deviation is overplotted.
        
        lmin : float
        Minimum wavelength.

        lmax : float
        Maximum wavelength.
        
        drawstyle : [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
        Drawstyle of the plot. 'default' connects the points with lines. 
        The steps variants produce step-plots. 'steps' is equivalent to 'steps-pre'.
        'steps-pre' by default.        
        """
        plt.ion()
        
        res = self.truncate(lmin,lmax)
        x = res.wave.coord()
        f = res.data*res.fscale
        if max != None:
            f = f*max/f.max()
        if res.var is  None:
            noise = False
        plt.clf() #Clear the current figure
        plt.semilogy(x, f, drawstyle=drawstyle)
        if noise: 
           plt.fill_between(x, f + np.sqrt(res.var)*res.fscale, f - np.sqrt(res.var)*res.fscale, color='0.75', facecolor='0.75', alpha=0.5)   
        if title is not None:
                plt.title(title)   
        if res.wave.cunit is not None:
            plt.xlabel(res.wave.cunit)
        if res.unit is not None:
            plt.ylabel(res.unit)
        plt.connect('motion_notify_event', self._on_move)

        
    def _on_move(self,event):
        """ prints x,y,i,lbda and data in the figure toolbar.
        """
        if event.inaxes is not None:
            xc, yc = event.xdata, event.ydata
            try:
                i = self.wave.pixel(xc, True)
                x = self.wave.coord(i)
                if isinstance(self.data,np.ma.core.MaskedArray):
                    val = self.data.data[i]*self.fscale
                else:
                    val = self.data[i]*self.fscale
                s = 'x= %g y=%g i=%d lbda=%g data=%g'%(xc,yc,i,x,val)
                plt.get_current_fig_manager().toolbar.set_message(s)
            except:
                pass
            
    def print_cursor_pos(self, filename='None'):
        """Prints cursor position.   
        To read cursor position, click on the left mouse button
        To remove a cursor position, click on the left mouse button + <shift>
        To quit the interactive mode, click on the right mouse button.
        At the end, clicks are saved in self.clicks as dictionary {'xc','yc','x','y'}.
        Parameters
        ----------
        
        filename : string
        If filename is not None, the cursor values are saved as a fits table.
        """
        if self._clicks is None:
            binding_id = plt.connect('button_press_event', self._on_click)
            plt.connect('motion_notify_event', self._on_move)
            self._clicks = SpectrumClicks(binding_id,filename)
        else:
            self._clicks.filename = filename
        
    def _on_click(self,event):
        """ prints x,y,i,lbda and data corresponding to the cursor position.
        """
        if event.key == 'shift':
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        xc, yc = event.xdata, event.ydata
                        i = np.argmin(np.abs(self._clicks.xc-xc))
                        line = self._clicks.id_lines[i]
                        del plt.gca().lines[line]
                        self._clicks.xc.pop(i)
                        self._clicks.yc.pop(i)
                        self._clicks.i.pop(i)
                        self._clicks.x.pop(i)
                        self._clicks.data.pop(i)
                        self._clicks.id_lines.pop(i)
                        for j in range(i,len(self._clicks.id_lines)):
                            self._clicks.id_lines[j] -= 1
                        plt.draw() 
                        print "new selection:"
                        if self.fscale == 1:
                            for i in range(len(self._clicks.xc)):
                                print 'x= %g\ty=%g\ti=%d\tlbda=%g\tdata=%g'%(self._clicks.xc[i],self._clicks.yc[i],self._clicks.i[i],self._clicks.x[i],self._clicks.data[i])
                        else:
                            for i in range(len(self._clicks.xc)):
                                print 'x= %g\ty=%g\ti=%d\tlbda=%g\tdata=%g\t[scaled=%g]'%(self._clicks.xc[i],self._clicks.yc[i],self._clicks.i[i],self._clicks.x[i],self._clicks.data[i],self._clicks.data[i]/self.fscale)                           
                    except:
                        pass 
        else:
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        xc, yc = event.xdata, event.ydata
                        i = self.wave.pixel(xc, True)
                        x = self.wave.coord(i)
                        val = self.data[i]*self.fscale
                        if len(self._clicks.x)==0:
                            print ''
                        plt.plot(xc,yc,'r+')
                        self._clicks.xc.append(xc)
                        self._clicks.yc.append(yc)
                        self._clicks.i.append(i)
                        self._clicks.x.append(x)
                        self._clicks.data.append(val)
                        self._clicks.id_lines.append(len(plt.gca().lines)-1)
                        if self.fscale == 1:
                            print 'x= %g\ty=%g\ti=%d\tlbda=%g\tdata=%g'%(xc,yc,i,x,val)
                        else:
                            print 'x= %g\ty=%g\ti=%d\tlbda=%g\tdata=%g\t[scaled=%g]'%(xc,yc,i,x,val,val/self.fscale)
                    except:
                        pass 
            else: 
                if self._clicks.filename != 'None':
                    c1 = pyfits.Column(name='XC', format='E', array=self._clicks.xc)
                    c2 = pyfits.Column(name='YC', format='E', array=self._clicks.yc)
                    c3 = pyfits.Column(name='I', format='I', array=self._clicks.i)
                    c4 = pyfits.Column(name='X', format='E', array=self._clicks.x)
                    c5 = pyfits.Column(name='DATA', format='E', array=self._clicks.data)
                    tbhdu=pyfits.new_table(pyfits.ColDefs([c1, c2, c3, c4, c5]))
                    tbhdu.writeto(self._clicks.filename, clobber=True)
                    print 'printing coordinates in fits table %s'%self._clicks.filename
                # save clicks in a dictionary {'xc','yc','x','y'}
                d = {'xc':self._clicks.xc, 'yc':self._clicks.yc, 'x':self._clicks.x, 'y':self._clicks.data}
                self.clicks = d
                print "disconnecting console coordinate printout..."
                plt.disconnect(self._clicks.binding_id)
                nlines =  len(self._clicks.id_lines)
                for i in range(nlines):
                    line = self._clicks.id_lines[nlines - i -1]
                    del plt.gca().lines[line]
                plt.draw()
                self._clicks = None
                
            
    def get_distance(self):
        """Gets distance and center from 2 cursor positions.
        """
        print 'Use 2 mouse clicks to get center and distance'
        if self._clicks is None:
            binding_id = plt.connect('button_press_event', self._on_click_dist)
            plt.connect('motion_notify_event', self._on_move)
            self._clicks = SpectrumClicks(binding_id)
    
    def _on_click_dist(self,event):
        """Prints distance and center between 2 cursor positions.
        """
        if event.button == 1:
            if event.inaxes is not None:
                try:
                    xc, yc = event.xdata, event.ydata
                    i = self.wave.pixel(xc, True)
                    x = self.wave.coord(i)
                    val = self.data[i]*self.fscale
                    if len(self._clicks.x)==0:
                        print ''
                    plt.plot(xc,yc,'r+')
                    self._clicks.xc.append(xc)
                    self._clicks.yc.append(yc)
                    self._clicks.i.append(i)
                    self._clicks.x.append(x)
                    self._clicks.data.append(val)
                    self._clicks.id_lines.insert(0,len(plt.gca().lines)-1)
                    if self.fscale == 1:
                        print 'x= %g\ty=%g\ti=%d\tlbda=%g\tdata=%g'%(xc,yc,i,x,val)
                    else:
                        print 'x= %g\ty=%g\ti=%d\tlbda=%g\tdata=%g\t[scaled=%g]'%(xc,yc,i,x,val,val/self.fscale)
                    if np.sometrue(np.mod( len(self._clicks.x), 2 )) == False:
                        dx = abs(self._clicks.xc[-1] - self._clicks.xc[-2])
                        xc = (self._clicks.xc[-1] + self._clicks.xc[-2])/2
                        print 'Center: %f Distance: %f' % (xc,dx)
                except:
                    pass 
        else: 
            print "disconnecting console distance printout..."
            plt.disconnect(self._clicks.binding_id)
            for i in self._clicks.id_lines:
                del plt.gca().lines[i]
            plt.draw()
            self._clicks = None
            
    def plot_mask(self):
        if isinstance(self.data,np.ma.core.MaskedArray):
            lbda = self.wave.coord()
#            ksel = np.where(self.data.mask==True)
#            x = lbda[ksel]
#            f = self.data.data[ksel]
            drawstyle = plt.gca().lines[0].get_drawstyle()
#            plt.plot(x, f, 'r+')
            plt.plot(lbda,self.data.data,drawstyle=drawstyle, hold = True, alpha=0.3)

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
    Lengths of data in Y and X (python notation: (ny,nx))

    var : array
    Array containing the variance

    fscale : float
    Flux scaling factor (1 by default)

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
        Lengths of data in Y and X. (101,101) by default.
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
        if filename is not None:
            f = pyfits.open(filename)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                # if the number of extension is 1, we just read the data from the primary header
                # test if image
                if hdr['NAXIS'] != 2:
                    raise IOError, '  not an image'
                self.unit = hdr.get('BUNIT', None)
                self.cards = hdr.ascardlist()
                self.shape = np.array([hdr['NAXIS2'],hdr['NAXIS1']])
                self.data = f[0].data
                self.var = None
                self.fscale = hdr.get('FSCALE', 1.0)
                try:
                    self.wcs = WCS(hdr) # WCS object from data header
                except:
                    print "Invalid wcs self.wcs=None"
                    self.wcs = None
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = f['DATA'].data
                else:
                    h = f[ext].header
                    d = f[ext].data
                if h['NAXIS'] != 2:
                    raise IOError, 'Wrong dimension number in DATA extension'
                self.unit = h.get('BUNIT', None)
                self.cards = h.ascardlist()
                self.shape = np.array([h['NAXIS2'],h['NAXIS1']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                try:
                    self.wcs = WCS(h) # WCS object from data header
                except:
                    print "Invalid wcs self.wcs=None"
                    self.wcs = None
                self.var = None
                if getnoise:
                    if f['STAT'].header['NAXIS'] != 2:
                        raise IOError, 'Wrong dimension number in STAT extension'
                    if f['STAT'].header['NAXIS1'] != self.shape[1] and f['STAT'].header['NAXIS2'] != self.shape[0]:
                        raise IOError, 'Number of points in STAT not equal to DATA'
                    self.var = f['STAT'].data
                # DQ extension
                try:
                    mask = np.ma.make_mask(f['DQ'].data)
                    self.data = np.ma.array(self.data, mask=mask)
                except:
                    pass
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
                if wcs.wcs.naxis1 == 0. and wcs.wcs.naxis2 == 0.:
                    self.wcs = wcs
                    self.wcs.wcs.naxis1 = self.shape[1]
                    self.wcs.wcs.naxis2 = self.shape[0]
                elif wcs.wcs.naxis1 != self.shape[1] or wcs.wcs.naxis2 != self.shape[0]:
                    print "warning: world coordinates and data have not the same dimensions."
                    self.wcs =  None
                else:
                    self.wcs = wcs
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

        if isinstance(self.data,np.ma.core.MaskedArray):
            hdulist = [prihdu]
            # create spectrum DATA in first extension
            tbhdu = pyfits.ImageHDU(name='DATA', data=self.data)
            for card in wcs_cards:
                tbhdu.header.update(card.key, card.value, card.comment)
            if self.unit is not None:
                tbhdu.header.update('BUNIT', self.unit, 'data unit type')
            tbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
            hdulist.append(tbhdu)
            if self.var is not None:
                # create spectrum STAT in second extension
                nbhdu = pyfits.ImageHDU(name='STAT', data=self.var)
                for card in wcs_cards:
                    nbhdu.header.update(card.key, card.value, card.comment)
    #            if self.unit is not None:
    #                nbhdu.header.update('UNIT', self.unit, 'data unit type')
    #            nbhdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
                hdulist.append(nbhdu)
            # DQ extension
            dqhdu = pyfits.ImageHDU(name='DQ', data=np.uint8(self.data.mask))
            hdulist.append(dqhdu)
        else:
            if self.var is None: # write simple fits file without extension
                prihdu.data = self.data
                for card in wcs_cards:
                    prihdu.header.update(card.key, card.value, card.comment)
                if self.unit is not None:
                    prihdu.header.update('BUNIT', self.unit, 'data unit type')
                prihdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
                hdulist = [prihdu]
            else: # write fits file with primary header and two extensions
                hdulist = [prihdu]
                # create spectrum DATA in first extension
                tbhdu = pyfits.ImageHDU(name='DATA', data=self.data)
                for card in wcs_cards:
                    tbhdu.header.update(card.key, card.value, card.comment)
                if self.unit is not None:
                    tbhdu.header.update('BUNIT', self.unit, 'data unit type')
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
        return result

    def __lt__ (self, item):
        """masks data array where greater or equal than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data, item/self.fscale)
        return result

    def __ge__ (self, item):
        """masks data array where less than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item/self.fscale)
        return result

    def __gt__ (self, item):
        """masks data array where less or equal than a given value.
        Returns an Image object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data, item/self.fscale)
        return result

    def remove_mask(self):
        """removes the mask on the array
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            self.data = self.data.data

    def resize(self):
        """resize the image to have a minimum number of masked values
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            ksel = np.where(self.data.mask==False)
            try:
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
            except:
                pass

    def __add__(self, other):
        """ adds other

        image1 + number = image2 (image2[j,i]=image1[j,i]+number)

        image1 + image2 = image3 (image3[j,i]=image1[j,i]+image2[j,i])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        image + cube1 = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
        The last two dimensions of cube1 must be equal to the image dimensions.
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
        The last two dimensions of cube1 must be equal to the image dimensions.
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
        The last two dimensions of cube1 must be equal to the image dimensions.
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
        The last two dimensions of cube1 must be equal to the image dimensions.
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
                res = Image(getnoise=getnoise, shape=shape, wcs = wcs, unit=self.unit, empty=True,fscale=self.fscale)
                res.data = data
                if getnoise:
                    res.var = var
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
        if wcs.wcs.naxis1 == 0. and wcs.wcs.naxis2 == 0.:
            self.wcs = wcs
            self.wcs.wcs.naxis1 = self.shape[1]
            self.wcs.wcs.naxis2 = self.shape[0]
        elif wcs.wcs.naxis1 != self.shape[1] or wcs.wcs.naxis2 != self.shape[0]:
            print "warning: world coordinates and data have not the same dimensions."
            self.wcs =  None
        else:
            self.wcs = wcs

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
    Lengths of data in X and Y and Z (python notation (nz,ny,nx)

    var : array
    Array containing the variance

    fscale : float
    Flux scaling factor (1 by default)

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

        shape : integer or (integer,integer,integer)
        Lengths of data in Z, Y and X. (101,101,101) by default.

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
        if filename is not None:
            f = pyfits.open(filename)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                # if the number of extension is 1, we just read the data from the primary header
                # test if image
                if hdr['NAXIS'] != 3:
                    raise IOError, 'Wrong dimension number: not a cube'
                self.unit = hdr.get('BUNIT', None)
                self.cards = hdr.ascardlist()
                self.shape = np.array([hdr['NAXIS3'],hdr['NAXIS2'],hdr['NAXIS1']])
                self.data = f[0].data
                self.var = None
                self.fscale = hdr.get('FSCALE', 1.0)
                # WCS object from data header
                try:
                    self.wcs = WCS(hdr)
                except:
                    print "Invalid wcs self.wcs=None"
                    self.wcs = None
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
                self.wave = WaveCoord(crpix, cdelt, crval, cunit,self.shape[0])
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = f['DATA'].data
                else:
                    h = f[ext].header
                    d = f[ext].data
                if h['NAXIS'] != 3:
                    raise IOError, 'Wrong dimension number in DATA extension'
                self.unit = h.get('BUNIT', None)
                self.cards = h.ascardlist()
                self.shape= np.array([h['NAXIS3'],h['NAXIS2'],h['NAXIS1']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                try:
                    self.wcs = WCS(h) # WCS object from data header
                except:
                    print "Invalid wcs self.wcs=None"
                    self.wcs = None
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
                self.wave = WaveCoord(crpix, cdelt, crval, cunit, self.shape[0])
                self.var = None
                if getnoise:
                    if f['STAT'].header['NAXIS'] != 3:
                        raise IOError, 'Wrong dimension number in STAT extension'
                    if f['STAT'].header['NAXIS1'] != self.shape[2] and f['STAT'].header['NAXIS2'] != self.shape[1] and f['STAT'].header['NAXIS3'] != self.shape[0]:
                        raise IOError, 'Number of points in STAT not equal to DATA'
                    self.var = f['STAT'].data
                # DQ extension
                try:
                    mask = np.ma.make_mask(f['DQ'].data)
                    self.data = np.ma.MaskedArray(self.data, mask=mask)
                except:
                    pass
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
                if wcs.wcs.naxis1 == 0. and wcs.wcs.naxis2 == 0.:
                    self.wcs = wcs
                    self.wcs.wcs.naxis1 = self.shape[2]
                    self.wcs.wcs.naxis2 = self.shape[1]
                elif wcs.wcs.naxis1 != self.shape[2] or wcs.wcs.naxis2 != self.shape[1]:
                    print "warning: world coordinates and data have not the same dimensions."
                    self.wcs =  None
                else:
                    self.wcs = wcs
            except :
                self.wcs = None
            try:
                if wave.shape is not None and wave.shape != self.shape[0]:
                    print "warning: wavelength coordinates and data have not the same dimensions."
                    self.wave = None
                else:
                    self.wave = wave
                    self.wave.shape = self.shape[0]
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

        #ToDo: pb with mask !!!!!!!!!!!!!!!!!

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
                prihdu.header.update('BUNIT', self.unit, 'data unit type')
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
                tbhdu.header.update('BUNIT', self.unit, 'data unit type')
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
        return result

    def __lt__ (self, item):
        """masks data array where greater or equal than a given value.
        Returns a cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data, item/self.fscale)
        return result

    def __ge__ (self, item):
        """masks data array where less than a given value.
        Returns a Cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item/self.fscale)
        return result

    def __gt__ (self, item):
        """masks data array where less or equal than a given value.
        Returns a Cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data, item/self.fscale)
        return result

    def remove_mask(self):
        """removes the mask on the array
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            self.data = self.data.data

    def resize(self):
        """resize the cube to have a minimum number of masked values
        """
        if self.data is not None and isinstance(self.data,np.ma.core.MaskedArray):
            ksel = np.where(self.data.mask==False)
            try:
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
            except:
                pass

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
                    res = Image(getnoise=getnoise, shape=shape, wcs = wcs, unit=self.unit, empty=True,fscale=self.fscale)
                    res.data = data
                    if getnoise:
                        res.var =var
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
                res = Spectrum(getnoise=getnoise, shape=shape, wave = wave, unit=self.unit, empty=True,fscale=self.fscale)
                res.data = data
                if getnoise:
                    res.var = var
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
                res = Cube(getnoise=getnoise, shape=shape, wcs = wcs, wave = wave, unit=self.unit, empty=True,fscale=self.fscale)
                res.data = data
                if getnoise:
                    res.var = var
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
        if wcs.wcs.naxis1 == 0. and wcs.wcs.naxis2 == 0.:
            self.wcs = wcs
            self.wcs.wcs.naxis1 = self.shape[2]
            self.wcs.wcs.naxis2 = self.shape[1]
        elif wcs.wcs.naxis1 != self.shape[2] or wcs.wcs.naxis2 != self.shape[1]:
            print "warning: world coordinates and data have not the same dimensions."
            self.wcs =  None
        else:
            self.wcs = wcs
        
        if wave.shape is not None and wave.shape != self.shape[0]:
            print "warning: wavelength coordinates and data have not the same dimensions."
            self.wave = None
        else:
            self.wave = wave