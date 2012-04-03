""" obj.py Manages cube, image and spectrum objects"""
import numpy as np
import pyfits
import datetime
from coords import WCS
from coords import WaveCoord

from scipy import integrate
from scipy import interpolate
from scipy.optimize import leastsq
from scipy import signal
from scipy import special

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
    """Object used to save click on spectrum plot.
    
    Attributes
    ---------- 
    filename : string
    Name of the table fits file where are saved the clicks values.
    
    binding_id : integer
    Connection id.
    
    xc : list of float
    Cursor position in spectrum (world coordinates).
    
    yc : list of float
    Cursor position in spectrum (world coordinates).
    
    i : list of integer
    Nearest pixel in spectrum.
    
    x : list of float
    Corresponding nearest position in spectrum (world coordinates)
    
    data : list of float
    Corresponding spectrum data value.
    
    id_lines : list of integer
    Plot id (cross for cursor positions).
    """
    def __init__(self, binding_id, filename=None):
        self.filename = filename
        self.binding_id = binding_id
        self.xc = []
        self.yc = []
        self.i = []
        self.x = []
        self.data = []
        self.id_lines = []
        
    def remove(self,xc):
        """removes a cursor position
        """
        i = np.argmin(np.abs(self.xc-xc))
        line = self.id_lines[i]
        del plt.gca().lines[line]
        self.xc.pop(i)
        self.yc.pop(i)
        self.i.pop(i)
        self.x.pop(i)
        self.data.pop(i)
        self.id_lines.pop(i)
        for j in range(i,len(self.id_lines)):
            self.id_lines[j] -= 1
        plt.draw()
        
    def add(self,xc,yc,i,x,data):
        plt.plot(xc,yc,'r+')
        self.xc.append(xc)
        self.yc.append(yc)
        self.i.append(i)
        self.x.append(x)
        self.data.append(data)        
        self.id_lines.append(len(plt.gca().lines)-1)
        
    def iprint(self,i,fscale):
        """prints a cursor positions
        """
        if fscale == 1:
            print 'x= %g\ty=%g\ti=%d\tlbda=%g\tdata=%g'%(self.xc[i],self.yc[i],self.i[i],self.x[i],self.data[i])
        else:
            print 'x= %g\ty=%g\ti=%d\tlbda=%g\tdata=%g\t[scaled=%g]'%(self.xc[i],self.yc[i],self.i[i],self.x[i],self.data[i],self.data[i]/fscale) 
           
    def write_fits(self): 
        """prints coordinates in fits table.
        """
        if self.filename != 'None':
            c1 = pyfits.Column(name='XC', format='E', array=self.xc)
            c2 = pyfits.Column(name='YC', format='E', array=self.yc)
            c3 = pyfits.Column(name='I', format='I', array=self.i)
            c4 = pyfits.Column(name='X', format='E', array=self.x)
            c5 = pyfits.Column(name='DATA', format='E', array=self.data)
            tbhdu=pyfits.new_table(pyfits.ColDefs([c1, c2, c3, c4, c5]))
            tbhdu.writeto(self.filename, clobber=True)
            print 'printing coordinates in fits table %s'%self.filename     
          
    def clear(self):
        """disconnects and clears
        """
        print "disconnecting console coordinate printout..."
        plt.disconnect(self.binding_id)
        nlines =  len(self.id_lines)
        for i in range(nlines):
            line = self.id_lines[nlines - i -1]
            del plt.gca().lines[line]
        plt.draw()                
        
class Gauss1D:
    """ Object used to saved 1d gaussian parameters
    
    Attributes
    ---------- 
    
    cont : float
    Continuum value.
    
    fwhm : float
    Gaussian fwhm.

    lpeak : float
    Gaussian center.

    peak : float
    Gaussian peak value.
    
    flux : float
    Gaussian integrated flux.
        
    err_fwhm : float
    Estimated error on Gaussian fwhm.
    
    err_lpeak : float
    Estimated error on Gaussian center.
    
    err_peak : float
    Estimated error on Gaussian peak value.
    
    flux : float
    Gaussian integrated flux.
    
    """
    def __init__(self, lpeak, peak, flux, fwhm, cont, err_lpeak, err_peak, err_flux,err_fwhm):
        self.cont = cont
        self.fwhm = fwhm
        self.lpeak = lpeak
        self.peak = peak
        self.flux = flux
        self.err_fwhm = err_fwhm
        self.err_lpeak = err_lpeak
        self.err_peak = err_peak
        self.err_flux = err_flux
        
    def copy(self):
        res = Gauss1D(self.lpeak, self.peak, self.flux, self.fwhm, self.cont, self.err_lpeak, self.err_peak, self.err_flux,self.err_fwhm)
        return res
        
    def print_param(self):
        print 'Gaussian center = %g (error:%g)' %(self.lpeak,self.err_lpeak)   
        print 'Gaussian integrated flux = %g (error:%g)' %(self.flux,self.err_flux)
        print 'Gaussian peak value = %g (error:%g)' %(self.peak,self.err_peak)
        print 'Gaussian fwhm = %g (error:%g)' %(self.fwhm,self.err_fwhm)
        print 'Gaussian continuum = %g' %self.cont
        
        
        
        print ''
        

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

    Selection: <, >, <=, >=

    Arithmetic: + - * / pow

    Info: info, []
    """

    def __init__(self, filename=None, ext = None, getnoise=False, shape=101, wave = None, unit=None, data=None, var=None,fscale=1.0,empty=False):
        """creates a Spectrum object

        Parameters
        ----------
        filename : string
        Possible FITS filename

        ext : integer or (integer,integer)
        Number of the data extension or numbers of the data and variance extensions.

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
                self.data = np.array(f[0].data,dtype=float)
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
                cunit = hdr.get('CUNIT1','')
                self.wave = WaveCoord(crpix, cdelt, crval, cunit, self.shape)
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = np.array(f['DATA'].data, dtype=float)
                else:
                    if isinstance(ext,int):
                        n = ext
                    else:
                        n = ext[0]
                    h = f[n].header
                    d = np.array(f[n].data, dtype=float)
                        
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
                cunit = h.get('CUNIT1','')
                self.wave = WaveCoord(crpix, cdelt, crval, cunit, self.shape)
                # STAT extension
                self.var = None
                if getnoise:
                    if ext is None:
                        fstat = f['STAT']
                    else:
                        n = ext[1]
                        fstat = f[n]
                    if fstat.header['NAXIS'] != 1:
                        raise IOError, 'Wrong dimension number in STAT extension'
                    if fstat.header['NAXIS1'] != self.shape:
                        raise IOError, 'Number of points in STAT not equal to DATA'
                    self.var = np.array(fstat.data, dtype=float)
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
        #Mask an array where invalid values occur (NaNs or infs).
        if self.data is not None:
            self.data = np.ma.masked_invalid(self.data)

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

        if np.ma.count_masked(self.data) != 0:
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
                prihdu.data = self.data.data
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
        print '%s (%s) fscale=%g, %s' %(data,unit,self.fscale,noise)
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

    def resize(self):
        """resizes the spectrum to have a minimum number of masked values
        """
        if np.ma.count_masked(self.data) != 0:
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
            
    def get_step(self):
        """returns the wavelength step
        """
        if self.wave is not None:
            return self.wave.cdelt
        else:
            return None
            
    def __setitem__(self,key,value):
        """ sets the corresponding part of data
        """
        self.data[key] = value

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
            self.data[pix_min:pix_max] = np.ma.masked  
            
    def unmask(self):
        """unmasks the spectrum
        """
        self.data.mask = False
        self.data = np.ma.masked_invalid(self.data)
        
    def _interp(self, wavelengths, spline=False):
        """ returns the interpolated values corresponding to the wavelength array
        
        Parameters
        ----------
        wavelengths : array of float
        wavelength values
        
        spline : boolean
        False: linear interpolation, True: spline interpolation 
        """
        lbda = self.wave.coord()
        ksel = np.where(self.data.mask==False)            
        d = np.zeros(np.shape(ksel)[1]+2, dtype= float)
        d[1:-1] = self.data.data[ksel]
        w = np.zeros(np.shape(ksel)[1]+2)      
        w[1:-1] = lbda[ksel]
        d[0] = d[1]
        d[-1] = d[-2]
        w[0] = (-self.wave.crpix + 1) * self.wave.cdelt + self.wave.crval - 0.5 * self.wave.cdelt
        w[-1] = (self.shape - self.wave.crpix ) * self.wave.cdelt + self.wave.crval + 0.5 * self.wave.cdelt
        if self.var is not None:
            weight = np.zeros(np.shape(ksel)[1]+2)
            weight[1:-1] = 1./self.var[ksel]
            weight[0] = self.var[1]
            weight[-1] = self.var[-2]
        else:
            weight = None
        if spline:
            tck = interpolate.splrep(w,d,w=weight)
            return interpolate.splev(wavelengths,tck,der=0)
        else:
            f = interpolate.interp1d(w, d)
            return f(wavelengths)

        
    def _interp_data(self, spline=False):
        """ returns data array with interpolated values for masked pixels
        
        Parameter
        ----------
        spline : boolean
        False: linear interpolation, True: spline interpolation 
        """
        if np.ma.count_masked(self.data) == 0:
            return self.data.data
        else:
            lbda = self.wave.coord()
            ksel = np.where(self.data.mask==True)
            wnew = lbda[ksel]
            data = self.data.data
            data[ksel] = self._interp(wnew,spline)
            return data
    
    def interp_mask(self, spline=False):
        """ returns a spectrum equal to the current spectrum with interpolated values for masked pixels.
        
        Parameter
        ----------
        
        spline : boolean
        False: linear interpolation, True: spline interpolation 
        """
        res = self.copy()
        res.data = np.ma.masked_invalid(self._interp_data(spline))
        return res
            
    def _rebin_factor(self, factor):
        '''shrinks the size of the spectrum by factor.
        New size is an integer multiple of the original size.
        
        Parameter
        ----------
        factor : integer
        Factor
        '''
        assert not np.sometrue(np.mod( self.shape, factor ))
        # new size is an integer multiple of the original size
        newshape = self.shape/factor
        data = self.data.reshape(newshape,factor).sum(1) / factor
        getnoise = False
        var = None
        if self.var is not None:
            getnoise = True
            var = self.var.reshape(newshape,factor).sum(1) / factor / factor
        try:
            #crval = self.wave.coord()[slice(0,factor,1)].sum()/factor
            crval = self.wave.coord()[0:factor].sum()/factor
            wave = WaveCoord(1, self.wave.cdelt*factor, crval, self.wave.cunit)
        except:
            wave = None
        res = Spectrum(getnoise=getnoise, shape=newshape, wave = wave, unit=self.unit, fscale=self.fscale, empty=True)
        res.data = data
        if getnoise:
            res.var = var
        return res

        
    def rebin_factor(self, factor, margin='center'):
        '''shrinks the size of the spectrum by factor.
        
        Parameter
        ----------
        factor : integer
        Factor
        
        margin : 'center' or 'right' or 'left'
        This parameters is used if new size is not an integer multiple of the original size.
        'center' : two pixels added, on the left and on the right of the spectrum.
        'right': one pixel added on the right of the spectrum.
        'left': one pixel added on the left of the spectrum.
        '''
        if factor<=1 or factor>=self.shape:
            raise ValueError, 'factor must be in ]1,shape['
        #assert not np.sometrue(np.mod( self.shape, factor ))
        if not np.sometrue(np.mod( self.shape, factor )):
            # new size is an integer multiple of the original size
            return self._rebin_factor(factor)
        else:
            newshape = self.shape/factor
            n = self.shape - newshape*factor
            if margin == 'center' and n==1:
                margin = 'right'
            if margin == 'center':
                n_left = n/2
                n_right = self.shape - n + n_left
                spe = self[n_left:n_right]._rebin_factor(factor)
                newshape = spe.shape + 2
                data = np.ones(newshape)
                data[1:-1] = spe.data
                data[0] = self.data[0:n_left].sum() / factor
                data[-1] = self.data[n_right:].sum() / factor
                if self.var is not None:
                    getnoise = True
                    var = np.ones(newshape)
                    var[1:-1] = spe.var
                    var[0] = self.var[0:n_left].sum() / factor / factor
                    var[-1] = self.var[n_right:].sum() / factor / factor
                try:
                    crval = spe.wave.crval - spe.wave.cdelt
                    wave = WaveCoord(1, spe.wave.cdelt, crval, spe.wave.cunit)
                except:
                    wave = None
                res = Spectrum(getnoise=getnoise, shape=newshape, wave = wave, unit=self.unit, fscale=self.fscale, empty=True)
                res.data = np.ma.masked_invalid(data)
                if getnoise:
                    res.var = var
                return res
            elif margin == 'right':
                spe = self[0:self.shape-n]._rebin_factor(factor)
                newshape = spe.shape + 1
                data = np.ones(newshape)
                data[:-1] = spe.data
                data[-1] = self.data[self.shape-n:].sum() / factor
                getnoise = False
                if self.var is not None:
                    getnoise = True
                    var = np.ones(newshape)
                    var[:-1] = spe.var
                    var[-1] = self.var[self.shape-n:].sum() / factor / factor
                try:
                    wave = WaveCoord(1, spe.wave.cdelt, spe.wave.crval, spe.wave.cunit)
                except:
                    wave = None
                res = Spectrum(getnoise=getnoise, shape=newshape, wave = wave, unit=self.unit, fscale=self.fscale, empty=True)
                res.data = np.ma.masked_invalid(data)
                if getnoise:
                    res.var = var
                return res
            elif margin == 'left':
                spe = self[n:]._rebin_factor(factor)
                newshape = spe.shape + 1
                data = np.ones(newshape)
                data[0] = self.data[0:n].sum() / factor
                data[1:] = spe.data
                getnoise = False
                var = None
                if self.var is not None:
                    getnoise = True
                    var = np.ones(newshape)
                    var[0] = self.var[0:n].sum() / factor / factor
                    var[1:] = spe.var
                try:
                    crval = spe.wave.crval - spe.wave.cdelt
                    wave = WaveCoord(1, spe.wave.cdelt, crval, spe.wave.cunit)
                except:
                    wave = None
                res = Spectrum(getnoise=getnoise, shape=newshape, wave = wave, unit=self.unit, fscale=self.fscale, empty=True)
                res.data = np.ma.masked_invalid(data)
                if getnoise:
                    res.var = var
                return res
            else:
                raise ValueError, 'margin must be center|right|left'
            pass
    
    def rebin(self, step, start=None, shape= None, spline = False):
        """returns a spectrum with data rebinned to different wavelength step size.
        
        Parameters
        ----------
        step: float
        New pixel size in spectral direction
        
        start: float
        Spectral position of the first new pixel.
        It can be set or kept at the edge of the old first one.     
        
        shape : integer
        Size of the new spectrum.
        
        spline : boolean
        linear/spline interpolation to interpolate masked values
        """
        data = self._interp_data(spline)

        f = lambda x: data[int(self.wave.pixel(x)+0.5)]
        
        newwave = self.wave.rebin(step,start)
        if shape is None:
            newshape = newwave.shape   
        else:
            newshape = min(shape, newwave.shape)
            newwave.shape = newshape
            
        newdata = np.zeros(newshape)        
        pix = np.arange(newshape+1,dtype=np.float)
        x = (pix - newwave.crpix + 1) * newwave.cdelt + newwave.crval - 0.5 * newwave.cdelt
        lbdamax = (self.shape - self.wave.crpix ) * self.wave.cdelt + self.wave.crval + 0.5 * self.wave.cdelt
        if x[-1]> lbdamax:
            x[-1] = lbdamax
        
        for i in range(newshape):
            newdata[i] = integrate.quad(f,x[i],x[i+1],full_output=1)[0] / newwave.cdelt
            
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
        data = self._interp_data(spline)

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
        data = self._interp_data(spline)

        if weight:
            flux = (i2-i1)*np.average(data[i1:i2], weights=1.0/self.var[i1:i2])*self.fscale
        else:
            flux = data[i1:i2].sum()*self.fscale
            if self.var is not None:
                err = np.sqrt(self.var[i1:i2].sum())*self.fscale
                return flux,err
        return flux

    def poly_fit(self, deg, weight=True):
        """ performs polynomial fit on spectrum

        Parameters
        ----------
        deg : integer
        Degree of the fitting polynomial

        weight : boolean
        if weight is True, the weight is computed as the inverse of variance
        """
        if self.shape <= deg+1:
            raise ValueError, 'Too few points to perform polynomial fit'

        if self.var is None:
            weight = False

        if weight:
            vec_weight = 1./self.var
        else:
            vec_weight = None

        mask = np.array(1 - self.data.mask,dtype=bool)
        d = self.data.compress(mask) * self.fscale
        w = self.wave.coord().compress(mask)
        if weight:
            vec_weight = vec_weight.compress(mask)

        #p = np.polyfit(w, d, deg, w=vec_weight) #numpy 1.7  
        
        order = int(deg) + 1
        x = np.asarray(w) + 0.0
        y = np.asarray(d) + 0.0

        # check arguments.
        if deg < 0 :
            raise ValueError("expected deg >= 0")
        if x.ndim != 1:
            raise TypeError("expected 1D vector for x")
        if x.size == 0:
            raise TypeError("expected non-empty vector for x")
        if y.ndim < 1 or y.ndim > 2 :
            raise TypeError("expected 1D or 2D array for y")
        if x.shape[0] != y.shape[0] :
            raise TypeError("expected x and y to have same length")

        rcond = len(x)*np.finfo(x.dtype).eps

        # set up least squares equation for powers of x
        lhs = np.vander(x, order)
        rhs = y

        # apply weighting
        if vec_weight is not None:
            w = np.asarray(vec_weight) + 0.0
            if w.ndim != 1:
                raise TypeError, "expected a 1-d array for weights"
            if w.shape[0] != y.shape[0] :
                raise TypeError, "expected w and y to have the same length"
            lhs *= w[:, np.newaxis]
            if rhs.ndim == 2:
                rhs *= w[:, np.newaxis]
            else:
                rhs *= w

        # scale lhs to improve condition number and solve
        scale = np.sqrt((lhs*lhs).sum(axis=0))
        lhs /= scale
        c, resids, rank, s = np.linalg.lstsq(lhs, rhs, rcond)
        c = (c.T/scale).T # broadcast scale coefficients

        # warn on rank reduction, which indicates an ill conditioned matrix
        if rank != order and not full:
            msg = "Polyfit may be poorly conditioned"
            warnings.warn(msg, RankWarning)

        return c
    
    def poly_val(self, z):
        """returns a spectrum containing polynomial fit values, from polynomial coefficients.     
        
        Parameter
        ---------
        z : array_like
        The polynomial coefficients, in decreasing powers.
        """
        l = self.wave.coord()
        p = np.poly1d(z)
        data = p(l)
        res = Spectrum(shape=self.shape, wave = self.wave, unit=self.unit, data=data, fscale=1.0)
        return res
    
    def poly_spec(self, deg, weight=True):
        """ performs polynomial fit on spectrum and returns a spectrum containing polynomial fit values.

        Parameters
        ----------
        deg : integer
        Degree of the fitting polynomial

        weight : boolean
        if weight is True, the weight is computed as the inverse of variance
        """
        z = self.poly_fit(deg, weight)
        return self.poly_val(z)
        
        

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
        data = self._interp_data(spline)
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
        data = self._interp_data(spline)
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
        if i1==i2:
            raise ValueError, 'Minimum and maximum wavelengths are equal'
        return self.__getitem__(slice(i1,i2,1))

    def fwhm(self, l0, cont=0, spline=False):
        """ Returns the fwhm of a peak

        Parameters
        ----------
        l0 : float
        wavelength value corresponding to the peak position.

        cont : integer
        The continuum [default 0].
        
        spline : boolean
        linear/spline interpolation to interpolate masked values
        """
        k0 = self.wave.pixel(l0, nearest=True)
        d = self._interp_data(spline)*self.fscale - cont
        f2 = d[k0]/2
        k2 = np.argwhere(d[k0:]<f2)[0][0] + k0
        i2 = np.interp(f2, d[k2:k2-2:-1], [k2,k2-1])
        k1 = k0 - np.argwhere(d[k0::-1]<f2)[0][0]
        i1 = np.interp(f2, d[k1:k1+2], [k1,k1+1])
        fwhm = (i2 - i1)*self.wave.cdelt
        return fwhm

    def gauss_fit(self, lmin, lmax, lpeak=None, flux=None, fwhm=None, cont=None, peak=False, spline=False, plot=False):
        """performs polynomial fit on spectrum.
        Returns Gauss1D

        Parameters
        ----------
        lmin : float or (float,float)
        Minimum wavelength value or wavelength range used to compute the gaussian left value.

        lmax : float or (float,float)
        Maximum wavelength value or wavelength range used to compute the gaussian right value.
        
        lpeak : float
        input gaussian center, if None it is estimated with the wavelength corresponding to the maximum value in [max(lmin), min(lmax)]
        
        flux : float
        integrated gaussian flux or gaussian peak value if peak is True.
        If None peak value is estimated.

        fwhm : float
        input gaussian fwhm, if None it is estimated.

        cont : float
        continuum value, if None it is estimated by the line through points (max(lmin),mean(data[lmin])) and (min(lmax),mean(data[lmax]))
        
        peak : boolean
        If true, flux contains the gaussian peak value 
        
        spline : boolean
        linear/spline interpolation to interpolate masked values
        
        plot : boolean
        If True, the gaussian is plotted.
        """

        # truncate the spectrum and compute right and left gaussian values
        if isinstance(lmin,int) or isinstance(lmin,float):
                fmin = None
        else:
            lmin = np.array(lmin, dtype=float)
            fmin = self.mean(lmin[0],lmin[1])
            lmin = lmin[1]
            
        if isinstance(lmax,int) or isinstance(lmax,float):
                fmax = None
        else:
            lmax = np.array(lmax, dtype=float)
            fmax = self.mean(lmax[0],lmax[1])
            lmax = lmax[0]
            
        spec = self.truncate(lmin, lmax)
        data = spec._interp_data(spline)
        l = spec.wave.coord()
        d = data * self.fscale
        
        if fmin is None:
            fmin = d[0]
        if fmax is None:
            fmax = d[-1]
            
        # weigth           
        if spec.var is None:
            dw = d
        else:
            dw = d * 1./spec.var

        # initial gaussian peak value 
        if lpeak is None:
            lpeak = l[dw.argmax()]
            
        # continuum value 
        if cont is None:
            cont = ((fmax-fmin)*lpeak +lmax*fmin-lmin*fmax)/(lmax-lmin)
        
        # initial sigma value    
        if fwhm is None:
            fwhm = spec.fwhm(lpeak, cont, spline)
        sigma = fwhm/(2.*np.sqrt(2.*np.log(2.0)))
            
        # intitial gaussian integrated flux
        if flux is None:
            pixel = spec.wave.pixel(lpeak,nearest=True)
            peak = d[pixel] - cont
            flux = peak * np.sqrt(2*np.pi*(sigma**2))
        elif peak is True:
            peak = flux - cont
            flux = peak * np.sqrt(2*np.pi*(sigma**2))
        else:
            pass
        
        # 1d gaussian function
        if cont is None:
            gaussfit = lambda p, x: ((fmax-fmin)*x +lmax*fmin-lmin*fmax)/(lmax-lmin) + p[0]*(1/np.sqrt(2*np.pi*(p[2]**2)))*np.exp(-(x-p[1])**2/(2*p[2]**2))
        else:
            gaussfit = lambda p, x: cont + p[0]*(1/np.sqrt(2*np.pi*(p[2]**2)))*np.exp(-(x-p[1])**2/(2*p[2]**2))
        # 1d Gaussian fit  
        e_gauss_fit = lambda p, x, y: (gaussfit(p,x) -y)
        
        # inital guesses for Gaussian Fit
        v0 = [flux,lpeak,sigma]
        # Minimize the sum of squares
        v,covar,info, mesg, success  = leastsq(e_gauss_fit, v0[:], args=(l, d), maxfev=100000, full_output=1) #Gauss Fit
          
        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        err = np.array([np.sqrt(covar[i, i]) * np.sqrt(chisq / dof) for i in range(len(v))])
        
        #plot
        if plot:
            xxx = np.arange(min(l),max(l),l[1]-l[0])
            ccc = gaussfit(v,xxx)
            plt.plot(xxx,ccc,'r--')

        # return a Gauss1D object
        flux = v[0]
        err_flux = err[0]
        lpeak = v[1]
        err_lpeak = err[1]
        sigma = v[2]
        err_sigma = err[2]
        fwhm = sigma*2*np.sqrt(2*np.log(2))
        err_fwhm = err_sigma*2*np.sqrt(2*np.log(2))
        peak = flux/np.sqrt(2*np.pi*(sigma**2))
        err_peak = np.abs(1./np.sqrt(2*np.pi)*(err_flux*sigma-flux*err_sigma)/sigma/sigma)            
        return Gauss1D(lpeak, peak, flux, fwhm, cont, err_lpeak, err_peak, err_flux,err_fwhm)        
    

    def add_gaussian(self, lpeak, flux, fwhm, cont=0, peak=False ):
        """adds a gausian on spectrum.

        Parameters
        ----------
        
        lpeak : float
        gaussian center
        
        flux : float
        gaussian integrated flux or gaussian peak value
        
        fwhm : float
        gaussian fwhm
        
        cont : float
        continuum value. O by default.
        
        peak : boolean
        If true, flux contains the gaussian peak value
        """
        gauss = lambda p, x: cont + p[0]*(1/np.sqrt(2*np.pi*(p[2]**2)))*np.exp(-(x-p[1])**2/(2*p[2]**2)) #1d Gaussian func

        sigma = fwhm/(2.*np.sqrt(2.*np.log(2.0)))
        
        
        if peak is True:
            flux = flux * np.sqrt(2*np.pi*(sigma**2))

        lmin = lpeak - 5*sigma
        lmax = lpeak + 5*sigma
        imin = self.wave.pixel(lmin, True)
        imax = self.wave.pixel(lmax, True)
        if imin == imax:
            if imin==0 or imin==self.shape:
                raise ValueError, 'Gaussian outside spectrum wavelength range'

        wave  = self.wave.coord()[imin:imax]
        v = [flux,lpeak,sigma]

        res = self.copy()
        res.data[imin:imax] = self.data[imin:imax] + gauss(v,wave)/self.fscale

        return res
    
    def median_filter(self, kernel_size=None, pixel=True):
        """performs a median filter on the spectrum
        
        Parameters
        ----------
        
        kernel_size : float
        Size of the median filter window.
        
        pixel : booelan
        If True, kernel_size is in pixels.
        If False, kernel_size is in spectrum coordinate unit.

        """
        if pixel is False:
            kernel_size = kernel_size / self.get_step()
        ks = int(kernel_size/2)*2 +1
        data = signal.medfilt(self.data, ks)
        res = Spectrum(shape=self.shape, wave = self.wave, unit=self.unit, data=data, fscale=1.0)
        return res
    
    def convolve(self, other):
        """convolves self and other.
        
        Parameter
        ---------
        
        other : 1d-array or Spectrum
        second Spectrum or 1d-array
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is np.array:
            res = self.copy()
            res.data = signal.convolve(self.data ,other ,mode='same')
            return res
        try:
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    print 'Operation forbidden for spectra with different sizes'
                    return None
                else:
                    res = self.copy()
                    res.data = signal.convolve(self.data ,other.data ,mode='same')
                    res.fscale = self.fscale * other.fscale
                    return res
        except:
            print 'Operation forbidden'
            return None
        
    def fftconvolve(self, other):
        """convolves self and other using fft.
        
        Parameter
        ---------
        
        other : 1d-array or Spectrum
        second Spectrum or 1d-array
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is np.array:
            res = self.copy()
            res.data = signal.fftconvolve(self.data ,other ,mode='same')
            return res
        try:
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    print 'Operation forbidden for spectra with different sizes'
                    return None
                else:
                    res = self.copy()
                    res.data = signal.fftconvolve(self.data ,other.data ,mode='same')
                    res.fscale = self.fscale * other.fscale
                    return res
        except:
            print 'Operation forbidden'
            return None
        
    def correlate(self, other):
        """cross-correlates self and other.
        
        Parameter
        ---------
        
        other : 1d-array or Spectrum
        second Spectrum or 1d-array
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is np.array:
            res = self.copy()
            res.data = signal.correlate(self.data ,other ,mode='same')
            return res
        try:
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    print 'Operation forbidden for spectra with different sizes'
                    return None
                else:
                    res = self.copy()
                    res.data = signal.correlate(self.data ,other.data ,mode='same')
                    res.fscale = self.fscale * other.fscale
                    return res
        except:
            print 'Operation forbidden'
            return None
                
    def fftconvolve_gauss(self, fwhm, nsig=5):
        """convolves the spectrum with a Gaussian using fft.
        
        Parameters
        ----------
        
        fwhm : float
        Gaussian fwhm.
        
        nsig : integer
        Number of standard deviations.
        """
        sigma = fwhm/(2.*np.sqrt(2.*np.log(2.0)))
        s = sigma/self.get_step()
        n = nsig * int(s+0.5)
        n = int(n/2)*2
        d = np.arange(-n,n+1)
        kernel = special.erf((1+2*d)/(2*np.sqrt(2)*s)) + special.erf((1-2*d)/(2*np.sqrt(2)*s))
        kernel /= kernel.sum()
        
        res = self.copy()
        res.data = signal.correlate(self.data ,kernel ,mode='same')
        return res
    
#    def peak_detector(self, threshold, kernel_size=None):
#        d = np.abs(self.data - signal.medfilt(self.data, kernel_size))
#        ksel = np.where(d>threshold)
#        wave  = self.wave.coord()
#        return wave[ksel]
        
    
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
            
        plt.plot(x, f, drawstyle=drawstyle)
        if noise: 
            plt.fill_between(x, f + np.sqrt(res.var)*res.fscale, f -np.sqrt(res.var)*res.fscale, color='0.75', facecolor='0.75', alpha=0.5) 
        if title is not None:
                plt.title(title)   
        if res.wave.cunit is not None:
            plt.xlabel(r'$\lambda$ (%s)' %res.wave.cunit)
        if res.unit is not None:
            plt.ylabel(res.unit)
        plt.connect('motion_notify_event', self._on_move)
        self._plot_id = len(plt.gca().lines)-1
        
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
            
        plt.semilogy(x, f, drawstyle=drawstyle)
        if noise: 
            plt.fill_between(x, f + np.sqrt(res.var)*res.fscale, f -np.sqrt(res.var)*res.fscale, color='0.75', facecolor='0.75', alpha=0.5) 
        if title is not None:
                plt.title(title)   
        if res.wave.cunit is not None:
            plt.xlabel(r'$\lambda$ (%s)' %res.wave.cunit)
        if res.unit is not None:
            plt.ylabel(res.unit)
        plt.connect('motion_notify_event', self._on_move)
        self._plot_id = len(plt.gca().lines)-1
        
    def _on_move(self,event):
        """ prints x,y,i,lbda and data in the figure toolbar.
        """
        if event.inaxes is not None:
            xc, yc = event.xdata, event.ydata
            try:
                i = self.wave.pixel(xc, True)
                x = self.wave.coord(i)
                val = self.data.data[i]*self.fscale
                s = 'x= %g y=%g i=%d lbda=%g data=%g'%(xc,yc,i,x,val)
                plt.get_current_fig_manager().toolbar.set_message(s)
            except:
                pass
            
    def ipos(self, filename='None'):
        """Interactive mode.
        Prints cursor position.   
        To read cursor position, click on the left mouse button
        To remove a cursor position, click on the left mouse button + <r>
        To quit the interactive mode, click on the right mouse button.
        At the end, clicks are saved in self.clicks as dictionary {'xc','yc','x','y'}.
        
        Parameter
        ---------
        
        filename : string
        If filename is not None, the cursor values are saved as a fits table.
        """
        print 'To read cursor position, click on the left mouse button'
        print 'To remove a cursor position, click on the left mouse button + <r>'
        print 'To quit the interactive mode, click on the right mouse button.'
        print 'After quit, clicks are saved in self.clicks as dictionary {xc,yc,x,y}.'
        
        if self._clicks is None:
            binding_id = plt.connect('button_press_event', self._on_click)
            self._clicks = SpectrumClicks(binding_id,filename)
        else:
            self._clicks.filename = filename
        
    def _on_click(self,event):
        """ prints x,y,i,lbda and data corresponding to the cursor position.
        """
        if event.key == 'r':
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        xc, yc = event.xdata, event.ydata
                        self._clicks.remove(xc)
                        print "new selection:"
                        for i in range(len(self._clicks.xc)):
                            self._clicks.iprint(i,self.fscale)
                    except:
                        pass 
        else:
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        xc, yc = event.xdata, event.ydata
                        i = self.wave.pixel(xc, True)
                        print i
                        x = self.wave.coord(i)
                        val = self.data[i]*self.fscale
                        if len(self._clicks.x)==0:
                            print ''
                        self._clicks.add(xc,yc,i,x,val)
                        self._clicks.iprint(len(self._clicks.x)-1, self.fscale)
                    except:
                        pass
            else:
                self._clicks.write_fits()
                # save clicks in a dictionary {'xc','yc','x','y'}
                d = {'xc':self._clicks.xc, 'yc':self._clicks.yc, 'x':self._clicks.x, 'y':self._clicks.data}
                self.clicks = d
                #clear
                self._clicks.clear()
                self._clicks = None
                
            
    def idist(self):
        """Interactive mode.
        Gets distance and center from 2 cursor positions.
        """
        print 'Use 2 mouse clicks to get center and distance.'
        print 'To quit the interactive mode, click on the right mouse button.'
        if self._clicks is None:
            binding_id = plt.connect('button_press_event', self._on_click_dist)
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
                    self._clicks.add(xc,yc,i,x,val)
                    self._clicks.iprint(len(self._clicks.x)-1, self.fscale)
                    if np.sometrue(np.mod( len(self._clicks.x), 2 )) == False:
                        dx = abs(self._clicks.xc[-1] - self._clicks.xc[-2])
                        xc = (self._clicks.xc[-1] + self._clicks.xc[-2])/2
                        print 'Center: %f Distance: %f' % (xc,dx)
                except:
                    pass 
        else: 
            self._clicks.clear()
            self._clicks = None
            
    def igauss_fit(self,nclicks=5):
        """Interactive mode. Performs polynomial fit on spectrum.
        Use 3 or 5 mouse clicks to get minimim, peak and maximum wavelengths.
        To quit the interactive mode, click on the right mouse button.
        The parameters of the last gaussian are saved in self.gauss.
        
        Parameter
        ---------
        nclicks : 3 or 5
        Number of clicks.
        
        
        """
        if nclicks==3:
            print 'Use 3 mouse clicks to get minimim, peak and maximum wavelengths.'
            print 'To quit the interactive mode, click on the right mouse button.'
            print 'The parameters of the last gaussian are saved in self.gauss.'
            if self._clicks is None:
                binding_id = plt.connect('button_press_event', self._on_3clicks_gauss_fit)
                self._clicks = SpectrumClicks(binding_id)
        else:
            print 'Use the 2 first mouse clicks to get the wavelength range to compute the gaussian left value.'
            print 'Use the next click to get the peak wavelength.'
            print 'Use the 2 last mouse clicks to get the wavelength range to compute the gaussian rigth value.'
            print 'To quit the interactive mode, click on the right mouse button.'
            print 'The parameters of the last gaussian are saved in self.gauss.'
            if self._clicks is None:
                binding_id = plt.connect('button_press_event', self._on_5clicks_gauss_fit)
                self._clicks = SpectrumClicks(binding_id)
    
    def _on_3clicks_gauss_fit(self,event):
        """Performs polynomial fit on spectrum (interactive mode).
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
                    self._clicks.add(xc,yc,i,x,val)
                    if np.sometrue(np.mod( len(self._clicks.x), 3 )) == False:
                        lmin = self._clicks.xc[-3]
                        lpeak = self._clicks.xc[-2]
                        lmax = self._clicks.xc[-1]
                        self.gauss = self.gauss_fit(lmin, lmax, lpeak=lpeak, plot=True)
                        self.gauss.print_param()
                        self._clicks.id_lines.append(len(plt.gca().lines)-1)
                except:
                    pass
        else: 
            self._clicks.clear()
            self._clicks = None
            
    def _on_5clicks_gauss_fit(self,event):
        """Performs polynomial fit on spectrum (interactive mode).
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
                    self._clicks.add(xc,yc,i,x,val)
                    if np.sometrue(np.mod( len(self._clicks.x), 5 )) == False:
                        lmin1 = self._clicks.xc[-5]
                        lmin2 = self._clicks.xc[-4]
                        lpeak = self._clicks.xc[-3]
                        lmax1 = self._clicks.xc[-2]
                        lmax2 = self._clicks.xc[-1]
                        self.gauss = self.gauss_fit((lmin1,lmin2), (lmax1,lmax2), lpeak=lpeak, plot=True)
                        self.gauss.print_param()
                        self._clicks.id_lines.append(len(plt.gca().lines)-1)
                except:
                    pass
        else: 
            self._clicks.clear()
            self._clicks = None
            
    def imask(self):
        """Interactive mode.
        Plots masked values.
        """
        try:
            try:
                del plt.gca().lines[self._plot_mask_id]
            except:
                pass
            lbda = self.wave.coord()
            drawstyle = plt.gca().lines[self._plot_id].get_drawstyle()
            plt.plot(lbda,self.data.data,drawstyle=drawstyle, hold = True, alpha=0.3)
            self._plot_mask_id = len(plt.gca().lines)-1
        except:
            pass
            

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
                self.data = np.array(f[0].data, dtype=float)
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
                    d = np.array(f['DATA'].data, dtype=float)
                else:
                    h = f[ext].header
                    d = np.array(f[ext].data, dtype=float)
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
                    self.var = np.array(f['STAT'].data, dtype=float)
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
        #Mask an array where invalid values occur (NaNs or infs).
        if self.data is not None:
            self.data = np.ma.masked_invalid(self.data)

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

        if np.ma.count_masked(self.data) != 0:
            hdulist = [prihdu]
            # create spectrum DATA in first extension
            tbhdu = pyfits.ImageHDU(name='DATA', data=self.data.data)
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
                prihdu.data = self.data.data
                for card in wcs_cards:
                    prihdu.header.update(card.key, card.value, card.comment)
                if self.unit is not None:
                    prihdu.header.update('BUNIT', self.unit, 'data unit type')
                prihdu.header.update('FSCALE', self.fscale, 'Flux scaling factor')
                hdulist = [prihdu]
            else: # write fits file with primary header and two extensions
                hdulist = [prihdu]
                # create spectrum DATA in first extension
                tbhdu = pyfits.ImageHDU(name='DATA', data=self.data.data)
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
        hdu.info()
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
        print '%s (%s) fscale=%g, %s' %(data,unit,self.fscale,noise)
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

    def resize(self):
        """resize the image to have a minimum number of masked values
        """
        if self.data is not None:
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
        
    def get_step(self):
        """ returns the image steps [dDec,dRa]
        """
        step = np.zeros(2)
        step[0] = self.wcs.cdelt[1]
        step[1] = self.wcs.cdelt[0]
        return step

    def __setitem__(self,key,value):
        """ sets the corresponding part of data
        """
        self.data[key] = value
        
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

    def truncate(self,ra_min,ra_max,dec_min,dec_max, mask=True):
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
        
        mask : boolean
        if True, pixels outside [ra_min,ra_max] and [dec_min,dec_max] are masked
        """
        skycrd = [[ra_min,dec_min],[ra_min,dec_max],[ra_max,dec_min],[ra_max,dec_max]]
        pixcrd = self.wcs.sky2pix(skycrd)
        imin = int(np.min(pixcrd[:,0]))
        if imin<0:
            imin = 0
        imax = int(np.max(pixcrd[:,0]))+1
        jmin = int(np.min(pixcrd[:,1]))
        if jmin<0:
            jmin = 0
        jmax = int(np.max(pixcrd[:,1]))+1
        
        res = self[jmin:jmax,imin:imax]
        if mask:
            #mask outside pixels
            m = np.ma.make_mask_none(res.data.shape)
            for j in range(res.shape[0]):
                pixcrd = np.array([np.arange(res.shape[1]),np.ones(res.shape[1])*j]).T
                skycrd = self.wcs.pix2sky(pixcrd)
                test_ra_min = np.array(skycrd[:,0]) < ra_min
                test_ra_max = np.array(skycrd[:,0]) > ra_max
                test_dec_min = np.array(skycrd[:,1]) < dec_min
                test_dec_max = np.array(skycrd[:,1]) > dec_max
                m[j,:] = test_ra_min + test_ra_max + test_dec_min + test_dec_max
            try:
                m = np.ma.mask_or(m,np.ma.getmask(res.data))
            except:
                pass
            res.data = np.ma.MaskedArray(res.data, mask=m)
        return res
            
    def _interp(self, grid, spline=False):
        """ returns the interpolated values corresponding to the grid points
        
        Parameters
        ----------
        grid : 
        pixel values
        
        spline : boolean
        False: linear interpolation, True: spline interpolation 
        """
        ksel = np.where(self.data.mask==False)
        x = ksel[0] 
        y = ksel[1]
        data = self.data.data[ksel]
        npoints = np.shape(data)[0]
        
        grid = np.array(grid)
        n = np.shape(grid)[0]
        
        if spline:
            if self.var is not None:    
                weight = np.zeros(n,dtype=float)
                for i in range(npoints):
                    weight[i] = 1./self.var[x[i],y[i]]
            else:
                weight = None
                
            tck = interpolate.bisplrep(x,y,data,w=weight)
            res = np.zeros(n,dtype=float)
            for i in range(n):
                res[i] = interpolate.bisplev(grid[i,0],grid[i,1],tck)
            return res
        else:
            #scipy 0.9 griddata
            #interpolate.interp2d segfaults when there are too many data points
            #f = interpolate.interp2d(x, y, data)
            points = np.zeros((npoints,2),dtype=float)
            points[:,0] = ksel[0]
            points[:,1] = ksel[1]
            res = np.zeros(n,dtype=float)
            for i in range(n):
                res[i] = interpolate.griddata(points, data, (grid[i,0],grid[i,1]), method='linear')
            return res
        
#    def moments(self):
#        """Returns (width_x, width_y) first moments of the 2D gaussian
#        """
#        total = np.abs(self.data).sum()
#        Y, X = np.indices(self.data.shape) # python convention: reverse x,y numpy.indices
#        y = np.argmax((X*np.abs(self.data)).sum(axis=1)/total)
#        x = np.argmax((Y*np.abs(self.data)).sum(axis=0)/total)
#        col = self.data[int(y),:]
#        # FIRST moment, not second!
#        width_x = np.sqrt(np.abs((np.arange(col.size)-y)*col).sum()/np.abs(col).sum())*self.wcs.cdelt[0]
#        row = self.data[:, int(x)]
#        width_y = np.sqrt(np.abs((np.arange(row.size)-x)*row).sum()/np.abs(row).sum())*self.wcs.cdelt[1]
#        return [width_x,width_y]
#            
#    def gauss_fit(self, ra_min , ra_max, dec_min, dec_max, fpeak=None, ra_peak=None, dec_peak=None, ra_width=None, dec_width=None, cont=None, plot = False):
#        """
#        """
#        """performs polynomial fit on spectrum.
#        Returns [[fwhm,lpeak,fpeak],[err_fwhm,err_lpeak,err_fpeak]]
#
#        Parameters
#        ----------
#        lmin : float
#        Minimum wavelength.
#
#        lmax : float
#        Maximum wavelength.
#
#        fwhm : float
#        input gaussian fwhm, if None it is estimated.
#
#        lpeak : float
#        input gaussian center, if None it is estimated.
#
#        fpeak : float
#        input gaussian peak value, if None it is estimated.
#        
#        cont : float
#        continuum value, if None it is estimated.
#        
#        plot : boolean
#        If True, the gaussian is plotted.
#        """
#        
#        if cont is None: #pb mask
#            cont = np.ma.median(self.data.ravel())
#            print cont
#            
#        gaussfit = lambda p, x, y: cont + p[0]*(1/np.sqrt(2*np.pi*(p[2]**2)))*np.exp(-(x-p[1])**2/(2*p[2]**2)) \
#                                              *(1/np.sqrt(2*np.pi*(p[4]**2)))*np.exp(-(y-p[3])**2/(2*p[4]**2)) #2d Gaussian func
#        e_gauss_fit = lambda p, x, y, data: (gaussfit(p,x,y) - data)
#
#        ima = self.truncate(ra_min, ra_max, dec_min, dec_max, mask = False)
#        ima.plot()
#        
#        if ima.var is None:
#            weight = None
#        else:
#            weight = 1./ima.var
#            
#        ksel = np.where(ima.data.mask==False)
#        pixcrd = np.zeros((np.shape(ksel[0])[0],2))
#        pixcrd[:,0] = ksel[1] #ra
#        pixcrd[:,1] = ksel[0] #dec
#        pixsky = ima.wcs.pix2sky(pixcrd)
#        x = pixsky[:,0] #ra
#        y = pixsky[:,1] #dec        
#        data = ima.data.data[ksel]*self.fscale
#        
#        if weight is not None:
#            dw = data * weight
#        else:
#            dw = data
#        
#        imax = dw.argmax()
#        if ra_peak is None:
#            ra_peak = x[imax]
#        if dec_peak is None:
#            dec_peak = y[imax]
#        if fpeak is None:
#            fpeak = data[imax] - cont
#        width= ima.moments()
#        if ra_width is None:
#            ra_width = width[0]
#        if dec_width is None:
#            dec_width = width[1]
#        
#        v0 = [fpeak, ra_peak, ra_width, dec_peak, dec_width]
#
#        out = leastsq(e_gauss_fit, v0[:], args=(x,y,data), maxfev=100000, full_output=1) #Gauss Fit
#        v = out[0] #fit parammeters out
#        
#        if plot:
#            xx = np.arange(x[0],x[-1],(x[1]-x[0]))
#            yy = np.arange(y[0],y[-1],10000*(y[0]-y[1]))
#            ff = np.zeros((np.shape(yy)[0],np.shape(xx)[0]))
#            for i in range(np.shape(xx)[0]):
#                xxx = np.zeros(np.shape(yy)[0])
#                xxx[:] = xx[i]
#                ff[:,i] = gaussfit(v,xxx,yy)
#            plt.contour(xx, yy, ff, 10)
#            
#        covar = out[1] #covariance matrix output
#
#        if ima.shape.prod() > len(v) and covar is not None:
#            s_sq = (e_gauss_fit(v, x , y, data)**2).sum()/(ima.shape.prod()-len(v))
#            err = np.diag(covar * s_sq)
#        else:
#            err = np.zeros(len(v))
#
#        fpeak = v[0]
#        err_fpeak = err[0]
#        ra_peak = v[1]
#        err_ra_peak = err[1]
#        ra_width = v[2]
#        err_ra_width = err[2]
#        dec_peak = v[3]
#        err_dec_peak = err[3]
#        dec_width = v[4]
#        err_dec_width = err[4]
#        
#        print 'fpeak',fpeak,err_fpeak
#        print 'ra_peak',ra_peak,err_ra_peak
#        print 'dec_peak',dec_peak,err_dec_peak
#        print 'ra_width',ra_width,err_ra_width
#        print 'dec_width',dec_width,err_dec_width
#
##        return[[fwhm,lpeak,fpeak],[err_fwhm,err_lpeak,err_fpeak]]

            
#    def get_line(self, dec, aver=0):
#        """returns a line or an average of line from image
#        """
#        
#        pixsky = self.wcs.pix2sky(pixcrd)
#        if aver==0:
#            return self.__getitem__((200, slice(None, None, None))
        
    def plot(self, max=None, title=None, noise=False): 
        """ plots the image.
        
        Parameters
        ----------
        
        max : boolean
        If max, the plot is normalized to peak at max value.
        
        title : string
        Figure tiltle (None by default).
        
        noise : boolean
        If noise is True, the +/- standard deviation is overplotted.
              
        """
        plt.ion()
        
        f = self.data*self.fscale
        if max != None:
            f = f*max/f.max()
            
        if self.wcs is None:
            xaxis = np.arange(self.shape[1], dtype=np.float)
            yaxis = np.arange(self.shape[0], dtype=np.float)
            xunit = 'pixel'
            yunit = 'pixel'
        else:
            pixcrd = np.zeros((self.shape[1],2))
            pixcrd[:,0] = np.arange(self.shape[1], dtype=np.float)
            pixsky = self.wcs.pix2sky(pixcrd)
            xaxis = pixsky[:,0]
            xunit = self.wcs.wcs.wcs.cunit[1]
        
            pixcrd = np.zeros((self.shape[0],2))
            pixcrd[:,1] = np.arange(self.shape[0], dtype=np.float)
            pixsky = self.wcs.pix2sky(pixcrd)
            yaxis = pixsky[:,1]
            yunit = self.wcs.wcs.wcs.cunit[0]
        
#        self._fig = plt.figure()
#        self._manager = plt.get_current_fig_manager()
#        ax = self._fig.add_subplot(111)
        
        if np.shape(xaxis)[0] == 1:
            #plot a  column
            plt.plot(yaxis,f)
            plt.xlabel('dec (%s)' %yunit)
            plt.ylabel(self.unit)
        elif np.shape(yaxis)[0] == 1:
            #plot a line
            plt.plot(xaxis,f)
            plt.xlabel('ra (%s)' %xunit)
            plt.ylabel(self.unit)
        else:
            cax = plt.contourf(xaxis, yaxis, f, 100)
            plt.colorbar(cax)
            plt.xlabel('ra (%s)' %xunit)
            plt.ylabel('dec (%s)' %yunit)
            
        if title is not None:
                plt.title(title)   
        plt.connect('motion_notify_event', self._on_move)
        
    def _on_move(self,event):
        """ prints x,y,i,j and data in the figure toolbar.
        """
        if event.inaxes is not None:
            xc, yc = event.xdata, event.ydata
            try:
                pixcrd = self.wcs.sky2pix([xc,yc])
                i = pixcrd[0][1]
                j = pixcrd[0][0]
                val = self.data.data[i,j]*self.fscale
                s = 'ra= %g dec=%g j=%i i=%i data=%g'%(xc,yc,j,i,val)
                plt.get_current_fig_manager().toolbar.set_message(s)
            except:
                pass    

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
                self.data = np.array(f[0].data, dtype=float)
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
                cunit = hdr.get('CUNIT3','')
                self.wave = WaveCoord(crpix, cdelt, crval, cunit,self.shape[0])
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = np.array(f['DATA'].data, dtype=float)
                else:
                    h = f[ext].header
                    d = np.array(f[ext].data, dtype=float)
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
                cunit = h.get('CUNIT3','')
                self.wave = WaveCoord(crpix, cdelt, crval, cunit, self.shape[0])
                self.var = None
                if getnoise:
                    if f['STAT'].header['NAXIS'] != 3:
                        raise IOError, 'Wrong dimension number in STAT extension'
                    if f['STAT'].header['NAXIS1'] != self.shape[2] and f['STAT'].header['NAXIS2'] != self.shape[1] and f['STAT'].header['NAXIS3'] != self.shape[0]:
                        raise IOError, 'Number of points in STAT not equal to DATA'
                    self.var = np.array(f['STAT'].data, dtype=float)
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
        #Mask an array where invalid values occur (NaNs or infs).
        if self.data is not None:
            self.data = np.ma.masked_invalid(self.data)

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
            prihdu.data = self.data.data
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
            tbhdu = pyfits.ImageHDU(name='DATA', data=self.data.data)
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
        print '%s (%s) fscale=%g, %s' %(data,unit,self.fscale,noise)
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

    def resize(self):
        """resize the cube to have a minimum number of masked values
        """
        if self.data is not None:
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
            
    def get_step(self):
        """ returns the cube steps [dLambda,dDec,dRa]
        """
        step = np.zeros(3)
        step[0] = self.wave.cdelt
        step[1] = self.wcs.cdelt[1]
        step[2] = self.wcs.cdelt[0]
        return step
            
    def __setitem__(self,key,value):
        """ sets the corresponding part of data
        """
        self.data[key] = value
            
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
            
    def sum(self,axis=None):
        """ Returns the sum over the given axis.
        axis = None returns a float
        axis = 0 returns an image
        axis = (1,2) returns a spectrum
        Other cases return None.
        """
        if axis is None:
            return self.data.sum()    
        elif axis==0:
            #return an image
            data = self.data.sum(axis)
            getnoise = False
            var = None
            if self.var is not None:
                getnoise = True
                var = self.var.sum(axis)
            res = Image(getnoise=getnoise, shape=data.shape, wcs = self.wcs, unit=self.unit, empty=True,fscale=self.fscale)
            res.data = data
            if getnoise:
                res.var =var
            return res
        elif axis==tuple([1,2]):
            #return a spectrum
            data = self.data.sum(axis=1).sum(axis=1)
            getnoise = False
            var = None
            if self.var is not None:
                getnoise = True
                var = self.var.sum(axis=1).sum(axis=1)
            res = Spectrum(getnoise=getnoise, shape=data.shape[0], wave = self.wave, unit=self.unit, empty=True,fscale=self.fscale)
            res.data = data
            if getnoise:
                res.var =var
            return res
        else:
            return None