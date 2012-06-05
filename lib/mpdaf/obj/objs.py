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
from scipy import ndimage
from scipy import special

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.colors

import ABmag_filters
import plt_norm
import plt_zscale


def is_float(x):
    if type(x) is float or type(x) is np.float32 or type(x) is np.float64:
        return True
    else:
        return False
    
def is_int(x):
     if type(x) is int or type(x) is np.int32 or type(x) is np.int64:
         return True
     else:
        return False

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
            print 'xc=%g\tyc=%g\ti=%d\tx=%g\tdata=%g'%(self.xc[i],self.yc[i],self.i[i],self.x[i],self.data[i])
        else:
            print 'xc=%g\tyc=%g\ti=%d\tx=%g\tdata=%g\t[scaled=%g]'%(self.xc[i],self.yc[i],self.i[i],self.x[i],self.data[i],self.data[i]/fscale) 
           
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

    def __init__(self, filename=None, ext = None, notnoise=False, shape=101, wave = None, unit=None, data=None, var=None,fscale=1.0):
        """creates a Spectrum object

        Parameters
        ----------
        filename : string
        Possible FITS filename

        ext : integer or (integer,integer) or string or (string,string)
        Number/name of the data extension or numbers/names of the data and variance extensions.

        notnoise: boolean
        True if the noise Variance spectrum is not read (if it exists)
        Use notnoise=True to create spectrum without variance extension

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

        fscale : float
        Flux scaling factor (1 by default)

        Examples
        --------
        Spectrum(filename="toto.fits",ext=1,nonoise=False): spectrum from file (extension number is 1).

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
                self.cards = hdr.ascard
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
                    if is_int(ext) or isinstance(ext,str):
                        n = ext
                    else:
                        n = ext[0]
                    h = f[n].header
                    d = np.array(f[n].data, dtype=float)
                        
                if h['NAXIS'] != 1:
                    raise IOError, 'Wrong dimension number: not a spectrum'
                self.unit = h.get('BUNIT', None)
                self.cards = h.ascard
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
                if not notnoise:
                    try:
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
                    except:
                        self.var = None
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
                self.data = None
                self.shape = shape
            else:
                self.data = np.array(data, dtype = float)
                self.shape = data.shape[0]

            if notnoise or var is None:
                self.var = None
            else:
                self.var = np.array(var, dtype = float)
            self.fscale = np.float(fscale)
            try:
                self.wave = wave
                if wave is not None:
                    if wave.shape is not None and wave.shape != self.shape:
                        print "warning: wavelength coordinates and data have not the same dimensions."
                    self.wave.shape = self.shape
            except :
                self.wave = None
                print "error: wavelength solution not copied."
        #Mask an array where invalid values occur (NaNs or infs).
        if self.data is not None:
            self.data = np.ma.masked_invalid(self.data)

    def copy(self):
        """copies spectrum object in a new one and returns it
        """
        spe = Spectrum()
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

        if np.ma.count_masked(self.data) != 0:
            hdulist = [prihdu]
            # create spectrum DATA in first extension
            tbhdu = pyfits.ImageHDU(name='DATA', data=self.data.data)
            if self.cards is not None:
                for card in self.cards:
                    try:
                        tbhdu.header.update(card.key, card.value, card.comment)
                    except:
                        pass
            tbhdu.header.update('date', str(datetime.datetime.now()), 'creation date')
            tbhdu.header.update('author', 'MPDAF', 'origin of the file')
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
                if self.cards is not None:
                    for card in self.cards:
                        try:
                            prihdu.header.update(card.key, card.value, card.comment)
                        except:
                            pass
                prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
                prihdu.header.update('author', 'MPDAF', 'origin of the file')
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
                if self.cards is not None:
                    for card in self.cards:
                        try:
                            tbhdu.header.update(card.key, card.value, card.comment)
                        except:
                            pass
                tbhdu.header.update('date', str(datetime.datetime.now()), 'creation date')
                tbhdu.header.update('author', 'MPDAF', 'origin of the file')
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
            print 'No wavelength solution'
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
                data = self.data[item]
                shape = data.shape[0]
                if self.var is not None:
                    var = self.var[item]
                    try:
                        wave = self.wave[item]
                    except:
                        wave = None
                        print "error: wavelength solution not copied."
                res = Spectrum(shape=shape, wave = wave, unit=self.unit, fscale=self.fscale)
                res.data = data
                if self.var is not None:
                    res.var = var
                return res 
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
        if is_float(other) or is_int(other):
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
                    res = Spectrum(shape=self.shape,fscale=self.fscale)
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
        if is_float(other) or is_int(other):
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
                    res = Spectrum(shape=self.shape,fscale=self.fscale)
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
                        res = Cube(shape=other.shape , wcs= other.wcs, fscale=self.fscale)
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
        if is_float(other) or is_int(other):
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
        if is_float(other) or is_int(other):
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
                    res = Spectrum(shape=self.shape,fscale=self.fscale*other.fscale)
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
        if is_float(other) or is_int(other):
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
                    res = Spectrum(shape=self.shape,fscale=self.fscale/other.fscale)
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
                        res = Cube(shape=other.shape , wcs= other.wcs, fscale=self.fscale/other.fscale)
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
        if is_float(other) or is_int(other):
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
        if is_float(other) or is_int(other):
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
        if is_int(item):
            return self.data[item]
        elif isinstance(item, slice):
            data = self.data[item]
            shape = data.shape[0]
            var = None
            if self.var is not None:
                var = self.var[item]
            try:
                wave = self.wave[item]
            except:
                wave = None
            res = Spectrum(shape=shape, wave = wave, unit=self.unit, fscale=self.fscale)
            res.data = data
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
            lmax = lmin
        if self.wave is None:
            raise ValueError, 'Operation forbidden without world coordinates along the spectral direction'
        else:
            pix_min = max(0,int(self.wave.pixel(lmin)))
            pix_max = min(self.shape,int(self.wave.pixel(lmax)) + 1)
            if (pix_min+1)==pix_max:
                return self.data[pix_min]
            else:
                return self[pix_min:pix_max]
            
    def get_step(self):
        """returns the wavelength step
        """
        if self.wave is not None:
            return self.wave.get_step()
        else:
            return None
        
    def get_start(self):
        """returns the value of the first pixel.
        """
        if self.wave is not None:
            return self.wave.get_start()
        else:
            return None
    
    def get_end(self):
        """returns the value of the last pixel.
        """
        if self.wave is not None:
            return self.wave.get_end()
        else:
            return None
        
    def get_range(self):
        """returns the wavelength range [Lambda_min,Lambda_max]
        """
        if self.wave is not None:
            return self.wave.get_range()
        else:
            return None
            
    def __setitem__(self,key,value):
        """ sets the corresponding part of data
        """
        self.data[key] = value

    def set_wcs(self, wave):
        """sets the world coordinates

        Parameter
        ---------
        wave : WaveCoord
        Wavelength coordinates
        """
        if wave.shape is not None and wave.shape != self.shape:
            print "warning: wavelength coordinates and data have not the same dimensions."
        self.wave = wave
        self.wave.shape = self.shape
            
    def set_var(self,var=None):
        """sets the variance array
        
        Parameter
        ---------
        var : float array
        Input variance array. If None, variance is set with zeros
        """
        if var is None:
            self.var = np.zeros(self.shape)
        else:
            if self.shape == np.shape(var)[0]:
                self.var = var
            else:
                raise ValueError, 'var and data have not the same dimensions.'
            
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
                pix_min = self.wave.pixel(lmin,nearest=True)
            if lmax is None:
                pix_max = self.shape
            else:
                pix_max = self.wave.pixel(lmax,nearest=True)
            self.data[pix_min:pix_max+1] = np.ma.masked  
            
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
        var = None
        if self.var is not None:
            var = self.var.reshape(newshape,factor).sum(1) / factor / factor
        try:
            #crval = self.wave.coord()[slice(0,factor,1)].sum()/factor
            crval = self.wave.coord()[0:factor].sum()/factor
            wave = WaveCoord(1, self.wave.cdelt*factor, crval, self.wave.cunit)
        except:
            wave = None
        res = Spectrum(shape=newshape, wave = wave, unit=self.unit, fscale=self.fscale)
        res.data = data
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
                data = np.ma.empty(newshape)
                data[1:-1] = spe.data
                data[0] = self.data[0:n_left].sum() / factor
                data[-1] = self.data[n_right:].sum() / factor
                var = None
                if self.var is not None:
                    var = np.ones(newshape)
                    var[1:-1] = spe.var
                    var[0] = self.var[0:n_left].sum() / factor / factor
                    var[-1] = self.var[n_right:].sum() / factor / factor
                try:
                    crval = spe.wave.crval - spe.wave.cdelt
                    wave = WaveCoord(1, spe.wave.cdelt, crval, spe.wave.cunit)
                except:
                    wave = None
                res = Spectrum(shape=newshape, wave = wave, unit=self.unit, fscale=self.fscale)
                res.data = data
                res.var = var
                return res
            elif margin == 'right':
                spe = self[0:self.shape-n]._rebin_factor(factor)
                newshape = spe.shape + 1
                data = np.ma.empty(newshape)
                data[:-1] = spe.data
                data[-1] = self.data[self.shape-n:].sum() / factor
                var = None
                if self.var is not None:
                    var = np.ones(newshape)
                    var[:-1] = spe.var
                    var[-1] = self.var[self.shape-n:].sum() / factor / factor
                try:
                    wave = WaveCoord(1, spe.wave.cdelt, spe.wave.crval, spe.wave.cunit)
                except:
                    wave = None
                res = Spectrum(shape=newshape, wave = wave, unit=self.unit, fscale=self.fscale)
                res.data = data
                res.var = var
                return res
            elif margin == 'left':
                spe = self[n:]._rebin_factor(factor)
                newshape = spe.shape + 1
                data = np.ma.empty(newshape)
                data[0] = self.data[0:n].sum() / factor
                data[1:] = spe.data
                var = None
                if self.var is not None:
                    var = np.ones(newshape)
                    var[0] = self.var[0:n].sum() / factor / factor
                    var[1:] = spe.var
                try:
                    crval = spe.wave.crval - spe.wave.cdelt
                    wave = WaveCoord(1, spe.wave.cdelt, crval, spe.wave.cunit)
                except:
                    wave = None
                res = Spectrum(shape=newshape, wave = wave, unit=self.unit, fscale=self.fscale)
                res.data = data
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
            
        res = Spectrum(notnoise=True, shape=newshape, wave = newwave, unit=self.unit, data=newdata,fscale=self.fscale)
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

        spline : booleanself.wave.pixel(lbda+dlbda/2,nearest=True)]
        linear/spline interpolation to interpolate masked values
        """
        data = self._interp_data(spline)
        vflux = data[self.wave.pixel(lbda-dlbda/2,nearest=True):self.wave.pixel(lbda+dlbda/2,nearest=True)].mean()*self.fscale
        print vflux
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
            return self.abmag_band(3663, 650, out)
        elif name == 'B':
            return self.abmag_band(4361, 890, out)
        elif name == 'V':
            return self.abmag_band(5448, 840, out)
        elif name == 'Rc':
            return self.abmag_band(6410, 1600., out)
        elif name == 'Ic':
            return self.abmag_band(7980, 1500., out)
        elif name == 'z':
            return self.abmag_band(8930, 1470., out)
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
        lb = (np.arange(imin,imax) - self.wave.crpix + 1) * self.wave.cdelt + self.wave.crval
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
        if is_int(lmin) or is_float(lmin):
                fmin = None
        else:
            lmin = np.array(lmin, dtype=float)
            fmin = self.mean(lmin[0],lmin[1])
            lmin = lmin[1]
            
        if is_int(lmax) or is_float(lmax):
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

        # initial gaussian peak position
        if lpeak is None:
            lpeak = l[dw.argmax()]
            
        # continuum value 
        if cont is None:
            cont = ((fmax-fmin)*lpeak +lmax*fmin-lmin*fmax)/(lmax-lmin)
        
        # initial sigma value    
        if fwhm is None:
            fwhm = spec.fwhm(lpeak, cont, spline)
        sigma = fwhm/(2.*np.sqrt(2.*np.log(2.0)))
            
        # initial gaussian integrated flux
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
        self._fig = plt.get_current_fig_manager()
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
            
        self._fig = plt.get_current_fig_manager()
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
                self._fig.toolbar.set_message(s)
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
        print 'To remove a cursor position, click on the left mouse button + <d>'
        print 'To quit the interactive mode, click on the right mouse button.'
        print 'After quit, clicks are saved in self.clicks as dictionary {xc,yc,x,data}.'
        
        if self._clicks is None:
            binding_id = plt.connect('button_press_event', self._on_click)
            self._clicks = SpectrumClicks(binding_id,filename)
        else:
            self._clicks.filename = filename
        
    def _on_click(self,event):
        """ prints x,y,i,lbda and data corresponding to the cursor position.
        """
        if event.key == 'd':
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
                # save clicks in a dictionary {'xc','yc','x','data'}
                d = {'xc':self._clicks.xc, 'yc':self._clicks.yc, 'x':self._clicks.x, 'data':self._clicks.data}
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
        
        
class ImageClicks:
    """Object used to save click on image plot.
    
    Attributes
    ---------- 
    filename : string
    Name of the table fits file where are saved the clicks values.
    
    binding_id : integer
    Connection id.
    
    i : list of integer
    Nearest pixel of the cursor position along the y-axis.
    
    j : list of integer
    Nearest pixel of the cursor position along the x-axis.
    
    ra : list of float
    Corresponding nearest position along the x-axis (world coordinates)
    
    dec : list of float
    Corresponding nearest position along the y-axis (world coordinates)
    
    data : list of float
    Corresponding image data value.
    
    id_lines : list of integer
    Plot id (cross for cursor positions).
    """
    def __init__(self, binding_id, filename=None):
        self.filename = filename
        self.binding_id = binding_id
        self.i = []
        self.j = []
        self.ra = []
        self.dec = []
        self.data = []
        self.id_lines = []
        
    def remove(self,ic,jc):
        """removes a cursor position
        """
        d2 = (self.i-ic)*(self.i-ic) + (self.j-jc)*(self.j-jc)
        i = np.argmin(d2)
        line = self.id_lines[i]
        del plt.gca().lines[line]
        self.i.pop(i)
        self.j.pop(i)
        self.ra.pop(i)
        self.dec.pop(i)
        self.data.pop(i)
        self.id_lines.pop(i)
        for j in range(i,len(self.id_lines)):
            self.id_lines[j] -= 1
        plt.draw()
        
    def add(self,i,j,x,y,data):
        plt.plot(j,i,'r+')
        self.i.append(i)
        self.j.append(j)
        self.ra.append(x)
        self.dec.append(y)
        self.data.append(data)        
        self.id_lines.append(len(plt.gca().lines)-1)
        
    def iprint(self,i,fscale):
        """prints a cursor positions
        """
        if fscale == 1:
            print 'dec=%g\tra=%g\ti=%d\tj=%d\tdata=%g'%(self.dec[i],self.ra[i],self.i[i],self.j[i],self.data[i])
        else:
            print 'dec=%g\tra=%g\ti=%d\tj=%d\tdata=%g\t[scaled=%g]'%(self.dec[i],self.ra[i],self.i[i],self.j[i],self.data[i],self.data[i]/fscale)
           
    def write_fits(self): 
        """prints coordinates in fits table.
        """
        if self.filename != 'None':
            c1 = pyfits.Column(name='I', format='I', array=self.i)
            c2 = pyfits.Column(name='J', format='I', array=self.j)
            c3 = pyfits.Column(name='RA', format='E', array=self.ra)
            c4 = pyfits.Column(name='DEC', format='E', array=self.dec)
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
                    
class Gauss2D:
    """ Object used to saved 2d gaussian parameters
    
    Attributes
    ---------- 
    
    center : (float,float)
    Gaussian center (dec,ra).
    
    flux : float
    Gaussian integrated flux.
    
    width : (float,float)
    Spreads of the Gaussian blob (dec_width,ra_width).
    
    cont : float
    Continuum value.
    
    rot : float
    Rotation in degrees.
    
    peak : float
    Gaussian peak value.
    
    err_center : (float,float)
    Estimated error on Gaussian center.
        
    err_flux : float
    Estimated error on Gaussian integrated flux.
    
    err_width : (float,float)
    Estimated error on Gaussian width.
    
    err_rot : float
    Estimated error on rotation.
    
    err_peak : float
    Estimated error on Gaussian peak value.  
    """
    def __init__(self, center, flux, width, cont, rot, peak, err_center, err_flux, err_width, err_rot, err_peak):
        self.center = center
        self.flux = flux
        self.width = width
        self.cont = cont
        self.rot = rot
        self.peak = peak
        self.err_center = err_center
        self.err_flux = err_flux
        self.err_width = err_width
        self.err_rot = err_rot
        self.err_peak = err_peak
        
    def copy(self):
        res = Gauss2D(self.center, self.flux, self.width, self.cont, self.rot, self.peak, self.err_center, self.err_flux, self.err_width, self.err_rot, self.err_peak)
        return res
        
    def print_param(self):
        print 'Gaussian center = (%g,%g) (error:(%g,%g))' %(self.center[0],self.center[1],self.err_center[0],self.err_center[1])   
        print 'Gaussian integrated flux = %g (error:%g)' %(self.flux,self.err_flux)
        print 'Gaussian peak value = %g (error:%g)' %(self.peak,self.err_peak)
        print 'Gaussian width = (%g,%g) (error:(%g,%g))' %(self.width[0],self.width[1],self.err_width[0],self.err_width[1])
        print 'Rotation in degree: %g (error:%g)' %(self.rot, self.err_rot)
        print 'Gaussian continuum = %g' %self.cont
        print ''
        
        
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

    def __init__(self, filename=None, ext = None, notnoise=False, shape=(101,101), wcs = None, unit=None, data=None, var=None,fscale=1.0):
        """creates a Image object

        Parameters
        ----------
        filename : string
        Possible FITS filename

        ext : integer or (integer,integer) or string or (string,string)
        Number/name of the data extension or numbers/names of the data and variance extensions.

        notnoise: boolean
        True if the noise Variance image is not read (if it exists)
        Use notnoise=True to create image without variance extension

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

        fscale : float
        Flux scaling factor (1 by default)

        Examples
        --------
        Image(filename="toto.fits",ext=1): image from file (extension number is 1).

        wcs = WCS(crval=0,cdelt=0.2)
        Image(shape=300, wcs=wcs) : image 300x300 filled with zeros
        Image(wcs=wcs, data = MyData) : image 300x300 filled with MyData
        """

        self.image = True
        self._clicks = None
        self._selector = None
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
                self.cards = hdr.ascard
                self.shape = np.array([hdr['NAXIS2'],hdr['NAXIS1']])
                self.data = np.array(f[0].data, dtype=float)
                self.var = None
                self.fscale = hdr.get('FSCALE', 1.0)
                try:
                    self.wcs = WCS(hdr) # WCS object from data header
                except:
                    print "Error: Invalid wcs. World coordonates are not copied."
                    self.wcs = None
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = np.array(f['DATA'].data, dtype=float)
                else:
                    if is_int(ext) or isinstance(ext,str):
                        n = ext
                    else:
                        n = ext[0]
                    h = f[n].header
                    d = np.array(f[n].data, dtype=float)
                        
                if h['NAXIS'] != 2:
                    raise IOError, 'Wrong dimension number in DATA extension'
                self.unit = h.get('BUNIT', None)
                self.cards = h.ascard
                self.shape = np.array([h['NAXIS2'],h['NAXIS1']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                try:
                    self.wcs = WCS(h) # WCS object from data header
                except:
                    print "Error: Invalid wcs. World coordonates are not copied."
                    self.wcs = None
                self.var = None
                if not notnoise:
                    try:
                        if ext is None:
                            fstat = f['STAT']
                        else:
                            n = ext[1]
                            fstat = f[n]
                            
                        if fstat.header['NAXIS'] != 2:
                            raise IOError, 'Wrong dimension number in STAT extension'
                        if fstat.header['NAXIS1'] != self.shape[1] and fstat.header['NAXIS2'] != self.shape[0]:
                            raise IOError, 'Number of points in STAT not equal to DATA'
                        self.var = np.array(fstat.data, dtype=float)
                    except:
                        self.var = None
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
            if is_int(shape):
                shape = (shape,shape)
            if data is None:
                self.data = None
                self.shape = np.array(shape)
            else:
                self.data = np.array(data, dtype = float)
                try:
                    self.shape = np.array(data.shape)
                except:
                    self.shape = np.array(shape)

            if notnoise or var is None:
                self.var = None
            else:
                self.var = np.array(var, dtype = float)
            self.fscale = np.float(fscale)
            try:
                self.wcs = wcs
                if wcs is not None:
                    self.wcs.wcs.naxis1 = self.shape[1]
                    self.wcs.wcs.naxis2 = self.shape[0]
                    if wcs.wcs.naxis1!=0 and wcs.wcs.naxis2 !=0 and ( wcs.wcs.naxis1!=self.shape[1] or wcs.wcs.naxis2 != self.shape[0]):
                        print "warning: world coordinates and data have not the same dimensions."
            except :
                self.wcs = None
                print "error: wcs not copied."
        #Mask an array where invalid values occur (NaNs or infs).
        if self.data is not None:
            self.data = np.ma.masked_invalid(self.data)

    def copy(self):
        """copies Image object in a new one and returns it
        """
        ima = Image()
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

        #world coordinates
        wcs_cards = self.wcs.to_header().ascard

        if np.ma.count_masked(self.data) != 0:
            hdulist = [prihdu]
            # create spectrum DATA in first extension
            tbhdu = pyfits.ImageHDU(name='DATA', data=self.data.data)
            if self.cards is not None:
                for card in self.cards:
                    try:
                        tbhdu.header.update(card.key, card.value, card.comment)
                    except:
                        pass
            tbhdu.header.update('date', str(datetime.datetime.now()), 'creation date')
            tbhdu.header.update('author', 'MPDAF', 'origin of the file')
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
                if self.cards is not None:
                    for card in self.cards:
                        try:
                            prihdu.header.update(card.key, card.value, card.comment)
                        except:
                            pass
                prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
                prihdu.header.update('author', 'MPDAF', 'origin of the file')
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
                if self.cards is not None:
                    for card in self.cards:
                        try:
                            tbhdu.header.update(card.key, card.value, card.comment)
                        except:
                            pass
                tbhdu.header.update('date', str(datetime.datetime.now()), 'creation date')
                tbhdu.header.update('author', 'MPDAF', 'origin of the file')
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
                data = self.data[item]
                if is_int(item[0]):
                    shape = (1,data.shape[0])
                elif is_int(item[1]):
                    shape = (data.shape[0],1)
                else:
                    shape = (data.shape[0],data.shape[1])
                if self.var is not None:
                    var = self.var[item]
                try:
                    wcs = self.wcs[item[0],item[1]]
                except:
                    wcs = None
                    print "error: wcs not copied."
                res = Image(shape=shape, wcs = wcs, unit=self.unit, fscale=self.fscale)
                res.data = data
                if self.var is not None:
                    res.var = var
                return res
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
        if is_float(other) or is_int(other):
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
                    res = Image(shape=self.shape,fscale=self.fscale)
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
        if is_float(other) or is_int(other):
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
                    res = Image(shape=self.shape,fscale=self.fscale)
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
                        res = Cube(shape=other.shape , wave= other.wave, fscale=self.fscale)
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
        if is_float(other) or is_int(other):
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
        if is_float(other) or is_int(other):
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
                    res = Image(shape=self.shape,fscale=self.fscale * other.fscale)
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
                            res = Cube(shape=shape , wave= other.wave, wcs = self.wcs, fscale=self.fscale * other.fscale)
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
        if is_float(other) or is_int(other):
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
                    res = Image(shape=self.shape,fscale=self.fscale / other.fscale)
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
                        res = Cube(shape=other.shape , wave= other.wave, fscale=self.fscale / other.fscale)
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
        if is_float(other) or is_int(other):
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
        if is_float(other) or is_int(other):
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
                if is_int(item[0]):
                    shape = (1,data.shape[0])
                elif is_int(item[1]):
                    shape = (data.shape[0],1)
                else:
                    shape = (data.shape[0],data.shape[1])
                var = None
                if self.var is not None:
                    var = self.var[item]
                try:
                    wcs = self.wcs[item]
                except:
                    wcs = None
                res = Image(shape=shape, wcs = wcs, unit=self.unit, fscale=self.fscale)
                res.data = data
                res.var = var
                return res
        else:
            raise ValueError, 'Operation forbidden'
        
    def get_step(self):
        """ returns the image steps [dDec,dRa]
        """
        return self.wcs.get_step()
    
    def get_range(self):
        """returns [ [dec_min,ra_min], [dec_max,ra_max] ]
        """
        return self.wcs.get_range()
    
    def get_start(self):
        """returns [dec,ra] corresponding to pixel (0,0)
        """
        return self.wcs.get_start()
    
    def get_end(self):
        """returns [dec,ra] corresponding to pixel (-1,-1)
        """
        return self.wcs.get_end()
    
    def get_rot(self):
        """returns the rotation angle
        """
        return self.wcs.get_rot()

    def __setitem__(self,key,value):
        """ sets the corresponding part of data
        """
        self.data[key] = value
        
    def set_wcs(self, wcs):
        """sets the world coordinates

        Parameter
        ---------
        wcs : WCS
        World coordinates
        """
        self.wcs = wcs
        self.wcs.wcs.naxis1 = self.shape[1]
        self.wcs.wcs.naxis2 = self.shape[0]
        if wcs.wcs.naxis1!=0 and wcs.wcs.naxis2 !=0 and (wcs.wcs.naxis1 != self.shape[1] or wcs.wcs.naxis2 != self.shape[0]):
            print "warning: world coordinates and data have not the same dimensions."
            
    def set_var(self, var):
        """sets the variance array
        
        Parameter
        ---------
        var : float array
        Input variance array. If None, variance is set with zeros
        """
        if var is None:
            self.var = np.zeros((self.shape[0],self.shape[1]))
        else:
            if self.shape[0] == np.shape(var)[0] and self.shape[1] == np.shape(var)[1]:
                self.var = var
            else:
                raise ValueError, 'var and data have not the same dimensions.'
            
    def mask(self, center, radius, pix=False, inside=False):
        """Masks values inside/outside the described region.
        
        Parameters
        ----------
        
        center : (float,float)
        Center of the explored region.
        If pix is False, center = (dec, ra) is in degrees.
        If pix is True, center = (i,j) is in pixels.
        
        radius : float or (float,float)
        Radius defined the explored region.
        If radius is float, it defined a circular region.
        If radius is (float,float), it defined a rectangular region.
        If pix is False, radius = (ddec/2, dra/2) is in arcsecs.
        If pix is True, radius = (di,dj) is in pixels.
        
        pix : boolean
        If pix is False, center and radius are in degrees and arcsecs.
        If pix is True, center and radius are in pixels.
        
        inside : boolean
        If inside is True, pixels inside the described region are masked.
        If inside is False, pixels outside the described region are masked.
        """
        if is_int(radius) or is_float(radius):
            circular = True
            radius2 = radius*radius
            radius = (radius,radius)
        else:
            circular = False
                
        if pix:
            imin = center[0] - radius[0]
            imax = center[0] + radius[0]
            jmin = center[1] - radius[1]
            jmax = center[1] + radius[1]
            if inside and not circular:
                self.data.mask[imin:imax,jmin:jmax] = 1
            elif inside and circular:
                ni = int(imax-imin)
                nj = int(jmax-jmin)
                m = np.ma.make_mask_none((ni,nj))
                for i_in in range(ni):
                    i = i_in + imin                 
                    pixcrd = np.array([np.ones(nj)*i,np.arange(nj)+jmin]).T
                    pixcrd[:,0] -= center[0]
                    pixcrd[:,1] -= center[1]
                    m[i_in,:] = ((np.array(pixcrd[:,0])*np.array(pixcrd[:,0]) + np.array(pixcrd[:,1])*np.array(pixcrd[:,1])) < radius2)
                try:
                    m = np.ma.mask_or(m,np.ma.getmask(self.data)[imin:imax,jmin:jmax])
                    self.data.mask[imin:imax,jmin:jmax] = m
                except:
                    pass
            elif not inside and circular:
                self.data.mask[0:imin,:] = 1
                self.data.mask[imax:,:] = 1
                self.data.mask[imin:imax,0:jmin] = 1
                self.data.mask[imin:imax:,jmax:] = 1
                ni = int(imax-imin)
                nj = int(jmax-jmin)
                m = np.ma.make_mask_none((ni,nj))
                for i_in in range(ni):
                    i = i_in + imin                 
                    pixcrd = np.array([np.ones(nj)*i,np.arange(nj)+jmin]).T
                    pixcrd[:,0] -= center[0]
                    pixcrd[:,1] -= center[1]
                    m[i_in,:] = ((np.array(pixcrd[:,0])*np.array(pixcrd[:,0]) + np.array(pixcrd[:,1])*np.array(pixcrd[:,1])) > radius2)
                try:
                    m = np.ma.mask_or(m,np.ma.getmask(self.data)[imin:imax,jmin:jmax])
                    self.data.mask[imin:imax,jmin:jmax] = m
                except:
                    pass
            else:
                self.data.mask[0:imin,:] = 1
                self.data.mask[imax:,:] = 1
                self.data.mask[imin:imax,0:jmin] = 1
                self.data.mask[imin:imax:,jmax:] = 1
        else:
            dec_min = center[0] - radius[0]/3600.0
            dec_max = center[0] + radius[0]/3600.0
            ra_min = center[1] - radius[1]/3600.0
            ra_max = center[1] + radius[1]/3600.0
            skycrd = [ [dec_min,ra_min], [dec_min,ra_max], [dec_max,ra_min], [dec_max,ra_max] ] 
            pixcrd = self.wcs.sky2pix(skycrd) 
            
            jmin = int(np.min(pixcrd[:,1]))
            if jmin<0:
                jmin = 0
            jmax = int(np.max(pixcrd[:,1]))+1
            if jmax > self.shape[1]:
                jmax = self.shape[1]
            imin = int(np.min(pixcrd[:,0]))
            if imin<0:
                imin = 0
            imax = int(np.max(pixcrd[:,0]))+1
            if imax > self.shape[0]:
                imax = self.shape[0]
                
            if inside and not circular:
                ni = int(imax-imin)
                nj = int(jmax-jmin)
                m = np.ma.make_mask_none((ni,nj))
                for i_in in range(ni):
                    i = i_in + imin                 
                    pixcrd = np.array([np.ones(nj)*i,np.arange(nj)+jmin]).T
                    skycrd = self.wcs.pix2sky(pixcrd)
                    test_ra_min = np.array(skycrd[:,1]) > ra_min
                    test_ra_max = np.array(skycrd[:,1]) < ra_max
                    test_dec_min = np.array(skycrd[:,0]) > dec_min
                    test_dec_max = np.array(skycrd[:,0]) < dec_max
                    m[i_in,:] = test_ra_min + test_ra_max + test_dec_min + test_dec_max
                try:
                    m = np.ma.mask_or(m,np.ma.getmask(self.data)[imin:imax,jmin:jmax])
                    self.data.mask[imin:imax,jmin:jmax] = m
                except:
                    pass
            elif inside and circular:
                ni = int(imax-imin)
                nj = int(jmax-jmin)
                m = np.ma.make_mask_none((ni,nj))
                for i_in in range(ni):
                    i = i_in + imin                 
                    pixcrd = np.array([np.ones(nj)*i,np.arange(nj)+jmin]).T
                    pixsky = self.wcs.pix2sky(pixcrd)
                    pixsky[:,0] -= center[0]
                    pixsky[:,1] -= center[1]
                    m[i_in,:] = (np.array(pixsky[:,0])*np.array(pixsky[:,0]) + np.array(pixsky[:,1])*np.array(pixsky[:,1])) < radius2/3600.0/3600.0
                try:
                    m = np.ma.mask_or(m,np.ma.getmask(self.data)[imin:imax,jmin:jmax])
                    self.data.mask[imin:imax,jmin:jmax] = m
                except:
                    pass
            elif not inside and circular:
                self.data.mask[0:imin,:] = 1
                self.data.mask[imax:,:] = 1
                self.data.mask[imin:imax,0:jmin] = 1
                self.data.mask[imin:imax:,jmax:] = 1
                ni = int(imax-imin)
                nj = int(jmax-jmin)
                m = np.ma.make_mask_none((ni,nj))
                for i_in in range(ni):
                    i = i_in + imin                 
                    pixcrd = np.array([np.ones(nj)*i,np.arange(nj)+jmin]).T
                    pixsky= self.wcs.pix2sky(pixcrd)
                    pixsky[:,0] -= center[0]
                    pixsky[:,1] -= center[1]
                    m[i_in,:] = (np.array(pixsky[:,0])*np.array(pixsky[:,0]) + np.array(pixsky[:,1])*np.array(pixsky[:,1])) > radius2/3600.0/3600.0
                try:
                    m = np.ma.mask_or(m,np.ma.getmask(self.data)[imin:imax,jmin:jmax])
                    self.data.mask[imin:imax,jmin:jmax] = m
                except:
                    pass
            else:               
                self.data.mask[0:imin,:] = 1
                self.data.mask[imax:,:] = 1
                self.data.mask[imin:imax,0:jmin] = 1
                self.data.mask[imin:imax:,jmax:] = 1
                ni = int(imax-imin)
                nj = int(jmax-jmin)
                m = np.ma.make_mask_none((ni,nj))
                for i_in in range(ni):
                    i = i_in + imin                 
                    pixcrd = np.array([np.ones(nj)*i,np.arange(nj)+jmin]).T
                    skycrd = self.wcs.pix2sky(pixcrd)
                    test_ra_min = np.array(skycrd[:,1]) < ra_min
                    test_ra_max = np.array(skycrd[:,1]) > ra_max
                    test_dec_min = np.array(skycrd[:,0]) < dec_min
                    test_dec_max = np.array(skycrd[:,0]) > dec_max
                    m[i_in,:] = test_ra_min + test_ra_max + test_dec_min + test_dec_max
                try:
                    m = np.ma.mask_or(m,np.ma.getmask(self.data)[imin:imax,jmin:jmax])
                    self.data.mask[imin:imax,jmin:jmax] = m
                except:
                    pass
        
    def truncate(self, dec_min, dec_max, ra_min, ra_max, mask=True):
        """ returns the corresponding sub-image

        Parameters
        ----------
        dec_min : float
        minimum declination in degrees

        dec_max : float
        maximum declination in degrees 
        
        ra_min : float
        minimum right ascension in degrees

        ra_max : float
        maximum right ascension in degrees

        mask : boolean
        if True, pixels outside [dec_min,dec_max] and [ra_min,ra_max] are masked
        """
        skycrd = [[dec_min,ra_min],[dec_min,ra_max],[dec_max,ra_min],[dec_max,ra_max]]
        pixcrd = self.wcs.sky2pix(skycrd)
        
        imin = int(np.min(pixcrd[:,0]))
        if imin<0:
            imin = 0
        imax = int(np.max(pixcrd[:,0]))+1
        if imax>self.shape[0]:
            imax = self.shape[0]
        jmin = int(np.min(pixcrd[:,1]))
        if jmin<0:
            jmin = 0
        jmax = int(np.max(pixcrd[:,1]))+1
        if jmax>self.shape[1]:
            jmax=self.shape[1]
        
        res = self[imin:imax,jmin:jmax]
        
        if mask:
            #mask outside pixels
            m = np.ma.make_mask_none(res.data.shape)
            for j in range(res.shape[0]):
                #pixcrd = np.array([np.arange(res.shape[1]),np.ones(res.shape[1])*j]).T
                pixcrd = np.array([np.ones(res.shape[1])*j,np.arange(res.shape[1])]).T
                skycrd = self.wcs.pix2sky(pixcrd)
                test_ra_min = np.array(skycrd[:,1]) < ra_min
                test_ra_max = np.array(skycrd[:,1]) > ra_max
                test_dec_min = np.array(skycrd[:,0]) < dec_min
                test_dec_max = np.array(skycrd[:,0]) > dec_max
                m[j,:] = test_ra_min + test_ra_max + test_dec_min + test_dec_max
            try:
                m = np.ma.mask_or(m,np.ma.getmask(res.data))
            except:
                pass
            res.data = np.ma.MaskedArray(res.data, mask=m)
        return res
    
    def rotate_wcs(self, theta):
        """rotates WCS coordinates to new orientation given by theta
        
        Parameter
        ---------
        
        theta : float
        Rotation in degrees.
        """
        res = self.copy()
        res.wcs.rotate(theta)
        return res
    
    def rotate(self, theta):
        """rotates the image using spline interpolation
        
        Parameter
        ---------
        
        theta : float
        Rotation in degrees.
        """
        res = self.copy()
        mask = np.array(1 - self.data.mask,dtype=bool)
        mask_rot = ndimage.rotate(mask, theta, reshape=False, order=0)
        data_rot = ndimage.rotate(self.data.filled(0), theta, reshape=False)
        mask_ma = np.ma.make_mask(1-mask_rot)
        res.data = np.ma.array(data_rot, mask=mask_ma)
        return res         
    
    def sum(self,axis=None):
        """ Returns the sum over the given axis.
        
        Parameter
        ---------
        axis = None returns a float
        axis = 0  or 1 returns a line or a column
        Other cases return None.
        """
        if axis is None:
            return self.data.sum()    
        elif axis==0 or axis==1:
            #return an image
            data = self.data.sum(axis)
            var = None
            if self.var is not None:
                var = self.var.sum(axis)
            if axis==0:
                wcs = self.wcs[0,:]
                shape = (1,data.shape[0])
            else:
                wcs = self.wcs[:,0]
                shape = (data.shape[0],1)
            res = Image(shape=shape, wcs = wcs, unit=self.unit, fscale=self.fscale)
            res.data = data
            res.var =var
            return res
        else:
            return None
        
    def norm(self, type='flux', value=1.0):
        """ Normalizes total flux to value (default 1).
            
        Parameters
        ----------
        type : 'flux' or 'sum' or 'max'
        If 'flux',the flux is normalized and the pixel area is taken into account.
        If 'sum', the flux is normalized to the sum of flux independantly of pixel size.
        If 'max', the flux is normalized so that the maximum of intensity will be 'value'.    
        
        value : float
        Normalized value.    
        """
        if type == 'flux':
            norm = value/(self.get_step().prod()*self.fscale*self.data.sum())
        elif type == 'sum':
            norm = value/(self.fscale*self.data.sum())
        elif type == 'max':
            norm = value/(self.fscale*self.data.max())
        else:
            raise ValueError, 'Error in type: only flux,sum,max permitted'
        res = self.copy()
        res *= norm
        return res
    
    def peak(self, center=None, radius=0, pix = False, dpix=2, plot=False):
        """ Finds image peak location.
        Returns a dictionary {'ra', 'dec', 'j', 'i', 'data'} containing the peak position and the peak intensity.
        
        Parameters
        ----------
        
        center : (float,float)
        Center of the explored region.
        If pix is False, center = (dec, ra) is in degrees.
        If pix is True, center = (i,j) is in pixels.
        If center is None, the full image is explored.
        
        radius : (float,float)
        Radius defined the explored region.
        If pix is False, radius = (ddec/2, dra/2) is in arcsecs.
        If pix is True, radius = (di,dj) is in pixels.
        
        pix : boolean
        If pix is False, center and radius are in degrees and arcsecs.
        If pix is True, center and radius are in pixels.
        
        dpix : integer
        Half size of the window to compute the center of gravity.
        
        plot : boolean
        If True, the peak center is overplotted on the image.
        """
        if center is None or radius==0:
            d = self.data
            imin = 0
            jmin = 0
        else:
            if is_int(radius) or is_float(radius):
                radius = (radius,radius)
            if pix:
                imin = center[0] - radius[0]
                imax = center[0] + radius[0] + 1
                jmin = center[1] - radius[1]
                jmax = center[1] + radius[1] + 1
            else:
                dec_min = center[0] - radius[0]/3600.0
                dec_max = center[0] + radius[0]/3600.0
                ra_min = center[1] - radius[1]/3600.0
                ra_max = center[1] + radius[1]/3600.0
                skycrd = [ [dec_min,ra_min], [dec_min,ra_max], [dec_max,ra_min], [dec_max,ra_max] ]
                pixcrd = self.wcs.sky2pix(skycrd)
                jmin = int(np.min(pixcrd[:,1]))
                if jmin<0:
                    jmin = 0
                jmax = int(np.max(pixcrd[:,1]))+1
                imin = int(np.min(pixcrd[:,0]))
                if imin<0:
                    imin = 0
                imax = int(np.max(pixcrd[:,0]))+1
            d = self.data[imin:imax,jmin:jmax]
            if np.shape(d)[0]==0 or np.shape(d)[1]==0:
                raise ValueError, 'Coord area outside image limits'
            
        ic,jc = ndimage.measurements.maximum_position(d)
        if dpix == 0:
            di = 0
            dj = 0
        else:
            if ic-dpix<0 or ic+dpix>self.shape[0] or jc-dpix<0 or jc+dpix>self.shape[1]:
                raise ValueError, 'Cannot compute center of mass, peak at the edges of the image'
            di,dj = ndimage.measurements.center_of_mass(d[ic-dpix:ic+dpix+1,jc-dpix:jc+dpix+1])
        ic = imin+ic-dpix+di
        jc = jmin+jc-dpix+dj
        [[dec,ra]] = self.wcs.pix2sky([[ic,jc]])
        maxv = self.fscale*self.data[int(round(ic)), int(round(jc))]
        if plot:
            plt.plot(jc,ic,'r+')
        return {'ra':ra, 'dec':dec, 'j':jc, 'i':ic, 'data': maxv}
    
    def fwhm(self, center=None, radius=0, pix = False):
        """ Computes the fwhm center. Returns [fwhm_dec,fwhm_ra].
        
        center : (float,float)
        Center of the explored region.
        If pix is False, center = (dec, ra) is in degrees.
        If pix is True, center = (i,j) is in pixels.
        If center is None, the full image is explored.
        
        radius : (float,float)
        Radius defined the explored region.
        If pix is False, radius = (ddec/2, dra/2) is in arcsecs.
        If pix is True, radius = (di,dj) is in pixels.
        
        pix : boolean
        If pix is False, center and radius are in degrees and arcsecs.
        If pix is True, center and radius are in pixels.
        """
        if center is None or radius==0:
            sigma = self.moments()
        else:
            if is_int(radius) or is_float(radius):
                radius = (radius,radius)
            if pix:
                imin = center[0] - radius[0]
                imax = center[0] + radius[0] + 1
                jmin = center[1] - radius[1]
                jmax = center[1] + radius[1] + 1
            else:
                dec_min = center[0] - radius[0]/3600.0
                dec_max = center[0] + radius[0]/3600.0
                ra_min = center[1] - radius[1]/3600.0
                ra_max = center[1] + radius[1]/3600.0
                skycrd = [ [dec_min,ra_min], [dec_min,ra_max], [dec_max,ra_min], [dec_max,ra_max] ]    
                pixcrd = self.wcs.sky2pix(skycrd)
                jmin = int(np.min(pixcrd[:,1]))
                if jmin<0:
                    jmin = 0
                jmax = int(np.max(pixcrd[:,1]))+1
                imin = int(np.min(pixcrd[:,0]))
                if imin<0:
                    imin = 0
                imax = int(np.max(pixcrd[:,0]))+1
            sigma = self[imin:imax,jmin:jmax].moments()
        fwhmx = sigma[0]*2.*np.sqrt(2.*np.log(2.0))
        fwhmy = sigma[1]*2.*np.sqrt(2.*np.log(2.0))
        return [fwhmx,fwhmy]
    
    def ee(self, center=None, radius=0, pix = False, frac = False):
        """ Computes ensquared energy.
        
        center : (float,float)
        Center of the explored region.
        If pix is False, center = (dec, ra) is in degrees.
        If pix is True, center = (i,j) is in pixels.
        If center is None, the full image is explored.
        
        radius : float or (float,float)
        Radius defined the explored region.
        If radius is float, it defined a circular region.
        If radius is (float,float), it defined a rectangular region.
        If pix is False, radius = (ddec/2, dra/2) is in arcsecs.
        If pix is True, radius = (di,dj) is in pixels.
        
        pix : boolean
        If pix is False, center and radius are in degrees and arcsecs.
        If pix is True, center and radius are in pixels.
        
        frac : boolean
        If frac is True, result is given relative to the total energy.
        """
        if center is None or radius==0:
            if frac:
                return 1.
            else:
                return self.data.sum()*self.fscale
        else:
            if is_int(radius) or is_float(radius):
                circular = True
                radius2 = radius*radius
                radius = (radius,radius)
            else:
                circular = False
                
            if pix:
                imin = center[0] - radius[0]
                imax = center[0] + radius[0]
                jmin = center[1] - radius[1]
                jmax = center[1] + radius[1]
            else:
                dec_min = center[0] - radius[0]/3600.0
                dec_max = center[0] + radius[0]/3600.0
                ra_min = center[1] - radius[1]/3600.0
                ra_max = center[1] + radius[1]/3600.0
                skycrd = [ [dec_min,ra_min], [dec_min,ra_max], [dec_max,ra_min], [dec_max,ra_max] ] 
                pixcrd = self.wcs.sky2pix(skycrd)
                jmin = int(np.min(pixcrd[:,1]))
                if jmin<0:
                    jmin = 0
                jmax = int(np.max(pixcrd[:,1]))+1
                imin = int(np.min(pixcrd[:,0]))
                if imin<0:
                    imin = 0
                imax = int(np.max(pixcrd[:,0]))+1
            ima = self[imin:imax,jmin:jmax]
            if circular:
                if pix:
                    xaxis = np.arange(ima.shape[0], dtype=np.float) - ima.shape[0]/2.
                    yaxis = np.arange(ima.shape[1], dtype=np.float) - ima.shape[1]/2.
                else:
                    step = self.get_step()
                    xaxis = (np.arange(ima.shape[0], dtype=np.float) - ima.shape[0]/2.) * step[0] * 3600.0
                    yaxis = (np.arange(ima.shape[1], dtype=np.float) - ima.shape[1]/2.) * step[1] * 3600.0
                gridx = np.zeros(ima.shape, dtype=np.float)
                gridy = np.zeros(ima.shape, dtype=np.float)
                for j in range(ima.shape[1]):
                    gridx[:,j] = xaxis
                for i in range(ima.shape[0]):
                    gridy[i,:] = yaxis
                r2 = gridx * gridx + gridy * gridy
                ksel = np.where(r2 < radius2)
                if frac:
                    return ima.data[ksel].sum()/self.data.sum()
                else:
                    return ima.data[ksel].sum()*self.fscale
            else:
                if frac:
                    return ima.data.sum()/self.data.sum()
                else:
                    return ima.data.sum()*self.fscale

    def ee_curve(self, center=None, pix = False, etot = None):
        """Returns Spectrum object containing enclosed energy as function of radius.
        
        center : (float,float)
        Center of the explored region.
        If pix is False, center = (dec, ra) is in degrees.
        If pix is True, center = (i,j) is in pixels.
        If center is None, the full image is explored.
        
        pix : boolean
        If pix is False, center is in degrees.
        If pix is True, center is in pixels.
        
        etot : float
        Total energy.
        If etot is not set it is computed from the full image.
        """
        if center is None:
            i = self.shape[0]/2
            j = self.shape[1]/2
        else:
            if pix:
                i = center[0]
                j = center[1]
            else:  
                pixcrd = self.wcs.sky2pix([[center[0],center[1]]])
                i = int(pixcrd[0][0]+0.5)
                j = int(pixcrd[0][1]+0.5)
        nmax = min(self.shape[0]-i,self.shape[1]-j,i,j)
        if etot is None:
            etot = self.fscale*self.data.sum()
        step = self.get_step()
        if nmax <= 1:
            raise ValueError, 'Coord area outside image limits'
        ee = np.zeros(nmax)
        for d in range(0, nmax):
            ee[d] = self.fscale*self.data[i-d:i+d+1, j-d:j+d+1].sum()/etot
        plt.plot(rad,ee)
        wave = WaveCoord(cdelt=np.sqrt(step[0]**2+step[1]**2), crval=0.0, cunit = '')
        return Spectrum(wave=wave, data = ee)
        
    def ee_size(self, center=None, pix = False, ee = None, frac = 0.90):
        """Computes the size of the square center on (dec,ra) containing the fraction of the energy.
        
        center : (float,float)
        Center of the explored region.
        If pix is False, center = (dec, ra) is in degrees.
        If pix is True, center = (i,j) is in pixels.
        If center is None, the full image is explored.
        
        pix : boolean
        If pix is False, center is in degrees.
        If pix is True, center is in pixels.
        
        ee : float
        Enclosed energy.
        If ee is not set it is computed from the full image that contain the fraction (frac) of the total energy
        
        frac : float
        Fraction of energy.
        """
        if center is None:
            i = self.shape[0]/2
            j = self.shape[1]/2
        else:
            if pix:
                i = center[0]
                j = center[1]
            else:  
                pixcrd = self.wcs.sky2pix([[center[0],center[1]]])
                i = int(pixcrd[0][0]+0.5)
                j = int(pixcrd[0][1]+0.5)
        nmax = min(self.shape[0]-i,self.shape[1]-j,i,j)
        if ee is None:
            ee = self.fscale*self.data.sum()
        step = self.get_step()
        
        if nmax <= 1:
            return step[0], step[1], 0, 0
        for d in range(1, nmax):
            ee2 = self.fscale*self.data[i-d:i+d+1, j-d:j+d+1].sum()/ee
            if ee2 > frac:
                break;
        d -= 1
        ee1 = self.fscale*self.data[i-d:i+d+1, i-d:i+d+1].sum()/ee
        d += (frac-ee1)/(ee2-ee1) # interpolate
        dx = d*step[0]*2
        dy = d*step[1]*2
        return [dx,dy]
                
    
       
            
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
            res = interpolate.bisplev(grid[:,0],grid[:,1],tck)
#            res = np.zeros(n,dtype=float)
#            for i in range(n):
#                res[i] = interpolate.bisplev(grid[i,0],grid[i,1],tck)
            return res
        else:
            #scipy 0.9 griddata
            #interpolate.interp2d segfaults when there are too many data points
            #f = interpolate.interp2d(x, y, data)
            points = np.zeros((npoints,2),dtype=float)
            points[:,0] = ksel[0]
            points[:,1] = ksel[1]
            res = interpolate.griddata(points, data, (grid[:,0],grid[:,1]), method='linear')
#            res = np.zeros(n,dtype=float)
#            for i in range(n):
#                res[i] = interpolate.griddata(points, data, (grid[i,0],grid[i,1]), method='linear')
            return res
        
    def moments(self):
        """Returns [width_dec, width_ra] first moments of the 2D gaussian
        """
        total = np.abs(self.data).sum()
        Y, X = np.indices(self.data.shape) # python convention: reverse x,y np.indices
        y = np.argmax((X*np.abs(self.data)).sum(axis=1)/total)
        x = np.argmax((Y*np.abs(self.data)).sum(axis=0)/total)
        col = self.data[int(y),:]
        # FIRST moment, not second!
        cdelt = self.wcs.get_step()
        width_x = np.sqrt(np.abs((np.arange(col.size)-y)*col).sum()/np.abs(col).sum())*cdelt[1]
        row = self.data[:, int(x)]
        width_y = np.sqrt(np.abs((np.arange(row.size)-x)*row).sum()/np.abs(row).sum())*cdelt[0]
        return [width_y,width_x]
        
    def gauss_fit(self, pos_min, pos_max, center=None, flux=None, width=None, cont=None, rot = 0, peak = False, factor = 1, plot = False):
        """performs gaussian fit on image.
        Returns Gauss2D object

        Parameters
        ----------
        
        pos_min : (float,float)
        Minimum declination and right ascension in degrees (dec_min, ra_min)

        pos_max : (float,float)
        Maximum declination and right ascension in degrees (dec_max,ra_max)
        
        center : (float,float)
        Initial gaussian center (dec_peak, ra_peak). If None they are estimated.
        
        flux : float
        Initial integrated gaussian flux or gaussian peak value if peak is True.
        If None peak value is estimated.

        width : (float,float)
        Initial spreads of the Gaussian blob (dec_width, ra_width). If None, they are estimated.

        cont : float
        Initial continuum value, if None it is estimated.
        
        rot : float
        Initial rotation in degree.
        
        peak : boolean
        If true, flux contains a gaussian peak value.
        
        factor : integer
        If factor<=1, gaussian is computed in the center of each pixel.
        If factor>1, for each pixel, gaussian value is the sum of the gaussian values on the factor*factor pixels divided by the pixel area.
        
        plot : boolean
        If True, the gaussian is plotted.
        """
        ra_min = pos_min[1]
        dec_min = pos_min[0]
        ra_max = pos_max[1]
        dec_max = pos_max[0]
        ima = self.truncate(dec_min, dec_max, ra_min, ra_max, mask = False)
        
        ksel = np.where(ima.data.mask==False)
        pixcrd = np.zeros((np.shape(ksel[0])[0],2))
        pixcrd[:,1] = ksel[1] #ra
        pixcrd[:,0] = ksel[0] #dec
        pixsky = ima.wcs.pix2sky(pixcrd)
        x = pixsky[:,1] #ra
        y = pixsky[:,0] #dec  
        data = ima.data.data[ksel]*self.fscale
        
        #weight
        if ima.var is None:
            dw = data
        else:
            dw = data / ima.var[ksel]
        
        # initial gaussian peak position
        if center is None:
            imax = dw.argmax()
            ra_peak = x[imax]
            dec_peak = y[imax]
        else:
            ra_peak = center[1]
            dec_peak = center[0]
            
        # continuum value 
        if cont is None:
            imin = dw.argmin()
            cont =  max(0,data[imin])
            
        # initial width value    
        if width is None:
            width= ima.moments()
        ra_width = width[1]
        dec_width = width[0]
        
        # initial gaussian integrated flux
        if flux is None:
            pixsky = [[dec_peak,ra_peak]]
            pixcrd = ima.wcs.sky2pix(pixsky)
            peak = ima.data.data[pixcrd[0,0],pixcrd[0,1]] - cont
            flux = peak * np.sqrt(2*np.pi*(ra_width**2)) * np.sqrt(2*np.pi*(dec_width**2))
        elif peak is True:
            peak = flux - cont
            flux = peak * np.sqrt(2*np.pi*(ra_width**2)) * np.sqrt(2*np.pi*(dec_width**2))
        else:
            pass
        
        #rotation angle in rad
        theta = np.pi * rot / 180.0
        
        
        if factor > 1:
            gaussfit = lambda p: gauss_image(ima.shape, ima.wcs, (p[3],p[1]), p[0], (p[4],p[2]), False, p[5], factor).data.data[ksel]*ima.fscale + cont
            e_gauss_fit = lambda p, data: (gaussfit(p) - data)
            v0 = [flux,ra_peak, ra_width, dec_peak, dec_width,rot]
            v,covar,info, mesg, success  = leastsq(e_gauss_fit, v0[:], args=(data), maxfev=100000, full_output=1)           
        else:                                             
            # 2d gaussian functio
            gaussfit = lambda p, x, y: cont + p[0]*(1/np.sqrt(2*np.pi*(p[2]**2)))*np.exp(-((x-p[1])*np.cos(p[5])-(y-p[3])*np.sin(p[5]))**2/(2*p[2]**2)) \
                                                  *(1/np.sqrt(2*np.pi*(p[4]**2)))*np.exp(-((x-p[1])*np.sin(p[5])+(y-p[3])*np.cos(p[5]))**2/(2*p[4]**2)) 
            # 2d Gaussian fit  
            e_gauss_fit = lambda p, x, y, data: (gaussfit(p,x,y) - data)
                
            # inital guesses for Gaussian Fit
            v0 = [flux,ra_peak, ra_width, dec_peak, dec_width,rot]
            # Minimize the sum of squares
            v,covar,info, mesg, success  = leastsq(e_gauss_fit, v0[:], args=(x,y,data), maxfev=100000, full_output=1)
    
        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        err = np.array([np.sqrt(covar[i, i]) * np.sqrt(chisq / dof) for i in range(len(v))])
        
        # plot
        if plot:
            gaussfit = lambda p, x, y: cont + p[0]*(1/np.sqrt(2*np.pi*(p[2]**2)))*np.exp(-((x-p[1])*np.cos(p[5])-(y-p[3])*np.sin(p[5]))**2/(2*p[2]**2)) \
                                                  *(1/np.sqrt(2*np.pi*(p[4]**2)))*np.exp(-((x-p[1])*np.sin(p[5])+(y-p[3])*np.cos(p[5]))**2/(2*p[4]**2)) 
            
            xmin = np.min(x)
            xmax = np.max(x)
            xx = np.arange(xmin,xmax,(xmax-xmin)/100)
            ymin = np.min(y)
            ymax = np.max(y)
            yy = np.arange(ymin,ymax,(ymax-ymin)/100)
            
            ff = np.zeros((np.shape(yy)[0],np.shape(xx)[0]))
            for i in range(np.shape(xx)[0]):
                xxx = np.zeros(np.shape(yy)[0])
                xxx[:] = xx[i]
                ff[:,i] = gaussfit(v,xxx,yy)
            
            pixsky = [[ymin,xmin],[ymax,xmax]]
            [[ymin,xmin],[ymax,xmax]] = self.wcs.sky2pix(pixsky)
            xx = np.arange(xmin,xmax,(xmax-xmin)/np.shape(xx)[0])
            yy = np.arange(ymin,ymax,(ymax-ymin)/np.shape(yy)[0])
            plt.contour(xx, yy, ff, 5)
            
        # return a Gauss2D object
        flux = v[0]
        err_flux = err[0]
        ra_peak = v[1]
        err_ra_peak = err[1]
        ra_width = v[2]
        err_ra_width = err[2]
        dec_peak = v[3]
        err_dec_peak = err[3]
        dec_width = v[4]
        err_dec_width = err[4]
        rot = (v[5] * 180.0 / np.pi)%180
        err_rot = err[5] * 180.0 / np.pi
        peak = flux / np.sqrt(2*np.pi*(ra_width**2)) / np.sqrt(2*np.pi*(dec_width**2))
        err_peak = (err_flux*ra_width*dec_width - flux*(err_ra_width*dec_width+err_dec_width*ra_width)) / (2*np.pi*ra_width*ra_width*dec_width*dec_width)
        return Gauss2D((dec_peak,ra_peak), flux, (dec_width,ra_width), cont, rot, peak, (err_dec_peak,err_ra_peak), err_flux, (err_dec_width,err_ra_width), err_rot, err_peak)
    
    def moffat_fit(self, pos_min, pos_max,  center=None, I=None, a=None , q=1, n=2.0, cont=None, rot=0, factor = 1, plot = False):
        """performs moffat fit on image.
        Returns Gauss2D object

        Parameters
        ----------
        
        pos_min : (float,float)
        Minimum declination and right ascension in degrees (dec_min, ra_min)

        pos_max : (float,float)
        Maximum declination and right ascension in degrees (dec_max,ra_max)
        
        center : (float,float)
        Initial Moffat center (dec_peak, ra_peak). If None they are estimated.
            
        I : float
        Initial intensity at image center. 
    
        a : float
        Initial half width at half maximum of the image in the absence of atmospheric scattering.
        
        q : float
        Initial axis ratio.
        
        n : integer
        Initial atmospheric scattering coefficient.
        
        cont : float
        Initial continuum value, if None it is estimated.
        
        rot : float
        Initial angle position in degree.
        
        factor : integer
        If factor<=1, gaussian is computed in the center of each pixel.
        If factor>1, for each pixel, gaussian value is the sum of the gaussian values on the factor*factor pixels divided by the pixel area.
        
        plot : boolean
        If True, the gaussian is plotted.
        """
        ra_min = pos_min[1]
        dec_min = pos_min[0]
        ra_max = pos_max[1]
        dec_max = pos_max[0]
        ima = self.truncate(dec_min, dec_max, ra_min, ra_max, mask = False)
        
        ksel = np.where(ima.data.mask==False)
        pixcrd = np.zeros((np.shape(ksel[0])[0],2))
        pixcrd[:,1] = ksel[1] #ra
        pixcrd[:,0] = ksel[0] #dec
        pixsky = ima.wcs.pix2sky(pixcrd)
        x = pixsky[:,1] #ra
        y = pixsky[:,0] #dec  
        data = ima.data.data[ksel]*self.fscale
        
        #weight
        if ima.var is None:
            dw = data
        else:
            dw = data / ima.var[ksel]
        
        # initial peak position
        if center is None:
            imax = dw.argmax()
            ra_peak = x[imax]
            dec_peak = y[imax]
        else:
            ra_peak = center[1]
            dec_peak = center[0]
            
        # continuum value 
        if cont is None:
            imin = dw.argmin()
            cont =  max(0,data[imin])
            
        # initial width value    
        if a is None:
            fwhm = ima.fwhm(center)
            a = fwhm[0]/(2*np.sqrt(2**(1.0/n)-1.0))
        
        # initial gaussian integrated flux
        if I is None:
           imax = dw.argmax()
           I = dw[imax]
        
        #rotation angle in rad
        theta = np.pi * rot / 180.0
        
        
        if factor > 1:
            moffatfit = lambda p: moffat_image(ima.shape, ima.wcs, (p[2],p[1]), p[0], p[3], p[5], p[4], p[6],factor).data.data[ksel]*ima.fscale + cont
            e_moffat_fit = lambda p, data: (moffatfit(p) - data)
            v0 = [I,ra_peak, dec_peak, a, n, q, rot]
            v,covar,info, mesg, success  = leastsq(e_moffat_fit, v0[:], args=(data), maxfev=100000, full_output=1)           
        else:                                             
            moffatfit = lambda p,x, y: cont + p[0]*(1+(((x-p[1])*np.cos(p[6])-(y-p[2])*np.sin(p[6]))/p[3])**2 \
                              +(((x-p[1])*np.sin(p[6])+(y-p[2])*np.cos(p[6]))/p[3]/p[5])**2)**p[4]
            e_moffat_fit = lambda p, x, y, data: (moffatfit(p,x,y) - data)                
            v0 = [I,ra_peak, dec_peak, a, n, q, rot]
            v,covar,info, mesg, success  = leastsq(e_moffat_fit, v0[:], args=(x,y,data), maxfev=100000, full_output=1)
    
        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        err = np.array([np.sqrt(covar[i, i]) * np.sqrt(chisq / dof) for i in range(len(v))])
        
        # plot
        if plot:
            moffatfit = lambda p,x, y: cont + p[0]*(1+(((x-p[1])*np.cos(p[6])-(y-p[2])*np.sin(p[6]))/p[3])**2 \
                              +(((x-p[1])*np.sin(p[6])+(y-p[2])*np.cos(p[6]))/p[3]/p[5])**2)**p[4]            
            xmin = np.min(x)
            xmax = np.max(x)
            xx = np.arange(xmin,xmax,(xmax-xmin)/100)
            ymin = np.min(y)
            ymax = np.max(y)
            yy = np.arange(ymin,ymax,(ymax-ymin)/100)
            
            ff = np.zeros((np.shape(yy)[0],np.shape(xx)[0]))
            for i in range(np.shape(xx)[0]):
                xxx = np.zeros(np.shape(yy)[0])
                xxx[:] = xx[i]
                ff[:,i] = moffatfit(v,xxx,yy)
            
            pixsky = [[ymin,xmin],[ymax,xmax]]
            [[ymin,xmin],[ymax,xmax]] = self.wcs.sky2pix(pixsky)
            xx = np.arange(xmin,xmax,(xmax-xmin)/np.shape(xx)[0])
            yy = np.arange(ymin,ymax,(ymax-ymin)/np.shape(yy)[0])
            plt.contour(xx, yy, ff, 5)

        I = v[0]
        err_I = err[0]
        ra = v[1]
        err_ra = err[1]
        dec = v[2]
        err_dec = err[2]
        a = v[3]
        err_a = err[3]
        n = v[4]
        err_n = err[4]
        q = v[5]
        err_q = err[5]
        rot = (v[6] * 180.0 / np.pi)%180
        err_rot = err[6] * 180.0 / np.pi
        print 'I',I,err_I
        print 'ra',ra,err_ra
        print 'dec',dec,err_dec
        print 'a',a,err_a
        print 'n',n,err_n
        print 'q',q,err_q
        print 'rot',rot,err_rot
    
    def _rebin_factor(self, factor):
        '''shrinks the size of the image by factor.
        New size is an integer multiple of the original size.
        
        Parameter
        ----------
        factor : (integer,integer)
        Factor in X and Y.
        Python notation: (ny,nx)
        '''
        assert not np.sometrue(np.mod( self.shape[0], factor[0] ))
        assert not np.sometrue(np.mod( self.shape[1], factor[1] ))
        # new size is an integer multiple of the original size
        newshape = (self.shape[0]/factor[0],self.shape[1]/factor[1])
        data = self.data.reshape(newshape[0],factor[0],newshape[1],factor[1]).sum(1).sum(2)/factor[0]/factor[1]
        var = None
        if self.var is not None:
            var = self.var.reshape(newshape[0],factor[0],newshape[1],factor[1]).sum(1).sum(2)/factor[0]/factor[1]/factor[0]/factor[1]
        cdelt = self.wcs.get_step()
        wcs = self.wcs.rebin(step=(cdelt[0]*factor[0],cdelt[1]*factor[1]),start=None)
        res = Image(shape=newshape, wcs = wcs, unit=self.unit, fscale=self.fscale)
        res.data = np.ma.masked_invalid(data)
        res.var = var
        return res

        
    def rebin_factor(self, factor, margin='center'):
        '''shrinks the size of the image by factor.
        
        Parameters
        ----------
        factor : integer or (integer,integer)
        Factor in X and Y.
        Python notation: (ny,nx)
        
        margin : 'center' or 'origin'
        This parameters is used if new size is not an integer multiple of the original size.
        'center' : pixels added, on the left and on the right, on the bottom and of the top of the image.
        'origin': pixels added on (n+1) line/column.
        '''
        if is_int(factor):
            factor = (factor,factor)
        if factor[0]<=1 or factor[0]>=self.shape[0] or factor[1]<=1 or factor[1]>=self.shape[1]:
            raise ValueError, 'factor must be in ]1,shape['
        if not np.sometrue(np.mod( self.shape[0], factor[0] )) and not np.sometrue(np.mod( self.shape[1], factor[1] )):
            # new size is an integer multiple of the original size
            return self._rebin_factor(factor)
        elif not np.sometrue(np.mod( self.shape[0], factor[0] )):
            newshape1 = self.shape[1]/factor[1]
            n1 = self.shape[1] - newshape1*factor[1]
            if margin == 'origin' or n1==1:
                ima = self[:,:-n1]._rebin_factor(factor)
                newshape = (ima.shape[0], ima.shape[1] + 1)
                data = np.ones(newshape)
                mask = np.zeros(newshape,dtype=bool)
                data[:,0:-1] = ima.data
                mask[:,0:-1] = ima.data.mask
                data[:,-1] = self.data[:,-n1:].sum() / factor[1]
                mask[:,-1] = self.data.mask[:,-n1:].any()
                var = None
                if self.var is not None:
                    var = np.ones(newshape)
                    var[:,0:-1] = ima.var
                    var[:,-1] = self.var[:,-n1:].sum() / factor[1] / factor[1]
                wcs = ima.wcs
                wcs.wcs.naxis1 = wcs.wcs.naxis1 +1
            else:
                n_left = n1/2
                n_right = self.shape[1] - n1 + n_left
                ima = self[:,n_left:n_right]._rebin_factor(factor)
                newshape = (ima.shape[0], ima.shape[1] + 2)
                data = np.ones(newshape)
                mask = np.zeros(newshape,dtype=bool)
                data[:,1:-1] = ima.data
                mask[:,1:-1] = ima.data.mask
                data[:,0] = self.data[:,0:n_left].sum() / factor[1]
                mask[:,0] = self.data.mask[:,0:n_left].any()
                data[:,-1] = self.data[:,n_right:].sum() / factor[1]
                mask[:,-1] = self.data.mask[:,n_right:].any()
                var = None
                if self.var is not None:
                    var = np.ones(newshape)
                    var[:,1:-1] = ima.var
                    var[:,0] = self.var[:,0:n_left].sum() / factor[1] / factor[1]
                    var[:,-1] = self.var[:,n_right:].sum() / factor[1] / factor[1]
                wcs = ima.wcs
                wcs.wcs.wcs.crval = [wcs.wcs.wcs.crval[0] - wcs.get_step()[1] , wcs.wcs.wcs.crval[1]]
                wcs.wcs.naxis1 = wcs.wcs.naxis1 +2
        elif not np.sometrue(np.mod( self.shape[1], factor[1] )):
            newshape0 = self.shape[0]/factor[0]
            n0 = self.shape[0] - newshape0*factor[0]
            if margin == 'origin' or n0==1:
                ima = self[:-n0,:]._rebin_factor(factor)
                newshape = (ima.shape[0] + 1, ima.shape[1])
                data = np.ones(newshape)
                mask = np.zeros(newshape,dtype=bool)
                data[0:-1,:] = ima.data
                mask[0:-1,:] = ima.data.mask
                data[-1,:] = self.data[-n0:,:].sum() / factor[0]
                mask[-1,:] = self.data.mask[-n0:,:].any()
                var = None
                if self.var is not None:
                    var = np.ones(newshape)
                    var[0:-1,:] = ima.var
                    var[-1,:] = self.var[-n0:,:].sum() / factor[0] / factor[0]
                wcs = ima.wcs
                wcs.wcs.naxis2 = wcs.wcs.naxis2 +1
            else:
                n_left = n0/2
                n_right = self.shape[0] - n0 + n_left
                ima = self[n_left:n_right,:]._rebin_factor(factor)
                newshape = (ima.shape[0] + 2, ima.shape[1])
                data = np.ones(newshape)
                mask = np.zeros(newshape,dtype=bool)
                data[1:-1,:] = ima.data
                mask[1:-1,:] = ima.data.mask
                data[0,:] = self.data[0:n_left,:].sum() / factor[0]
                mask[0,:] = self.data.mask[0:n_left,:].any()
                data[-1,:] = self.data[n_right:,:].sum() / factor[0]
                mask[-1,:] = self.data.mask[n_right:,:].any()
                var = None
                if self.var is not None:
                    var = np.ones(newshape)
                    var[1:-1,:] = ima.var
                    var[0,:] = self.var[0:n_left,:].sum() / factor[0] / factor[0]
                    var[-1,:] = self.var[n_right:,:].sum() / factor[0] / factor[0]
                wcs = ima.wcs
                wcs.wcs.wcs.crval = [wcs.wcs.wcs.crval[0] , wcs.wcs.wcs.crval[1] - wcs.get_step()[0]]
                wcs.wcs.naxis2 = wcs.wcs.naxis2 +2
        else:
            factor = np.array(factor)
            newshape = self.shape/factor
            n = self.shape - newshape*factor
            if n[0]==1 and n[1]==1:
                margin = 'origin'
            if margin == 'center':
                n_left = n/2
                n_right = self.shape - n + n_left
                ima = self[n_left[0]:n_right[0],n_left[1]:n_right[1]]._rebin_factor(factor)
                if n_left[0]!=0 and n_left[1]!=0:
                    newshape = ima.shape + 2
                    data = np.ones(newshape)
                    mask = np.zeros(newshape,dtype=bool)
                    data[1:-1,1:-1] = ima.data
                    mask[1:-1,1:-1] = ima.data.mask
                    data[0,:] = self.data[0:n_left[0],:].sum() / factor[0]
                    mask[0,:] = self.data.mask[0:n_left[0],:].any()
                    data[-1,:] = self.data[n_right[0]:,:].sum() / factor[0]
                    mask[-1,:] = self.data.mask[n_right[0]:,:].any()
                    data[:,0] = self.data[:,0:n_left[1]].sum() / factor[1]
                    mask[:,0] = self.data.mask[:,0:n_left[1]].any()
                    data[:,-1] = self.data[:,n_right[1]:].sum() / factor[1]
                    mask[:,-1] = self.data.mask[:,n_right[1]:].any()
                    var = None
                    if self.var is not None:
                        var = np.ones(newshape)
                        var[1:-1,1:-1] = var.data
                        var[0,:] = self.var[0:n_left[0],:].sum() / factor[0] / factor[0]
                        var[-1,:] = self.var[n_right[0]:,:].sum() / factor[0] / factor[0]
                        var[:,0] = self.var[:,0:n_left[1]].sum() / factor[1] / factor[1]
                        var[:,-1] = self.var[:,n_right[1]:].sum() / factor[1] / factor[1]
                    wcs = ima.wcs
                    step = wcs.get_step()
                    wcs.wcs.wcs.crval = wcs.wcs.wcs.crval - np.array([step[1],step[0]])
                    wcs.wcs.naxis1 = wcs.wcs.naxis1 +2
                    wcs.wcs.naxis2 = wcs.wcs.naxis2 +2
                elif n_left[0]==0:
                    newshape = (ima.shape[0] + 1, ima.shape[1] + 2)
                    data = np.ones(newshape)
                    mask = np.zeros(newshape,dtype=bool)
                    data[0:-1,1:-1] = ima.data
                    mask[0:-1,1:-1] = ima.data.mask
                    data[-1,:] = self.data[n_right[0]:,:].sum() / factor[0]
                    mask[-1,:] = self.data.mask[n_right[0]:,:].any()
                    data[:,0] = self.data[:,0:n_left[1]].sum() / factor[1]
                    mask[:,0] = self.data.mask[:,0:n_left[1]].any()
                    data[:,-1] = self.data[:,n_right[1]:].sum() / factor[1]
                    mask[:,-1] = self.data.mask[:,n_right[1]:].any()
                    var = None
                    if self.var is not None:
                        var = np.ones(newshape)
                        var[0:-1,1:-1] = var.data
                        var[-1,:] = self.var[n_right[0]:,:].sum() / factor[0] / factor[0]
                        var[:,0] = self.var[:,0:n_left[1]].sum() / factor[1] / factor[1]
                        var[:,-1] = self.var[:,n_right[1]:].sum() / factor[1] / factor[1]
                    wcs = ima.wcs
                    wcs.wcs.wcs.crval = [wcs.wcs.wcs.crval[0] - wcs.get_step()[1] , wcs.wcs.wcs.crval[1]]
                    wcs.wcs.naxis1 = wcs.wcs.naxis1 +2
                    wcs.wcs.naxis2 = wcs.wcs.naxis2 +1
                else:
                    newshape = (ima.shape[0] + 2, ima.shape[1] + 1)
                    data = np.ones(newshape)
                    mask = np.zeros(newshape,dtype=bool)
                    data[1:-1,0:-1] = ima.data
                    mask[1:-1,0:-1] = ima.data.mask
                    data[0,:] = self.data[0:n_left[0],:].sum() / factor[0]
                    mask[0,:] = self.data.mask[0:n_left[0],:].any()
                    data[-1,:] = self.data[n_right[0]:,:].sum() / factor[0]
                    mask[-1,:] = self.data.mask[n_right[0]:,:].any()
                    data[:,-1] = self.data[:,n_right[1]:].sum() / factor[1]
                    mask[:,-1] = self.data.mask[:,n_right[1]:].any()
                    var = None
                    if self.var is not None:
                        var = np.ones(newshape)
                        var[1:-1,0:-1] = var.data
                        var[0,:] = self.var[0:n_left[0],:].sum() / factor[0] / factor[0]
                        var[-1,:] = self.var[n_right[0]:,:].sum() / factor[0] / factor[0]
                        var[:,-1] = self.var[:,n_right[1]:].sum() / factor[1] / factor[1]
                    wcs = ima.wcs
                    wcs.wcs.wcs.crval = [wcs.wcs.wcs.crval[0] , wcs.wcs.wcs.crval[1] - wcs.get_step()[0]] 
                    wcs.wcs.naxis1 = wcs.wcs.naxis1 +1
                    wcs.wcs.naxis2 = wcs.wcs.naxis2 +2
            elif margin=='origin':
                n_right = self.shape - n
                ima = self[0:n_right[0],0:n_right[1]]._rebin_factor(factor)
                newshape = ima.shape + 1
                data = np.ones(newshape)
                mask = np.zeros(newshape,dtype=bool)
                data[0:-1,0:-1] = ima.data
                mask[0:-1,0:-1] = ima.data.mask
                data[-1,:] = self.data[n_right[0]:,:].sum() / factor[0]
                mask[-1,:] = self.data.mask[n_right[0]:,:].any()
                data[:,-1] = self.data[:,n_right[1]:].sum() / factor[1]
                mask[:,-1] = self.data.mask[:,n_right[1]:].any()
                var = None
                if self.var is not None:
                    var = np.ones(newshape)
                    var[0:-1,0:-1] = ima.var
                    var[-1,:] = self.var[n_right[0]:,:].sum() / factor[0] / factor[0]
                    var[:,-1] = self.var[:,n_right[1]:].sum() / factor[1] / factor[1]
                wcs = ima.wcs
                wcs.wcs.naxis1 = wcs.wcs.naxis1 +1
                wcs.wcs.naxis2 = wcs.wcs.naxis2 +1
            else:
                raise ValueError, 'margin must be center|origin'
        res = Image(shape=newshape, wcs = wcs, unit=self.unit, fscale=self.fscale)
        res.data = np.ma.array(data, mask=mask)
        res.var = var
        return res
    
    def rebin(self, newdim, newstart, newstep, flux=False, order=3):
        """rebins the image to a new coordinate system.
        
        Parameters
        ----------
        newdim : integer or (integer,integer)
        New dimensions. Python notation: (ny,nx)
        
        newstart : float or (float, float)
        New positions (dec,ra) for the pixel (0,0). If None, old position is used.
        
        newstep : float or (float, float)
        New step (ddec,dra).
        
        flux : boolean
        if flux is True, the flux is conserved.
        
        order : integer
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.   
        """
        if is_int(newdim):
            newdim = (newdim,newdim)
        if newstart is None:
            newstart = self.wcs.get_start()
        elif is_int(newstart) or is_float(newstart):
            newstart = (newstart,newstart)
        else:
            pass
        if is_int(newstep) or is_float(newstep):
            newstep = (newstep,newstep)
        newdim = np.array(newdim)
        newstart = np.array(newstart)
        newstep = np.array(newstep)
                   
        wcs =WCS(crpix=(1.0,1.0),crval=newstart,cdelt=newstep,deg=self.wcs.is_deg(),rot=self.wcs.get_rot(), shape = newdim)
        pstep = newstep/self.wcs.get_step()   
        poffset = (newstart-self.wcs.get_start())/newstep
        data = ndimage.affine_transform(self.data.filled(0), pstep, poffset,output_shape=newdim, order=order)
        mask = np.array(1 - self.data.mask,dtype=bool)
        newmask = ndimage.affine_transform(mask, pstep, poffset,output_shape=newdim, order=0)
        mask = np.ma.make_mask(1-newmask)
        
        if flux:
            rflux = self.wcs.get_step().prod()/newstep.prod()
            data *= rflux
        res = Image(notnoise=True, shape=newdim, wcs = wcs, unit=self.unit, fscale=self.fscale)
        res.data = np.ma.array(data, mask=mask)
        return res 

    def gaussian_filter(self, sigma=3):
        """Applies gaussian filter to the image.
        
        Parameter
        ---------
        sigma : float
        Standard deviation for Gaussian kernel.
        """
        res = self.copy()
        res.data = np.ma.array(ndimage.gaussian_filter(res.data.filled(0), sigma),mask=res.data.mask)
        return res
            
    def median_filter(self, size=3):
        """Applies median filter to the image.
        
        Parameter
        ---------
        size : float
        Shape that is taken from the input array, at every element position, to define the input to the filter function.

        """
        res = self.copy()
        res.data = np.ma.array(ndimage.median_filter(res.data.filled(0), size),mask=res.data.mask)
        return res
    
    def maximum_filter(self, size=3):
        """Applies maximum filter to the image.
        
        Parameter
        ---------
        size : float
        Shape that is taken from the input array, at every element position, to define the input to the filter function.

        """
        res = self.copy()
        res.data = np.ma.array(ndimage.maximum_filter(res.data.filled(0), size),mask=res.data.mask)
        return res     
    
    def minimum_filter(self, size=3):
        """Applies minimum filter to the image.
        
        Parameter
        ---------
        size : float
        Shape that is taken from the input array, at every element position, to define the input to the filter function.

        """
        res = self.copy()
        res.data = np.ma.array(ndimage.minimum_filter(res.data.filled(0), size),mask=res.data.mask)
        return res   
    
    def add(self, other):
        """ Adds the image other to the current image.
        The coordinate are taken into account.
        
        Parameter
        ---------
        other : Image
        Second image to add.
        """
        try:
            if other.image:
                ima = other.copy()
                self_rot = self.wcs.get_rot()
                ima_rot = ima.wcs.get_rot()
                if self_rot != ima_rot:
                    ima = ima.rotate(self_rot-ima_rot)
                self_cdelt = self.wcs.get_step()
                ima_cdelt = ima.wcs.get_step()
                if (self_cdelt != ima_cdelt).all():
                    try :
                        factor = self_cdelt/ima_cdelt
                        if not np.sometrue(np.mod( self_cdelt[0],  ima_cdelt[0])) and not np.sometrue(np.mod( self_cdelt[1],  ima_cdelt[1] )):
                            # ima.step is an integer multiple of the self.step
                            ima = ima.rebin_factor(factor)
                        else:
                            raise ValueError, 'steps are not integer multiple'
                    except:
                        newdim = ima.shape/factor
                        ima = ima.rebin(newdim, None, self_cdelt, flux=True)
                # here ima and self have the same step
                [[k1,l1]] = self.wcs.sky2pix(ima.wcs.pix2sky([[0,0]]))
                l1 = int(l1 + 0.5)
                k1 = int(k1 + 0.5)
                if k1 < 0:
                    nk1 = -k1
                    k1 = 0
                else:
                    nk1 = 0
                k2 = k1 + ima.shape[0] 
                if k2 > self.shape[0]:
                    nk2 = ima.shape[0] - (k2 - self.shape[0])
                    k2 = self.shape[0] 
                else:
                    nk2 = ima.shape[0]
                
                if l1 < 0:
                    nl1 = -l1
                    l1 = 0
                else:
                    nl1 = 0                    
                l2 = l1 + ima.shape[1] 
                if l2 > self.shape[1]:
                    nl2 = ima.shape[1] - (l2 - self.shape[1])
                    l2 = self.shape[1] 
                else:
                    nl2 = ima.shape[1]
        
                data = self.data.filled(0)  
                
                data[k1:k2,l1:l2] += (ima.data.filled(0)[nk1:nk2,nl1:nl2] * ima.fscale / self.fscale)
                res = Image(notnoise=True, shape=self.shape, wcs = self.wcs, unit=self.unit, fscale=self.fscale)
                res.data = np.ma.array(data, mask=self.data.mask)
                return res 
        except:
            print 'Operation forbidden'
            return None
        
    def segment(self, shape=(2,2), minsize=20, background = 20):
        """ Segments the image in a number of smaller images.
        Returns a list of images.
        
        Parameters
        ----------
        
        shape : (integer,integer)
        Shape used for connectivity.
        
        minsize : integer
        Minimmum size of the images.
        
        background : float
        Under this value, flux is considered as background.
        """
        structure = ndimage.morphology.generate_binary_structure(shape[0], shape[1])
        expanded = ndimage.morphology.grey_dilation(self.data.filled(0), (minsize,minsize))
        ksel = np.where(expanded<background)
        expanded[ksel] = 0
        
        lab = ndimage.measurements.label(expanded, structure)
        slices = ndimage.measurements.find_objects(lab[0])

        imalist = []
        for i in range(lab[1]):
            [[starty,startx]] = self.wcs.pix2sky(ima.wcs.pix2sky([[slices[i][0].start,slices[i][1].start]]))
            wcs = WCS(crpix=(1.0,1.0),crval=(starty,startx),cdelt=self.wcs.get_step(),deg=self.wcs.is_deg(),rot=self.wcs.get_rot())
            res = Image(data=self.data[slices[i]],wcs=wcs)
            imalist.append(res)
        return imalist
    
    def add_gaussian_noise(self, sigma):
        """ Adds gaussian noise to image
        
        Parameter
        ---------
        
        sigma : float
        Standard deviation.
        """
        res = self.copy()
        res.data = np.ma.array(np.random.normal(res.data.filled(0), sigma),mask=res.data.mask)
        return res
    
    def add_poisson_noise(self):
        """ Adds poisson noise to image
        """
        res = self.copy()
        res.data = np.ma.array(np.random.poisson(res.data.filled(0)),mask=res.data.mask)
        return res
    
    def inside(self, coord):
        """ Returns True if coord is inside image
        """
        pixcrd = [ [0,0], [self.shape[0]-1,0], [ self.shape[0]-1,self.shape[1]-1], [0,self.shape[1]-1]]
        pixsky = self.wcs.pix2sky(pixcrd)
        #Compute the cross product
        if ((coord[0]-pixsky[0][0])*(pixsky[1][1]-pixsky[0][1])-(coord[1]-pixsky[0][1])*(pixsky[1][0]-pixsky[0][0]))<0 :
            Normal1IsPositive = False
        if ((coord[0]-pixsky[1][0])*(pixsky[2][1]-pixsky[1][1])-(coord[1]-pixsky[1][1])*(pixsky[2][0]-pixsky[1][0]))<0 :
            Normal2IsPositive = False
        if ((coord[0]-pixsky[2][0])*(pixsky[3][1]-pixsky[2][1])-(coord[1]-pixsky[2][1])*(pixsky[3][0]-pixsky[2][0]))<0 :
            Normal3IsPositive = False
        if ((coord[0]-pixsky[3][0])*(pixsky[0][1]-pixsky[3][1])-(coord[1]-pixsky[3][1])*(pixsky[0][0]-pixsky[3][0]))<0 :
            Normal4IsPositive = False;
        if (Normal1IsPositive==Normal2IsPositive) and (Normal2IsPositive==Normal3IsPositive) and (Normal3IsPositive==Normal4IsPositive) :
            return True
        else:
            return False
        
    def fftconvolve(self, other):
        """convolves self and other using fft.
        
        Parameter
        ---------
        
        other : 2d-array or Image
        Second Image or 2d-array
        """
        if self.data is None:
            raise ValueError, 'empty data array'
        if type(other) is np.array:
            res = self.copy()
            res.data = np.ma.array(signal.fftconvolve(self.data.filled(0) ,other ,mode='same'), mask=self.data.mask)
            return res
        try:
            if other.image:
                if other.data is None or self.shape[0] != other.shape[0] or self.shape[1] != other.shape[1]:
                    print 'Operation forbidden for images with different sizes'
                    return None
                else:
                    res = self.copy()
                    res.data = np.ma.array(signal.fftconvolve(self.data.filled(0) ,other.data.filled(0) ,mode='same'), mask=self.data.mask)
                    res.fscale = self.fscale * other.fscale
                    return res
        except:
            print 'Operation forbidden'
            return None
        
    def fftconvolve_gauss(self,center=None, flux=1., width=(1.,1.), peak=False, rot = 0., factor=1):
        """Convolves image with a 2D gaussian.
        
        Parameters
        ----------
        
        center : (float,float)
        Gaussian center (dec_peak, ra_peak). If None the center of the image is used.
        
        flux : float
        Integrated gaussian flux or gaussian peak value if peak is True.

        width : (float,float)
        Spreads of the Gaussian blob (dec_width,ra_width).
        
        peak : boolean
        If true, flux contains a gaussian peak value.
    
        rot : float
        Angle position in degree.
      
        factor : integer
        If factor<=1, gaussian is computed in the center of each pixel.
        If factor>1, for each pixel, gaussian value is the sum of the gaussian values on the factor*factor pixels divided by the pixel area.
        """
        ima = gauss_image(self.shape, self.wcs, center, flux, width, peak, rot, factor)
        ima.norm(type='sum')
        return self.fftconvolve(ima)
    
    def fftconvolve_moffat(self, center=None, I=1., a=1.0, q=1.0, n=2, rot = 0., factor=1):
        """Convolves image with a 2D moffat.
        
        Parameters
        ----------
        
        center : (float,float)
        Moffat center (dec_peak, ra_peak). If None the center of the image is used.
            
        I : float
        Intensity at image center. 1 by default.
    
        a : float
        Half width at half maximum of the image in the absence of atmospheric scattering. 1 by default.
        
        q : float
        axis ratio, 1 by default.
        
        n : integer
        Atmospheric scattering coefficient. 2 by default.
        
        rot : float
        Angle position in degree.
          
        factor : integer
        If factor<=1, moffat value is computed in the center of each pixel.
        If factor>1, for each pixel, moffat value is the sum of the moffat values on the factor*factor pixels divided by the pixel area.
        """
        ima = moffat_image(self.shape, self.wcs, center, I, a, q, n, rot, factor)
        ima.norm(type='sum')
        return self.fftconvolve(ima)
    
    def plot(self, title=None, scale='linear', vmin=None, vmax=None, zscale = False): 
        """ plots the image.
        
        Parameter
        ---------     
        title : string
        Figure title (None by default).
        
        scale : linear' | 'log' | 'sqrt' | 'arcsinh' | 'power' 
        The stretch function to use for the scaling (default is 'linear').
        
        vmin: float
        Minimum pixel value to use for the scaling.
        If None, vmin is set to min of data.
 
        vmax: float
        Maximum pixel value to use for the scaling.
        If None, vmax is set to max of data.
        
        zscale : boolean
        If true, compute vmin and vmax using the IRAF zscale algorithm.
        """
        plt.ion()
        
        f = self.data*self.fscale
        xaxis = np.arange(self.shape[1], dtype=np.float)
        yaxis = np.arange(self.shape[0], dtype=np.float)
        xunit = 'pixel'
        yunit = 'pixel'
        
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
            if zscale:
                vmin,vmax = plt_zscale.zscale(self.data.filled(0))
            if scale=='log':
                from matplotlib.colors import LogNorm
                norm = LogNorm(vmin=vmin, vmax=vmax)
            elif scale=='arcsinh':
                norm = plt_norm.ArcsinhNorm(vmin=vmin, vmax=vmax)
            elif scale=='power':
                norm = plt_norm.PowerNorm(vmin=vmin, vmax=vmax)
            elif scale=='sqrt':
                norm = plt_norm.SqrtNorm(vmin=vmin, vmax=vmax)
            else:
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

            cax = plt.imshow(f,interpolation='nearest',origin='lower',extent=(xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]),norm=norm)
            plt.colorbar(cax)
            plt.xlabel('ra (%s)' %xunit)
            plt.ylabel('dec (%s)' %yunit)
            self._ax = cax
            
        if title is not None:
                plt.title(title)   
                
        self._fig = plt.get_current_fig_manager()
        plt.connect('motion_notify_event', self._on_move)
        
    def _on_move(self,event):
        """ prints y,x,i,j and data in the figure toolbar.
        """
        if event.inaxes is not None:
            j, i = event.xdata, event.ydata
            try:
                pixsky = self.wcs.pix2sky([i,j])
                yc = pixsky[0][0]
                xc = pixsky[0][1]
                val = self.data.data[i,j]*self.fscale
                s = 'dec= %g ra=%g i=%i j=%i data=%g'%(yc,xc,i,j,val)
                self._fig.toolbar.set_message(s)
            except:
                pass    
            
    def ipos(self, filename='None'):
        """Interactive mode.
        Prints cursor position.   
        To read cursor position, click on the left mouse button
        To remove a cursor position, click on the left mouse button + <r>
        To quit the interactive mode, click on the right mouse button.
        At the end, clicks are saved in self.clicks as dictionary {'ra','dec','i','j','data'}.
        
        Parameter
        ---------
        
        filename : string
        If filename is not None, the cursor values are saved as a fits table.
        """
        print 'To read cursor position, click on the left mouse button'
        print 'To remove a cursor position, click on the left mouse button + <d>'
        print 'To quit the interactive mode, click on the right mouse button.'
        print 'After quit, clicks are saved in self.clicks as dictionary {ra,dec,i,j,data}.'
        
        if self._clicks is None:
            binding_id = plt.connect('button_press_event', self._on_click)
            self._clicks = ImageClicks(binding_id,filename)
        else:
            self._clicks.filename = filename
        
    def _on_click(self,event):
        """ prints dec,ra,i,j and data corresponding to the cursor position.
        """
        if event.key == 'd':
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        j, i = event.xdata, event.ydata
                        self._clicks.remove(i,j)
                        print "new selection:"
                        for i in range(len(self._clicks.ra)):
                            self._clicks.iprint(i,self.fscale)
                    except:
                        pass 
        else:
            if event.button == 1:
                if event.inaxes is not None:
                    j, i = event.xdata, event.ydata
                    try:
                        i = int(i)
                        j = int(j)
                        [[y,x]] = self.wcs.pix2sky([i,j])
                        val = self.data[i,j]*self.fscale
                        if len(self._clicks.ra)==0:
                            print ''
                        self._clicks.add(i,j,x,y,val)
                        self._clicks.iprint(len(self._clicks.ra)-1, self.fscale)
                    except:
                        pass
            else:
                self._clicks.write_fits()
                # save clicks in a dictionary {'i','j','x','y','data'}
                d = {'i':self._clicks.i, 'j':self._clicks.j, 'ra':self._clicks.ra, 'dec':self._clicks.dec, 'data':self._clicks.data}
                self.clicks = d
                #clear
                self._clicks.clear()
                self._clicks = None
                
            
    def idist(self):
        """Interactive mode.
        Gets distance and center from 2 cursor positions.
        """
        print 'Use left mouse button to define the line.'
        print 'To quit the interactive mode, click on the right mouse button.'
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_dist, drawtype='line')
            
    def _on_select_dist(self, eclick, erelease):
        """Prints distance and center between 2 cursor positions.
        """
        if eclick.button == 1:
            try:
                j1, i1 = int(eclick.xdata), int(eclick.ydata)
                [[y1,x1]] = self.wcs.pix2sky([i1,j1])
                j2, i2 = int(erelease.xdata), int(erelease.ydata)
                [[y2,x2]] = self.wcs.pix2sky([i2,j2])
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                xc = (x1 + x2)/2
                yc = (y1 + y2)/2
                print 'Center: (%g,%g)\tDistance: %g' % (xc,yc,dist)
            except:
                pass
        else: 
            print 'idist deactivated.'
            self._selector.set_active(False)
            self._selector = None

            
    def istat(self):
        """Interactive mode.
        Computes image statistics from windows defined by 2 cursor positions.
        mean is the mean value, median the median value
        std is the rms standard deviation, sum the sum, peak the peak value
        npts is the total number of points.
        """
        print 'Use left mouse button to define the box.'
        print 'To quit the interactive mode, click on the right mouse button.'
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_stat, drawtype='box')
            
    def _on_select_stat(self,eclick, erelease):
        """Prints image statistics from windows defined by 2 cursor positions.
        """
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata,erelease.xdata))
                j2 = int(max(eclick.xdata,erelease.xdata))
                i1 = int(min(eclick.ydata,erelease.ydata))
                i2 = int(max(eclick.ydata,erelease.ydata))
                d = self.data[i1:i2, j1:j2]
                mean = self.fscale*np.mean(d)
                median = self.fscale*np.median(np.ravel(d))
                vsum = self.fscale*d.sum()
                std = self.fscale*np.std(d)
                npts = d.shape[0]*d.shape[1]
                peak = self.fscale*d.max()
                print 'mean=%g\tmedian=%g\tstd=%g\tsum=%g\tpeak=%g\tnpts=%d' % (mean, median, std, vsum, peak, npts)
            except:
                pass
        else: 
            print 'istat deactivated.'
            self._selector.set_active(False)
            self._selector = None
            
    def ipeak(self):
        """Interactive mode.
        Prints peak location in windows defined by 2 cursor positions.
        """
        print 'Use left mouse button to define the box.'
        print 'To quit the interactive mode, click on the right mouse button.'
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_peak, drawtype='box')
            
    def _on_select_peak(self,eclick, erelease):
        """Prints image peak location in windows defined by 2 cursor positions.
        """
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata,erelease.xdata))
                j2 = int(max(eclick.xdata,erelease.xdata))
                i1 = int(min(eclick.ydata,erelease.ydata))
                i2 = int(max(eclick.ydata,erelease.ydata))
                center = ((i2+i1)/2,(j2+j1)/2)
                radius = (np.abs(i2-i1)/2,np.abs(j2-j1)/2)
                peak =  self.peak(center,radius,True)
                print 'peak: dec=%g\tra=%g\ti=%d\tj=%d\tdata=%g' % (peak['dec'], peak['ra'], peak['i'], peak['j'], peak['data'])
            except:
                pass
        else: 
            print 'ipeak deactivated.'
            self._selector.set_active(False)
            self._selector = None
            
    def ifwhm(self):
        """Interactive mode.
        Computes fwhm in windows defined by 2 cursor positions.
        """
        print 'Use left mouse button to define the box.'
        print 'To quit the interactive mode, click on the right mouse button.'
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_fwhm, drawtype='box')
            
    def _on_select_fwhm(self,eclick, erelease):
        """Prints image peak location in windows defined by 2 cursor positions.
        """
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata,erelease.xdata))
                j2 = int(max(eclick.xdata,erelease.xdata))
                i1 = int(min(eclick.ydata,erelease.ydata))
                i2 = int(max(eclick.ydata,erelease.ydata))
                center = ((i2+i1)/2,(j2+j1)/2)
                radius = (np.abs(i2-i1)/2,np.abs(j2-j1)/2)
                fwhm =  self.fwhm(center,radius,True)
                print 'fwhm_dec=%g\tfwhm_ra=%g' % (fwhm[0], fwhm[1])
            except:
                pass
        else: 
            print 'ifwhm deactivated.'
            self._selector.set_active(False)
            self._selector = None
            
    def iee(self):
        """Interactive mode.
        Computes encloded energy in windows defined by 2 cursor positions.
        """
        print 'Use left mouse button to define the box.'
        print 'To quit the interactive mode, click on the right mouse button.'
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_ee, drawtype='box')
            
    def _on_select_ee(self,eclick, erelease):
        """Prints image peak location in windows defined by 2 cursor positions.
        """
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata,erelease.xdata))
                j2 = int(max(eclick.xdata,erelease.xdata))
                i1 = int(min(eclick.ydata,erelease.ydata))
                i2 = int(max(eclick.ydata,erelease.ydata))
                center = ((i2+i1)/2,(j2+j1)/2)
                radius = (np.abs(i2-i1)/2,np.abs(j2-j1)/2)
                ee =  self.ee(center,radius,True)
                print 'ee=%g' %ee
            except:
                pass
        else: 
            print 'iee deactivated.'
            self._selector.set_active(False)
            self._selector = None
            
    def imask(self):
        """Interactive mode.
        Plots masked values.
        """
        try:
            try:
                self._plot_mask_id.remove()
                #plt.draw()
            except:
                pass
            xaxis = np.arange(self.shape[1], dtype=np.float)
            yaxis = np.arange(self.shape[0], dtype=np.float)
            
            if np.shape(xaxis)[0] == 1:
                #plot a  column
                plt.plot(yaxis,self.data.data,alpha=0.3)
            elif np.shape(yaxis)[0] == 1:
                #plot a line
                plt.plot(xaxis,self.data.data,alpha=0.3)
            else:
                mask = np.array(1 - self.data.mask,dtype=bool)
                data = np.ma.MaskedArray(self.data.data*self.fscale, mask=mask)
                self._plot_mask_id = plt.imshow(data,interpolation='nearest',origin='lower',extent=(0,self.shape[1]-1,0,self.shape[0]-1),vmin=self.data.min(),vmax=self.data.max(), alpha=0.9)
        except:
            pass
            
            
def gauss_image(shape=(101,101), wcs=WCS(), center=None, flux=1., width=(1.,1.), peak=False, rot = 0., factor=1):
    """creates a new image from a 2D gaussian.
    Returns Image object

    Parameters
    ----------
        
    shape : integer or (integer,integer)
    Lengths of the image in Y and X. (101,101) by default.
    python notation: (ny,nx)
    if wcs object contains dimensions, theses dimensions are used.

    wcs : WCS
    World coordinates 
    
    center : (float,float)
    Gaussian center (dec_peak, ra_peak). If None the center of the image is used.
        
    flux : float
    Integrated gaussian flux or gaussian peak value if peak is True.

    width : (float,float)
    Spreads of the Gaussian blob (dec_width,ra_width).
        
    peak : boolean
    If true, flux contains a gaussian peak value.
    
    rot : float
    Angle position in degree.
      
    factor : integer
    If factor<=1, gaussian is computed in the center of each pixel.
    If factor>1, for each pixel, gaussian value is the sum of the gaussian values on the factor*factor pixels divided by the pixel area.
    """
    if is_int(shape):
        shape = (shape,shape)
    shape = np.array(shape)
    
    if wcs.wcs.naxis1 != 0. or wcs.wcs.naxis2 != 0.:
        shape[1] = wcs.wcs.naxis1
        shape[0] = wcs.wcs.naxis2
    
    if center is None:
        pixcrd = [[(shape[0]-1)/2.0,(shape[1]-1)/2.0]]
        pixsky = wcs.pix2sky(pixcrd)
        center = [0,0]
        center[0] = pixsky[0,0]
        center[1] = pixsky[0,1]
    else:
        center = [center[0],center[1]]
    
    data = np.zeros(shape=shape, dtype = float)
    
    ra_width = width[1]
    dec_width = width[0]
    
    #rotation angle in rad
    theta = np.pi * rot / 180.0
        
    if peak is True:
        I = flux * np.sqrt(2*np.pi*(ra_width**2)) * np.sqrt(2*np.pi*(dec_width**2))
    else:
        I = flux
        
    gauss = lambda x, y: I*(1/np.sqrt(2*np.pi*(ra_width**2)))*np.exp(-((x-center[1])*np.cos(theta)-(y-center[0])*np.sin(theta))**2/(2*ra_width**2)) \
                          *(1/np.sqrt(2*np.pi*(dec_width**2)))*np.exp(-((x-center[1])*np.sin(theta)+(y-center[0])*np.cos(theta))**2/(2*dec_width**2))  
    
    if factor>1:
        if rot == 0:
            X,Y = np.meshgrid(xrange(shape[0]),xrange(shape[1]))
            pixcrd = np.array(zip(X.ravel(),Y.ravel())) -0.5
            pixsky_min = wcs.pix2sky(pixcrd)               
            xmin = (pixsky_min[:,1]-center[1])/np.sqrt(2.0)/ra_width
            ymin = (pixsky_min[:,0]-center[0])/np.sqrt(2.0)/dec_width
                    
            pixcrd = np.array(zip(X.ravel(),Y.ravel())) +0.5
            pixsky_max = wcs.pix2sky(pixcrd)
            xmax = (pixsky_max[:,1]-center[1])/np.sqrt(2.0)/ra_width
            ymax = (pixsky_max[:,0]-center[0])/np.sqrt(2.0)/dec_width
            
            dx = pixsky_max[:,1] - pixsky_min[:,1]
            dy = pixsky_max[:,0] - pixsky_min[:,0]
            data = I * 0.25 / dx / dy * (special.erf(xmax)-special.erf(xmin)) * (special.erf(ymax)-special.erf(ymin))
            data = np.reshape(data,(shape[1],shape[0])).T
        else:
            X,Y = np.meshgrid(xrange(shape[0]*factor),xrange(shape[1]*factor))
            factor = float(factor)
            pixcrd = zip(X.ravel()/factor,Y.ravel()/factor)
            pixsky = wcs.pix2sky(pixcrd)
            data = gauss(pixsky[:,1],pixsky[:,0])
            data = (data.reshape(shape[1],factor,shape[0],factor).sum(1).sum(2)/factor/factor).T
    else:       
        X,Y = np.meshgrid(xrange(shape[0]),xrange(shape[1]))
        pixcrd = zip(X.ravel(),Y.ravel())
        pixsky = wcs.pix2sky(pixcrd)        
        data = gauss(pixsky[:,1],pixsky[:,0])
        data = np.reshape(data,(shape[1],shape[0])).T
            
    return Image(data=data, wcs=wcs)

def moffat_image(shape=(101,101), wcs=WCS(), center=None, I=1., a=1.0, q=1.0, n=2, rot = 0., factor=1):
    """creates a new image from a 2D Moffat function.
    Returns Image object

    Parameters
    ----------
        
    shape : integer or (integer,integer)
    Lengths of the image in Y and X. (101,101) by default.
    python notation: (ny,nx)
    if wcs object contains dimensions, theses dimensions are used.

    wcs : WCS
    World coordinates 
    
    center : (float,float)
    Moffat center (dec_peak, ra_peak). If None the center of the image is used.
        
    I : float
    Intensity at image center. 1 by default.

    a : float
    Half width at half maximum of the image in the absence of atmospheric scattering. 1 by default.
    
    q : float
    axis ratio, 1 by default.
    
    n : integer
    Atmospheric scattering coefficient. 2 by default.
    
    rot : float
    Angle position in degree.
      
    factor : integer
    If factor<=1, moffat value is computed in the center of each pixel.
    If factor>1, for each pixel, moffat value is the sum of the moffat values on the factor*factor pixels divided by the pixel area.
    """
    if is_int(shape):
        shape = (shape,shape)
    shape = np.array(shape)
    
    if wcs.wcs.naxis1 != 0. or wcs.wcs.naxis2 != 0.:
        shape[1] = wcs.wcs.naxis1
        shape[0] = wcs.wcs.naxis2
    
    if center is None:
        pixcrd = [[(shape[0]-1)/2.0,(shape[1]-1)/2.0]]
        pixsky = wcs.pix2sky(pixcrd)
        center = [0,0]
        center[0] = pixsky[0,0]
        center[1] = pixsky[0,1]
    else:
        center = [center[0],center[1]]
    
    data = np.zeros(shape=shape, dtype = float)
    
    #rotation angle in rad
    theta = np.pi * rot / 180.0
        
    moffat = lambda x, y: I*(1+(((x-center[1])*np.cos(theta)-(y-center[0])*np.sin(theta))/a)**2 \
                              +(((x-center[1])*np.sin(theta)+(y-center[0])*np.cos(theta))/a/q)**2)**n
    
    if factor>1:
        X,Y = np.meshgrid(xrange(shape[0]*factor),xrange(shape[1]*factor))
        factor = float(factor)
        pixcrd = zip(X.ravel()/factor,Y.ravel()/factor)
        pixsky = wcs.pix2sky(pixcrd)
        data = moffat(pixsky[:,1],pixsky[:,0])
        data = (data.reshape(shape[0],factor,shape[1],factor).sum(1).sum(2)/factor/factor).T
    else:       
        X,Y = np.meshgrid(xrange(shape[0]),xrange(shape[1]))
        pixcrd = zip(X.ravel(),Y.ravel())
        pixsky = wcs.pix2sky(pixcrd)        
        data = moffat(pixsky[:,1],pixsky[:,0])
        data = np.reshape(data,(shape[0],shape[1])).T
            
    return Image(data=data, wcs=wcs)

def make_image(x, y, z, steps, deg=True, limits=None, spline=False, order=3, smooth=0):
    """ interpolates z(x,y) and returns an image.
    
    Parameters
    ----------
    
    x : float array
    Coordinate array corresponding to the declinaison.
    
    y : float arry
    Coordinate array corresponding to the right ascension.
    
    z : float array
    Input data.
    
    steps : (float,float)
    Steps of the output image (dDec,dRa).
    
    deg : boolean
    If True, world coordinates are in decimal degrees (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg')
    If False (by default), world coordinates are linear (CTYPE1=CTYPE2='LINEAR')
    
    limits : (float,float,float,float)
    (dec_min,ra_min,dec_max,ra_max). If None, minum and maximum values of x,y arrays are used.
    
    spline : boolean
    False: bilinear interpolation, True: spline interpolation 
    
    order : integer
    Polynomial order for spline interpolation (default 3)
    
    smooth : float
    Smoothing parameter for spline interpolation (default 0: no smoothing)
    """
    if limits == None:
        x1 = x.min()
        x2 = x.max()
        y1 = y.min()
        y2 = y.max()
    else:
        x1,x2,y1,y2 = limits
    dx,dy = steps
    nx = int((x2-x1)/dx + 1.5)
    ny = int((y2-y1)/dy + 1.5)
    
    wcs = WCS(crpix=(1,1), crval=(x1,y1), cdelt=(dx,dy), deg=deg, shape=(nx,ny))
    
    xi = np.arange(nx)*dx + x1
    yi = np.arange(ny)*dy + y1 
    
    Y,X = np.meshgrid(y,x)
    
    if spline:
        tck = interpolate.bisplrep(X, Y, z, s=smooth, kx=order, ky=order)
        data = interpolate.bisplev(xi, yi, tck)
    else:
        n = np.shape(x)[0]*np.shape(y)[0]
        points = np.zeros((n,2),dtype=float)
        points[:,0] = X.ravel()[:]
        points[:,1] = Y.ravel()[:]
        Yi,Xi = np.meshgrid(yi,xi)
        data = interpolate.griddata(points, z.ravel(), (Xi,Yi), method='linear')
    
    return Image(data=data, wcs=wcs)

def composite_image(ImaColList, mode='lin', cuts=(10,90), bar=False):
    """ builds composite image from a list of image and colors. 
    Returns a PIL RGB image (or 2 PIL images if bar is True).
    
    Parameters
    ----------
    ImaColList : list of tuple (Image,float,float)
    List of images and colors [(Ima, hue, saturation)] 
    
    mode : 'lin' or 'sqrt'
    Intensity mode. Use 'lin' for linear and 'sqrt' for root square.
    
    cut : (float,float)
    Minimum and maximum in percent.
    
    bar : boolean
    If bar is True a color bar image is created.
    
    Example
    -------
    
    imalist = [stars, lowz, highz]
    tab = zip(imalist,linspace(250,0,3),ones(3)*100)
    p1 = composite_image(tab,cuts=(0,99.5),mode='sqrt')
    p1.show()
    p1.save('test_composite.jpg')
    """
    from PIL import Image as PILima
    from PIL import Image, ImageColor, ImageChops
    
    # compute statistic of intensity and derive cuts
    first = True
    for ImaCol in ImaColList:
        ima,col,sat = ImaCol
        if mode == 'lin':
            f = ima.data.filled(0)
        elif mode == 'sqrt':
            f = np.sqrt(np.clip(ima.data.filled(0), 0, 1.e99))
        else:
            raise ValueError, 'Wrong cut mode'
        if first:
            d = f.ravel()
            first = False
        else:
            d = np.concatenate([d, f.ravel()])
    d.sort()
    k1,k2 = cuts
    d1 = d[max(int(0.01*k1*len(d)+0.5),0)]
    d2 = d[min(int(0.01*k2*len(d)+0.5),len(d)-1)]

    # first image
    ima,col,sat = ImaColList[0]
    p1 = PILima.new('RGB', (ima.shape[0],ima.shape[1]))
    if mode == 'lin':
        f = ima.data.filled(0)
    elif mode == 'sqrt':
        f = np.sqrt(np.clip(ima.data.filled(0), 0, 1.e99))
    lum = np.clip((f-d1)*100/(d2 - d1), 0, 100)
    for i in range(ima.shape[0]):
        for j in range(ima.shape[1]):
            p1.putpixel((i,j), ImageColor.getrgb('hsl(%d,%d%%,%d%%)'%(int(col),int(sat),int(lum[i,j]))))
            
    for ImaCol in ImaColList[1:]:
        ima,col,sat = ImaCol
        p2 = PILima.new('RGB', (ima.shape[0],ima.shape[1]))
        if mode == 'lin':
            f = ima.data.filled(0)
        elif mode == 'sqrt':
            f = np.sqrt(np.clip(ima.data.filled(0), 0, 1.e99))
        lum = np.clip((f-d1)*100/(d2 - d1), 0, 100)
        for i in range(ima.shape[0]):
            for j in range(ima.shape[1]):
                p2.putpixel((i,j), ImageColor.getrgb('hsl(%d,%d%%,%d%%)'%(int(col),int(sat),int(lum[i,j]))))
        p1 = ImageChops.add(p1, p2)

    if bar:
        nxb = ima.shape[0]
        nyb = 50
        dx = nxb/len(ImaColList)
        p3 = PILima.new('RGB', (nxb,nyb))
        i1 = 0
        for ImaCol in ImaColList:
            ima,col,sat = ImaCol    
            for i in range(i1,i1+dx):
                for j in range(nyb):
                    p3.putpixel((i,j), ImageColor.getrgb('hsl(%d,%d%%,%d%%)'%(int(col),int(sat),50)))
            i1 += dx

    if bar:
        return p1,p3
    else:
        return p1

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
    
    def __init__(self, filename=None, ext = None, notnoise=False, shape=(101,101,101), wcs = None, wave = None, unit=None, data=None, var=None,fscale=1.0):
        """creates a Cube object

        Parameters
        ----------
        filename : string
        Possible FITS filename

        ext : integer or (integer,integer) or string or (string,sting)
        Number/name of the data extension or numbers/names of the data and variance extensions.

        notnoise: boolean
        True if the noise Variance image is not read (if it exists)
        Use notnoise=True to create image without variance extension

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
                self.cards = hdr.ascard
                self.shape = np.array([hdr['NAXIS3'],hdr['NAXIS2'],hdr['NAXIS1']])
                self.data = np.array(f[0].data, dtype=float)
                self.var = None
                self.fscale = hdr.get('FSCALE', 1.0)
                # WCS object from data header
                try:
                    self.wcs = WCS(hdr)
                except:
                    print "error: wcs not copied."
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
                    if isinstance(ext,int) or isinstance(ext,str):
                        n = ext
                    else:
                        n = ext[0]
                    h = f[n].header
                    d = np.array(f[n].data, dtype=float)
                if h['NAXIS'] != 3:
                    raise IOError, 'Wrong dimension number in DATA extension'
                self.unit = h.get('BUNIT', None)
                self.cards = h.ascard
                self.shape= np.array([h['NAXIS3'],h['NAXIS2'],h['NAXIS1']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                try:
                    self.wcs = WCS(h) # WCS object from data header
                except:
                    print "error: wcs not copied."
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
                if not notnoise:
                    try:
                        if ext is None:
                            fstat = f['STAT']
                        else:
                            n = ext[1]
                            fstat = f[n]
                        if fstat.header['NAXIS'] != 3:
                            raise IOError, 'Wrong dimension number in variance extension'
                        if fstat.header['NAXIS1'] != self.shape[2] and fstat.header['NAXIS2'] != self.shape[1] and fstat.header['NAXIS3'] != self.shape[0]:
                            raise IOError, 'Number of points in STAT not equal to DATA'
                        self.var = np.array(fstat.data, dtype=float)
                    except:
                        self.var = None
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
            if is_int(shape):
                shape = (shape,shape,shape)
            elif len(shape) == 2:
                shape = (shape[0],shape[1],shape[1])
            elif len(shape) == 3:
                pass
            else:
                raise ValueError, 'dim with dimension > 3'
            if data is None:
                self.data = None
                self.shape = np.array(shape)
            else:
                self.data = np.array(data, dtype = float)
                try:
                    self.shape = np.array(data.shape)
                except:
                    self.shape = np.array(shape)

            if notnoise or var is None:
                self.var = None
            else:
                self.var = np.array(var, dtype = float)
            self.fscale = np.float(fscale)
            try:
                self.wcs = wcs
                if wcs is not None:
                    self.wcs.wcs.naxis1 = self.shape[2]
                    self.wcs.wcs.naxis2 = self.shape[1]
                    if wcs.wcs.naxis1 !=0 and wcs.wcs.naxis2 != 0 and ( wcs.wcs.naxis1 != self.shape[2] or wcs.wcs.naxis2 != self.shape[1]):
                        print "warning: world coordinates and data have not the same dimensions."
            except :
                self.wcs = None
                print "error: world coordinates not copied."
            try:
                self.wave = wave
                if wave is not None:
                    if wave.shape is not None and wave.shape != self.shape[0]:
                        print "warning: wavelength coordinates and data have not the same dimensions."
                    self.wave.shape = self.shape[0]
            except :
                self.wave = None
                print "error: wavelength solution not copied."
        #Mask an array where invalid values occur (NaNs or infs).
        if self.data is not None:
            self.data = np.ma.masked_invalid(self.data)

    def copy(self):
        """copies Cube object in a new one and returns it
        """
        cub = Cube()
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

        #world coordinates
        wcs_cards = self.wcs.to_header().ascard

        if self.var is None: # write simple fits file without extension
            prihdu.data = self.data.data
            if self.cards is not None:
                for card in self.cards:
                    try:
                        prihdu.header.update(card.key, card.value, card.comment)
                    except:
                        pass
            prihdu.header.update('date', str(datetime.datetime.now()), 'creation date')
            prihdu.header.update('author', 'MPDAF', 'origin of the file')
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
            if self.cards is not None:
                for card in self.cards:
                    try:
                        tbhdu.header.update(card.key, card.value, card.comment)
                    except:
                        pass
            tbhdu.header.update('date', str(datetime.datetime.now()), 'creation date')
            tbhdu.header.update('author', 'MPDAF', 'origin of the file')
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
                data = self.data[item]
                if is_int(item[0]):
                    if is_int(item[1]):
                        shape = (1,1,data.shape[0])
                    elif is_int(item[2]):
                        shape = (1,data.shape[0],1)
                    else:
                        shape = (1,data.shape[0],data.shape[1])
                elif is_int(item[1]):
                    if is_int(item[2]):
                        shape = (data.shape[0],1,1)
                    else:
                        shape = (data.shape[0],1,data.shape[1])
                elif is_int(item[2]):
                        shape = (data.shape[0],data.shape[1],1)
                else:
                    shape = data.shape
                if self.var is not None:
                    var = self.var[item]
                try:
                    wcs = self.wcs[item[1],item[2]]
                except:
                    wcs = None
                    print "error: wcs not copied."
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                    print "error: wavelength solution not copied."
                res = Cube(shape=shape, wcs=wcs, wave=wave, unit=self.unit, data=None, var=None,fscale=self.fscale)
                res.data = data
                if self.var is not None:
                    res.var = var
                return res
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
        if is_float(other) or is_int(other):
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
                    res = Cube(shape=self.shape , fscale=self.fscale)
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
                        res = Cube(shape=self.shape , wave= self.wave, fscale=self.fscale)
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
                            res = Cube(shape=self.shape , wcs= self.wcs, fscale=self.fscale)
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
        if is_float(other) or is_int(other):
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
                    res = Cube(shape=self.shape , fscale=self.fscale)
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
                        res = Cube(shape=self.shape , wave= self.wave, fscale=self.fscale)
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
                            res = Cube(shape=self.shape , wcs= self.wcs, fscale=self.fscale)
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
        if is_float(other) or is_int(other):
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
        if is_float(other) or is_int(other):
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
                    res = Cube(shape=self.shape , fscale=self.fscale*other.fscale)
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
                        res = Cube(shape=self.shape , wave= self.wave, fscale=self.fscale * other.fscale)
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
                            res = Cube(shape=self.shape , wcs= self.wcs, fscale=self.fscale*other.fscale)
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
        if is_float(other) or is_int(other):
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
                    res = Cube(shape=self.shape , fscale=self.fscale/other.fscale)
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
                        res = Cube(shape=self.shape , wave= self.wave, fscale=self.fscale / other.fscale)
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
                            res = Cube(shape=self.shape , wcs= self.wcs, fscale=self.fscale/other.fscale)
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
        if is_float(other) or is_int(other):
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
        if is_float(other) or is_int(other):
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
            if is_int(item[0]):
                if is_int(item[1]) and is_int(item[2]):
                    #return a float
                    return data
                else:
                    #return an image
                    if is_int(item[1]):
                        shape = (1,data.shape[0])
                    elif is_int(item[2]):
                        shape = (data.shape[0],1)
                    else:
                        shape = data.shape
                    var = None
                    if self.var is not None:
                        var = self.var[item]
                    try:
                        wcs = self.wcs[item[1],item[2]]
                    except:
                        wcs = None
                    res = Image(shape=shape, wcs = wcs, unit=self.unit, fscale=self.fscale)
                    res.data = data
                    res.var =var
                    return res
            elif is_int(item[1]) and is_int(item[2]):
                #return a spectrum
                shape = data.shape[0]
                var = None
                if self.var is not None:
                    var = self.var[item]
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Spectrum(shape=shape, wave = wave, unit=self.unit, fscale=self.fscale)
                res.data = data
                res.var = var
                return res
            else:
                #return a cube
                if is_int(item[1]):
                    shape = (data.shape[0],1,data.shape[1])
                elif is_int(item[2]):
                    shape = (data.shape[0],data.shape[1],1)
                else:
                    shape = data.shape
                var = None
                if self.var is not None:
                    var = self.var[item]
                try:
                    wcs = self.wcs[item[1],item[2]]
                except:
                    wcs = None
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Cube(shape=shape, wcs = wcs, wave = wave, unit=self.unit, fscale=self.fscale)
                res.data = data
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
            pix_min = max(0,int(self.wave.pixel(lbda_min)))
            pix_max = min(self.shape[0],int(self.wave.pixel(lbda_max)) + 1)
            if (pix_min+1)==pix_max:
                return self.data[pix_min,:,:]
            else:
                return self[pix_min:pix_max,:,:]
            
    def get_step(self):
        """ returns the cube steps [dLambda,dDec,dRa]
        """
        step = np.zeros(3)
        step[0] = self.wave.cdelt
        step[1:] = self.wcs.get_step()
        return step
    
    def get_range(self):
        """returns [ [lambda_min,dec_min,ra_min], [lambda_max,dec_max,ra_max] ]
        """
        range = np.zeros((2,3))
        range[:,0] = self.wave.get_range()
        range[:,1:] = self.wcs.get_range()
        return range
    
    def get_start(self):
        """returns [lambda,dec,ra] corresponding to pixel (0,0,0)
        """
        start = np.zeros(3)
        start[0] = self.wave.get_start()
        start[1:] = self.wcs.get_start()
        return start
    
    def get_end(self):
        """returns [lambda,dec,ra] corresponding to pixel (-1,-1,-1)
        """
        end = np.zeros(3)
        end[0] = self.wave.get_end()
        end[1:] = self.wcs.get_end()
        return end
    
    def get_rot(self):
        """returns the rotation angle
        """
        return self.wcs.get_rot()
        
            
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
        self.wcs = wcs
        self.wcs.wcs.naxis1 = self.shape[2]
        self.wcs.wcs.naxis2 = self.shape[1]
        if wcs.wcs.naxis1 !=0 and wcs.wcs.naxis2 != 0 and (wcs.wcs.naxis1 != self.shape[2] or wcs.wcs.naxis2 != self.shape[1]):
            print "warning: world coordinates and data have not the same dimensions."
        
        if wave.shape is not None and wave.shape != self.shape[0]:
            print "warning: wavelength coordinates and data have not the same dimensions."
        self.wave = wave
        self.wave.shape = self.shape[0]
            
    def set_var(self,var):
        """sets the variance array
        
        Parameter
        ---------
        var : float array
        Input variance array. If None, variance is set with zeros
        """
        if var is None:
            self.var = np.zeros((self.shape[0],self.shape[1], self.shape[2]))
        else:
            if self.shape[0] == np.shape(var)[0] and self.shape[1] == np.shape(var)[1] and self.shape[2] == np.shape(var)[2]:
                self.var = var
            else:
                raise ValueError, 'var and data have not the same dimensions.'
            
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
            res = Image(notnoise=True, shape=data.shape, wcs = self.wcs, unit=self.unit, fscale=self.fscale)
            res.data = data
            return res
        elif axis==tuple([1,2]):
            #return a spectrum
            data = self.data.sum(axis=1).sum(axis=1)
            res = Spectrum(notnoise=True, shape=data.shape[0], wave = self.wave, unit=self.unit, fscale=self.fscale)
            res.data = data
            return res
        else:
            return None
        
    def mean(self,axis=None):
        """ Returns the mean over the given axis.
        axis = None returns a float
        axis = 0 returns an image
        axis = (1,2) returns a spectrum
        Other cases return None.
        """
        if axis is None:
            return self.data.mean()    
        elif axis==0:
            #return an image
            data = self.data.mean(axis)
            res = Image(notnoise=True, shape=data.shape, wcs = self.wcs, unit=self.unit, fscale=self.fscale)
            res.data = data
            return res
        elif axis==tuple([1,2]):
            #return a spectrum
            data = self.data.mean(axis=1).mean(axis=1)
            res = Spectrum(notnoise=True, shape=data.shape[0], wave = self.wave, unit=self.unit, fscale=self.fscale)
            res.data = data
            return res
        else:
            return None
