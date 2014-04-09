""" spectrum.py defines Spectrum objects."""

import numpy as np
try:
    from astropy.io import fits as pyfits
except:
    import pyfits
import datetime
import types

from scipy import integrate
from scipy import interpolate
from scipy.optimize import leastsq
from scipy import signal
from scipy import special

import matplotlib.pyplot as plt

from coords import WaveCoord
from objs import is_float
from objs import is_int
from objs import flux2mag
import ABmag_filters

import warnings

import logging
FORMAT = "WARNING mpdaf corelib %(class)s.%(method)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('mpdaf corelib')


class SpectrumClicks:  # Object used to save click on spectrum plot.

    def __init__(self, binding_id, filename=None):
        self.filename = filename  # Name of the table fits file where are
                                  # saved the clicks values.
        self.binding_id = binding_id  # Connection id.
        self.xc = []  # Cursor position in spectrum (world coordinates).
        self.yc = []  # Cursor position in spectrum (world coordinates).
        self.k = []  # Nearest pixel in spectrum.
        self.lbda = []  # Corresponding nearest position in spectrum
                        # (world coordinates)
        self.data = []  # Corresponding spectrum data value.
        self.id_lines = []  # Plot id (cross for cursor positions).

    def remove(self, xc):
        # removes a cursor position
        i = np.argmin(np.abs(self.xc - xc))
        line = self.id_lines[i]
        del plt.gca().lines[line]
        self.xc.pop(i)
        self.yc.pop(i)
        self.k.pop(i)
        self.lbda.pop(i)
        self.data.pop(i)
        self.id_lines.pop(i)
        for j in range(i, len(self.id_lines)):
            self.id_lines[j] -= 1
        plt.draw()

    def add(self, xc, yc, i, x, data):
        plt.plot(xc, yc, 'r+')
        self.xc.append(xc)
        self.yc.append(yc)
        self.k.append(i)
        self.lbda.append(x)
        self.data.append(data)
        self.id_lines.append(len(plt.gca().lines) - 1)

    def iprint(self, i, fscale):
        # prints a cursor positions
        if fscale == 1:
            print 'xc=%g\tyc=%g\tk=%d\tlbda=%g\tdata=%g' \
            % (self.xc[i], self.yc[i], self.k[i], self.lbda[i], self.data[i])
        else:
            print 'xc=%g\tyc=%g\tk=%d\tlbda=%g\tdata=%g\t[scaled=%g]' \
            % (self.xc[i], self.yc[i], self.k[i], self.lbda[i], \
               self.data[i], self.data[i] / fscale)

    def write_fits(self):
        # prints coordinates in fits table.
        if self.filename != 'None':
            c1 = pyfits.Column(name='xc', format='E', array=self.xc)
            c2 = pyfits.Column(name='yc', format='E', array=self.yc)
            c3 = pyfits.Column(name='k', format='I', array=self.k)
            c4 = pyfits.Column(name='lbda', format='E', array=self.lbda)
            c5 = pyfits.Column(name='data', format='E', array=self.data)
            #tbhdu = pyfits.new_table(pyfits.ColDefs([c1, c2, c3, c4, c5]))
            coltab = pyfits.ColDefs([c1, c2, c3, c4, c5])
            tbhdu = pyfits.TableHDU(pyfits.FITS_rec.from_columns(coltab))
            tbhdu.writeto(self.filename, clobber=True)
            print 'printing coordinates in fits table %s' % self.filename

    def clear(self):
        # disconnects and clears
        print "disconnecting console coordinate printout..."
        plt.disconnect(self.binding_id)
        nlines = len(self.id_lines)
        for i in range(nlines):
            line = self.id_lines[nlines - i - 1]
            del plt.gca().lines[line]
        plt.draw()


class Gauss1D:
    """ This class stores 1D Gaussian parameters.

    Attributes
    ----------

    cont (float) : Continuum value.

    fwhm (float) : Gaussian fwhm.

    lpeak (float) : Gaussian center.

    peak (float) : Gaussian peak value.

    flux (float) : Gaussian integrated flux.

    err_fwhm (float) : Estimated error on Gaussian fwhm.

    err_lpeak (float) : Estimated error on Gaussian center.

    err_peak (float) : Estimated error on Gaussian peak value.

    flux (float) : Gaussian integrated flux.
    """
    def __init__(self, lpeak, peak, flux, fwhm, cont, err_lpeak, \
                 err_peak, err_flux, err_fwhm):
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
        """Copies Gauss1D object in a new one and returns it.
        """
        res = Gauss1D(self.lpeak, self.peak, self.flux, self.fwhm, \
                      self.cont, self.err_lpeak, self.err_peak, \
                      self.err_flux, self.err_fwhm)
        return res

    def print_param(self):
        """Prints Gaussian parameters.
        """
        print 'Gaussian center = %g (error:%g)' % (self.lpeak, self.err_lpeak)
        print 'Gaussian integrated flux = %g (error:%g)' % \
        (self.flux, self.err_flux)
        print 'Gaussian peak value = %g (error:%g)' % \
        (self.peak, self.err_peak)
        print 'Gaussian fwhm = %g (error:%g)' % (self.fwhm, self.err_fwhm)
        print 'Gaussian continuum = %g' % self.cont
        print ''


class Spectrum(object):
    """Spectrum class manages spectrum, optionally including a
    variance and a bad pixel mask.

    :param filename: Possible FITS filename.
    :type filename: string
    :param ext: Number/name of the data extension or numbers/names
    of the data and variance extensions.
    :type ext: integer or (integer,integer) or string or (string,string)
    :param notnoise: True if the noise Variance spectrum is not read
    (if it exists).

           Use notnoise=True to create spectrum without variance extension.
    :type notnoise: bool
    :param shape: size of the spectrum. 101 by default.
    :type shape: integer
    :param wave: Wavelength coordinates.
    :type wave: :class:`mpdaf.obj.WaveCoord`
    :param unit: Possible data unit type. None by default.
    :type unit: string
    :param data: Array containing the pixel values of the spectrum.
    None by default.
    :type data: float array
    :param var: Array containing the variance. None by default.
    :type var: float array
    :param fscale: Flux scaling factor (1 by default).
    :type fscale: float

    Attributes
    ----------

    filename (string) : Possible FITS filename.

    unit (string) : Possible data unit type.

    primary_header (pyfits.Header) : Possible FITS primary header instance.

    data_header (pyfits.Header) : Possible FITS data header instance.

    data (masked array) : Array containing the pixel values of the spectrum.

    shape (integer) : Size of spectrum.

    var (array) : Array containing the variance.

    fscale (float) : Flux scaling factor (1 by default).

    wave (:class:`mpdaf.obj.WaveCoord`) : Wavelength coordinates.
    """

    def __init__(self, filename=None, ext=None, notnoise=False, shape=101, \
                 wave=None, pix=False, unit=None, data=None, var=None, \
                 fscale=1.0):
        """Creates a Spectrum object.

        :param filename: Possible FITS filename.
        :type filename: string
        :param ext: Number/name of the data extension or numbers/names
        of the data and variance extensions.
        :type ext: integer or (integer,integer) or string or (string,string)
        :param notnoise: True if the noise Variance spectrum is not read
        (if it exists).

           Use notnoise=True to create spectrum without variance extension.
        :type notnoise: bool
        :param shape: size of the spectrum. 101 by default.
        :type shape: integer
        :param wave: Wavelength coordinates.
        :type wave: :class:`mpdaf.obj.WaveCoord`
        :param unit: Possible data unit type. None by default.
        :type unit: string
        :param data: Array containing the pixel values of the spectrum.
        None by default.
        :type data: float array
        :param var: Array containing the variance. None by default.
        :type var: float array
        :param fscale: Flux scaling factor (1 by default).
        :type fscale: float
        """
        self._clicks = None
        self.spectrum = True
        # possible FITS filename
        self.filename = filename
        if filename is not None:
            f = pyfits.open(filename)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                self.primary_header = pyfits.Header()
                # if the number of extension is 1,
                # we just read the data from the primary header
                # test if spectrum
                if hdr['NAXIS'] != 1:
                    raise IOError('Wrong dimension number: not a spectrum')
                self.unit = hdr.get('BUNIT', None)
                #self.data_header = hdr.ascard
                self.data_header = hdr
                self.shape = hdr['NAXIS1']
                self.data = np.array(f[0].data, dtype=float)
                self.var = None
                self.fscale = hdr.get('FSCALE', 1.0)
                if wave is None:
                    if 'CDELT1' in hdr:
                        cdelt = hdr.get('CDELT1')
                    elif 'CD1_1' in hdr:
                        cdelt = hdr.get('CD1_1')
                    else:
                        cdelt = 1.0
                    crpix = hdr.get('CRPIX1')
                    crval = hdr.get('CRVAL1')
                    cunit = hdr.get('CUNIT1', '')
                    self.wave = WaveCoord(crpix, cdelt, crval, \
                                          cunit, self.shape)
                else:
                    self.wave = wave
                    if wave.shape is not None and wave.shape != self.shape:
                        d = {'class': 'Spectrum', 'method': '__init__'}
                        logger.warning('wavelength coordinates and data '\
                                       'have not the same dimension: %s', \
                                       'shape of WaveCoord object'\
                                       ' is modified', extra=d)
                    self.wave.shape = self.shape
            else:
                self.primary_header = hdr
                if ext is None:
                    h = f['DATA'].header
                    d = np.array(f['DATA'].data, dtype=float)
                else:
                    if is_int(ext) or isinstance(ext, str):
                        n = ext
                    else:
                        n = ext[0]
                    h = f[n].header
                    d = np.array(f[n].data, dtype=float)

                if h['NAXIS'] != 1:
                    raise IOError('Wrong dimension number: not a spectrum')
                self.unit = h.get('BUNIT', None)
                self.data_header = h
                self.shape = h['NAXIS1']
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                if wave is None:
                    if 'CDELT1' in h:
                        cdelt = h.get('CDELT1')
                    elif 'CD1_1' in h:
                        cdelt = h.get('CD1_1')
                    else:
                        cdelt = 1.0
                    crpix = h.get('CRPIX1')
                    crval = h.get('CRVAL1')
                    cunit = h.get('CUNIT1', '')
                    self.wave = WaveCoord(crpix, cdelt, crval, \
                                          cunit, self.shape)
                else:
                    self.wave = wave
                    if wave.shape is not None and wave.shape != self.shape:
                        d = {'class': 'Spectrum', 'method': '__init__'}
                        logger.warning('wavelength coordinates and data '\
                                       'have not the same dimension: %s', \
                                       'shape of WaveCoord object '\
                                       'is modified', extra=d)
                    self.wave.shape = self.shape
                # STAT extension
                self.var = None
                if not notnoise:
                    if ext is None:
                        try:
                            fstat = f['STAT']
                        except:
                            fstat = None
                    else:
                        try:
                            n = ext[1]
                            fstat = f[n]
                        except:
                            fstat = None
                    if fstat is None:
                        self.var = None
                    else:
                        if fstat.header['NAXIS'] != 1:
                            raise IOError('Wrong dimension number '\
                                          'in STAT extension')
                            if fstat.header['NAXIS1'] != self.shape:
                                raise IOError('Number of points in STAT '\
                                              'not equal to DATA')
                        self.var = np.array(fstat.data, dtype=float)
                # DQ extension
                try:
                    mask = np.ma.make_mask(f['DQ'].data)
                    self.data = np.ma.array(self.data, mask=mask)
                except:
                    pass
            f.close()
        else:
            # possible data unit type
            self.unit = unit
            # possible FITS header instance
            self.data_header = pyfits.Header()
            self.primary_header = pyfits.Header()
            # data
            if data is None:
                self.data = None
                self.shape = shape
            else:
                self.data = np.array(data, dtype=float)
                self.shape = data.shape[0]

            if notnoise or var is None:
                self.var = None
            else:
                self.var = np.array(var, dtype=float)
            self.fscale = np.float(fscale)
            try:
                self.wave = wave
                if wave is not None:
                    if wave.shape is not None and wave.shape != self.shape:
                        d = {'class': 'Spectrum', 'method': '__init__'}
                        logger.warning('wavelength coordinates and data '\
                                       'have not the same dimension: %s', \
                                       'shape of WaveCoord object '\
                                       'is modified', extra=d)
                    self.wave.shape = self.shape
            except:
                self.wave = None
                d = {'class': 'Spectrum', 'method': '__init__'}
                logger.warning("wavelength solution not copied", extra=d)
        # Mask an array where invalid values occur (NaNs or infs).
        if self.data is not None:
            self.data = np.ma.masked_invalid(self.data)

    def copy(self):
        """Returns a new copy of a Spectrum object.

        :rtype: Spectrum object.
        """
        spe = Spectrum()
        spe.filename = self.filename
        spe.unit = self.unit
        spe.data_header = pyfits.Header(self.data_header)
        spe.primary_header = pyfits.Header(self.primary_header)
        spe.shape = self.shape
        try:
            spe.data = self.data.copy()
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

    def clone(self, var=False):
        """Returns a new spectrum of the same shape and coordinates,
        filled with zeros.

        :param var: Presence of the variance extension.
        :type var: bool
        """
        try:
            wave = self.wave.copy()
        except:
            wave = None
        if var is False:
            spe = Spectrum(wave=wave, data=np.zeros(shape=self.shape), \
                           unit=self.unit)
        else:
            spe = Spectrum(wave=wave, data=np.zeros(shape=self.shape), \
                           var=np.zeros(shape=self.shape), unit=self.unit)
        return spe

    def write(self, filename):
        """ Saves the object in a FITS file.

          :param filename: The FITS filename.
          :type filename: string
        """

        assert self.data is not None
        # create primary header
        prihdu = pyfits.PrimaryHDU()
        for card in self.primary_header.cards:
            try:
                prihdu.header[card.keyword] = (card.value, card.comment)
            except:
                try:
                    card.verify('fix')
                    prihdu.header[card.keyword] =  (card.value, card.comment)
                except:
                    try:
                        if isinstance(card.value, str):
                            n = 80 - len(card.keyword) - 14
                            s = card.value[0:n]
                            prihdu.header['hierarch %s' % card.keyword] = \
                                                 (s, card.comment)
                        else:
                            prihdu.header['hierarch %s' % card.keyword] = \
                                                 (card.value, card.comment)
                    except:
                        d = {'class': 'Spectrum', 'method': 'write'}
                        logger.warning("%s not copied in primary header", \
                                       card.keyword, extra=d)
                        pass
        prihdu.header['date'] = (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        hdulist = [prihdu]

        # create spectrum DATA extension
        tbhdu = pyfits.ImageHDU(name='DATA', data=self.data.data)
        for card in self.data_header.cards:
            try:
                if tbhdu.header.keys().count(card.keyword) == 0:
                    tbhdu.header[card.keyword] = (card.value, card.comment)
            except:
                try:
                    card.verify('fix')
                    if tbhdu.header.keys().count(card.keyword) == 0:
                        tbhdu.header[card.keyword] = \
                        (card.value, card.comment)
                except:
                    d = {'class': 'Spectrum', 'method': 'write'}
                    logger.warning("%s not copied in data header",\
                                    card.keyword, extra=d)
                    pass
        tbhdu.header['CRVAL1'] = \
        (self.wave.crval, 'Start in world coordinate')
        tbhdu.header['CRPIX1'] = (self.wave.crpix, 'Start in pixel')
        tbhdu.header['CDELT1'] = (self.wave.cdelt, 'Step in world coordinate')
        tbhdu.header['CTYPE1'] = ('LINEAR', 'world coordinate type')
        tbhdu.header['CUNIT1'] = (self.wave.cunit, 'world coordinate units')
        if self.unit is not None:
            tbhdu.header['BUNIT'] = (self.unit, 'data unit type')
        tbhdu.header['FSCALE'] = (self.fscale, 'Flux scaling factor')
        hdulist.append(tbhdu)

        # create spectrum STAT extension
        if self.var is not None:
            nbhdu = pyfits.ImageHDU(name='STAT', data=self.var)
            nbhdu.header['CRVAL1'] = \
            (self.wave.crval, 'Start in world coordinate')
            nbhdu.header['CRPIX1'] = (self.wave.crpix, 'Start in pixel')
            nbhdu.header['CDELT1'] = \
            (self.wave.cdelt, 'Step in world coordinate')
            nbhdu.header['CUNIT1'] = \
            (self.wave.cunit, 'world coordinate units')
            hdulist.append(nbhdu)

        # create spectrum DQ extension
        if np.ma.count_masked(self.data) != 0:
            dqhdu = pyfits.ImageHDU(name='DQ', data=np.uint8(self.data.mask))
            dqhdu.header['CRVAL1'] = \
            (self.wave.crval, 'Start in world coordinate')
            dqhdu.header['CRPIX1'] = (self.wave.crpix, 'Start in pixel')
            dqhdu.header['CDELT1'] = \
            (self.wave.cdelt, 'Step in world coordinate')
            dqhdu.header['CUNIT1'] = \
            (self.wave.cunit, 'world coordinate units')
            hdulist.append(dqhdu)

        # save to disk
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(filename, clobber=True, output_verify='fix')

        self.filename = filename

    def info(self):
        """Prints information.
        """
        if self.filename != None:
            print 'spectrum of %i elements (%s)' % (self.shape, self.filename)
        else:
            print 'spectrum of %i elements (no name)' % self.shape
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
        print '%s (%s) fscale=%g, %s' % (data, unit, self.fscale, noise)
        if self.wave is None:
            print 'No wavelength solution'
        else:
            self.wave.info()

    def __le__(self, item):
        """Masks data array where greater than a given value (operator <=).

          :param x: minimum value.
          :type x: float
          :rtype: Spectrum object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater(self.data, item / self.fscale)
        return result

    def __lt__(self, item):
        """Masks data array where greater or equal than a given value
        (operator <).

          :param x: minimum value.
          :type x: float
          :rtype: Spectrum object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data, item \
                                                     / self.fscale)
        return result

    def __ge__(self, item):
        """Masks data array where less than a given value (operator >=).

          :param x: maximum value.
          :type x: float
          :rtype: Spectrum object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item / self.fscale)
        return result

    def __gt__(self, item):
        """Masks data array where less or equal than a given value
        (operator >).

          :param x: maximum value.
          :type x: float
          :rtype: Spectrum object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data, item \
                                                  / self.fscale)
        return result

    def resize(self):
        """Resizes the spectrum to have a minimum number of masked values.
        """
        if np.ma.count_masked(self.data) != 0:
            ksel = np.where(self.data.mask == False)
            try:
                item = slice(ksel[0][0], ksel[0][-1] + 1, None)
                self.data = self.data[item]
                self.shape = self.data.shape[0]
                if self.var is not None:
                    self.var = self.var[item]
                try:
                    self.wave = self.wave[item]
                except:
                    wave = None
                    d = {'class': 'Spectrum', 'method': 'resize'}
                    logger.warning("wavelength solution not copied", extra=d)
            except:
                pass

    def __add__(self, other):
        """ Operator +.

              :param x:
                  x is Spectrum : Dimensions and wavelength
                                  coordinates must be the same.
                  x is Cube : The first dimension of cube1 must be
                              equal to the spectrum dimension.
                              Wavelength coordinates must be the same.
              :type x: number or Spectrum or Cube object.
              :rtype: Spectrum or Cube object.

              spectrum1 + number = spectrum2
              (spectrum2[k] = spectrum1[k] + number)

              spectrum1 + spectrum2 = spectrum3
              (spectrum3[k] = spectrum1[k] + spectrum2[k])

              spectrum + cube1 = cube2
              (cube2[k,p,q] = cube1[k,p,q] + spectrum[k])
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # spectrum1 + number = spectrum2 (spectrum2[k]=spectrum1[k]+number)
            res = self.copy()
            res.data = self.data + (other / np.double(self.fscale))
            return res
        try:
            # spectrum1 + spectrum2 = spectrum3
            # (spectrum3[k]=spectrum1[k]+spectrum2[k])
            # Dimension must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '\
                                  'with different sizes')
                else:
                    res = Spectrum(shape=self.shape, fscale=self.fscale)
                    # coordinates
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        raise IOError('Operation forbidden for spectra '\
                                      'with different world coordinates')
                    # data
                    res.data = self.data + other.data \
                    * np.double(other.fscale / self.fscale)
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var \
                        * np.double(other.fscale * other.fscale \
                                    / self.fscale / self.fscale)
                    elif other.var is None:
                        res.var = self.var
                    else:
                        res.var = self.var + other.var \
                        * np.double(other.fscale * other.fscale \
                                    / self.fscale / self.fscale)
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    #return
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # spectrum + cube1 = cube2
                # (cube2[k,j,i]=cube1[k,j,i]+spectrum[k])
                # The last dimension of cube1 must be equal
                # to the spectrum dimension.
                # If not equal to None, world coordinates
                # in spectral direction must be the same.
                if other.cube:
                    res = other.__add__(self)
                    return res
            except IOError as e:
                raise e
            except:
                raise IOError('Operation forbidden')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """ Operator -.

            :param x:   x is Spectrum : Dimensions and wavelength coordinates
                                        must be the same.
                        x is Cube : The first dimension of cube1 must be equal
                                    to the spectrum dimension.
                                    Wavelength coordinates must be the same.
            :type x: number or Spectrum or Cube object.
            :rtype: Spectrum or Cube object.

            spectrum1 - number = spectrum2
            (spectrum2[k] = spectrum1[k] - number)

            spectrum1 - spectrum2 = spectrum3
            (spectrum3[k] = spectrum1[k] - spectrum2[k])

            spectrum - cube1 = cube2
            (cube2[k,p,q] = spectrum[k] - cube1[k,p,q])
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # spectrum1 - number = spectrum2
            # (spectrum2[k]=spectrum1[k]-number)
            res = self.copy()
            res.data = self.data - (other / np.double(self.fscale))
            return res
        try:
            # spectrum1 + spectrum2 = spectrum3
            # (spectrum3[k]=spectrum1[k]-spectrum2[k])
            # Dimension must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '\
                                  'with different sizes')
                else:
                    res = Spectrum(shape=self.shape, fscale=self.fscale)
                    # coordinates
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        raise IOError('Operation forbidden for spectra '\
                                      'with different world coordinates')
                    # data
                    res.data = self.data \
                    - (other.data * np.double(other.fscale / self.fscale))
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var * np.double(other.fscale \
                                                        * other.fscale \
                                                        / self.fscale \
                                                        / self.fscale)
                    elif other.var is None:
                        res.var = self.var
                    else:
                        res.var = self.var + other.var \
                        * np.double(other.fscale * other.fscale \
                                    / self.fscale / self.fscale)
                    #unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # spectrum - cube1 = cube2
                # (cube2[k,j,i]=spectrum[k]-cube1[k,j,i])
                # The last dimension of cube1 must be equal
                # to the spectrum dimension.
                # If not equal to None, world coordinates
                # in spectral direction must be the same.
                if other.cube:
                    if other.data is None or self.shape != other.shape[0]:
                        raise IOError('Operation forbidden for objects'\
                                      ' with different sizes')
                    else:
                        from cube import Cube
                        res = Cube(shape=other.shape, wcs=other.wcs, \
                                   fscale=self.fscale)
                        # coordinates
                        if self.wave is None or other.wave is None:
                            res.wave = None
                        elif self.wave.isEqual(other.wave):
                            res.wave = self.wave
                        else:
                            raise IOError('Operation forbidden for spectra '\
                                          'with different world coordinates')
                        # data
                        res.data = self.data[:, np.newaxis, np.newaxis] \
                        - (other.data * np.double(other.fscale / self.fscale))
                        # variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = other.var \
                            * np.double(other.fscale * other.fscale \
                                        / self.fscale / self.fscale)
                        elif other.var is None:
                            res.var = np.ones(res.shape) \
                            * self.var[:, np.newaxis, np.newaxis]
                        else:
                            res.var = self.var[:, np.newaxis, np.newaxis] \
                            + other.var \
                            * np.double(other.fscale * other.fscale \
                                        / self.fscale / self.fscale)
                        # unit
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except IOError as e:
                raise e
            except:
                raise IOError('Operation forbidden')

    def __rsub__(self, other):
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            res = self.copy()
            res.data = (other / np.double(self.fscale)) - self.data
            return res
        try:
            if other.spectrum:
                return other.__sub__(self)
        except IOError as e:
            raise e
        except:
            try:
                if other.cube:
                    return other.__sub__(self)
            except IOError as e:
                raise e
            except:
                raise IOError('Operation forbidden')

    def __mul__(self, other):
        """ Operator \*.

          :param x: x is Spectrum : Dimensions and wavelength coordinates
                                    must be the same.
                    x is Cube : The first dimension of cube1 must be equal
                                to the spectrum dimension.
                                Wavelength coordinates must be the same.
          :type x: number or Spectrum or Image or Cube object.
          :rtype: Spectrum or Cube object.

          spectrum1 \* number = spectrum2
          (spectrum2[k] = spectrum1[k] \* number)

          spectrum1 \* spectrum2 = spectrum3
          (spectrum3[k] = spectrum1[k] \* spectrum2[k])

          spectrum \* cube1 = cube2
          (cube2[k,p,q] = spectrum[k] \* cube1[k,p,q])

          spectrum \* image = cube (cube[k,p,q]=image[p,q] \* spectrum[k]
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # spectrum1 * number = spectrum2
            # (spectrum2[k]=spectrum1[k]*number)
            res = self.copy()
            res.fscale *= other
            return res
        try:
            # spectrum1 * spectrum2 = spectrum3
            # (spectrum3[k]=spectrum1[k]*spectrum2[k])
            # Dimension must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '\
                                  'with different sizes')
                else:
                    res = Spectrum(shape=self.shape, \
                                   fscale=self.fscale * other.fscale)
                    # coordinates
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        raise IOError('Operation forbidden for spectra '\
                                      'with different world coordinates')
                    # data
                    res.data = self.data * other.data
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var * self.data * self.data
                    elif other.var is None:
                        res.var = self.var * other.data * other.data
                    else:
                        res.var = other.var * self.data * self.data \
                        + self.var * other.data * other.data
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    # return
                    return res
        except IOError as e:
            raise e
        except:
            try:
                res = other.__mul__(self)
                return res
            except IOError as e:
                raise e
            except:
                raise IOError('Operation forbidden')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """ Operator /.

          Note : divide functions that have a validity domain returns
          the masked constant whenever the input is masked or falls
          outside the validity domain.

          :param x: x is Spectrum : Dimensions and wavelength coordinates
                                    must be the same.
                    x is Cube : The first dimension of cube1 must be equal
                                to the spectrum dimension.
                                Wavelength coordinates must be the same.
          :type x: number or Spectrum or Cube object.
          :rtype: Spectrum or Cube object.

          spectrum1 / number = spectrum2
          (spectrum2[k] = spectrum1[k] / number)

          spectrum1 / spectrum2 = spectrum3
          (spectrum3[k] = spectrum1[k] / spectrum2[k])

          spectrum / cube1 = cube2
          (cube2[k,p,q] = spectrum[k] / cube1[k,p,q])
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # spectrum1 / number =
            # spectrum2 (spectrum2[k]=spectrum1[k]/number)
            res = self.copy()
            res.fscale /= other
            return res
        try:
            # spectrum1 / spectrum2 = spectrum3
            # (spectrum3[k]=spectrum1[k]/spectrum2[k])
            # Dimension must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '\
                                  'with different sizes')
                else:
                    res = Spectrum(shape=self.shape, \
                                   fscale=self.fscale / other.fscale)
                    # coordinates
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        raise IOError('Operation forbidden for spectra '\
                                      'with different world coordinates')
                    # data
                    res.data = self.data / other.data
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var * self.data * self.data \
                        / (other.data ** 4)
                    elif other.var is None:
                        res.var = self.var * other.data * other.data \
                        / (other.data ** 4)
                    else:
                        res.var = (other.var * self.data * self.data \
                                   + self.var * other.data * other.data) \
                                   / (other.data ** 4)
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # spectrum / cube1 = cube2
                # (cube2[k,j,i]=spectrum[k]/cube1[k,j,i])
                # The last dimension of cube1 must be equal
                # to the spectrum dimension.
                # If not equal to None, world coordinates
                # in spectral direction must be the same.
                if other.cube:
                    if other.data is None or self.shape != other.shape[0]:
                        raise IOError('Operation forbidden for objects '\
                                      'with different sizes')
                    else:
                        from cube import Cube
                        res = Cube(shape=other.shape, wcs=other.wcs, \
                                   fscale=self.fscale / other.fscale)
                        # coordinates
                        if self.wave is None or other.wave is None:
                            res.wave = None
                        elif self.wave.isEqual(other.wave):
                            res.wave = self.wave
                        else:
                            raise IOError('Operation forbidden for spectra '\
                                          'with different world coordinates')
                        # data
                        res.data = self.data[:, np.newaxis, np.newaxis] \
                        / other.data
                        # variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = other.var \
                            * self.data[:, np.newaxis, np.newaxis] \
                            * self.data[:, np.newaxis, np.newaxis] \
                            / (other.data ** 4)
                        elif other.var is None:
                            res.var = self.var[:, np.newaxis, np.newaxis] \
                            * other.data * other.data / (other.data ** 4)
                        else:
                            res.var = \
                            (other.var \
                             * self.data[:, np.newaxis, np.newaxis] \
                             * self.data[:, np.newaxis, np.newaxis] \
                             + self.var[:, np.newaxis, np.newaxis] \
                             * other.data * other.data) / (other.data ** 4)
                        # unit
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except IOError as e:
                raise e
            except:
                raise IOError('Operation forbidden')

    def __rdiv__(self, other):
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            res = self.copy()
            res.fscale = other / res.fscale
            return res
        try:
            if other.spectrum:
                return other.__div__(self)
        except IOError as e:
            raise e
        except:
            try:
                if other.cube:
                    return other.__div__(self)
            except IOError as e:
                raise e
            except:
                raise IOError('Operation forbidden')
                return None

    def __pow__(self, other):
        """Computes the power exponent of data extensions (operator \*\*).
        """
        if self.data is None:
            raise ValueError('empty data array')
        res = self.copy()
        if is_float(other) or is_int(other):
            res.data = self.data ** other
            res.fscale = res.fscale ** other
            res.var = None
        else:
            raise ValueError('Operation forbidden')
        return res

    def _sqrt(self):
        """Computes the positive square-root of data extension.
        """
        if self.data is None:
            raise ValueError('empty data array')
        if self.var is not None:
            self.var = 3 * self.var * self.fscale ** 5 / self.data ** 4
        self.fscale = np.sqrt(self.fscale)
        self.data = np.ma.sqrt(self.data)

    def sqrt(self):
        """Returns a spectrum containing the positive
        square-root of data extension

        :rtype: Spectrum
        """
        res = self.copy()
        res._sqrt()
        return res

    def _abs(self):
        """Computes the absolute value of data extension.
        """
        if self.data is None:
            raise ValueError('empty data array')
        self.data = np.ma.abs(self.data)
        self.fscale = np.abs(self.fscale)

    def abs(self):
        """Returns a spectrum containing the absolute value of data extension.

        :rtype: Spectrum
        """
        res = self.copy()
        res._abs()
        return res

    def __getitem__(self, item):
        """ Returns the corresponding value or sub-spectrum.
        """
        if is_int(item):
            return self.data[item] * self.fscale
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
            res = Spectrum(shape=shape, wave=wave, unit=self.unit, \
                           fscale=self.fscale)
            res.data = data
            res.var = var
            return res
        else:
            raise ValueError('Operation forbidden')

    def get_lambda(self, lmin, lmax=None):
        """ Returns the flux value corresponding to a wavelength,
        or returns the sub-spectrum corresponding to a wavelength range.

          :param lmin: minimum wavelength.
          :type lmin: float
          :param lmax: maximum wavelength.
          :type lmax: float
          :rtype: float or Spectrum
        """
        if lmax is None:
            lmax = lmin
        if self.wave is None:
            raise ValueError('Operation forbidden without world coordinates '\
                             'along the spectral direction')
        else:
            pix_min = max(0, self.wave.pixel(lmin, nearest=True))
            pix_max = min(self.shape, self.wave.pixel(lmax, nearest=True) + 1)
            if (pix_min + 1) == pix_max:
                return self.data[pix_min] * self.fscale
            else:
                return self[pix_min:pix_max]

    def get_step(self):
        """Returns the wavelength step.

          :rtype: float
        """
        if self.wave is not None:
            return self.wave.get_step()
        else:
            return None

    def get_start(self):
        """Returns the wavelength value of the first pixel.

          :rtype: float
        """
        if self.wave is not None:
            return self.wave.get_start()
        else:
            return None

    def get_end(self):
        """Returns the wavelength value of the last pixel.

          :rtype: float
        """
        if self.wave is not None:
            return self.wave.get_end()
        else:
            return None

    def get_range(self):
        """Returns the wavelength range [Lambda_min,Lambda_max].

          :rtype: float array
        """
        if self.wave is not None:
            return self.wave.get_range()
        else:
            return None

    def __setitem__(self, key, other):
        """Sets the corresponding part of data
        """
        if self.data is None:
            raise ValueError('empty data array')
        try:
            self.data[key] = other / np.double(self.fscale)
        except:
            try:
                # other is a spectrum
                if other.spectrum:
                    if self.wave is not None and other.wave is not None \
                    and (self.wave.get_step() != other.wave.get_step()):
                        d = {'class': 'Spectrum', 'method': '__setitem__'}
                        logger.warning("spectra with different steps", \
                                       extra=d)
                    self.data[key] = other.data \
                    * np.double(other.fscale / self.fscale)
            except:
                raise IOError('Operation forbidden')

    def set_wcs(self, wave):
        """Sets the world coordinates.

          :param wave: Wavelength coordinates.
          :type wave: :class:`mpdaf.obj.WaveCoord`
        """
        if wave.shape is not None and wave.shape != self.shape:
            d = {'class': 'Spectrum', 'method': 'set_wcs'}
            logger.warning('wavelength coordinates and data have '\
                           'not the same dimensions', extra=d)
        self.wave = wave
        self.wave.shape = self.shape

    def set_var(self, var=None):
        """Sets the variance array.

          :param var: Input variance array.
          If None, variance is set with zeros.
          :type var: float array
        """
        if var is None:
            self.var = np.zeros(self.shape)
        else:
            if self.shape == np.shape(var)[0]:
                self.var = var
            else:
                raise ValueError('var and data have not the same dimensions.')

    def mask(self, lmin=None, lmax=None):
        """Masks the spectrum on [lmin,lmax].

          :param lmin: minimum wavelength.
          :type lmin: float
          :param lmax: maximum wavelength.
          :type lmax: float
        """
        if self.wave is None:
            raise ValueError('Operation forbidden without world coordinates '\
                             'along the spectral direction')
        else:
            if lmin is None:
                pix_min = 0
            else:
                pix_min = max(0, self.wave.pixel(lmin, nearest=True))
            if lmax is None:
                pix_max = self.shape
            else:
                pix_max = min(self.shape, \
                              self.wave.pixel(lmax, nearest=True) + 1)

            self.data[pix_min:pix_max] = np.ma.masked

    def unmask(self):
        """Unmasks the spectrum (just invalid data (nan,inf) are masked).
        """
        self.data.mask = False
        self.data = np.ma.masked_invalid(self.data)

    def mask_variance(self, threshold):
        """Masks pixels with a variance upper than threshold value.

        :param threshold: Threshold value.
        :type threshold: float
        """
        if self.var is None:
            raise ValueError('Operation forbidden '\
                             'without variance extension.')
        else:
            ksel = np.where(self.var > threshold)
            self.data[ksel] = np.ma.masked

    def mask_selection(self, ksel):
        """Masks pixels corresponding to the selection.

        :param ksel: elements depending on a condition (output of np.where)
        :type ksel: ndarray or tuple of ndarrays
        """
        self.data[ksel] = np.ma.masked

    def _interp(self, wavelengths, spline=False):
        """ returns the interpolated values corresponding
        to the wavelength array

        :param wavelengths: wavelength values
        :type wavelengths : array of float

        :param spline: False: linear interpolation
        (scipy.interpolate.interp1d used),
        True: spline interpolation (scipy.interpolate.splrep/splev used).
        :type spline : bool
        """
        lbda = self.wave.coord()
        ksel = np.where(self.data.mask == False)
        d = np.empty(np.shape(ksel)[1] + 2, dtype=float)
        d[1:-1] = self.data.data[ksel]
        w = np.empty(np.shape(ksel)[1] + 2)
        w[1:-1] = lbda[ksel]
        d[0] = d[1]
        d[-1] = d[-2]
        w[0] = (-self.wave.crpix + 1) * self.wave.cdelt + self.wave.crval \
        - 0.5 * self.wave.cdelt
        w[-1] = (self.shape - self.wave.crpix) * self.wave.cdelt \
        + self.wave.crval + 0.5 * self.wave.cdelt
        if self.var is not None:
            weight = np.empty(np.shape(ksel)[1] + 2)
            weight[1:-1] = 1. / self.var[ksel]
            weight[0] = 1. / self.var[1]
            weight[-1] = 1. / self.var[-2]
            np.ma.fix_invalid(weight, copy=False, fill_value=0)
        else:
            weight = None
        if spline:
            tck = interpolate.splrep(w, d, w=weight)
            return interpolate.splev(wavelengths, tck, der=0)
        else:
            f = interpolate.interp1d(w, d)
            return f(wavelengths)

    def _interp_data(self, spline=False):
        """ Returns data array with interpolated values for masked pixels.

        :param spline: False: linear interpolation
        (scipy.interpolate.interp1d used),
        True: spline interpolation
        (scipy.interpolate.splrep/splev used).
        :type spline : bool
        """
        if np.ma.count_masked(self.data) == 0:
            return self.data.data
        else:
            lbda = self.wave.coord()
            ksel = np.where(self.data.mask == True)
            wnew = lbda[ksel]
            data = self.data.data.__copy__()
            data[ksel] = self._interp(wnew, spline)
            return data

    def interp_mask(self, spline=False):
        """ Interpolates masked pixels.

        :param spline: False: linear interpolation
        (it uses `scipy.interpolate.interp1d <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`), True: spline interpolation ( it uses `scipy.interpolate.splrep <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html>`_ and `scipy.interpolate.splev <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html>`_).
        :type spline: bool
        """
        self.data = np.ma.masked_invalid(self._interp_data(spline))

    def _rebin_factor_(self, factor):
        '''Shrinks the size of the spectrum by factor.
        New size is an integer multiple of the original size.

        :param factor: Factor.
        :type factor: integer
        '''
        assert not np.sometrue(np.mod(self.shape, factor))
        # new size is an integer multiple of the original size
        self.shape = self.shape / factor
        self.data = \
        np.ma.array(self.data.reshape(self.shape, factor).sum(1) / factor, \
                    mask=self.data.mask.reshape(self.shape, factor).sum(1))
        if self.var is not None:
            self.var = self.var.reshape(self.shape, factor).sum(1) \
            / factor / factor
        try:
            crval = self.wave.coord()[0:factor].sum() / factor
            self.wave = WaveCoord(1, self.wave.cdelt * factor, crval, \
                                  self.wave.cunit, self.shape)
        except:
            self.wave = None

    def _rebin_factor(self, factor, margin='center'):
        '''Shrinks the size of the spectrum by factor.

          :param factor: factor
          :type factor: integer
          :param margin: This parameters is used if new size is not
          an integer multiple of the original size.

            'center' : two pixels added, on the left and
                      on the right of the spectrum.

            'right': one pixel added on the right of the spectrum.

            'left': one pixel added on the left of the spectrum.

          :type margin: string in 'center'|'right'|'left'
        '''
        if factor <= 1 or factor >= self.shape:
            raise ValueError('factor must be in ]1,shape[')
        # assert not np.sometrue(np.mod( self.shape, factor))
        if not np.sometrue(np.mod(self.shape, factor)):
            # new size is an integer multiple of the original size
            self._rebin_factor_(factor)
        else:
            newshape = self.shape / factor
            n = self.shape - newshape * factor
            if margin == 'center' and n == 1:
                margin = 'right'
            if margin == 'center':
                n_left = n / 2
                n_right = self.shape - n + n_left
                spe = self[n_left:n_right]
                spe._rebin_factor_(factor)
                newshape = spe.shape + 2
                data = np.ma.empty(newshape)
                data[1:-1] = spe.data
                data[0] = self.data[0:n_left].sum() / factor
                data[-1] = self.data[n_right:].sum() / factor
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[1:-1] = spe.var
                    var[0] = self.var[0:n_left].sum() / factor / factor
                    var[-1] = self.var[n_right:].sum() / factor / factor
                try:
                    crval = spe.wave.crval - spe.wave.cdelt
                    wave = WaveCoord(1, spe.wave.cdelt, crval, \
                                     spe.wave.cunit, shape=newshape)
                except:
                    wave = None
                self.shape = newshape
                self.wave = wave
                self.data = np.ma.masked_invalid(data)
                self.var = var
            elif margin == 'right':
                spe = self[0:self.shape - n]
                spe._rebin_factor_(factor)
                newshape = spe.shape + 1
                data = np.ma.empty(newshape)
                data[:-1] = spe.data
                data[-1] = self.data[self.shape - n:].sum() / factor
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[:-1] = spe.var
                    var[-1] = self.var[self.shape - n:].sum() / factor / factor
                try:
                    wave = WaveCoord(1, spe.wave.cdelt, spe.wave.crval, \
                                     spe.wave.cunit, shape=newshape)
                except:
                    wave = None
                self.shape = newshape
                self.wave = wave
                self.data = np.ma.masked_invalid(data)
                self.var = var
            elif margin == 'left':
                spe = self[n:]
                spe._rebin_factor_(factor)
                newshape = spe.shape + 1
                data = np.ma.empty(newshape)
                data[0] = self.data[0:n].sum() / factor
                data[1:] = spe.data
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[0] = self.var[0:n].sum() / factor / factor
                    var[1:] = spe.var
                try:
                    crval = spe.wave.crval - spe.wave.cdelt
                    wave = WaveCoord(1, spe.wave.cdelt, crval, \
                                     spe.wave.cunit, shape=newshape)
                except:
                    wave = None
                self.shape = newshape
                self.wave = wave
                self.data = np.ma.masked_invalid(data)
                self.var = var
            else:
                raise ValueError('margin must be center|right|left')
            pass

    def rebin_factor(self, factor, margin='center'):
        '''Returns a spectrum that shrinks the size
        of the current spectrum by factor.

          :param factor: factor
          :type factor: integer
          :param margin: This parameters is used if new size is not
                         an integer multiple of the original size.

            'center' : two pixels added, on the left
                       and on the right of the spectrum.

            'right': one pixel added on the right of the spectrum.

            'left': one pixel added on the left of the spectrum.

          :type margin: string in 'center'|'right'|'left'
          :rtype: Spectrum
        '''
        res = self.copy()
        res._rebin_factor(factor, margin)
        return res

    def _rebin_median_(self, factor):
        '''Shrinks the size of the spectrum by factor. Median values used.
        New size is an integer multiple of the original size.

        :param factor: Factor.
        :type factor: integer
        '''
        assert not np.sometrue(np.mod(self.shape, factor))
        # new size is an integer multiple of the original size
        self.shape = self.shape / factor
        self.data = \
        np.ma.array(np.ma.median(self.data.reshape(self.shape, factor), 1), \
                    mask=self.data.mask.reshape(self.shape, factor).sum(1))
        self.var = None
        try:
            crval = self.wave.coord()[0:factor].sum() / factor
            self.wave = WaveCoord(1, self.wave.cdelt * factor, crval, \
                                  self.wave.cunit, self.shape)
        except:
            self.wave = None

    def rebin_median(self, factor, margin='center'):
        '''Shrinks the size of the spectrum by factor. Median values are used

          :param factor: factor
          :type factor: integer
          :param margin: This parameters is used if new size
                         is not an integer multiple of the original size.

            'center' : pixels truncated, on the left
            and on the right of the spectrum.

            'right': pixel truncated on the right of the spectrum.

            'left': pixel truncated on the left of the spectrum.

          :type margin: string in 'center'|'right'|'left'
          :rtype: :class:`mpdaf.obj.Spectrum`
        '''
        if factor <= 1 or factor >= self.shape:
            raise ValueError('factor must be in ]1,shape[')
        # assert not np.sometrue(np.mod( self.shape, factor ))
        if not np.sometrue(np.mod(self.shape, factor)):
            # new size is an integer multiple of the original size
            res = self.copy()
        else:
            newshape = self.shape / factor
            n = self.shape - newshape * factor
            if margin == 'center' and n == 1:
                margin = 'right'
            if margin == 'center':
                n_left = n / 2
                n_right = self.shape - n + n_left
                res = self[n_left:n_right]
            elif margin == 'right':
                res = self[0:self.shape - n]
            elif margin == 'left':
                res = self[n:]
            else:
                raise ValueError('margin must be center|right|left')
        res._rebin_median_(factor)
        return res

    def _rebin(self, step, start=None, shape=None, \
               spline=False, notnoise=False):
        """Rebins spectrum data to different wavelength step size.
        Uses `scipy.integrate.quad <http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html>`_.

          :param step: New pixel size in spectral direction.
          :type step: float
          :param start: Spectral position of the first new pixel.
          It can be set or kept at the edge of the old first one.
          :type start: float
          :param shape: Size of the new spectrum.
          :type shape: integer
          :param spline: Linear/spline interpolation
          to interpolate masked values.
          :type spline: bool
          :param notnoise: True if the noise Variance
          spectrum is not interpolated (if it exists).
          :type notnoise: bool
        """
        lrange = self.get_range()
        if start is not None and start > lrange[1]:
            raise ValueError('Start value outside the spectrum range')
        if start is not None and start < lrange[0]:
            n = int((lrange[0] - start) / step)
            start = lrange[0] + (n + 1) * step

        newwave = self.wave.rebin(step, start)
        if shape is None:
            newshape = newwave.shape
        else:
            newshape = min(shape, newwave.shape)
            newwave.shape = newshape

        dmin = np.min(self.data)
        dmax = np.max(self.data)
        if dmin == dmax:
            self.data = np.ones(newshape, dtype=np.float) * dmin
        else:
            data = self._interp_data(spline)
            f = lambda x: data[int(self.wave.pixel(x) + 0.5)]
            self.data = np.empty(newshape, dtype=np.float)
            pix = np.arange(newshape + 1, dtype=np.float)
            x = (pix - newwave.crpix + 1) * newwave.cdelt \
            + newwave.crval - 0.5 * newwave.cdelt

            lbdamax = (self.shape - self.wave.crpix) * self.wave.cdelt \
            + self.wave.crval + 0.5 * self.wave.cdelt
            if x[-1] > lbdamax:
                x[-1] = lbdamax

            for i in range(newshape):
                self.data[i] = \
                integrate.quad(f, x[i], x[i + 1], full_output=1)[0] \
                / newwave.cdelt

        if self.var is not None and not notnoise:
            f = lambda x: self.var[int(self.wave.pixel(x) + 0.5)]
            var = np.empty(newshape, dtype=np.float)
            for i in range(newshape):
                var[i] = \
                integrate.quad(f, x[i], x[i + 1], full_output=1)[0] \
                / newwave.cdelt
            self.var = var
        else:
            self.var = None

        self.data = np.ma.masked_invalid(self.data)
        self.shape = newshape
        self.wave = newwave

    def rebin(self, step, start=None, shape=None, \
              spline=False, notnoise=False):
        """Returns a spectrum with data rebin to different wavelength step size.
        Uses `scipy.integrate.quad <http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html>`_.

          :param step: New pixel size in spectral direction.
          :type step: float
          :param start: Spectral position of the first new pixel.
          It can be set or kept at the edge of the old first one.
          :type start: float
          :param shape: Size of the new spectrum.
          :type shape: integer
          :param spline: Linear/spline interpolation
          to interpolate masked values.
          :type spline: bool
          :param notnoise: True if the noise Variance
          spectrum is not interpolated (if it exists).
          :type notnoise: bool
          :rtype: Spectrum
        """
        res = self.copy()
        res._rebin(step, start, shape, spline, notnoise)
        return res

    def mean(self, lmin=None, lmax=None, weight=True, spline=False):
        """ Computes the mean flux value over a wavelength range.

          :param lmin: Minimum wavelength.
          :type lmin: float
          :param lmax: Maximum wavelength.
          :type lmax: float
          :param weight: If weight is True, compute the weighted average
          with the inverse of variance as weight.
          :type weight: bool
          :param spline: Linear/spline interpolation
          to interpolate masked values.
          :type spline: bool
          :rtype: Returns mean value (float).
        """
        if self.var is None:
            weight = False
        if lmin is None:
            i1 = 0
        else:
            i1 = max(0, self.wave.pixel(lmin, nearest=True))
        if lmax is None:
            i2 = self.shape
        else:
            i2 = min(self.shape, self.wave.pixel(lmax, nearest=True) + 1)

        # replace masked values by interpolated values
        data = self._interp_data(spline)

        if weight:
            weights=1.0 / self.var[i1:i2]
            np.ma.fix_invalid(weights, copy=False, fill_value=0)
            flux = np.average(data[i1:i2], \
                              weights=weights) * self.fscale
        else:
            flux = data[i1:i2].mean() * self.fscale
        return flux

    def sum(self, lmin=None, lmax=None, weight=True, spline=False):
        """ Computes the flux value over [lmin,lmax].

          :param lmin: Minimum wavelength.
          :type lmin: float
          :param lmax: Maximum wavelength.
          :type lmax: float
          :param weight: If weight is True, compute the weighted sum
          with the inverse of variance as weight.
          :type weight: bool
          :param spline: Linear/spline interpolation
          to interpolate masked values.
          :type spline: bool
          :rtype: Returns flux value (float).
        """
        if self.var is None:
            weight = False
        if lmin is None:
            i1 = 0
        else:
            i1 = max(0, self.wave.pixel(lmin, True))
        if lmax is None:
            i2 = self.shape
        else:
            i2 = min(self.shape, self.wave.pixel(lmax, True) + 1)

        # replace masked values by interpolated values
        data = self._interp_data(spline)

        if weight:
            weights=1.0 / self.var[i1:i2]
            np.ma.fix_invalid(weights, copy=False, fill_value=0)
            flux = (i2 - i1) * np.average(data[i1:i2], weights=weights) * self.fscale
        else:
            flux = data[i1:i2].sum() * self.fscale
        return flux

    def poly_fit(self, deg, weight=True, maxiter=0, \
                 nsig=(-3.0, 3.0), verbose=False):
        """Performs polynomial fit on normalized spectrum
        and returns polynomial coefficients.

          :param deg: Polynomial degree.
          :type deg: integer
          :param weight:  If weight is True, the weight is computed
          as the inverse of variance.
          :type weight: bool
          :param maxiter: Maximum allowed iterations (0)
          :type maxiter: int
          :param nsig: the low and high rejection factor
          in std units (-3.0,3.0)
          :type nsig: (float,float)
          :rtype: ndarray, shape.
          Polynomial coefficients ordered from low to high.
        """
        if self.shape <= deg + 1:
            raise ValueError('Too few points to perform polynomial fit')

        if self.var is None:
            weight = False

        if weight:
            vec_weight = 1. / self.var
            np.ma.fix_invalid(vec_weight, copy=False, fill_value=0)
        else:
            vec_weight = None

        mask = np.array(1 - self.data.mask, dtype=bool)
        d = self.data.compress(mask)
        w = self.wave.coord().compress(mask)
        if weight:
            vec_weight = vec_weight.compress(mask)

        # normalize w
        w0 = np.min(w)
        dw = np.max(w) - w0
        w = (w - w0) / dw

        p = np.polynomial.polynomial.polyfit(w, d, deg, w=vec_weight)

        if maxiter > 0:
            err = d - np.polynomial.polynomial.polyval(w, p)
            sig = np.std(err)
            n_p = len(d)
            for iter in range(maxiter):
                ind = np.where((err >= nsig[0] * sig) \
                               & (np.abs(err) <= nsig[1] * sig))
                if len(ind[0]) == n_p:
                    break
                if len(ind[0]) <= deg + 1:
                    raise ValueError('Too few points to perform '\
                                     'polynomial fit')
                if vec_weight is not None:
                    vec_weight = vec_weight[ind]
                p = np.polynomial.polynomial.polyfit(w[ind], d[ind], \
                                                     deg, w=vec_weight)
                err = d[ind] - np.polynomial.polynomial.polyval(w[ind], p)
                sig = np.std(err)
                n_p = len(ind[0])
                if verbose:
                    print 'Number of iteration: '\
                    '%d Std: %10.4e Np: %d Frac: %4.2f' \
                    % (iter + 1, sig, n_p, 100. * n_p / self.shape)

#        a = p
#        from scipy.misc import comb
#        c = np.zeros((deg+1,deg+1))
#        for i in range(deg+1):
#            c[deg-i,deg-i:] = np.array(map(lambda p: comb(i,p)
#            * (-w0)**p /dw**i * a[i],range(i+1)))
#
#        return c.sum(axis=0)[::-1] * self.fscale

        return p * self.fscale

    def poly_val(self, z):
        """Updates in place the spectrum data from polynomial coefficients.
        Uses `numpy.poly1d <http://docs.scipy.org/doc/numpy/reference/generated/numpy.poly1d.html>`_.

          :param z: The polynomial coefficients, in increasing powers.

          data = z0 + z1(lbda-min(lbda))/(max(lbda)-min(lbda)) + ...
          + zn ((lbda-min(lbda))/(max(lbda)-min(lbda)))**n
          :type z: array
        """
        l = self.wave.coord()
        w0 = np.min(l)
        dw = np.max(l) - w0
        w = (l - w0) / dw
        val = np.polynomial.polynomial.polyval(w, z)
        self.data = np.ma.masked_invalid(val)
        self.fscale = 1.0
        self.var = None

    def poly_spec(self, deg, weight=True, maxiter=0, \
                  nsig=(-3.0, 3.0), verbose=False):
        """Returns a spectrum containing a polynomial fit.

          :param deg: Polynomial degree.
          :type deg: integer
          :param weight:  If weight is True, the weight is computed
          as the inverse of variance.
          :type weight: bool
          :param maxiter: Maximum allowed iterations (0)
          :type maxiter: int
          :param nsig: the low and high rejection factor in std units
          (-3.0,3.0)
          :type nsig: (float,float)
          :rtype: Spectrum
        """
        fscale = self.fscale
        self.fscale = 1.0
        z = self.poly_fit(deg, weight, maxiter, nsig, verbose)
        self.fscale = fscale
        res = self.clone()
        res.poly_val(z)
        res.fscale = fscale
        return res

    def abmag_band(self, lbda, dlbda, out=1, spline=False):
        """Computes AB magnitude corresponding to the wavelength band.

          :param lbda: Mean wavelength.
          :type lbda: float
          :param dlbda: Width of the wavelength band.
          :type dlbda: float
          :param out: 1: the magnitude is returned,
          2: the magnitude, mean flux and mean wavelength are returned.
          :type out: 1 or 2
          :param spline: Linear/spline interpolation
          to interpolate masked values.
          :type spline: bool
          :rtype: magnitude value (out=1)
          or float array containing magnitude,
          mean flux and mean wavelength (out=2).
        """
        data = self._interp_data(spline)
        i1 = max(0, self.wave.pixel(lbda - dlbda / 2, nearest=True))
        i2 = min(self.shape, self.wave.pixel(lbda + dlbda / 2, nearest=True))
        if i1 == i2:
            return 99
        else:
            vflux = data[i1:i2 + 1].mean() * self.fscale
            mag = flux2mag(vflux, lbda)
            if out == 1:
                return mag
            if out == 2:
                return np.array([mag, vflux, lbda])

    def abmag_filter_name(self, name, out=1, spline=False):
        """Computes AB magnitude using the filter name.

          :param name: 'U', 'B', 'V', 'Rc', 'Ic', 'z', 'R-Johnson','F606W'
          :type name: string
          :param out: 1: the magnitude is returned,
          2: the magnitude, mean flux and mean wavelength are returned.
          :type out: 1 or 2
          :param spline: Linear/spline interpolation
          to interpolate masked values.
          :type spline: bool
          :rtype: magnitude value (out=1) or magnitude,
          mean flux and mean wavelength (out=2).
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
            (l0, lmin, lmax, tck) = ABmag_filters.mag_RJohnson()
            return self._filter(l0, lmin, lmax, tck, out, spline)
        elif name == 'F606W':
            (l0, lmin, lmax, tck) = ABmag_filters.mag_F606W()
            return self._filter(l0, lmin, lmax, tck, out, spline)
        else:
            pass

    def abmag_filter(self, lbda, eff, out=1, spline=False):
        """Computes AB magnitude using array filter.

          :param lbda: Wavelength values.
          :type lbda: float array
          :param eff: Efficiency values.
          :type eff: float array
          :param out: 1: the magnitude is returned,
          2: the magnitude, mean flux and mean wavelength are returned.
          :type out: 1 or 2
          :param spline: Linear/spline interpolation
          to interpolate masked values.
          :type spline: bool
          :rtype: magnitude value (out=1) or magnitude,
          mean flux and mean wavelength (out=2).
        """
        lbda = np.array(lbda)
        eff = np.array(eff)
        if np.shape(lbda) != np.shape(eff):
            raise TypeError('lbda and eff inputs have not the same size.')
        l0 = np.average(lbda, weights=eff)
        lmin = lbda[0]
        lmax = lbda[-1]
        if np.shape(lbda)[0] > 3:
            tck = interpolate.splrep(lbda, eff, k=min(np.shape(lbda)[0], 3))
        else:
            tck = interpolate.splrep(lbda, eff, k=1)
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
        t = the knot-points, c = coefficients and k = the order of the spline.

        out : 1 or 2
        1: the magnitude is returned
        2: the magnitude, mean flux and mean lbda are returned

        spline : bool
        linear/spline interpolation to interpolate masked values
        """
        imin = self.wave.pixel(lmin, True)
        imax = self.wave.pixel(lmax, True)
        if imin == imax:
            if imin == 0 or imin == self.shape:
                raise ValueError('Spectrum outside Filter band')
            else:
                raise ValueError('filter band smaller than spectrum step')
        lb = (np.arange(imin, imax) - self.wave.crpix + 1) \
        * self.wave.cdelt + self.wave.crval
        w = interpolate.splev(lb, tck, der=0)
        data = self._interp_data(spline)
        vflux = np.average(data[imin:imax], weights=w) * self.fscale
        mag = flux2mag(vflux, l0)
        if out == 1:
            return mag
        if out == 2:
            return np.array([mag, vflux, l0])

    def truncate(self, lmin=None, lmax=None):
        """Truncates a spectrum in place.

          :param lmin: Minimum wavelength.
          :type lmin: float
          :param lmax: Maximum wavelength.
          :type lmax: float
        """
        if lmin is None:
            i1 = 0
        else:
            i1 = max(0, self.wave.pixel(lmin, nearest=True))
        if lmax is None:
            i2 = self.shape
        else:
            i2 = min(self.shape, self.wave.pixel(lmax, nearest=True) + 1)

        if i1 == i2:
            raise ValueError('Minimum and maximum wavelengths are equal')

        if i2 == i1 + 1:
            raise ValueError('Minimum and maximum wavelengths '\
                             'are outside the spectrum range')

        res = self.__getitem__(slice(i1, i2, 1))
        self.shape = res.shape
        self.data = res.data
        self.wave = res.wave
        self.var = res.var

    def fwhm(self, l0, cont=0, spline=False):
        """Returns the fwhm of a peak.

          :param l0: Wavelength value corresponding to the peak position.
          :type l0: float
          :param cont: The continuum [default 0].
          :type cont: integer
          :param spline: Linear/spline interpolation
          to interpolate masked values.
          :type spline: bool
          :rtype: float
        """
        try:
            k0 = self.wave.pixel(l0, nearest=True)
            d = self._interp_data(spline) - cont / self.fscale
            f2 = d[k0] / 2
            k2 = np.argwhere(d[k0:-1] < f2)[0][0] + k0
            i2 = np.interp(f2, d[k2:k2 - 2:-1], [k2, k2 - 1])
            k1 = k0 - np.argwhere(d[k0:-1] < f2)[0][0]
            i1 = np.interp(f2, d[k1:k1 + 2], [k1, k1 + 1])
            fwhm = (i2 - i1) * self.wave.cdelt
            return fwhm
        except:
            try:
                k0 = self.wave.pixel(l0, nearest=True)
                d = self._interp_data(spline) - cont / self.fscale
                f2 = d[k0] / 2
                k2 = np.argwhere(d[k0:-1] > f2)[0][0] + k0
                i2 = np.interp(f2, d[k2:k2 - 2:-1], [k2, k2 - 1])
                k1 = k0 - np.argwhere(d[k0:-1] > f2)[0][0]
                i1 = np.interp(f2, d[k1:k1 + 2], [k1, k1 + 1])
                fwhm = (i2 - i1) * self.wave.cdelt
                return fwhm
            except:
                raise ValueError('Error in fwhm estimation')

    def gauss_fit(self, lmin, lmax, lpeak=None, flux=None, fwhm=None, \
                  cont=None, peak=False, spline=False, weight=True, plot=False):
        """Performs Gaussian fit on spectrum.
        Uses `scipy.optimize.leastsq <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html>`_ to minimize the sum of squares.

          :param lmin: Minimum wavelength value or wavelength range
          used to compute the gaussian left value.
          :type lmin: float or (float,float)
          :param lmax: Maximum wavelength or wavelength range
          used to compute the gaussian right value.
          :type lmax: float or (float,float)
          :param lpeak: Input gaussian center, if None it is estimated
          with the wavelength corresponding to the maximum value
          in [max(lmin), min(lmax)]
          :type lpeak: float
          :param flux: Integrated gaussian flux
          or gaussian peak value if peak is True.
          :type flux: float
          :param fwhm: Input gaussian fwhm, if None it is estimated.
          :type fwhm: float
          :param peak: If true, flux contains the gaussian peak value .
          :type peak: bool
          :param cont: Continuum value, if None it is estimated
          by the line through points (max(lmin),mean(data[lmin]))
          and (min(lmax),mean(data[lmax])).
          :type cont: float
          :param spline: Linear/spline interpolation
          to interpolate masked values.
          :type spline: bool
          :param weight:  If weight is True, the weight
          is computed as the inverse of variance.
          :type weight: bool
          :param plot: If True, the Gaussian is plotted.
          :type plot: bool
          :returns: :class:`mpdaf.obj.Gauss1D`
        """
        # truncate the spectrum and compute right and left gaussian values
        if is_int(lmin) or is_float(lmin):
                fmin = None
        else:
            lmin = np.array(lmin, dtype=float)
            fmin = self.mean(lmin[0], lmin[1])
            lmin = lmin[1]

        if is_int(lmax) or is_float(lmax):
                fmax = None
        else:
            lmax = np.array(lmax, dtype=float)
            fmax = self.mean(lmax[0], lmax[1])
            lmax = lmax[0]

        # spec = self.truncate(lmin, lmax)
        spec = self.get_lambda(lmin, lmax)
        data = spec._interp_data(spline)
        l = spec.wave.coord()
        d = data

        lmin = l[0]
        lmax = l[-1]
        if fmin is None:
            fmin = d[0]
        if fmax is None:
            fmax = d[-1]

        # initial gaussian peak position
        if lpeak is None:
            lpeak = l[d.argmax()]

        # continuum value
        if cont is None:
            cont0 = ((fmax - fmin) * lpeak + lmax \
                     * fmin - lmin * fmax) / (lmax - lmin)
        else:
            cont /= self.fscale
            cont0 = cont

        # initial sigma value
        if fwhm is None:
            try:
                fwhm = spec.fwhm(lpeak, cont0 * self.fscale, spline)
            except:
                lpeak = l[d.argmin()]
                fwhm = spec.fwhm(lpeak, cont0 * self.fscale, spline)
        sigma = fwhm / (2. * np.sqrt(2. * np.log(2.0)))

        # initial gaussian integrated flux
        if flux is None:
            pixel = spec.wave.pixel(lpeak, nearest=True)
            peak = d[pixel] - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        elif peak is True:
            peak = flux / self.fscale - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        else:
            pass

        # 1d gaussian function
        if cont is None:
            gaussfit = lambda p, x: \
            ((fmax - fmin) * x + lmax * fmin - lmin * fmax) \
            / (lmax - lmin) + p[0] \
            * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) \
            * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))
        else:
            gaussfit = lambda p, x: \
            cont + p[0] * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) \
            * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))
        # 1d Gaussian fit
        if spec.var is not None and weight:
            wght = 1 / spec.var
            np.ma.fix_invalid(wght, copy=False, fill_value=0)
        else:
            wght = np.ones(spec.shape)
        e_gauss_fit = lambda p, x, y, w: w * (gaussfit(p, x) - y)

        # inital guesses for Gaussian Fit
        v0 = [flux, lpeak, sigma]
        # Minimize the sum of squares
        v, covar, info, mesg, success = leastsq(e_gauss_fit, v0[:], \
                                                 args=(l, d, wght), \
                                                 maxfev=100000, full_output=1)

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i])) \
                            * np.sqrt(np.abs(chisq / dof)) \
                            for i in range(len(v))])
        else:
            err = None

        # plot
        if plot:
            xxx = np.arange(l[0], l[-1], l[1] - l[0])
            ccc = gaussfit(v, xxx) * self.fscale
            plt.plot(xxx, ccc, 'r--')

        # return a Gauss1D object
        flux = v[0] * self.fscale
        lpeak = v[1]
        sigma = np.abs(v[2])
        fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
        peak = flux / np.sqrt(2 * np.pi * (sigma ** 2))
        cont0 *= self.fscale
        if err is not None:
            err_flux = err[0] * self.fscale
            err_lpeak = err[1]
            err_sigma = err[2]
            err_fwhm = err_sigma * 2 * np.sqrt(2 * np.log(2))
            err_peak = np.abs(1. / np.sqrt(2 * np.pi) \
                              * (err_flux * sigma - flux * err_sigma) \
                              / sigma / sigma)
        else:
            err_flux = np.NAN
            err_lpeak = np.NAN
            err_sigma = np.NAN
            err_fwhm = np.NAN
            err_peak = np.NAN

        return Gauss1D(lpeak, peak, flux, fwhm, cont0, err_lpeak, \
                       err_peak, err_flux, err_fwhm)

    def add_gaussian(self, lpeak, flux, fwhm, cont=0, peak=False):
        """Adds a gaussian on spectrum in place.

          :param lpeak: Gaussian center.
          :type lpeak: float
          :param flux: Integrated gaussian flux
          or gaussian peak value if peak is True.
          :type flux: float
          :param fwhm: Gaussian fwhm.
          :type fwhm: float
          :param cont: Continuum value.
          :type cont: float
          :param peak: If true, flux contains the gaussian peak value
          :type peak: bool
        """
        gauss = lambda p, x: cont \
        + p[0] * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) \
        * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))

        sigma = fwhm / (2. * np.sqrt(2. * np.log(2.0)))

        if peak is True:
            flux = flux * np.sqrt(2 * np.pi * (sigma ** 2))

        lmin = lpeak - 5 * sigma
        lmax = lpeak + 5 * sigma
        imin = self.wave.pixel(lmin, True)
        imax = self.wave.pixel(lmax, True)
        if imin == imax:
            if imin == 0 or imin == self.shape:
                raise ValueError('Gaussian outside spectrum wavelength range')

        wave = self.wave.coord()[imin:imax]
        v = [flux, lpeak, sigma]

        self.data[imin:imax] = self.data[imin:imax] \
        + gauss(v, wave) / self.fscale

    def _median_filter(self, kernel_size=1., pixel=True, spline=False):
        """Performs a median filter on the spectrum.
        Uses `scipy.signal.medfilt <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html>`_.

          :param kernel_size: Size of the median filter window.
          :type kernel_size: float
          :param pixel: True: kernel_size is in pixels,
          False: kernel_size is in spectrum coordinate unit.
          :type pixel: bool
        """
        if pixel is False:
            kernel_size = kernel_size / self.get_step()
        ks = int(kernel_size / 2) * 2 + 1

        data = np.empty(self.shape + 2 * ks)
        data[ks:-ks] = self._interp_data(spline)
        data[:ks] = data[ks:2 * ks][::-1]
        data[-ks:] = data[-2 * ks:-ks][::-1]
        data = signal.medfilt(data, ks)
        self.data = np.ma.array(data[ks:-ks], mask=self.data.mask)

    def median_filter(self, kernel_size=1., pixel=True, spline=False):
        """Returns a spectrum resulted on a median filter
        on the current spectrum.
        Uses `scipy.signal.medfilt <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html>`_.

          :param kernel_size: Size of the median filter window.
          :type kernel_size: float
          :param pixel: True: kernel_size is in pixels,
          False: kernel_size is in spectrum coordinate unit.
          :type pixel: bool
          :rtype: Spectrum
        """
        res = self.copy()
        res._median_filter(kernel_size, pixel, spline)
        return res

    def _convolve(self, other):
        """Convolves the spectrum with a other spectrum or an array.
        Uses `scipy.signal.convolve <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html>`_. self and other must have the same size.

          :param other: Second spectrum or 1d-array.
          :type other: 1d-array or Spectrum
        """
        if self.data is None:
            raise ValueError('empty data array')

        try:
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '\
                                  'with different sizes')
                else:
                    self.data = \
                    np.ma.array(signal.convolve(self.data, other.data, \
                                                mode='same'), \
                                mask=self.data.mask)
                    if self.var is not None:
                        self.var = signal.convolve(self.var, \
                                                   other.data, mode='same') \
                                                   * other.fscale \
                                                   * other.fscale
                    self.fscale = self.fscale * other.fscale
        except IOError as e:
            raise e
        except:
            try:
                self.data = \
                np.ma.array(signal.convolve(self.data, other, mode='same'), \
                            mask=self.data.mask)
                if self.var is not None:
                    self.var = signal.convolve(self.var, other, mode='same')
            except:
                raise IOError('Operation forbidden')
                return None

    def convolve(self, other):
        """Returns the convolution of the spectrum
        with a other spectrum or an array.
        Uses `scipy.signal.convolve <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html>`_. self and other must have the same size.

          :param other: Second spectrum or 1d-array.
          :type other: 1d-array or Spectrum
          :rtype: Spectrum
        """
        res = self.copy()
        res._convolve(other)
        return res

    def _fftconvolve(self, other):
        """Convolves the spectrum with a other spectrum or an array using fft.
        Uses `scipy.signal.fftconvolve <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html>`_. self and other must have the same size.

          :param other: Second spectrum or 1d-array.
          :type other: 1d-array or Spectrum
        """
        if self.data is None:
            raise ValueError('empty data array')

        try:
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    raise IOError('Operation forbidden '\
                                  'for spectra with different sizes')
                else:
                    self.data = \
                    np.ma.array(signal.fftconvolve(self.data, other.data, \
                                                   mode='same'), \
                                mask=self.data.mask)
                    if self.var is not None:
                        self.var = signal.fftconvolve(self.var, other.data, \
                                                      mode='same') \
                                                      * other.fscale \
                                                      * other.fscale
                    self.fscale = self.fscale * other.fscale
        except IOError as e:
            raise e
        except:
            try:
                self.data = np.ma.array(signal.fftconvolve(self.data, other, \
                                                           mode='same'), \
                                        mask=self.data.mask)
                if self.var is not None:
                    self.var = signal.fftconvolve(self.var, other, mode='same')
            except:
                raise IOError('Operation forbidden')

    def fftconvolve(self, other):
        """Returns the convolution of the spectrum
        with a other spectrum or an array using fft.
        Uses `scipy.signal.fftconvolve <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html>`_. self and other must have the same size.

          :param other: Second spectrum or 1d-array.
          :type other: 1d-array or Spectrum
          :rtype: Spectrum
        """
        res = self.copy()
        res._fftconvolve(other)
        return res

    def _correlate(self, other):
        """Cross-correlates the spectrum with a other spectrum or an array.
        Uses `scipy.signal.correlate <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html>`_. self and other must have the same size.

          :param other: Second spectrum or 1d-array.
          :type other: 1d-array or Spectrum
        """
        if self.data is None:
            raise ValueError('empty data array')

        try:
            if other.spectrum:
                if other.data is None or self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '\
                                  'with different sizes')
                else:
                    self.data = \
                    np.ma.array(signal.correlate(self.data, other.data, \
                                                 mode='same'), \
                                mask=self.data.mask)
                    if self.var is not None:
                        self.var = signal.correlate(self.var, other.data, \
                                                    mode='same') \
                                                    * other.fscale \
                                                    * other.fscale
                    self.fscale = self.fscale * other.fscale
        except IOError as e:
            raise e
        except:
            try:
                self.data = np.ma.array(signal.correlate(self.data, \
                                                         other, mode='same'),\
                                         mask=self.data.mask)
                if self.var is not None:
                    self.var = signal.correlate(self.var, other, mode='same')
            except:
                raise IOError('Operation forbidden')

    def correlate(self, other):
        """Returns the cross-correlation of the spectrum
        with a other spectrum or an array.
        Uses `scipy.signal.correlate <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html>`_. self and other must have the same size.

          :param other: Second spectrum or 1d-array.
          :type other: 1d-array or Spectrum
          :rtype: Spectrum
        """
        res = self.copy()
        res._correlate(other)
        return res

    def _fftconvolve_gauss(self, fwhm, nsig=5):
        """Convolves the spectrum with a Gaussian using fft.

          :param fwhm: Gaussian fwhm.
          :type fwhm: float
          :param nsig: Number of standard deviations.
          :type nsig: integer
        """
        sigma = fwhm / (2. * np.sqrt(2. * np.log(2.0)))
        s = sigma / self.get_step()
        n = nsig * int(s + 0.5)
        n = int(n / 2) * 2
        d = np.arange(-n, n + 1)
        kernel = special.erf((1 + 2 * d) / (2 * np.sqrt(2) * s)) \
        + special.erf((1 - 2 * d) / (2 * np.sqrt(2) * s))
        kernel /= kernel.sum()

        self.data = np.ma.array(signal.correlate(self.data, kernel, \
                                                 mode='same'), \
                                mask=self.data.mask)
        if self.var is not None:
            self.var = signal.correlate(self.var, kernel, mode='same')

    def fftconvolve_gauss(self, fwhm, nsig=5):
        """Returns the convolution of the spectrum with a Gaussian using fft.

          :param fwhm: Gaussian fwhm.
          :type fwhm: float
          :param nsig: Number of standard deviations.
          :type nsig: integer
          :rtype: Spectrum
        """
        res = self.copy()
        res._fftconvolve_gauss(fwhm, nsig)
        return res

    def LSF_convolve(self, lsf, size, **kargs):
        """Convolve spectrum with LSF.

        :param lsf: :class:`mpdaf.MUSE.LSF` object
        or function f describing the LSF.

            The first three parameters of the function f must be lbda
            (wavelength value in A), step (in A) and size (odd integer).

            f returns an np.array with shape=2*(size/2)+1 and centered in lbda

            Example: from mpdaf.MUSE import LSF
        :type f: python function
        :param size: size of LSF in pixels.
        :type size: odd integer
        :param kargs: kargs can be used to set function arguments.
        :rtype: :class:`mpdaf.obj.Spectrum`
        """
        res = self.clone()
        if self.data.sum() == 0:
            return res
        step = self.get_step()
        lbda = self.wave.coord()

        if size % 2 == 0:
            raise ValueError('Size must be an odd number')
        else:
            k = size / 2

        if isinstance(lsf, types.FunctionType):
            f = lsf
        else:
            try:
                f = getattr(lsf, 'get_LSF')
            except:
                raise ValueError('lsf parameter is not valid')

        data = np.empty(len(self.data) + 2 * k)
        data[k:-k] = self.data
        data[:k] = self.data[k:0:-1]
        data[-k:] = self.data[-2:-k - 2:-1]

        res.data = np.ma.array(map(lambda i: (f(lbda[i], step, \
                                                size, **kargs) \
                                              * data[i:i + size]).sum(), \
                                   range(self.shape)), mask=self.data.mask)
        res.fscale = self.fscale

        if self.var is None:
            res.var = None
        else:
            res.var = np.array(map(lambda i: (f(lbda[i], step, size, **kargs)\
                                               * data[i:i + size]).sum(), \
                                   range(self.shape)))
        return res

    def peak_detection(self, kernel_size=None, pix=False):
        """Returns a list of peak locations.
        Uses `scipy.signal.medfilt <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html>`_.

        :param kernel_size: size of the median filter window
        :type kernel_size: float
        :param pix: If pix is True, returned positions are in pixels.
        If pix is False, it is wavelenths values.
        :type pix: bool
        """
        d = np.abs(self.data - signal.medfilt(self.data, kernel_size))
        cont = self.poly_spec(5)
        ksel = np.where(d > cont.data)
        if pix:
            return ksel[0]
        else:
            wave = self.wave.coord()
            return wave[ksel]

    def plot(self, max=None, title=None, noise=False, \
             lmin=None, lmax=None, **kargs):
        """Plots the spectrum. By default, drawstyle is 'steps-mid'.

          :param max: If max is True, the plot is normalized
          to peak at max value.
          :type max: bool
          :param title: Figure tiltle (None by default).
          :type title: string
          :param noise: If noise is True
          the +/- standard deviation is overplotted.
          :type noise: bool
          :param lmin: Minimum wavelength.
          :type lmin: float
          :param lmax: Maximum wavelength.
          :type lmax: float
          :param kargs: kargs can be used to set line properties:
          line label (for auto legends), linewidth,
          anitialising, marker face color, etc.
          :type  kargs: matplotlib.lines.Line2D
        """
        plt.ion()

        res = self.copy()
        res.truncate(lmin, lmax)

        x = res.wave.coord()
        f = res.data * res.fscale
        if max != None:
            f = f * max / f.max()
        if res.var is  None:
            noise = False

        # default plot arguments
        plotargs = dict(drawstyle='steps-mid')
        plotargs.update(kargs)

        plt.plot(x, f, **plotargs)

        if noise:
            plt.fill_between(x, f + np.sqrt(res.var) * res.fscale, \
                             f - np.sqrt(res.var) * res.fscale, \
                             color='0.75', facecolor='0.75', alpha=0.5)
        if title is not None:
            plt.title(title)
        if res.wave.cunit is not None:
            plt.xlabel(r'$\lambda$ (%s)' % res.wave.cunit)
        if res.unit is not None:
            plt.ylabel(res.unit)
        self._fig = plt.get_current_fig_manager()
        plt.connect('motion_notify_event', self._on_move)
        self._plot_id = len(plt.gca().lines) - 1

    def log_plot(self, max=None, title=None, noise=False, \
                 lmin=None, lmax=None, **kargs):
        """Plots the spectrum with y logarithmic scale.
        By default, drawstyle is 'steps-mid'.

          :param max: If max is True, the plot is normalized
          to peak at max value.
          :type max: bool
          :param title: Figure tiltle (None by default).
          :type title: string
          :param noise: If noise is True,
          the +/- standard deviation is overplotted.
          :type noise: bool
          :param lmin: Minimum wavelength.
          :type lmin: float
          :param lmax: Maximum wavelength.
          :type lmax: float
          :param kargs: kargs can be used to set additional line properties:
          line label (for auto legends), linewidth, anitialising,
          marker face color, etc.
          :type  kargs: matplotlib.lines.Line2D
        """
        plt.ion()

        res = self.copy()
        res.truncate(lmin, lmax)

        x = res.wave.coord()
        f = res.data * res.fscale
        if max != None:
            f = f * max / f.max()
        if res.var is  None:
            noise = False

        # default plot arguments
        plotargs = dict(drawstyle='steps-mid')
        plotargs.update(kargs)

        plt.semilogy(x, f, **plotargs)
        if noise:
            plt.fill_between(x, f + np.sqrt(res.var) * res.fscale, \
                             f - np.sqrt(res.var) * res.fscale, \
                             color='0.75', facecolor='0.75', alpha=0.5)
        if title is not None:
            plt.title(title)
        if res.wave.cunit is not None:
            plt.xlabel(r'$\lambda$ (%s)' % res.wave.cunit)
        if res.unit is not None:
            plt.ylabel(res.unit)

        self._fig = plt.get_current_fig_manager()
        plt.connect('motion_notify_event', self._on_move)
        self._plot_id = len(plt.gca().lines) - 1

    def _on_move(self, event):
        """ prints xc,yc,k,lbda and data in the figure toolbar.
        """
        if event.inaxes is not None:
            xc, yc = event.xdata, event.ydata
            try:
                i = self.wave.pixel(xc, True)
                x = self.wave.coord(i)
                val = self.data.data[i] * self.fscale
                s = 'xc= %g yc=%g k=%d lbda=%g data=%g' % (xc, yc, i, x, val)
                self._fig.toolbar.set_message(s)
            except:
                pass

    def ipos(self, filename='None'):
        """Prints cursor position in interactive mode
        (xc and yc correspond to the cursor position,
        k is the nearest pixel, lbda contains the wavelength value
        and data contains spectrum data value.)

          To read cursor position, click on the left mouse button.

          To remove a cursor position, click on the left mouse button + <d>

          To quit the interactive mode, click on the right mouse button.

          At the end, clicks are saved in self.clicks as dictionary
          {'xc','yc','k','lbda','data'}.

          :param filename: If filename is not None, the cursor values
          are saved as a fits table with columns labeled
          'XC'|'YC'|'I'|'X'|'DATA'
          :type filename: string
        """
        print 'To read cursor position, click on the left mouse button'
        print 'To remove a cursor position, '\
        'click on the left mouse button + <d>'
        print 'To quit the interactive mode, click on the right mouse button.'
        print 'After quit, clicks are saved in self.clicks '\
        'as dictionary {xc,yc,k,lbda,data}.'

        if self._clicks is None:
            binding_id = plt.connect('button_press_event', self._on_click)
            self._clicks = SpectrumClicks(binding_id, filename)
            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")
        else:
            self._clicks.filename = filename

    def _on_click(self, event):
        """ prints xc,yc,k,lbda and data corresponding to the cursor position.
        """
        if event.key == 'd':
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        xc, yc = event.xdata, event.ydata
                        self._clicks.remove(xc)
                        print "new selection:"
                        for i in range(len(self._clicks.xc)):
                            self._clicks.iprint(i, self.fscale)
                    except:
                        pass
        else:
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        xc, yc = event.xdata, event.ydata
                        i = self.wave.pixel(xc, True)
                        x = self.wave.coord(i)
                        val = self.data[i] * self.fscale
                        if len(self._clicks.k) == 0:
                            print ''
                        self._clicks.add(xc, yc, i, x, val)
                        self._clicks.iprint(len(self._clicks.k) - 1, \
                                            self.fscale)
                    except:
                        pass
            else:
                self._clicks.write_fits()
                # save clicks in a dictionary {'xc','yc','x','data'}
                d = {'xc': self._clicks.xc, 'yc': self._clicks.yc, \
                     'k': self._clicks.k, 'lbda': self._clicks.lbda, \
                     'data': self._clicks.data}
                self.clicks = d
                #clear
                self._clicks.clear()
                self._clicks = None
                fig = plt.gcf()
                fig.canvas.stop_event_loop_default()

    def idist(self):
        """Gets distance and center from 2 cursor positions (interactive mode)

          To quit the interactive mode, click on the right mouse button.
        """
        print 'Use 2 mouse clicks to get center and distance.'
        print 'To quit the interactive mode, click on the right mouse button.'
        if self._clicks is None:
            binding_id = plt.connect('button_press_event', \
                                     self._on_click_dist)
            self._clicks = SpectrumClicks(binding_id)

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_click_dist(self, event):
        """Prints distance and center between 2 cursor positions.
        """
        if event.button == 1:
            if event.inaxes is not None:
                try:
                    xc, yc = event.xdata, event.ydata
                    i = self.wave.pixel(xc, True)
                    x = self.wave.coord(i)
                    val = self.data[i] * self.fscale
                    if len(self._clicks.k) == 0:
                        print ''
                    self._clicks.add(xc, yc, i, x, val)
                    self._clicks.iprint(len(self._clicks.k) - 1, self.fscale)
                    if np.sometrue(np.mod(len(self._clicks.k), 2)) == False:
                        dx = np.abs(self._clicks.xc[-1] - self._clicks.xc[-2])
                        xc = (self._clicks.xc[-1] + self._clicks.xc[-2]) / 2
                        print 'Center: %f Distance: %f' % (xc, dx)
                except:
                    pass
        else:
            self._clicks.clear()
            self._clicks = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def igauss_fit(self, nclicks=5):
        """Performs and plots a polynomial fit on spectrum.

          To select minimum, peak and maximum wavelengths,
          click on the left mouse button.

          To quit the interactive mode,
          click on the right mouse button.

          The parameters of the last gaussian are saved
          in self.gauss (:class:`mpdaf.obj.Gauss1D`)

          :param nclicks: 3 or 5 clicks.

          Use 3 mouse clicks to get minimim, peak and maximum wavelengths.

          Use 5 mouse clicks: the two first select a range
          of minimum wavelengths, the 3th selects the peak wavelength and
          the two last clicks select a range of maximum wavelengths
          - see :func:`mpdaf.obj.Spectrum.gauss_fit`.

          :type nclicks: integer
        """
        if nclicks == 3:
            print 'Use 3 mouse clicks to get minimim, '\
            'peak and maximum wavelengths.'
            print 'To quit the interactive mode, '\
            'click on the right mouse button.'
            print 'The parameters of the last '\
            'gaussian are saved in self.gauss.'
            if self._clicks is None:
                binding_id = plt.connect('button_press_event', \
                                         self._on_3clicks_gauss_fit)
                self._clicks = SpectrumClicks(binding_id)
                warnings.filterwarnings(action="ignore")
                fig = plt.gcf()
                fig.canvas.start_event_loop_default(timeout=-1)
                warnings.filterwarnings(action="default")
        else:
            print 'Use the 2 first mouse clicks to get the wavelength '\
            'range to compute the gaussian left value.'
            print 'Use the next click to get the peak wavelength.'
            print 'Use the 2 last mouse clicks to get the wavelength range '\
            'to compute the gaussian rigth value.'
            print 'To quit the interactive mode, '\
            'click on the right mouse button.'
            print 'The parameters of the last gaussian '\
            'are saved in self.gauss.'
            if self._clicks is None:
                binding_id = plt.connect('button_press_event', \
                                         self._on_5clicks_gauss_fit)
                self._clicks = SpectrumClicks(binding_id)
                warnings.filterwarnings(action="ignore")
                fig = plt.gcf()
                fig.canvas.start_event_loop_default(timeout=-1)
                warnings.filterwarnings(action="default")

    def _on_3clicks_gauss_fit(self, event):
        """Performs polynomial fit on spectrum (interactive mode).
        """
        if event.button == 1:
            if event.inaxes is not None:
                try:
                    xc, yc = event.xdata, event.ydata
                    i = self.wave.pixel(xc, True)
                    x = self.wave.coord(i)
                    val = self.data[i] * self.fscale
                    if len(self._clicks.k) == 0:
                        print ''
                    self._clicks.add(xc, yc, i, x, val)
                    if np.sometrue(np.mod(len(self._clicks.k), 3)) == False:
                        lmin = self._clicks.xc[-3]
                        lpeak = self._clicks.xc[-2]
                        lmax = self._clicks.xc[-1]
                        self.gauss = self.gauss_fit(lmin, lmax, \
                                                    lpeak=lpeak, plot=True)
                        self.gauss.print_param()
                        self._clicks.id_lines.append(len(plt.gca().lines) - 1)
                except:
                    pass
        else:
            self._clicks.clear()
            self._clicks = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def _on_5clicks_gauss_fit(self, event):
        """Performs polynomial fit on spectrum (interactive mode).
        """
        if event.button == 1:
            if event.inaxes is not None:
                try:
                    xc, yc = event.xdata, event.ydata
                    i = self.wave.pixel(xc, True)
                    x = self.wave.coord(i)
                    val = self.data[i] * self.fscale
                    if len(self._clicks.k) == 0:
                        print ''
                    self._clicks.add(xc, yc, i, x, val)
                    if np.sometrue(np.mod(len(self._clicks.k), 5)) == False:
                        lmin1 = self._clicks.xc[-5]
                        lmin2 = self._clicks.xc[-4]
                        lpeak = self._clicks.xc[-3]
                        lmax1 = self._clicks.xc[-2]
                        lmax2 = self._clicks.xc[-1]
                        self.gauss = self.gauss_fit((lmin1, lmin2), \
                                                    (lmax1, lmax2), \
                                                    lpeak=lpeak, plot=True)
                        self.gauss.print_param()
                        self._clicks.id_lines.append(len(plt.gca().lines) - 1)
                except:
                    pass
        else:
            self._clicks.clear()
            self._clicks = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def imask(self):
        """Over-plots masked values (interactive mode).
        """
        try:
            try:
                del plt.gca().lines[self._plot_mask_id]
            except:
                pass
            lbda = self.wave.coord()
            drawstyle = plt.gca().lines[self._plot_id].get_drawstyle()
            plt.plot(lbda, self.data.data, drawstyle=drawstyle, \
                     hold=True, alpha=0.3)
            self._plot_mask_id = len(plt.gca().lines) - 1
        except:
            pass
