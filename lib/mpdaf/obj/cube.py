""" cube.py manages Cube objects"""

import datetime
import logging
import multiprocessing
import numpy as np
import types
import warnings
from astropy.io import fits as pyfits

from .coords import WCS, WaveCoord
from .objs import is_float, is_int

FORMAT = "WARNING mpdaf corelib %(class)s.%(method)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('mpdaf corelib')


class iter_spe(object):

    def __init__(self, cube, index=False):
        self.cube = cube
        self.p = cube.shape[1]
        self.q = cube.shape[2]
        self.index = index

    def next(self):
        """Returns the next spectrum."""
        if self.q == 0:
            self.p -= 1
            self.q = self.cube.shape[2]
        self.q -= 1
        if self.p == 0:
            raise StopIteration
        if self.index is False:
            return self.cube[:, self.p - 1, self.q]
        else:
            return (self.cube[:, self.p - 1, self.q], (self.p - 1, self.q))

    def __iter__(self):
        """Returns the iterator itself."""
        return self


class iter_ima(object):

    def __init__(self, cube, index=False):
        self.cube = cube
        self.k = cube.shape[0]
        self.index = index

    def next(self):
        """Returns the next image."""
        if self.k == 0:
            raise StopIteration
        self.k -= 1
        if self.index is False:
            return self.cube[self.k, :,:]
        else:
            return (self.cube[self.k, :,:], self.k)

    def __iter__(self):
        """Returns the iterator itself."""
        return self


class Cube(object):

    """This class manages Cube objects.

    Parameters
    ----------
    filename : string
               Possible FITS file name. None by default.
    ext      : integer or (integer,integer) or string or (string,string)
               Number/name of the data extension
               or numbers/names of the data and variance extensions.
    notnoise : boolean
               True if the noise Variance cube is not read (if it exists).
               Use notnoise=True to create cube without variance extension.

    shape    : integer or (integer,integer,integer)
               Lengths of data in Z, Y and X.
               Python notation is used (nz,ny,nx).
    wcs      : :class:`mpdaf.obj.WCS`
               World coordinates.
    wave     : :class:`mpdaf.obj.WaveCoord`
               Wavelength coordinates.
    unit     : string
               Possible data unit type. None by default.
    data     : float arry
               Array containing the pixel values of the cube. None by default.
    var      : float array
               Array containing the variance. None by default.
    fscale   : float
               Flux scaling factor (1 by default).

    Attributes
    ----------
    filename       : string
                     Possible FITS filename.
    unit           : string
                     Possible data unit type
    primary_header : pyfits.Header
                     FITS primary header instance.
    data_header    : pyfits.Header
                     FITS data header instance.
    data           : masked array numpy.ma
                     Array containing the cube pixel values.
    shape          : array of 3 integers
                     Lengths of data in Z and Y and X
                     (python notation (nz,ny,nx)).
    var            : float array
                     Array containing the variance.
    fscale         : float
                     Flux scaling factor (1 by default).
    wcs            : :class:`mpdaf.obj.WCS`
                     World coordinates.
    wave           : :class:`mpdaf.obj.WaveCoord`
                     Wavelength coordinates
    ima            : dict{string,:class:`mpdaf.obj.Image`}
                     dictionary of images
    """

    def __init__(self, filename=None, ext=None, notnoise=False,
                 shape=(101, 101, 101), wcs=None, wave=None, unit=None,
                 data=None, var=None, fscale=1.0, ima=True):
        """Creates a Cube object.

        Parameters
        ----------
        filename : string
                   Possible FITS file name. None by default.
        ext      : integer or (integer,integer) or string or (string,string)
                   Number/name of the data extension
                   or numbers/names of the data and variance extensions.
        notnoise : boolean
                   True if the noise Variance cube is not read (if it exists).
                   Use notnoise=True to create cube without variance extension.

        shape    : integer or (integer,integer,integer)
                   Lengths of data in Z, Y and X.
                   Python notation is used (nz,ny,nx).
        wcs      : :class:`mpdaf.obj.WCS`
                   World coordinates.
        wave     : :class:`mpdaf.obj.WaveCoord`
                   Wavelength coordinates.
        unit     : string
                   Possible data unit type. None by default.
        data     : float arry
                   Array containing the pixel values of the cube. None by default.
        var      : float array
                   Array containing the variance. None by default.
        fscale   : float
                   Flux scaling factor (1 by default).
        """
        # possible FITS filename
        self.cube = True
        self.filename = filename
        self.ima = {}
        if filename is not None:
            f = pyfits.open(filename)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                # if the number of extension is 1,
                # we just read the data from the primary header
                # test if image
                if hdr['NAXIS'] != 3:
                    raise IOError('Wrong dimension number: not a cube')
                self.unit = hdr.get('BUNIT', None)
                self.primary_header = pyfits.Header()
                self.data_header = hdr
                self.shape = np.array([hdr['NAXIS3'], hdr['NAXIS2'],
                                       hdr['NAXIS1']])
                self.data = np.array(f[0].data, dtype=float)
                self.var = None
                self.fscale = hdr.get('FSCALE', 1.0)
                # WCS object from data header
                if wcs is None:
                    self.wcs = WCS(hdr)
                else:
                    self.wcs = wcs
                    if wcs.naxis1 != 0 and wcs.naxis2 != 0 and \
                        (wcs.naxis1 != self.shape[2] or
                         wcs.naxis2 != self.shape[1]):
                        d = {'class': 'Cube', 'method': '__init__'}
                        logger.warning('world coordinates and data have not'
                                       ' the same dimensions: %s',
                                       "shape of WCS object is modified",
                                       extra=d)
                    self.wcs.naxis1 = self.shape[2]
                    self.wcs.naxis2 = self.shape[1]
                # Wavelength coordinates
                if wave is None:
                    if 'CRPIX3' not in hdr or 'CRVAL3' not in hdr:
                        self.wave = None
                    else:
                        if 'CDELT3' in hdr:
                            cdelt = hdr.get('CDELT3')
                        elif 'CD3_3' in hdr:
                            cdelt = hdr.get('CD3_3')
                        else:
                            cdelt = 1.0
                        crpix = hdr.get('CRPIX3')
                        crval = hdr.get('CRVAL3')
                        cunit = hdr.get('CUNIT3', '')
                        self.wave = WaveCoord(crpix, cdelt, crval, cunit,
                                              self.shape[0])
                else:
                    self.wave = wave
                    if wave.shape is not None and wave.shape != self.shape[0]:
                        d = {'class': 'Cube', 'method': '__init__'}
                        logger.warning('wavelength coordinates and data have '
                                       'not the same dimensions: %s',
                                       'shape of WaveCoord object is '
                                       'modified', extra=d)
                    self.wave.shape = self.shape[0]
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = np.array(f['DATA'].data, dtype=float)
                else:
                    if isinstance(ext, int) or isinstance(ext, str):
                        n = ext
                    else:
                        n = ext[0]
                    h = f[n].header
                    d = np.array(f[n].data, dtype=float)
                if h['NAXIS'] != 3:
                    raise IOError('Wrong dimension number in DATA extension')
                self.unit = h.get('BUNIT', None)
                self.primary_header = hdr
                self.data_header = h
                self.shape = np.array([h['NAXIS3'], h['NAXIS2'], h['NAXIS1']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                if wcs is None:
                    self.wcs = WCS(h)  # WCS object from data header
                else:
                    self.wcs = wcs
                    if wcs.naxis1 != 0 and wcs.naxis2 != 0 and \
                        (wcs.naxis1 != self.shape[2] or
                         wcs.naxis2 != self.shape[1]):
                        d = {'class': 'Cube', 'method': '__init__'}
                        logger.warning('world coordinates and data have not '
                                       'the same dimensions: %s',
                                       'shape of WCS object is modified',
                                       extra=d)
                    self.wcs.naxis1 = self.shape[2]
                    self.wcs.naxis2 = self.shape[1]
                # Wavelength coordinates
                if wave is None:
                    if 'CRPIX3' not in h or 'CRVAL3' not in h:
                        self.wave = None
                    else:
                        if 'CDELT3' in h:
                            cdelt = h.get('CDELT3')
                        elif 'CD3_3' in h:
                            cdelt = h.get('CD3_3')
                        else:
                            cdelt = 1.0
                        crpix = h.get('CRPIX3')
                        crval = h.get('CRVAL3')
                        cunit = h.get('CUNIT3', '')
                        self.wave = WaveCoord(crpix, cdelt, crval, cunit,
                                              self.shape[0])
                else:
                    self.wave = wave
                    if wave.shape is not None and \
                            wave.shape != self.shape[0]:
                        d = {'class': 'Cube', 'method': '__init__'}
                        logger.warning('wavelength coordinates and data have '
                                       'not the same dimensions: %s',
                                       'shape of WaveCoord object is '
                                       'modified', extra=d)
                    self.wave.shape = self.shape[0]
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
                        if fstat.header['NAXIS'] != 3:
                            raise IOError('Wrong dimension number '
                                          'in variance extension')
                        if fstat.header['NAXIS1'] != self.shape[2] and \
                                fstat.header['NAXIS2'] != self.shape[1] and \
                                fstat.header['NAXIS3'] != self.shape[0]:
                            raise IOError('Number of points in STAT '
                                          'not equal to DATA')
                        self.var = np.array(fstat.data, dtype=float)
                # DQ extension
                try:
                    mask = np.ma.make_mask(f['DQ'].data)
                    self.data = np.ma.MaskedArray(self.data, mask=mask)
                except:
                    pass
                if ima:
                    from image import Image
                    for i in range(len(f)):
                        try:
                            hdr = f[i].header
                            if hdr['NAXIS'] != 2:
                                raise IOError('not an image')
                            self.ima[hdr.get('EXTNAME')] = \
                                Image(filename, ext=hdr.get('EXTNAME'),
                                      notnoise=True)
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
            if is_int(shape):
                shape = (shape, shape, shape)
            elif len(shape) == 2:
                shape = (shape[0], shape[1], shape[1])
            elif len(shape) == 3:
                pass
            else:
                raise ValueError('dim with dimension > 3')
            if data is None:
                self.data = None
                self.shape = np.array(shape)
            else:
                self.data = np.array(data, dtype=float)
                try:
                    self.shape = np.array(data.shape)
                except:
                    self.shape = np.array(shape)

            if notnoise or var is None:
                self.var = None
            else:
                self.var = np.array(var, dtype=float)
            self.fscale = np.float(fscale)
            try:
                self.wcs = wcs
                if wcs is not None:
                    if wcs.naxis1 != 0 and wcs.naxis2 != 0 and \
                        (wcs.naxis1 != self.shape[2] or
                         wcs.naxis2 != self.shape[1]):
                        d = {'class': 'Cube', 'method': '__init__'}
                        logger.warning('world coordinates and data have not '
                                       'the same dimensions: %s',
                                       'shape of WCS object is modified',
                                       extra=d)
                    self.wcs.naxis1 = self.shape[2]
                    self.wcs.naxis2 = self.shape[1]
            except:
                self.wcs = None
                d = {'class': 'Cube', 'method': '__init__'}
                logger.warning("world coordinates not copied: %s",
                               "wcs attribute is None", extra=d)
            try:
                self.wave = wave
                if wave is not None:
                    if wave.shape is not None and wave.shape != self.shape[0]:
                        d = {'class': 'Cube', 'method': '__init__'}
                        logger.warning('wavelength coordinates and data '
                                       'have not the same dimensions: %s',
                                       'shape of WaveCoord object is '
                                       'modified', extra=d)
                    self.wave.shape = self.shape[0]
            except:
                self.wave = None
                d = {'class': 'Cube', 'method': '__init__'}
                logger.warning("wavelength solution not copied: %s",
                               "wave attribute is None", extra=d)
        # Mask an array where invalid values occur (NaNs or infs).
        if self.data is not None:
            self.data = np.ma.masked_invalid(self.data)

    def copy(self):
        """Returns a new copy of a Cube object.
        """
        cub = Cube()
        cub.filename = self.filename
        cub.unit = self.unit
        cub.data_header = pyfits.Header(self.data_header)
        cub.primary_header = pyfits.Header(self.primary_header)
        cub.shape = self.shape.__copy__()
        try:
            cub.data = self.data.copy()
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

    def clone(self, var=False):
        """Returns a new cube of the same shape and coordinates,
        filled with zeros.

        Parameters
        ----------
        var : bool
        Presence of the variance extension.
        """
        try:
            wcs = self.wcs.copy()
        except:
            wcs = None
        try:
            wave = self.wave.copy()
        except:
            wave = None
        if var is False:
            cube = Cube(wcs=wcs, wave=wave, data=np.zeros(shape=self.shape),
                        unit=self.unit)
        else:
            cube = Cube(wcs=wcs, wave=wave, data=np.zeros(shape=self.shape),
                        var=np.zeros(shape=self.shape), unit=self.unit)
        return cube

    def write(self, filename, fscale=None, savemask=True):
        """ Saves the cube in a FITS file.

        Parameters
        ----------
        filename : string
                The FITS filename.
        fscale   : float
                Flux scaling factor.
        savemask : boolean
                If True,Cube mask is saved in DQ extension

        """
        # update fscale
        if fscale is None:
            fscale = self.fscale

        # create primary header
        warnings.simplefilter("ignore")
        prihdu = pyfits.PrimaryHDU()
        for card in self.primary_header.cards:
            try:
                prihdu.header[card.keyword] = (card.value, card.comment)
            except:
                try:
                    card.verify('fix')
                    prihdu.header[card.keyword] = (card.value, card.comment)
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
                        d = {'class': 'Cube', 'method': 'write'}
                        logger.warning("%s not copied in primary header",
                                       card.keyword, extra=d)
                        pass
        prihdu.header['date'] = \
            (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        hdulist = [prihdu]
        warnings.simplefilter("default")

        # world coordinates
        wcs_cards = self.wcs.to_header().cards

        # create spectrum DATA extension
        tbhdu = pyfits.ImageHDU(name='DATA', data=(self.data.data
                                                   * np.double(self.fscale / fscale))
                                .astype(np.float32))
        for card in self.data_header.cards:
            try:
                if card.keyword != 'CD1_1' and card.keyword != 'CD1_2' and \
                        card.keyword != 'CD2_1' and card.keyword != 'CD2_2' and \
                        card.keyword != 'CDELT1' and card.keyword != 'CDELT2' and \
                        tbhdu.header.keys().count(card.keyword) == 0:
                    tbhdu.header[card.keyword] = (card.value, card.comment)
            except:
                try:
                    card.verify('fix')
                    if card.keyword != 'CD1_1' and card.keyword != 'CD1_2' and\
                            card.keyword != 'CD2_1' and card.keyword != 'CD2_2' and \
                            card.keyword != 'CDELT1' and card.keyword != 'CDELT2' and \
                            tbhdu.header.keys().count(card.keyword) == 0:
                        prihdu.header[card.keyword] = \
                            (card.value, card.comment)
                except:
                    d = {'class': 'Cube', 'method': 'write'}
                    logger.warning("%s not copied in data header", card.keyword,
                                   extra=d)
                    pass
        # add world coordinate
        cd = self.wcs.get_cd()
        tbhdu.header['CTYPE1'] = \
            (wcs_cards['CTYPE1'].value, wcs_cards['CTYPE1'].comment)
        tbhdu.header['CUNIT1'] = \
            (wcs_cards['CUNIT1'].value, wcs_cards['CUNIT1'].comment)
        tbhdu.header['CRVAL1'] = \
            (wcs_cards['CRVAL1'].value, wcs_cards['CRVAL1'].comment)
        tbhdu.header['CRPIX1'] = \
            (wcs_cards['CRPIX1'].value, wcs_cards['CRPIX1'].comment)
        tbhdu.header['CD1_1'] = \
            (cd[0, 0], 'partial of first axis coordinate w.r.t. x ')
        tbhdu.header['CD1_2'] = \
            (cd[0, 1], 'partial of first axis coordinate w.r.t. y')
        tbhdu.header['CTYPE2'] = \
            (wcs_cards['CTYPE2'].value, wcs_cards['CTYPE2'].comment)
        tbhdu.header['CUNIT2'] = \
            (wcs_cards['CUNIT2'].value, wcs_cards['CUNIT2'].comment)
        tbhdu.header['CRVAL2'] = \
            (wcs_cards['CRVAL2'].value, wcs_cards['CRVAL2'].comment)
        tbhdu.header['CRPIX2'] = \
            (wcs_cards['CRPIX2'].value, wcs_cards['CRPIX2'].comment)
        tbhdu.header['CD2_1'] = \
            (cd[1, 0], 'partial of second axis coordinate w.r.t. x')
        tbhdu.header['CD2_2'] = \
            (cd[1, 1], 'partial of second axis coordinate w.r.t. y')
        tbhdu.header['CRVAL3'] = \
            (self.wave.crval, 'Start in world coordinate')
        tbhdu.header['CRPIX3'] = (self.wave.crpix, 'Start in pixel')
        tbhdu.header['CDELT3'] = \
            (self.wave.cdelt, 'Step in world coordinate')
        tbhdu.header['CTYPE3'] = ('LINEAR', 'world coordinate type')
        tbhdu.header['CUNIT3'] = (self.wave.cunit, 'world coordinate units')
        if self.unit is not None:
            tbhdu.header['BUNIT'] = (self.unit, 'data unit type')
        tbhdu.header['FSCALE'] = (fscale, 'Flux scaling factor')
        hdulist.append(tbhdu)

        self.wcs = WCS(tbhdu.header)

        # create spectrum STAT extension
        if self.var is not None:
            nbhdu = pyfits.ImageHDU(name='STAT', data=(self.var
                                                       * np.double(self.fscale * self.fscale
                                                                   / fscale / fscale)).astype(np.float32))
            # add world coordinate
#            for card in wcs_cards:
#                nbhdu.header.update(card.keyword, card.value, card.comment)
            nbhdu.header['CTYPE1'] = \
                (wcs_cards['CTYPE1'].value, wcs_cards['CTYPE1'].comment)
            nbhdu.header['CUNIT1'] = \
                (wcs_cards['CUNIT1'].value, wcs_cards['CUNIT1'].comment)
            nbhdu.header['CRVAL1'] = \
                (wcs_cards['CRVAL1'].value, wcs_cards['CRVAL1'].comment)
            nbhdu.header['CRPIX1'] = \
                (wcs_cards['CRPIX1'].value, wcs_cards['CRPIX1'].comment)
            nbhdu.header['CD1_1'] = \
                (cd[0, 0], 'partial of first axis coordinate w.r.t. x ')
            nbhdu.header['CD1_2'] = \
                (cd[0, 1], 'partial of first axis coordinate w.r.t. y')
            nbhdu.header['CTYPE2'] = \
                (wcs_cards['CTYPE2'].value, wcs_cards['CTYPE2'].comment)
            nbhdu.header['CUNIT2'] = \
                (wcs_cards['CUNIT2'].value, wcs_cards['CUNIT2'].comment)
            nbhdu.header['CRVAL2'] = \
                (wcs_cards['CRVAL2'].value, wcs_cards['CRVAL2'].comment)
            nbhdu.header['CRPIX2'] = \
                (wcs_cards['CRPIX2'].value, wcs_cards['CRPIX2'].comment)
            nbhdu.header['CD2_1'] = \
                (cd[1, 0], 'partial of second axis coordinate w.r.t. x')
            nbhdu.header['CD2_2'] = \
                (cd[1, 1], 'partial of second axis coordinate w.r.t. y')
            nbhdu.header['CRVAL3'] = \
                (self.wave.crval, 'Start in world coordinate')
            nbhdu.header['CRPIX3'] = (self.wave.crpix, 'Start in pixel')
            nbhdu.header['CDELT3'] = \
                (self.wave.cdelt, 'Step in world coordinate')
            nbhdu.header['CUNIT3'] = \
                (self.wave.cunit, 'world coordinate units')
            hdulist.append(nbhdu)

        # create DQ extension
        if savemask and np.ma.count_masked(self.data) != 0:
            dqhdu = pyfits.ImageHDU(name='DQ', data=np.uint8(self.data.mask))
            for card in wcs_cards:
                dqhdu.header[card.keyword] = (card.value, card.comment)
            hdulist.append(dqhdu)

        # save to disk
        hdu = pyfits.HDUList(hdulist)
        warnings.simplefilter("ignore")
        hdu.writeto(filename, clobber=True, output_verify='fix')
        warnings.simplefilter("default")

        self.filename = filename

    def info(self):
        """Prints information.
        """
        if self.filename is None:
            print '%i X %i X %i cube (no name)' % (self.shape[0],
                                                   self.shape[1], self.shape[2])
        else:
            print '%i X %i X %i cube (%s)' % (self.shape[0], self.shape[1],
                                              self.shape[2], self.filename)
        data = '.data(%i,%i,%i)' % (self.shape[0], self.shape[1], self.shape[2])
        if self.data is None:
            data = 'no data'
        noise = '.var(%i,%i,%i)' % (self.shape[0], self.shape[1], self.shape[2])
        if self.var is None:
            noise = 'no noise'
        if self.unit is None:
            unit = 'no unit'
        else:
            unit = self.unit
        print '%s (%s) fscale=%g, %s' % (data, unit, self.fscale, noise)
        if self.wcs is None:
            print 'no world coordinates for spatial direction'
        else:
            self.wcs.info()
        if self.wave is None:
            print 'no world coordinates for spectral direction'
        else:
            self.wave.info()
        print ".ima: ",
        for k in self.ima.keys():
            print k,
        print '\n'

    def __le__(self, item):
        """Masks data array where greater than a given value.
        Returns a cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater(self.data, item / self.fscale)
        return result

    def __lt__(self, item):
        """Masks data array where greater or equal than a given value.
        Returns a cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data,
                                                     item / self.fscale)
        return result

    def __ge__(self, item):
        """Masks data array where less than a given value.
        Returns a Cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item / self.fscale)
        return result

    def __gt__(self, item):
        """Masks data array where less or equal than a given value.
        Returns a Cube object containing a masked array
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data,
                                                  item / self.fscale)
        return result

    def resize(self):
        """Resizes the cube to have a minimum number of masked values.
        """
        if self.data is not None:
            ksel = np.where(self.data.mask == False)
            try:
                item = (slice(ksel[0][0], ksel[0][-1] + 1, None),
                        slice(ksel[1][0], ksel[1][-1] + 1, None),
                        slice(ksel[2][0], ksel[2][-1] + 1, None))
                self.data = self.data[item]
                if is_int(item[0]):
                    if is_int(item[1]):
                        self.shape = np.array((1, 1, self.data.shape[0]))
                    elif is_int(item[2]):
                        self.shape = np.array((1, self.data.shape[0], 1))
                    else:
                        self.shape = np.array((1, self.data.shape[0],
                                               self.data.shape[1]))
                elif is_int(item[1]):
                    if is_int(item[2]):
                        self.shape = np.array((self.data.shape[0], 1, 1))
                    else:
                        self.shape = np.array((self.data.shape[0], 1,
                                               self.data.shape[1]))
                elif is_int(item[2]):
                    self.shape = np.array((self.data.shape[0],
                                           self.data.shape[1], 1))
                else:
                    self.shape = self.data.shape
                if self.var is not None:
                    self.var = self.var[item]
                try:
                    self.wcs = self.wcs[item[1], item[2]]
                except:
                    self.wcs = None
                    d = {'class': 'Cube', 'method': 'resize'}
                    logger.warning("wcs not copied: %s",
                                   "wcs attribute is None", extra=d)
                try:
                    self.wave = self.wave[item[0]]
                except:
                    self.wave = None
                    d = {'class': 'Cube', 'method': 'resize'}
                    logger.warning("wavelength solution not copied: %s",
                                   "wave attribute is None", extra=d)
            except:
                pass

    def unmask(self):
        """Unmasks the cube (just invalid data (nan,inf) are masked).
        """
        self.data.mask = False
        self.data = np.ma.masked_invalid(self.data)

    def mask_variance(self, threshold):
        """Masks pixels with a variance upper than threshold value.

        Parameters
        ----------
        threshold : float
                    Threshold value.
        """
        if self.var is None:
            raise ValueError('Operation forbidden'
                             ' without variance extension.')
        else:
            ksel = np.where(self.var > threshold)
            self.data[ksel] = np.ma.masked

    def mask_selection(self, ksel):
        """Masks pixels corresponding to the selection.

        Parameters
        ----------
        ksel : output of np.where
               elements depending on a condition
        """
        self.data[ksel] = np.ma.masked

    def __add__(self, other):
        """Adds other

        cube1 + number = cube2 (cube2[k,p,q]=cube1[k,p,q]+number)

        cube1 + cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]+cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 + image = cube2 (cube2[k,p,q]=cube1[k,p,q]+image[p,q])
        The first two dimensions of cube1 must be equal
        to the image dimensions.
        If not equal to None, world coordinates in spatial
        directions must be the same.

        cube1 + spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]+spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates
        in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # cube + number = cube (add pixel per pixel)
            res = self.copy()
            res.data = self.data + (other / np.double(self.fscale))
            return res
        try:
            # cube1 + cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]+cube2[k,j,i])
            # dimensions must be the same
            # if not equal to None, world coordinates must be the same
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1] \
                        or self.shape[2] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                else:
                    res = Cube(shape=self.shape, fscale=self.fscale)
                    # coordinate
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        raise IOError('Operation forbidden for cubes with '
                                      'different world coordinates '
                                      'in spectral direction')
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        raise IOError('Operation forbidden for cubes with '
                                      'different world coordinates '
                                      'in spatial directions')
                    # data
                    res.data = self.data + (other.data *
                                            np.double(other.fscale /
                                                      self.fscale))
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var * np.double(other.fscale *
                                                        other.fscale /
                                                        self.fscale /
                                                        self.fscale)
                    elif other.var is None:
                        res.var = self.var
                    else:
                        res.var = self.var + other.var * \
                            np.double(other.fscale * other.fscale /
                                      self.fscale / self.fscale)
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # cube1 + image = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
                # the 2 first dimensions of cube1 must be equal
                # to the image dimensions
                # if not equal to None, world coordinates
                # in spatial directions must be the same
                if other.image:
                    if other.data is None or self.shape[2] != other.shape[1] \
                            or self.shape[1] != other.shape[0]:
                        raise IOError('Operation forbidden for objects '
                                      'with different sizes')
                    else:
                        res = Cube(shape=self.shape, wave=self.wave,
                                   fscale=self.fscale)
                        # coordinates
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            raise IOError('Operation forbidden for objects '
                                          'with different world coordinates')
                        # data
                        res.data = self.data + (other.data[np.newaxis, :,:] *
                                                np.double(other.fscale /
                                                          self.fscale))
                        # variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = np.ones(self.shape) \
                            * other.var[np.newaxis, :,:] \
                                * np.double(other.fscale
                                            * other.fscale / self.fscale
                                            / self.fscale)
                        elif other.var is None:
                            res.var = self.var
                        else:
                            res.var = self.var + other.var[np.newaxis, :,:] \
                                * np.double(other.fscale * other.fscale
                                            / self.fscale / self.fscale)
                        # unit
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except IOError as e:
                raise e
            except:
                try:
                    # cube1 + spectrum = cube2
                    # (cube2[k,j,i]=cube1[k,j,i]+spectrum[k])
                    # the last dimension of cube1 must be equal
                    # to the spectrum dimension
                    # if not equal to None, world coordinates
                    # in spectral direction must be the same
                    if other.spectrum:
                        if other.data is None or other.shape != self.shape[0]:
                            raise IOError('Operation forbidden for objects '
                                          'with different sizes')
                        else:
                            res = Cube(shape=self.shape, wcs=self.wcs,
                                       fscale=self.fscale)
                            # coordinates
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                raise IOError('Operation forbidden for '
                                              'spectra with different '
                                              'world coordinates')
                            # data
                            res.data = self.data + \
                                (other.data[:, np.newaxis, np.newaxis]
                                 * np.double(other.fscale / self.fscale))
                            # variance
                            if self.var is None and other.var is None:
                                res.var = None
                            elif self.var is None:
                                res.var = np.ones(self.shape) \
                                    * other.var[:, np.newaxis, np.newaxis] \
                                    * np.double(other.fscale * other.fscale
                                                / self.fscale / self.fscale)
                            elif other.var is None:
                                res.var = self.var
                            else:
                                res.var = self.var \
                                    + other.var[:, np.newaxis, np.newaxis] \
                                    * np.double(other.fscale * other.fscale
                                                / self.fscale / self.fscale)
                            # unit
                            if self.unit == other.unit:
                                res.unit = self.unit
                            return res
                except IOError as e:
                    raise e
                except:
                    raise IOError('Operation forbidden')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtracts other

        cube1 - number = cube2 (cube2[k,p,q]=cube1[k,p,q]-number)

        cube1 - cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]-cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 - image = cube2 (cube2[k,p,q]=cube1[k,p,q]-image[p,q])
        The first two dimensions of cube1 must be equal
        to the image dimensions.
        If not equal to None, world coordinates
        in spatial directions must be the same.

        cube1 - spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]-spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates
        in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # cube1 - number = cube2 (cube2[k,j,i]=cube1[k,j,i]-number)
            res = self.copy()
            res.data = self.data - (other / np.double(self.fscale))
            return res
        try:
            # cube1 - cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]-cube2[k,j,i])
            # Dimensions must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] \
                    or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                else:
                    res = Cube(shape=self.shape, fscale=self.fscale)
                    # coordinates
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        raise IOError('Operation forbidden for cubes '
                                      'with different world coordinates '
                                      'in spectral direction')
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        raise IOError('Operation forbidden for cubes '
                                      'with different world coordinates '
                                      'in spatial directions')
                    # data
                    res.data = self.data - (other.data
                                            * np.double(other.fscale
                                                        / self.fscale))
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var * np.double(other.fscale
                                                        * other.fscale
                                                        / self.fscale
                                                        / self.fscale)
                    elif other.var is None:
                        res.var = self.var
                    else:
                        res.var = self.var + other.var \
                            * np.double(other.fscale * other.fscale
                                        / self.fscale / self.fscale)
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # cube1 - image = cube2 (cube2[k,j,i]=cube1[k,j,i]-image[j,i])
                # The first two dimensions of cube1 must be equal
                # to the image dimensions.
                # If not equal to None, world coordinates
                # in spatial directions must be the same.
                if other.image:
                    if other.data is None or self.shape[2] != other.shape[1] \
                            or self.shape[1] != other.shape[0]:
                        raise IOError('Operation forbidden for images '
                                      'with different sizes')
                    else:
                        res = Cube(shape=self.shape, wave=self.wave,
                                   fscale=self.fscale)
                        # coordinates
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            raise IOError('Operation forbidden for objects '
                                          'with different world coordinates')
                        # data
                        res.data = self.data - (other.data[np.newaxis, :,:]
                                                * np.double(other.fscale
                                                            / self.fscale))
                        # variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = np.ones(self.shape) \
                            * other.var[np.newaxis, :,:] \
                                * np.double(other.fscale * other.fscale
                                            / self.fscale / self.fscale)
                        elif other.var is None:
                            res.var = self.var
                        else:
                            res.var = self.var + other.var[np.newaxis, :,:] \
                                * np.double(other.fscale * other.fscale
                                            / self.fscale / self.fscale)
                        # unit
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except IOError as e:
                raise e
            except:
                try:
                    # cube1 - spectrum = cube2
                    #(cube2[k,j,i]=cube1[k,j,i]-spectrum[k])
                    # The last dimension of cube1 must be equal
                    # to the spectrum dimension.
                    # If not equal to None, world coordinates
                    # in spectral direction must be the same.
                    if other.spectrum:
                        if other.data is None or other.shape != self.shape[0]:
                            raise IOError('Operation forbidden '
                                          'for objects with different sizes')
                        else:
                            res = Cube(shape=self.shape, wcs=self.wcs,
                                       fscale=self.fscale)
                            # coordinates
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                raise IOError('Operation forbidden for '
                                              'spectra with different '
                                              'world coordinates')
                            # data
                            res.data = self.data - \
                                (other.data[:, np.newaxis, np.newaxis]
                                 * np.double(other.fscale / self.fscale))
                            # variance
                            if self.var is None and other.var is None:
                                res.var = None
                            elif self.var is None:
                                res.var = np.ones(self.shape) \
                                    * other.var[:, np.newaxis, np.newaxis] \
                                    * np.double(other.fscale * other.fscale
                                                / self.fscale / self.fscale)
                            elif other.var is None:
                                res.var = self.var
                            else:
                                res.var = self.var \
                                    + other.var[:, np.newaxis, np.newaxis] \
                                    * np.double(other.fscale * other.fscale
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
            if other.cube:
                return other.__sub__(self)
        except IOError as e:
            raise e
        except:
            try:
                if other.image:
                    return other.__sub__(self)
            except IOError as e:
                raise e
            except:
                try:
                    if other.spectrum:
                        return other.__sub__(self)
                except IOError as e:
                    raise e
                except:
                    raise IOError('Operation forbidden')

    def __mul__(self, other):
        """Multiplies by other

        cube1 * number = cube2 (cube2[k,p,q]=cube1[k,p,q]*number)

        cube1 * cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]*cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 * image = cube2 (cube2[k,p,q]=cube1[k,p,q]*image[p,q])
        The first two dimensions of cube1 must be equal
        to the image dimensions.
        If not equal to None, world coordinates
        in spatial directions must be the same.

        cube1 * spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]*spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates
        in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # cube1 * number = cube2 (cube2[k,j,i]=cube1[k,j,i]*number)
            res = self.copy()
            res.data *= other
            return res
        try:
            # cube1 * cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]*cube2[k,j,i])
            # Dimensions must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] \
                    or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                else:
                    res = Cube(shape=self.shape, fscale=self.fscale)
                    # coordinates
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        raise IOError('Operation forbidden for cubes with '
                                      'different world coordinates '
                                      'in spectral direction')
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        raise IOError('Operation forbidden for cubes with '
                                      'different world coordinates '
                                      'in spatial directions')
                    # data
                    res.data = self.data * other.data * other.fscale
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var * self.data * self.data \
                            * other.fscale * other.fscale
                    elif other.var is None:
                        res.var = self.var * other.data * other.data \
                            * other.fscale * other.fscale
                    else:
                        res.var = (other.var * self.data * self.data
                                   + self.var * other.data * other.data) \
                            * other.fscale * other.fscale
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # cube1 * image = cube2 (cube2[k,j,i]=cube1[k,j,i]*image[j,i])
                # The first two dimensions of cube1 must be equal
                # to the image dimensions.
                # If not equal to None, world coordinates
                # in spatial directions must be the same.
                if other.image:
                    if other.data is None or self.shape[2] != other.shape[1] \
                            or self.shape[1] != other.shape[0]:
                        raise IOError('Operation forbidden for images '
                                      'with different sizes')
                    else:
                        res = Cube(shape=self.shape, wave=self.wave,
                                   fscale=self.fscale)
                        # coordinates
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            raise IOError('Operation forbidden for objects '
                                          'with different world coordinates')
                        # data
                        res.data = self.data * other.data[np.newaxis, :,:] \
                                             * other.fscale
                        # variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = other.var[np.newaxis, :,:] \
                                * self.data * self.data \
                                * other.fscale * other.fscale
                        elif other.var is None:
                            res.var = self.var * other.data[np.newaxis, :,:] \
                            * other.data[np.newaxis, :,:] \
                                * other.fscale * other.fscale
                        else:
                            res.var = (other.var[np.newaxis, :,:]
                                       * self.data * self.data
                            + self.var * other.data[np.newaxis, :,:]
                            * other.data[np.newaxis, :,:]) \
                                * other.fscale * other.fscale
                        # unit
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except IOError as e:
                raise e
            except:
                try:
                    # cube1 * spectrum = cube2
                    #(cube2[k,j,i]=cube1[k,j,i]*spectrum[k])
                    # The last dimension of cube1 must be equal
                    # to the spectrum dimension.
                    # If not equal to None, world coordinates
                    # in spectral direction must be the same.
                    if other.spectrum:
                        if other.data is None or other.shape != self.shape[0]:
                            raise IOError('Operation forbidden for objects '
                                          'with different sizes')
                        else:
                            res = Cube(shape=self.shape, wcs=self.wcs,
                                       fscale=self.fscale)
                            # coordinates
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                raise IOError('Operation forbidden for '
                                              'spectra with different '
                                              'world coordinates')
                            # data
                            res.data = self.data * other.fscale \
                                * other.data[:, np.newaxis, np.newaxis]
                            # variance
                            if self.var is None and other.var is None:
                                res.var = None
                            elif self.var is None:
                                res.var = other.var[:, np.newaxis, np.newaxis] \
                                    * self.data * self.data  \
                                    * other.fscale * other.fscale
                            elif other.var is None:
                                res.var = self.var \
                                    * other.data[:, np.newaxis, np.newaxis] \
                                    * other.data[:, np.newaxis, np.newaxis] \
                                    * other.fscale * other.fscale
                            else:
                                res.var = (other.var[:, np.newaxis, np.newaxis]
                                           * self.data * self.data + self.var
                                           * other.data[:, np.newaxis, np.newaxis]
                                           * other.data[:, np.newaxis, np.newaxis]) \
                                    * other.fscale * other.fscale
                            # unit
                            if self.unit == other.unit:
                                res.unit = self.unit
                            return res
                except IOError as e:
                    raise e
                except:
                    raise IOError('Operation forbidden')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """Divides by other

        cube1 / number = cube2 (cube2[k,p,q]=cube1[k,p,q]/number)

        cube1 / cube2 = cube3 (cube3[k,p,q]=cube1[k,p,q]/cube2[k,p,q])
        Dimensions must be the same.
        If not equal to None, world coordinates must be the same.

        cube1 / image = cube2
        (cube2[k,p,q]=cube1[k,p,q]/image[p,q])
        The first two dimensions of cube1 must be equal
        to the image dimensions.
        If not equal to None, world coordinates
        in spatial directions must be the same.

        cube1 / spectrum = cube2 (cube2[k,p,q]=cube1[k,p,q]/spectrum[k])
        The last dimension of cube1 must be equal to the spectrum dimension.
        If not equal to None, world coordinates
        in spectral direction must be the same.
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # cube1 / number = cube2 (cube2[k,j,i]=cube1[k,j,i]/number)
            res = self.copy()
            res.data /= other
            return res
        try:
            # cube1 / cube2 = cube3 (cube3[k,j,i]=cube1[k,j,i]/cube2[k,j,i])
            # Dimensions must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] \
                    or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                else:
                    res = Cube(shape=self.shape,
                               fscale=self.fscale)
                    # coordinates
                    if self.wave is None or other.wave is None:
                        res.wave = None
                    elif self.wave.isEqual(other.wave):
                        res.wave = self.wave
                    else:
                        raise IOError('Operation forbidden for cubes with '
                                      'different world coordinates '
                                      'in spectral direction')
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        raise ValueError('Operation forbidden for cubes '
                                         'with different world coordinates'
                                         ' in spatial directions')
                    # data
                    res.data = self.data / other.data / other.fscale
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var * self.data * self.data \
                            / (other.data ** 4) \
                            / other.fscale / other.fscale
                    elif other.var is None:
                        res.var = self.var * other.data * other.data \
                            / (other.data ** 4) \
                            / other.fscale / other.fscale
                    else:
                        res.var = (other.var * self.data * self.data
                                   + self.var * other.data * other.data) \
                            / (other.data ** 4) \
                            / other.fscale / other.fscale
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # cube1 / image = cube2 (cube2[k,j,i]=cube1[k,j,i]/image[j,i])
                # The first two dimensions of cube1 must be equal
                # to the image dimensions.
                # If not equal to None, world coordinates
                # in spatial directions must be the same.
                if other.image:
                    if other.data is None or self.shape[2] != other.shape[1] \
                            or self.shape[1] != other.shape[0]:
                        raise IOError('Operation forbidden for images '
                                      'with different sizes')
                    else:
                        res = Cube(shape=self.shape, wave=self.wave,
                                   fscale=self.fscale)
                        # coordinates
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            raise IOError('Operation forbidden for objects '
                                          'with different world coordinates')
                        # data
                        res.data = self.data / other.data[np.newaxis, :,:] \
                                             / other.fscale
                        # variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = other.var[np.newaxis, :,:] \
                                * self.data * self.data \
                            / (other.data[np.newaxis, :,:] ** 4) \
                                / other.fscale / other.fscale
                        elif other.var is None:
                            res.var = self.var * other.data[np.newaxis, :,:] \
                            * other.data[np.newaxis, :,:] \
                            / (other.data[np.newaxis, :,:] ** 4) \
                                / other.fscale / other.fscale
                        else:
                            res.var = (other.var[np.newaxis, :,:]
                                       * self.data * self.data + self.var
                                       * other.data[np.newaxis, :,:]
                                       * other.data[np.newaxis, :,:]) \
                                       / (other.data[np.newaxis, :,:] ** 4) \
                                       / other.fscale / other.fscale
                        # unit
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except IOError as e:
                raise e
            except:
                try:
                    # cube1 / spectrum = cube2
                    #(cube2[k,j,i]=cube1[k,j,i]/spectrum[k])
                    # The last dimension of cube1 must be equal
                    # to the spectrum dimension.
                    # If not equal to None, world coordinates
                    # in spectral direction must be the same.
                    if other.spectrum:
                        if other.data is None or other.shape != self.shape[0]:
                            raise IOError('Operation forbidden for objects '
                                          'with different sizes')
                        else:
                            res = Cube(shape=self.shape, wcs=self.wcs,
                                       fscale=self.fscale)
                            # coordinates
                            if self.wave is None or other.wave is None:
                                res.wave = None
                            elif self.wave.isEqual(other.wave):
                                res.wave = self.wave
                            else:
                                raise IOError('Operation forbidden for '
                                              'spectra with different '
                                              'world coordinates')
                            # data
                            res.data = self.data  / other.fscale \
                                / other.data[:, np.newaxis, np.newaxis]
                            # variance
                            if self.var is None and other.var is None:
                                res.var = None
                            elif self.var is None:
                                res.var = other.var[:, np.newaxis, np.newaxis] \
                                    * self.data * self.data \
                                    / (other.data[:, np.newaxis, np.newaxis] ** 4) \
                                    / other.fscale / other.fscale
                            elif other.var is None:
                                res.var = self.var \
                                    * other.data[:, np.newaxis, np.newaxis] \
                                    * other.data[:, np.newaxis, np.newaxis] \
                                    / (other.data[:, np.newaxis, np.newaxis] ** 4) \
                                    / other.fscale / other.fscale
                            else:
                                res.var = (other.var[:, np.newaxis, np.newaxis]
                                           * self.data * self.data + self.var
                                           * other.data[:, np.newaxis,
                                                        np.newaxis]
                                           * other.data[:, np.newaxis,
                                                        np.newaxis]) \
                                    / (other.data[:, np.newaxis,
                                                  np.newaxis] ** 4) \
                                    / other.fscale / other.fscale
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
            # cube1 / number = cube2 (cube2[k,j,i]=cube1[k,j,i]/number)
            res = self.copy()
            res.fscale = other / res.fscale
            return res
        try:
            if other.cube:
                if other.data is None or self.shape[0] != other.shape[0] \
                    or self.shape[1] != other.shape[1] \
                   or self.shape[2] != other.shape[2]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                else:
                    return other.__div__(self)
        except IOError as e:
            raise e
        except:
            try:
                if other.image:
                    return other.__div__(self)
            except IOError as e:
                raise e
            except:
                try:
                    if other.spectrum:
                        return other.__div__(self)
                except IOError as e:
                    raise e
                except:
                    raise IOError('Operation forbidden')

    def __pow__(self, other):
        """Computes the power exponent.
        """
        if self.data is None:
            raise ValueError('empty data array')
        res = self.copy()
        if is_float(other) or is_int(other):
            res.data = (self.data ** other) * (self.fscale ** (other - 1))
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
            self.var = 3 * self.var * self.fscale ** 4 / self.data ** 4
        self.data = np.ma.sqrt(self.data) / np.sqrt(self.fscale)

    def sqrt(self):
        """Returns a cube containing the positive square-root
        of data extension.
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
        self.var = None

    def abs(self):
        """Returns a cube containing the absolute value of data extension.
        """
        res = self.copy()
        res._abs()
        return res

    def __getitem__(self, item):
        """Returns the corresponding object:
        cube[k,p,k] = value
        cube[k,:,:] = spectrum
        cube[:,p,q] = image
        cube[:,:,:] = sub-cube
        """
        if isinstance(item, tuple) and len(item) == 3:
            data = self.data[item]
            if is_int(item[0]):
                if is_int(item[1]) and is_int(item[2]):
                    # return a float
                    return data * self.fscale
                else:
                    # return an image
                    from image import Image
                    if is_int(item[1]):
                        shape = (1, data.shape[0])
                    elif is_int(item[2]):
                        shape = (data.shape[0], 1)
                    else:
                        shape = data.shape
                    var = None
                    if self.var is not None:
                        var = self.var[item]
                    try:
                        wcs = self.wcs[item[1], item[2]]
                    except:
                        wcs = None
                    res = Image(shape=shape, wcs=wcs, unit=self.unit,
                                fscale=self.fscale)
                    res.data = data
                    res.var = var
                    return res
            elif is_int(item[1]) and is_int(item[2]):
                # return a spectrum
                from spectrum import Spectrum
                shape = data.shape[0]
                var = None
                if self.var is not None:
                    var = self.var[item]
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Spectrum(shape=shape, wave=wave, unit=self.unit,
                               fscale=self.fscale)
                res.data = data
                res.var = var
                return res
            else:
                # return a cube
                if is_int(item[1]):
                    shape = (data.shape[0], 1, data.shape[1])
                elif is_int(item[2]):
                    shape = (data.shape[0], data.shape[1], 1)
                else:
                    shape = data.shape
                var = None
                if self.var is not None:
                    var = self.var[item]
                try:
                    wcs = self.wcs[item[1], item[2]]
                except:
                    wcs = None
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Cube(shape=shape, wcs=wcs, wave=wave, unit=self.unit,
                           fscale=self.fscale)
                res.data = data
                res.var = var
                return res
        else:
            raise ValueError('Operation forbidden')

    def get_lambda(self, lbda_min, lbda_max=None):
        """Returns the sub-cube corresponding to a wavelength range.

        Parameters
        ----------
        lbda_min : float
                   Minimum wavelength.
        lbda_max : float
                   Maximum wavelength.
        """
        if lbda_max is None:
            lbda_max = lbda_min
        if self.wave is None:
            raise ValueError('Operation forbidden without world coordinates '
                             'along the spectral direction')
        else:
            pix_min = max(0, int(self.wave.pixel(lbda_min)))
            pix_max = min(self.shape[0], int(self.wave.pixel(lbda_max)) + 1)
            if (pix_min + 1) == pix_max:
                return self.data[pix_min, :,:] * self.fscale
            else:
                return self[pix_min:pix_max, :,:]

    def get_step(self):
        """Returns the cube steps [dlbda,dy,dx].
        """
        step = np.empty(3)
        step[0] = self.wave.cdelt
        step[1:] = self.wcs.get_step()
        return step

    def get_range(self):
        """Returns [ [lbda_min,y_min,x_min], [lbda_max,y_max,x_max] ].
        """
        r = np.empty((2, 3))
        r[:, 0] = self.wave.get_range()
        r[:, 1:] = self.wcs.get_range()
        return r

    def get_start(self):
        """Returns [lbda,y,x] corresponding to pixel (0,0,0).
        """
        start = np.empty(3)
        start[0] = self.wave.get_start()
        start[1:] = self.wcs.get_start()
        return start

    def get_end(self):
        """Returns [lbda,y,x] corresponding to pixel (-1,-1,-1).
        """
        end = np.empty(3)
        end[0] = self.wave.get_end()
        end[1:] = self.wcs.get_end()
        return end

    def get_rot(self):
        """Returns the rotation angle.
        """
        return self.wcs.get_rot()

    def get_np_data(self):
        """ Returns numpy masked array containing the flux
            multiplied by scaling factor
        """
        return self.data * self.fscale

    def __setitem__(self, key, other):
        """Sets the corresponding part of data.
        """
        #self.data[key] = value
        if self.data is None:
            raise ValueError('empty data array')
        try:
            self.data[key] = other / np.double(self.fscale)
        except:
            try:
                # other is a cube
                if other.cube:
                    try:
                        if (self.wcs is not None and other.wcs is not None
                            and (self.wcs.get_step() != other.wcs.get_step())
                            .any()) \
                            or (self.wave is not None and other.wave is not None
                                and (self.wave.get_step() !=
                                     other.wave.get_step())):
                            d = {'class': 'Cube', 'method': '__setitem__'}
                            logger.warning("cubes with different steps",
                                           extra=d)
                        self.data[key] = other.data \
                            * np.double(other.fscale / self.fscale)
                    except:
                        self.data[key] = other.data \
                            * np.double(other.fscale / self.fscale)
            except:
                try:
                    # other is an image
                    if other.image:
                        try:
                            if self.wcs is not None and other.wcs is not None \
                                    and (self.wcs.get_step() != other.wcs.get_step())\
                                    .any():
                                d = {'class': 'Cube', 'method': '__setitem__'}
                                logger.warning("cube & image with different '\
                                'steps", extra=d)
                            self.data[key] = other.data \
                                * np.double(other.fscale / self.fscale)
                        except:
                            self.data[key] = other.data \
                                * np.double(other.fscale / self.fscale)
                except:
                    try:
                        # other is a spectrum
                        if other.spectrum:
                            if self.wave is not None \
                                and other.wave is not None \
                                and (self.wave.get_step() !=
                                     other.wave.get_step()):
                                d = {'class': 'Cube', 'method': '__setitem__'}
                                logger.warning('cube & spectrum with '
                                               'different steps', extra=d)
                            self.data[key] = other.data \
                                * np.double(other.fscale / self.fscale)
                    except:
                        raise IOError('Operation forbidden')

    def set_wcs(self, wcs=None, wave=None):
        """Sets the world coordinates (spatial and/or spectral).

        Parameters
        ----------
        wcs : :class:`mpdaf.obj.WCS`
              World coordinates.
        wave : :class:`mpdaf.obj.WaveCoord`
               Wavelength coordinates.
        """
        if wcs is not None:
            self.wcs = wcs
            self.wcs.naxis1 = self.shape[2]
            self.wcs.naxis2 = self.shape[1]
            if wcs.naxis1 != 0 and wcs.naxis2 != 0 \
                and (wcs.naxis1 != self.shape[2]
                     or wcs.naxis2 != self.shape[1]):
                d = {'class': 'Cube', 'method': 'set_wcs'}
                logger.warning('world coordinates and data have not the same '
                               'dimensions', extra=d)
        if wave is not None:
            if wave.shape is not None and wave.shape != self.shape[0]:
                d = {'class': 'Cube', 'method': 'set_wcs'}
                logger.warning('wavelength coordinates and data have not '
                               'the same dimensions', extra=d)
            self.wave = wave
            self.wave.shape = self.shape[0]

    def set_var(self, var):
        """Sets the variance array.

        Parameters
        ----------
        var : float array
              Input variance array. If None, variance is set with zeros.
        """
        if var is None:
            self.var = np.zeros((self.shape[0], self.shape[1], self.shape[2]))
        else:
            if self.shape[0] == np.shape(var)[0] \
                    and self.shape[1] == np.shape(var)[1] \
                    and self.shape[2] == np.shape(var)[2]:
                self.var = var
            else:
                raise ValueError('var and data have not the same dimensions.')

    def sum(self, axis=None):
        """Returns the sum over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints
               Axis or axes along which a sum is performed.

               The default (axis = None) is perform a sum over all
               the dimensions of the cube and returns a float.

               axis = 0  is perform a sum over the wavelength dimension
               and returns an image.

               axis = (1,2) is perform a sum over the (X,Y) axes and
               returns a spectrum.

               Other cases return None.
        """
        if axis is None:
            return self.data.sum() * self.fscale
        elif axis == 0:
            # return an image
            from image import Image
            data = np.ma.sum(self.data, axis)
            if self.var is not None:
                var = (np.ma.sum(np.ma.masked_invalid(self.var), axis)).filled(np.NaN)
            else:
                var = None
            res = Image(shape=data.shape, wcs=self.wcs, unit=self.unit,
                        fscale=self.fscale)
            res.data = data
            res.var = var
            return res
        elif axis == tuple([1, 2]):
            # return a spectrum
            from spectrum import Spectrum
            data = np.ma.sum(self.data, axis=1).sum(axis=1)
            if self.var is not None:
                var = np.ma.sum(np.ma.masked_invalid(self.var), axis=1).sum(axis=1).filled(np.NaN)
            else:
                var = None
            res = Spectrum(shape=data.shape[0], wave=self.wave,
                           unit=self.unit, fscale=self.fscale)
            res.data = data
            res.var = var
            return res
        else:
            return None

    def mean(self, axis=None):
        """ Returns the mean over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints
               Axis or axes along which a mean is performed.

               The default (axis = None) is perform a mean over all the
               dimensions of the cube and returns a float.

               axis = 0  is perform a mean over the wavelength dimension
               and returns an image.

               axis = (1,2) is perform a mean over the (X,Y) axes and
               returns a spectrum.

               Other cases return None.
        """
        if axis is None:
            return self.data.mean() * self.fscale
        elif axis == 0:
            # return an image
            from image import Image
            data = np.ma.mean(self.data, axis)
            if self.var is not None:
                var = np.ma.mean(np.ma.masked_invalid(self.var), axis).filled(np.NaN)
            else:
                var = None
            res = Image(shape=data.shape, wcs=self.wcs,
                        unit=self.unit, fscale=self.fscale)
            res.data = data
            res.var = var
            return res
        elif axis == tuple([1, 2]):
            # return a spectrum
            from spectrum import Spectrum
            data = np.ma.mean(self.data, axis=1).mean(axis=1)
            if self.var is not None:
                var = np.ma.mean(np.ma.masked_invalid(self.var), axis=1).mean(axis=1).filled(np.NaN)
            else:
                var = None
            res = Spectrum(notnoise=True, shape=data.shape[0],
                           wave=self.wave,
                           unit=self.unit, fscale=self.fscale)
            res.data = data
            res.var = var
            return res
        else:
            return None

    def median(self, axis=None):
        """ Returns the median over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints
               Axis or axes along which a mean is performed.

               The default (axis = None) is perform a mean over all the
               dimensions of the cube and returns a float.

               axis = 0  is perform a mean over the wavelength dimension
               and returns an image.

               axis = (1,2) is perform a mean over the (X,Y) axes and
               returns a spectrum.

               Other cases return None.
        """
        if axis is None:
            return self.data.median() * self.fscale
        elif axis == 0:
            # return an image
            from image import Image
            data = np.ma.median(self.data, axis)
            if self.var is not None:
                var = np.ma.median(np.ma.masked_invalid(self.var), axis).filled(np.NaN)
            else:
                var = None
            res = Image(shape=data.shape, wcs=self.wcs,
                        unit=self.unit, fscale=self.fscale)
            res.data = data
            res.var = var
            return res
        elif axis == tuple([1, 2]):
            # return a spectrum
            from spectrum import Spectrum
            data = np.ma.median(self.data, axis=1).median(axis=1)
            if self.var is not None:
                var = np.ma.median(np.ma.masked_invalid(self.var), axis=1).median(axis=1).filled(np.NaN)
            else:
                var = None
            res = Spectrum(notnoise=True, shape=data.shape[0],
                           wave=self.wave,
                           unit=self.unit, fscale=self.fscale)
            res.data = data
            res.var = var
            return res
        else:
            return None

    def truncate(self, coord, mask=True):
        """ Truncates the cube and return a sub-cube.

        Parameters
        ----------
        coord : array
                array containing the sub-cube boundaries
                [[lbda_min,y_min,x_min], [lbda_max,y_max,x_max]]
                (output of `mpdaf.obj.cube.get_range`)
                x and y in degrees
        mask  : boolean
                if True, pixels outside [y_min,y_max]
                and [x_min,x_max] are masked.
        """
        lmin = coord[0][0]
        y_min = coord[0][1]
        x_min = coord[0][2]
        lmax = coord[1][0]
        y_max = coord[1][1]
        x_max = coord[1][2]

        skycrd = [[y_min, x_min], [y_min, x_max], [y_max, x_min], [y_max, x_max]]
        pixcrd = self.wcs.sky2pix(skycrd)

        imin = int(np.min(pixcrd[:, 0]))
        if imin < 0:
            imin = 0
        imax = int(np.max(pixcrd[:, 0])) + 1
        if imax > self.shape[1]:
            imax = self.shape[1]
        jmin = int(np.min(pixcrd[:, 1]))
        if jmin < 0:
            jmin = 0
        jmax = int(np.max(pixcrd[:, 1])) + 1
        if jmax > self.shape[2]:
            jmax = self.shape[2]

        kmin = max(0, self.wave.pixel(lmin, nearest=True))
        kmax = min(self.shape[0], self.wave.pixel(lmax, nearest=True) + 1)

        if kmin == kmax:
            raise ValueError('Minimum and maximum wavelengths are equal')

        if kmax == kmin + 1:
            raise ValueError('Minimum and maximum wavelengths are outside'
                             ' the spectrum range')

        data = self.data[kmin:kmax, imin:imax, jmin:jmax]
        shape = np.array((data.shape[0], data.shape[1], data.shape[2]))

        if self.var is not None:
            var = self.var[kmin:kmax, imin:imax, jmin:jmax]
        else:
            var = None
        try:
            wcs = self.wcs[imin:imax, jmin:jmax]
        except:
            wcs = None
        try:
            wave = self.wave[kmin:kmax]
        except:
            wave = None

        if mask:
            # mask outside pixels
            m = np.ma.make_mask_none(data.shape)
            for j in range(shape[1]):
                pixcrd = np.array([np.ones(shape[2]) * j,
                                   np.arange(shape[2])]).T
                skycrd = self.wcs.pix2sky(pixcrd)
                test_ra_min = np.array(skycrd[:, 1]) < x_min
                test_ra_max = np.array(skycrd[:, 1]) > x_max
                test_dec_min = np.array(skycrd[:, 0]) < y_min
                test_dec_max = np.array(skycrd[:, 0]) > y_max
                m[:, j, :] = test_ra_min + test_ra_max \
                    + test_dec_min + test_dec_max
            try:
                m = np.ma.mask_or(m, np.ma.getmask(data))
                data = np.ma.MaskedArray(data, mask=m)
            except:
                pass

        res = Cube(shape=shape, wcs=wcs, wave=wave,
                   unit=self.unit, fscale=self.fscale)
        res.data = data
        res.var = var
        return res

    def _rebin_factor_(self, factor):
        '''Shrinks the size of the cube by factor.
        New size is an integer multiple of the original size.

        Parameters
        ----------
        factor : (integer,integer,integer)
                 Factor in z, y and x.
                 Python notation: (nz,ny,nx)
        '''
        # new size is an integer multiple of the original size
        assert not np.sometrue(np.mod(self.shape[0], factor[0]))
        assert not np.sometrue(np.mod(self.shape[1], factor[1]))
        assert not np.sometrue(np.mod(self.shape[2], factor[2]))
        # shape
        self.shape = np.array((self.shape[0] / factor[0],
                               self.shape[1] / factor[1],
                               self.shape[2] / factor[2]))
        # data
        self.data = self.data.reshape(self.shape[0], factor[0],
                                      self.shape[1], factor[1],
                                      self.shape[2], factor[2])\
            .sum(1).sum(2).sum(3) \
            / factor[0] / factor[1] / factor[2]
        # variance
        if self.var is not None:
            self.var = self.var.reshape(self.shape[0], factor[0],
                                        self.shape[1], factor[1],
                                        self.shape[2], factor[2])\
                .sum(1).sum(2).sum(3) \
                / factor[0] / factor[1] / factor[2] \
                / factor[0] / factor[1] / factor[2]
        # coordinates
        #cdelt = self.wcs.get_step()
        self.wcs = self.wcs.rebin_factor(factor[1:])
        crval = self.wave.coord()[0:factor[0]].sum() / factor[0]
        self.wave = WaveCoord(1, self.wave.cdelt * factor[0], crval,
                              self.wave.cunit, self.shape[0])

    def _rebin_factor(self, factor, margin='center', flux=False):
        '''Shrinks the size of the cube by factor.

          Parameters
          ----------
          factor : integer or (integer,integer,integer)
                   Factor in z, y and x. Python notation: (nz,ny,nx).
          flux   : boolean
                   This parameters is used if new size is not an integer
                   multiple of the original size.

                   If Flux is False, the cube is truncated and the flux
                   is not conserved.

                   If Flux is True, margins are added to the cube to
                   conserve the flux.
          margin : 'center' or 'origin'
                   This parameters is used if new size is not an
                   integer multiple of the original size.

                   In 'center' case, cube is truncated/pixels are added on the left
                   and on the right, on the bottom and of the top of the cube.

                   In 'origin'case, cube is truncated/pixels are added at the end
                   along each direction
        '''
        if is_int(factor):
            factor = (factor, factor, factor)
        if not np.sometrue(np.mod(self.shape[0], factor[0])) \
                and not np.sometrue(np.mod(self.shape[1], factor[1])) \
                and not np.sometrue(np.mod(self.shape[2], factor[2])):
            # new size is an integer multiple of the original size
            self._rebin_factor_(factor)
            return None
        else:
            factor = np.array(factor)
            newshape = self.shape / factor
            n = self.shape - newshape * factor

            if n[0] == 0:
                n0_left = 0
                n0_right = self.shape[0]
            else:
                if margin == 'origin' or n[0] == 1:
                    n0_left = 0
                    n0_right = -n[0]
                else:
                    n0_left = n[0] / 2
                    n0_right = self.shape[0] - n[0] + n0_left
            if n[1] == 0:
                n1_left = 0
                n1_right = self.shape[1]
            else:
                if margin == 'origin' or n[1] == 1:
                    n1_left = 0
                    n1_right = -n[1]
                else:
                    n1_left = n[1] / 2
                    n1_right = self.shape[1] - n[1] + n1_left
            if n[2] == 0:
                n2_left = 0
                n2_right = self.shape[2]
            else:
                if margin == 'origin' or n[2] == 1:
                    n2_left = 0
                    n2_right = -n[2]
                else:
                    n2_left = n[2] / 2
                    n2_right = self.shape[2] - n[2] + n2_left

            cub = self[n0_left:n0_right, n1_left:n1_right, n2_left:n2_right]
            cub._rebin_factor_(factor)

            if flux is False:
                self.shape = cub.shape
                self.data = cub.data
                self.var = cub.var
                self.wave = cub.wave
                self.wcs = cub.wcs
                return None
            else:
                newshape = cub.shape
                wave = cub.wave
                wcs = cub.wcs
                if n0_left != 0:
                    newshape[0] += 1
                    wave.crpix += 1
                    wave.shape += 1
                    l_left = 1
                else:
                    l_left = 0
                if n0_right != self.shape[0]:
                    newshape[0] += 1
                    l_right = newshape[0] - 1
                else:
                    l_right = newshape[0]

                if n1_left != 0:
                    newshape[1] += 1
                    wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                    wcs.set_naxis2(wcs.naxis2 + 1)
                    p_left = 1
                else:
                    p_left = 0
                if n1_right != self.shape[1]:
                    newshape[1] += 1
                    wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                    p_right = newshape[1] - 1
                else:
                    p_right = newshape[1]

                if n2_left != 0:
                    newshape[2] += 1
                    wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                    wcs.set_naxis1(wcs.naxis1 + 1)
                    q_left = 1
                else:
                    q_left = 0
                if n2_right != self.shape[2]:
                    newshape[2] += 1
                    wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                    q_right = newshape[2] - 1
                else:
                    q_right = newshape[2]

                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[l_left:l_right, p_left:p_right, q_left:q_right] = cub.data
                mask[l_left:l_right, p_left:p_right, q_left:q_right] = \
                    cub.data.mask

                if self.var is None:
                    var = None
                else:
                    var = np.empty(newshape)
                    var[l_left:l_right, p_left:p_right, q_left:q_right] = cub.var

                F = factor[0] * factor[1] * factor[2]
                F2 = F * F

                if cub.shape[0] != newshape[0]:
                    d = self.data[n0_right:, n1_left:n1_right, n2_left:n2_right]\
                        .sum(axis=0)\
                        .reshape(cub.shape[1], factor[1], cub.shape[2], factor[2])\
                        .sum(1).sum(2) / F
                    data[-1, p_left:q_left, q_left:q_right] = d.data
                    mask[-1, p_left:q_left, q_left:q_right] = d.mask
                    if var is not None:
                        var[-1, p_left:q_left, q_left:q_right] = \
                            self.var[n0_right:, n1_left:n1_right, n2_left:n2_right]\
                            .sum(axis=0)\
                            .reshape(cub.shape[1], factor[1],
                                     cub.shape[2], factor[2])\
                            .sum(1).sum(2) / F2
                if l_left == 1:
                    d = self.data[:n0_left, n1_left:n1_right, n2_left:n2_right]\
                        .sum(axis=0)\
                        .reshape(cub.shape[1], factor[1], cub.shape[2], factor[2])\
                        .sum(1).sum(2) / F
                    data[0, p_left:q_left, q_left:q_right] = d.data
                    mask[0, p_left:q_left, q_left:q_right] = d.mask
                    if var is not None:
                        var[0, p_left:q_left, q_left:q_right] = \
                            self.var[:n0_left, n1_left:n1_right, n2_left:n2_right]\
                            .sum(axis=0)\
                            .reshape(cub.shape[1], factor[1],
                                     cub.shape[2], factor[2])\
                            .sum(1).sum(2) / F2
                if cub.shape[1] != newshape[1]:
                    d = self.data[n0_left:n0_right, n1_right:,
                                  n2_left:n2_right]\
                        .sum(axis=1).reshape(cub.shape[0], factor[0],
                                             cub.shape[2], factor[2])\
                        .sum(1).sum(2) / F
                    data[l_left:l_right, -1, q_left:q_right] = d.data
                    mask[l_left:l_right, -1, q_left:q_right] = d.mask
                    if var is not None:
                        var[l_left:l_right, -1, q_left:q_right] = \
                            self.var[n0_left:n0_right, n1_right:, n2_left:n2_right]\
                            .sum(axis=1)\
                            .reshape(cub.shape[0], factor[0],
                                     cub.shape[2], factor[2])\
                            .sum(1).sum(2) / F2
                if p_left == 1:
                    d = self.data[n0_left:n0_right, :n1_left, n2_left:n2_right]\
                        .sum(axis=1).reshape(cub.shape[0], factor[0],
                                             cub.shape[2], factor[2])\
                        .sum(1).sum(2) / F
                    data[l_left:l_right, 0, q_left:q_right] = d.data
                    mask[l_left:l_right, 0, q_left:q_right] = d.mask
                    if var is not None:
                        var[l_left:l_right, 0, q_left:q_right] = \
                            self.var[n0_left:n0_right, :n1_left, n2_left:n2_right]\
                            .sum(axis=1).reshape(cub.shape[0], factor[0],
                                                 cub.shape[2], factor[2])\
                            .sum(1).sum(2) / F2

                if cub.shape[2] != newshape[2]:
                    d = self.data[n0_left:n0_right,
                                  n1_left:n1_right:, n2_right:]\
                        .sum(axis=2)\
                        .reshape(cub.shape[0], factor[0], cub.shape[1], factor[1])\
                        .sum(1).sum(2) / F
                    data[l_left:l_right, p_left:p_right, -1] = d.data
                    mask[l_left:l_right, p_left:p_right, -1] = d.mask
                    if var is not None:
                        var[l_left:l_right, p_left:p_right, -1] = \
                            self.var[n0_left:n0_right,
                                     n1_left:n1_right:, n2_right:]\
                            .sum(axis=2).reshape(cub.shape[0], factor[0],
                                                 cub.shape[1], factor[1])\
                            .sum(1).sum(2) / F2
                if q_left == 1:
                    d = self.data[n0_left:n0_right,
                                  n1_left:n1_right:, :n2_left]\
                        .sum(axis=2).reshape(cub.shape[0], factor[0],
                                             cub.shape[1], factor[1])\
                        .sum(1).sum(2) / F
                    data[l_left:l_right, p_left:p_right, 0] = d.data
                    mask[l_left:l_right, p_left:p_right, 0] = d.mask
                    if var is not None:
                        var[l_left:l_right, p_left:p_right, 0] = \
                            self.var[n0_left:n0_right, n1_left:n1_right:, :n2_left]\
                            .sum(axis=2)\
                            .reshape(cub.shape[0], factor[0],
                                     cub.shape[1], factor[1])\
                            .sum(1).sum(2) / F2

                if l_left == 1 and p_left == 1 and q_left == 1:
                    data[0, 0, 0] = \
                        self.data[:n0_left, :n1_left, :n2_left].sum() / F
                    mask[0, 0, 0] = self.mask[:n0_left, :n1_left, :n2_left].any()
                    if var is not None:
                        var[0, 0, 0] = \
                            self.var[:n0_left, :n1_left, :n2_left].sum() / F2
                if l_left == 1 and p_right == (newshape[1] - 1) \
                        and q_left == 1:
                    data[0, -1, 0] = \
                        self.data[:n0_left, n1_right:, :n2_left].sum() / F
                    mask[0, -1, 0] = \
                        self.mask[:n0_left, n1_right:, :n2_left].any()
                    if var is not None:
                        var[0, -1, 0] = \
                            self.var[:n0_left, n1_right:, :n2_left].sum() / F2
                if l_left == 1 and p_right == (newshape[1] - 1) \
                        and q_right == (newshape[2] - 1):
                    data[0, -1, -1] = \
                        self.data[:n0_left, n1_right:, n2_right:].sum() / F
                    mask[0, -1, -1] = \
                        self.mask[:n0_left, n1_right:, n2_right:].any()
                    if var is not None:
                        var[0, -1, -1] = \
                            self.var[:n0_left, n1_right:, n2_right:].sum() / F2
                if l_left == 1 and p_left == 1 and \
                        q_right == (newshape[2] - 1):
                    data[0, 0, -1] = \
                        self.data[:n0_left, :n1_left, n2_right:].sum() / F
                    mask[0, 0, -1] = \
                        self.mask[:n0_left, :n1_left, n2_right:].any()
                    if var is not None:
                        var[0, 0, -1] = \
                            self.var[:n0_left, :n1_left, n2_right:].sum() / F2
                if l_left == (newshape[0] - 1) and p_left == 1 \
                        and q_left == 1:
                    data[-1, 0, 0] = \
                        self.data[n0_right:, :n1_left, :n2_left].sum() / F
                    mask[-1, 0, 0] = \
                        self.mask[n0_right:, :n1_left, :n2_left].any()
                    if var is not None:
                        var[-1, 0, 0] = \
                            self.var[n0_right:, :n1_left, :n2_left].sum() / F2
                if l_left == (newshape[0] - 1) \
                        and p_right == (newshape[1] - 1) and q_left == 1:
                    data[-1, -1, 0] = \
                        self.data[n0_right:, n1_right:, :n2_left].sum() / F
                    mask[-1, -1, 0] = \
                        self.mask[n0_right:, n1_right:, :n2_left].any()
                    if var is not None:
                        var[-1, -1, 0] = \
                            self.var[n0_right:, n1_right:, :n2_left].sum() / F2
                if l_left == (newshape[0] - 1) \
                        and p_right == (newshape[1] - 1) \
                        and q_right == (newshape[2] - 1):
                    data[-1, -1, -1] = \
                        self.data[n0_right:, n1_right:, n2_right:].sum() / F
                    mask[-1, -1, -1] = \
                        self.mask[n0_right:, n1_right:, n2_right:].any()
                    if var is not None:
                        var[-1, -1, -1] = \
                            self.var[n0_right:, n1_right:, n2_right:].sum() / F2
                if l_left == (newshape[0] - 1) and p_left == 1 \
                        and q_right == (newshape[2] - 1):
                    data[-1, 0, -1] = \
                        self.data[n0_right:, :n1_left, n2_right:].sum() / F
                    mask[-1, 0, -1] = \
                        self.mask[n0_right:, :n1_left, n2_right:].any()
                    if var is not None:
                        var[-1, 0, -1] = \
                            self.var[n0_right:, :n1_left, n2_right:].sum() / F2

                if p_left == 1 and q_left == 1:
                    d = self.data[n0_left:n0_right, :n1_left, :n2_left]\
                        .sum(axis=2).sum(axis=1)\
                        .reshape(cub.shape[0], factor[0]).sum(1) / F
                    data[l_left:l_right, 0, 0] = d.data
                    mask[l_left:l_right, 0, 0] = d.mask
                    if var is not None:
                        var[l_left:l_right, 0, 0] = \
                            self.var[n0_left:n0_right, :n1_left, :n2_left]\
                            .sum(axis=2).sum(axis=1)\
                            .reshape(cub.shape[0], factor[0]).sum(1) / F2
                if l_left == 1 and p_left == 1:
                    d = self.data[:n0_left, :n1_left, n2_left:n2_right]\
                        .sum(axis=0).sum(axis=0)\
                        .reshape(cub.shape[2], factor[2]).sum(1) / F
                    data[0, 0, q_left:q_right] = d.data
                    mask[0, 0, q_left:q_right] = d.mask
                    if var is not None:
                        var[0, 0, q_left:q_right] = \
                            self.var[:n0_left, :n1_left, n2_left:n2_right]\
                            .sum(axis=0).sum(axis=0)\
                            .reshape(cub.shape[2], factor[2]).sum(1) / F2
                if l_left == 1 and q_left == 1:
                    d = self.data[:n0_left, n1_left:n1_right, :n2_left]\
                        .sum(axis=2).sum(axis=0)\
                        .reshape(cub.shape[1], factor[1]).sum(1) / F
                    data[0, p_left:p_right, 0] = d.data
                    mask[0, p_left:p_right, 0] = d.mask
                    if var is not None:
                        var[0, p_left:p_right, 0] = \
                            self.var[:n0_left, n1_left:n1_right, :n2_left]\
                            .sum(axis=2).sum(axis=0)\
                            .reshape(cub.shape[1], factor[1]).sum(1) / F2

                if p_left == 1 and q_right == (newshape[2] - 1):
                    d = self.data[n0_left:n0_right, :n1_left, n2_right:]\
                        .sum(axis=2).sum(axis=1)\
                        .reshape(cub.shape[0], factor[0]).sum(1) / F
                    data[l_left:l_right, 0, -1] = d.data
                    mask[l_left:l_right, 0, -1] = d.mask
                    if var is not None:
                        var[l_left:l_right, 0, -1] = \
                            self.var[n0_left:n0_right, :n1_left, n2_right:]\
                            .sum(axis=2).sum(axis=1)\
                            .reshape(cub.shape[0], factor[0]).sum(1) / F2
                if l_left == 1 and p_right == (newshape[1] - 1):
                    d = self.data[:n0_left, n1_right:, n2_left:n2_right]\
                        .sum(axis=0).sum(axis=0)\
                        .reshape(cub.shape[2], factor[2]).sum(1) / F
                    data[0, -1, q_left:q_right] = d.data
                    mask[0, -1, q_left:q_right] = d.mask
                    if var is not None:
                        var[0, -1, q_left:q_right] = \
                            self.var[:n0_left, n1_right:, n2_left:n2_right]\
                            .sum(axis=0).sum(axis=0)\
                            .reshape(cub.shape[2], factor[2]).sum(1) / F2
                if l_left == 1 and q_right == (newshape[2] - 1):
                    d = self.data[:n0_left, n1_left:n1_right, n2_right:]\
                        .sum(axis=2).sum(axis=0)\
                        .reshape(cub.shape[1], factor[1]).sum(1) / F
                    data[0, p_left:p_right, -1] = d.data
                    mask[0, p_left:p_right, -1] = d.mask
                    if var is not None:
                        var[0, p_left:p_right, -1] = \
                            self.var[:n0_left, n1_left:n1_right, n2_right:]\
                            .sum(axis=2).sum(axis=0)\
                            .reshape(cub.shape[1], factor[1]).sum(1) / F2

                if p_right == (newshape[1] - 1) and q_left == 1:
                    d = self.data[n0_left:n0_right, n1_right:, :n2_left]\
                        .sum(axis=2).sum(axis=1)\
                        .reshape(cub.shape[0], factor[0]).sum(1) / F
                    data[l_left:l_right, -1, 0] = d.data
                    mask[l_left:l_right, -1, 0] = d.mask
                    if var is not None:
                        var[l_left:l_right, -1, 0] = \
                            self.var[n0_left:n0_right, n1_right:, :n2_left]\
                            .sum(axis=2).sum(axis=1)\
                            .reshape(cub.shape[0], factor[0]).sum(1) / F2
                if l_right == (newshape[0] - 1) and p_left == 1:
                    d = self.data[n0_right:, :n1_left, n2_left:n2_right]\
                        .sum(axis=0).sum(axis=0)\
                        .reshape(cub.shape[2], factor[2]).sum(1) / F
                    data[-1, 0, q_left:q_right] = d.data
                    mask[-1, 0, q_left:q_right] = d.mask
                    if var is not None:
                        var[-1, 0, q_left:q_right] = \
                            self.var[n0_right:, :n1_left, n2_left:n2_right]\
                            .sum(axis=0).sum(axis=0)\
                            .reshape(cub.shape[2], factor[2]).sum(1) / F2
                if l_right == (newshape[0] - 1) and q_left == 1:
                    d = self.data[n0_right:, n1_left:n1_right, :n2_left]\
                        .sum(axis=2).sum(axis=0)\
                        .reshape(cub.shape[1], factor[1]).sum(1) / F
                    data[-1, p_left:p_right, 0] = d.data
                    mask[-1, p_left:p_right, 0] = d.mask
                    if var is not None:
                        var[-1, p_left:p_right, 0] = \
                            self.var[n0_right:, n1_left:n1_right, :n2_left]\
                            .sum(axis=2).sum(axis=0)\
                            .reshape(cub.shape[1], factor[1]).sum(1) / F2

                if p_right == (newshape[1] - 1) \
                        and q_right == (newshape[2] - 1):
                    d = self.data[n0_left:n0_right, n1_right:, n2_right:]\
                        .sum(axis=2).sum(axis=1)\
                        .reshape(cub.shape[0], factor[0]).sum(1) / F
                    data[l_left:l_right, -1, -1] = d.data
                    mask[l_left:l_right, -1, -1] = d.mask
                    if var is not None:
                        var[l_left:l_right, -1, -1] = \
                            self.var[n0_left:n0_right, n1_right:, n2_right:]\
                            .sum(axis=2).sum(axis=1)\
                            .reshape(cub.shape[0], factor[0]).sum(1) / F2
                if l_right == (newshape[0] - 1) \
                        and p_right == (newshape[1] - 1):
                    d = self.data[n0_right:, n1_right:, n2_left:n2_right]\
                        .sum(axis=0).sum(axis=0)\
                        .reshape(cub.shape[2], factor[2]).sum(1) / F
                    data[-1, -1, q_left:q_right] = d.data
                    mask[-1, -1, q_left:q_right] = d.mask
                    if var is not None:
                        var[-1, -1, q_left:q_right] = \
                            self.var[n0_right:, n1_right:, n2_left:n2_right]\
                            .sum(axis=0).sum(axis=0)\
                            .reshape(cub.shape[2], factor[2]).sum(1) / F2
                if l_right == (newshape[0] - 1) \
                        and q_right == (newshape[2] - 1):
                    d = self.data[n0_right:, n1_left:n1_right, n2_right:]\
                        .sum(axis=2).sum(axis=0)\
                        .reshape(cub.shape[1], factor[1]).sum(1) / F
                    data[-1, p_left:p_right, -1] = d.data
                    mask[-1, p_left:p_right, -1] = d.mask
                    if var is not None:
                        var[-1, p_left:p_right, -1] = \
                            self.var[n0_right:, n1_left:n1_right, n2_right:]\
                            .sum(axis=2).sum(axis=0)\
                            .reshape(cub.shape[1], factor[1]).sum(1) / F2

                self.shape = newshape
                self.wcs = wcs
                self.wave = wave
                self.data = np.ma.array(data, mask=mask)
                self.var = var
                return None

    def rebin_factor(self, factor, margin='center', flux=False):
        '''Shrinks the size of the cube by factor.

          Parameters
          ----------
          factor : integer or (integer,integer,integer)
                   Factor in z, y and x. Python notation: (nz,ny,nx).
          flux   : boolean
                   This parameters is used if new size is
                   not an integer multiple of the original size.

                   If Flux is False, the cube is truncated and the flux
                   is not conserved.

                   If Flux is True, margins are added to the cube
                   to conserve the flux.
          margin : 'center' or 'origin'
                    This parameters is used if new size is not
                    an integer multiple of the original size.

                    In 'center' case, cube is truncated/pixels are added on the left
                    and on the right, on the bottom and of the top of the cube.

                    In 'origin'case, cube is truncated/pixels are added
                    at the end along each direction
        '''
        if is_int(factor):
            factor = (factor, factor, factor)
        factor = np.array(factor)
        if factor[0] < 1:
            factor[0] = 1
        if factor[0] > self.shape[0]:
            factor[0] = self.shape[0]
        if factor[1] < 1:
            factor[1] = 1
        if factor[1] > self.shape[1]:
            factor[1] = self.shape[1]
        if factor[2] < 1:
            factor[2] = 1
        if factor[2] > self.shape[2]:
            factor[2] = self.shape[2]

        res = self.copy()
        res._rebin_factor(factor, margin, flux)
        return res

    def _med_(self, k, p, q, kfactor, pfactor, qfactor):
        return np.ma.median(self.data[k * kfactor:(k + 1) * kfactor,
                                      p * pfactor:(p + 1) * pfactor,
                                      q * qfactor:(q + 1) * qfactor])

    def _rebin_median_(self, factor):
        '''Shrinks the size of the cube by factor.
        New size is an integer multiple of the original size.

        Parameter
        ---------
        factor : (integer,integer,integer)
                 Factor in z, y and x.
                Python notation: (nz,ny,nx)
        '''
        # new size is an integer multiple of the original size
        assert not np.sometrue(np.mod(self.shape[0], factor[0]))
        assert not np.sometrue(np.mod(self.shape[1], factor[1]))
        assert not np.sometrue(np.mod(self.shape[2], factor[2]))
        # shape
        self.shape = np.array((self.shape[0] / factor[0],
                               self.shape[1] / factor[1],
                               self.shape[2] / factor[2]))
        # data
        grid = np.lib.index_tricks.nd_grid()
        g = grid[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]]
        vfunc = np.vectorize(self._med_)
        data = vfunc(g[0], g[1], g[2], factor[0], factor[1], factor[2])
        mask = self.data.mask.reshape(self.shape[0], factor[0],
                                      self.shape[1], factor[1],
                                      self.shape[2], factor[2])\
            .sum(1).sum(2).sum(3)
        self.data = np.ma.array(data, mask=mask)
        # variance
        self.var = None
        # coordinates
        #cdelt = self.wcs.get_step()
        self.wcs = self.wcs.rebin_factor(factor[1:])
        crval = self.wave.coord()[0:factor[0]].sum() / factor[0]
        self.wave = WaveCoord(1, self.wave.cdelt * factor[0],
                              crval, self.wave.cunit, self.shape[0])

    def rebin_median(self, factor, margin='center'):
        '''Shrinks the size of the cube by factor.

        Parameters
        ----------
        factor : integer or (integer,integer,integer)
                Factor in z, y and x. Python notation: (nz,ny,nx).
        margin : 'center' or 'origin'
                This parameters is used if new size is not an
                integer multiple of the original size.

                In 'center' case, cube is truncated on the left and on the right,
                on the bottom and of the top of the cube.

                In 'origin'case, cube is truncatedat the end along each direction

        Returns
        -------
        out : :class:`mpdaf.obj.Cube`
        '''
        if is_int(factor):
            factor = (factor, factor, factor)
        factor = np.array(factor)
        if factor[0] < 1:
            factor[0] = 1
        if factor[0] > self.shape[0]:
            factor[0] = self.shape[0]
        if factor[1] < 1:
            factor[1] = 1
        if factor[1] > self.shape[1]:
            factor[1] = self.shape[1]
        if factor[2] < 1:
            factor[2] = 1
        if factor[2] > self.shape[2]:
            factor[2] = self.shape[2]
        if not np.sometrue(np.mod(self.shape[0], factor[0])) \
                and not np.sometrue(np.mod(self.shape[1], factor[1])) \
                and not np.sometrue(np.mod(self.shape[2], factor[2])):
            # new size is an integer multiple of the original size
            res = self.copy()
        else:
            newshape = self.shape / factor
            n = self.shape - newshape * factor

            if n[0] == 0:
                n0_left = 0
                n0_right = self.shape[0]
            else:
                if margin == 'origin' or n[0] == 1:
                    n0_left = 0
                    n0_right = -n[0]
                else:
                    n0_left = n[0] / 2
                    n0_right = self.shape[0] - n[0] + n0_left
            if n[1] == 0:
                n1_left = 0
                n1_right = self.shape[1]
            else:
                if margin == 'origin' or n[1] == 1:
                    n1_left = 0
                    n1_right = -n[1]
                else:
                    n1_left = n[1] / 2
                    n1_right = self.shape[1] - n[1] + n1_left
            if n[2] == 0:
                n2_left = 0
                n2_right = self.shape[2]
            else:
                if margin == 'origin' or n[2] == 1:
                    n2_left = 0
                    n2_right = -n[2]
                else:
                    n2_left = n[2] / 2
                    n2_right = self.shape[2] - n[2] + n2_left

            res = self[n0_left:n0_right, n1_left:n1_right, n2_left:n2_right]

        res._rebin_median_(factor)
        return res

    def loop_spe_multiprocessing(self, f, cpu=None, verbose=True, **kargs):
        """loops over all spectra to apply a function/method.
        Returns the resulting cube.
        Multiprocessing is used.

        Parameters
        ----------
        f       : function or :class:`mpdaf.obj.Spectrum` method
                  Spectrum method or function that the first argument
                  is a spectrum object.
        cpu     : integer
                  number of CPUs. It is also possible to set
                  the mpdaf.CPU global variable.
        verbose : boolean
                  if True, progression is printed.
        kargs   : kargs
                  can be used to set function arguments.

        Returns
        -------
        out : :class:`mpdaf.obj.Cube` if f returns :class:`mpdaf.obj.Spectrum`,
        out : :class:`mpdaf.obj.Image` if f returns a number,
        out : np.array(dtype=object) in others cases.
        """
        from mpdaf import CPU
        if cpu is not None and cpu < multiprocessing.cpu_count():
            cpu_count = cpu
        elif CPU != 0 and CPU < multiprocessing.cpu_count():
            cpu_count = CPU
        else:
            cpu_count = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(processes=cpu_count)
        processlist = list()

        if isinstance(f, types.MethodType):
            f = f.__name__

        for sp, pos in iter_spe(self, index=True):
            processlist.append([sp, pos, f, kargs])
        num_tasks = len(processlist)

        processresult = pool.imap_unordered(_process_spe, processlist)
        pool.close()

        if verbose:
            print "loop_spe_multiprocessing (%s): %i tasks" % (f, num_tasks)
            import time
            import sys
            while (True):
                time.sleep(5)
                completed = processresult._index
                if completed == num_tasks:
                    output = ""
                    sys.stdout.write("\r\x1b[K" + output.__str__())
                    break
                output = "\r Waiting for %i tasks to complete '\
                '(%i%% done) ..." \
                % (num_tasks - completed, float(completed)
                   / float(num_tasks) * 100.0)
                sys.stdout.write("\r\x1b[K" + output.__str__())
                sys.stdout.flush()

        init = True
        for pos, out in processresult:
            if is_float(out) or is_int(out):
                # f returns a number -> iterator returns an image
                if init:
                    from image import Image
                    result = Image(wcs=self.wcs.copy(),
                                   data=np.zeros(shape=(self.shape[1],
                                                        self.shape[2])),
                                   unit=self.unit)
                    init = False
                p, q = pos
                result[p, q] = out
            else:
                try:
                    if out.spectrum:
                        # f returns a spectrum -> iterator returns a cube
                        if init:
                            if self.var is None:
                                result = Cube(wcs=self.wcs.copy(),
                                              wave=out.wave.copy(),
                                              data=np.zeros(shape=(out.shape,
                                                                   self.shape[1],
                                                                   self.shape[2])),
                                              unit=self.unit,
                                              fscale=out.fscale)
                            else:
                                result = Cube(wcs=self.wcs.copy(),
                                              wave=out.wave.copy(),
                                              data=np.zeros(
                                              shape=(out.shape, self.shape[1],
                                                     self.shape[2])),
                                              var=np.zeros(
                                              shape=(out.shape, self.shape[1],
                                                     self.shape[2])),
                                              unit=self.unit,
                                              fscale=out.fscale)
                            init = False
                        p, q = pos
                        result[:, p, q] = out

                except:
                    # f returns dtype -> iterator returns an array of dtype
                    if init:
                        result = np.empty((self.shape[1], self.shape[2]),
                                          dtype=type(out))
                        init = False
                    p, q = pos
                    result[p, q] = out

        return result

    def loop_ima_multiprocessing(self, f, cpu=None, verbose=True, **kargs):
        """loops over all images to apply a function/method.
        Returns the resulting cube.
        Multiprocessing is used.

        Parameters
        ----------
        f       : function or :class:`mpdaf.obj.Image` method
                  Image method or function that the first argument
                  is a Image object. It should return an Image object.
        cpu     : integer
                  number of CPUs. It is also possible to set
        verbose : boolean
                  if True, progression is printed.
        kargs   : kargs
                  can be used to set function arguments.

        Returns
        -------
        out : :class:`mpdaf.obj.Cube` if f returns :class:`mpdaf.obj.Image`,
        out : :class:`mpdaf.obj.Spectrum` if f returns a number,
        out : np.array(dtype=object) in others cases.
        """
        from mpdaf import CPU
        if cpu is not None and cpu < multiprocessing.cpu_count():
            cpu_count = cpu
        elif CPU != 0 and CPU < multiprocessing.cpu_count():
            cpu_count = CPU
        else:
            cpu_count = multiprocessing.cpu_count() - 1

        pool = multiprocessing.Pool(processes=cpu_count)
        processlist = list()

        if isinstance(f, types.MethodType):
            f = f.__name__

        for ima, k in iter_ima(self, index=True):
            header = ima.wcs.to_header()
            processlist.append([ima, k, f, header, kargs])
        num_tasks = len(processlist)

        processresult = pool.imap_unordered(_process_ima, processlist)
        pool.close()

        if verbose:
            print "loop_ima_multiprocessing (%s): %i tasks" % (f, num_tasks)
            import time
            import sys
            while (True):
                time.sleep(5)
                completed = processresult._index
                if completed == num_tasks:
                    output = ""
                    sys.stdout.write("\r\x1b[K" + output.__str__())
                    break
                output = "\r Waiting for %i tasks to complete '\
                '(%i%% done) ..." % (num_tasks - completed,
                                     float(completed) / float(num_tasks)
                                     * 100.0)
                sys.stdout.write("\r\x1b[K" + output.__str__())
                sys.stdout.flush()

        init = True
        for k, out in processresult:
            if is_float(out) or is_int(out):
                # f returns a number -> iterator returns a spectrum
                if init:
                    from spectrum import Spectrum
                    result = Spectrum(wave=self.wave.copy(),
                                      data=np.zeros(shape=self.shape[0]),
                                      unit=self.unit)
                    init = False
                result[k] = out
            else:
                try:
                    if out.image:
                        # f returns an image -> iterator returns a cube
                        if init:
                            if self.var is None:
                                result = Cube(wcs=out.wcs,
                                              wave=self.wave.copy(),
                                              data=np.zeros(shape=(self.shape[0], out.shape[0], out.shape[1])),
                                              unit=self.unit,
                                              fscale=self.fscale)
                            else:
                                result = Cube(wcs=out.wcs,
                                              wave=self.wave.copy(),
                                              data=np.zeros(shape=(self.shape[0], out.shape[0], out.shape[1])),
                                              var=np.zeros(shape=(self.shape[0], out.shape[0], out.shape[1])),
                                              unit=self.unit,
                                              fscale=self.fscale)
                            init = False
                        result[k, :,:] = out
                except:
                    # f returns dtype -> iterator returns an array of dtype
                    if init:
                        result = np.empty(self.shape[0], dtype=type(out))
                        init = False
                    result[k] = out

        return result

    def get_image(self, wave, is_sum=False, verbose=True):
        """ Extracts an image from the datacube.

        Parameters
        ----------
        wave : (float, float)
            (lbda1,lbda2) interval of wavelength.
        is_sum : boolean
                if True the sum is computes, otherwise this is the average.
        """
        l1, l2 = wave
        k1, k2 = self.wave.pixel(wave, nearest=True).astype(int)
        if verbose:
            print 'Computing image for lbda %g-%g [%d-%d]' % (l1, l2, k1, k2)
        if is_sum:
            ima = self[k1:k2+1, :,:].sum(axis=0)
        else:
            ima = self[k1:k2+1, :,:].mean(axis=0)
        return ima

    def aperture(self, center, radius, verbose=True):
        """ Extracts a spectra from an aperture of fixed radius.

        Parameters
        ----------
        center : (float,float)
                Center of the aperture.
                (dec,ra) is in degrees.
        radius : float
                Radius of the aperture in arcsec.
        """
        center = self.wcs.sky2pix(center)[0]
        radius = radius / np.abs(self.wcs.get_step()[0]) / 3600.
        radius2 = radius * radius
        if radius > 0:
            imin = max(0, center[0] - radius)
            imax = min(center[0] + radius + 1, self.shape[1])
            jmin = max(0, center[1] - radius)
            jmax = min(center[1] + radius + 1, self.shape[2])
            data = self.data[:, imin:imax, jmin:jmax].copy()

            ni = data.shape[1]
            nj = data.shape[2]
            m = np.ma.make_mask_none(data.shape)
            for i_in in range(ni):
                i = i_in + imin
                pixcrd = np.array([np.ones(nj) * i, np.arange(nj) + jmin]).T
                pixcrd[:, 0] -= center[0]
                pixcrd[:, 1] -= center[1]
                m[:, i_in, :] = ((np.array(pixcrd[:, 0])
                                 * np.array(pixcrd[:, 0])
                                 + np.array(pixcrd[:, 1])
                                 * np.array(pixcrd[:, 1])) < radius2)
            m = np.ma.mask_or(m, np.ma.getmask(data))
            data.mask[:, :,:] = m
            if self.var is not None:
                var = (np.ma.sum(np.ma.masked_invalid(self.var[:, imin:imax, jmin:jmax]), axis=(1, 2))).filled(np.NaN)
            else:
                var = None
            from spectrum import Spectrum
            spec = Spectrum(wave=self.wave, unit=self.unit, data=data.sum(axis=(1, 2)), var=var, fscale=self.fscale)
            if verbose: print '%d spaxels summed' % (data.shape[1] * data.shape[2])
        else:
            spec = self[:, int(center[0] + 0.5), int(center[1] + 0.5)]
            if verbose: print 'returning spectrum at nearest spaxel'
        return spec


def _process_spe(arglist):
    try:
        pos = arglist[1]
        obj = arglist[0]
        f = arglist[2]
        kargs = arglist[3]
        if isinstance(f, types.FunctionType):
            obj_result = f(obj, **kargs)
        else:
            obj_result = getattr(obj, f)(**kargs)

        return (pos, obj_result)
    except Exception as inst:
        raise type(inst), str(inst) + \
            '\n The error occurred for the spectrum '\
            '[:,%i,%i]' % (pos[0], pos[1])


def _process_ima(arglist):
    try:
        pos = arglist[1]
        obj = arglist[0]

        f = arglist[2]
        kargs = arglist[4]
        if isinstance(f, types.FunctionType):
            obj_result = f(obj, **kargs)
        else:
            obj_result = getattr(obj, f)(**kargs)

        return (pos, obj_result)
    except Exception as inst:
        raise type(inst), str(inst) + '\n The error occurred '\
            'for the image [%i,:,:]' % pos


class CubeDisk(object):

    """Sometimes, MPDAF users may want to open fairly large datacubes
    (> 4 Gb or so).
    This can be difficult to handle with limited RAM.
    This class provides a way to open datacube fits files with memory mapping.
    The methods of the class can extract a spectrum, an image or a smaller
    datacube from the larger one.

    Parameters
    ----------
    filename : string
               Possible FITS filename.
    ext      : integer or (integer,integer) or string or (string,string)
               Number/name of the data extension or numbers/names
               of the data and variance extensions.
    notnoise : bool
               True if the noise Variance cube is not read (if it exists).
               Use notnoise=True to create cube without variance extension.

    Attributes
    ----------
    filename       : string
                     Fits file
    data           : int or string
                     Data extension
    unit           : string
                     Possible data unit type
    primary_header : pyfits.Header
                     Possible FITS primary header instance.
    data_header    : pyfits.Header
                     Possible FITS data header instance.
    shape          : array of 3 integers)
                     Lengths of data in Z and Y and X
                     (python notation (nz,ny,nx)).
    var            : int or string
                     Variance extension (-1 if any).
    fscale         : float
                     Flux scaling factor (1 by default).
    wcs            : :class:`mpdaf.obj.WCS`
                     World coordinates.
    wave           : :class:`mpdaf.obj.WaveCoord`)
                     Wavelength coordinates
    ima            : dict{string, :class:`mpdaf.obj.Image`}
                     dictionary of images
    """

    def __init__(self, filename=None, ext=None, notnoise=False, ima=True):
        """Creates a CubeDisk object.

    Parameters
    ----------
    filename : string
               Possible FITS filename.
    ext      : integer or (integer,integer) or string or (string,string)
               Number/name of the data extension or numbers/names
               of the data and variance extensions.
    notnoise : bool
               True if the noise Variance cube is not read (if it exists).
               Use notnoise=True to create cube without variance extension.
        """
        self.filename = filename
        self.ima = {}
        if filename is not None:
            f = pyfits.open(filename, memmap=True)
            # primary header
            hdr = f[0].header
            if len(f) == 1:
                # if the number of extension is 1,
                # we just read the data from the primary header
                # test if image
                if hdr['NAXIS'] != 3:
                    raise IOError('Wrong dimension number: not a cube')
                self.unit = hdr.get('BUNIT', None)
                self.primary_header = pyfits.Header()
                self.data_header = hdr
                self.shape = np.array([hdr['NAXIS3'], hdr['NAXIS2'],
                                       hdr['NAXIS1']])
                self.data = 0
                self.var = -1
                self.fscale = hdr.get('FSCALE', 1.0)
                # WCS object from data header
                try:
                    self.wcs = WCS(hdr)
                except:
                    d = {'class': 'CubeDisk', 'method': '__init__'}
                    logger.warning("wcs not copied", extra=d)
                    self.wcs = None
                # Wavelength coordinates
                if 'CRPIX3' not in hdr or 'CRVAL3' not in hdr:
                    self.wave = None
                else:
                    if 'CDELT3' in hdr:
                        cdelt = hdr.get('CDELT3')
                    elif 'CD3_3' in hdr:
                        cdelt = hdr.get('CD3_3')
                    else:
                        cdelt = 1.0
                    crpix = hdr.get('CRPIX3')
                    crval = hdr.get('CRVAL3')
                    cunit = hdr.get('CUNIT3', '')
                    self.wave = WaveCoord(crpix, cdelt, crval, cunit, self.shape[0])
            else:
                if ext is None:
                    h = f['DATA'].header
                    d = 'DATA'
                else:
                    if isinstance(ext, int) or isinstance(ext, str):
                        n = ext
                    else:
                        n = ext[0]
                    h = f[n].header
                    d = n
                if h['NAXIS'] != 3:
                    raise IOError('Wrong dimension number in DATA extension')
                self.unit = h.get('BUNIT', None)
                self.primary_header = hdr
                self.data_header = h
                self.shape = np.array([h['NAXIS3'], h['NAXIS2'], h['NAXIS1']])
                self.data = d
                self.fscale = h.get('FSCALE', 1.0)
                try:
                    self.wcs = WCS(h)  # WCS object from data header
                except:
                    d = {'class': 'CubeDisk', 'method': '__init__'}
                    logger.warning("wcs not copied", extra=d)
                    self.wcs = None
                # Wavelength coordinates
                if 'CRPIX3' not in h or 'CRVAL3' not in h:
                    self.wave = None
                else:
                    if 'CDELT3' in h:
                        cdelt = h.get('CDELT3')
                    elif 'CD3_3' in h:
                        cdelt = h.get('CD3_3')
                    else:
                        cdelt = 1.0
                    crpix = h.get('CRPIX3')
                    crval = h.get('CRVAL3')
                    cunit = h.get('CUNIT3', '')
                    self.wave = WaveCoord(crpix, cdelt, crval,
                                          cunit, self.shape[0])
                self.var = -1
                if not notnoise:
                    try:
                        if ext is None:
                            fstat = 'STAT'
                        else:
                            n = ext[1]
                            fstat = n
                        if f[fstat].header['NAXIS'] != 3:
                            raise IOError('Wrong dimension number '
                                          'in variance extension')
                        if f[fstat].header['NAXIS1'] != self.shape[2] and \
                                f[fstat].header['NAXIS2'] != self.shape[1] and \
                                f[fstat].header['NAXIS3'] != self.shape[0]:
                            raise IOError('Number of points '
                                          'in STAT not equal to DATA')
                        self.var = fstat
                    except:
                        self.var = -1
                if ima:
                    from image import Image
                    for i in range(len(f)):
                        try:
                            hdr = f[i].header
                            if hdr['NAXIS'] != 2:
                                raise IOError(' not an image')
                            self.ima[hdr.get('EXTNAME')] = \
                                Image(filename=filename, ext=hdr.get('EXTNAME'), notnoise=True)
                        except:
                            pass
            # DQ
            f.close()

    def info(self):
        """Prints information.
        """
        if self.filename is None:
            print '%i X %i X %i cube (no name)' % (self.shape[0],
                                                   self.shape[1],
                                                   self.shape[2])
        else:
            print '%i X %i X %i cube (%s)' % (self.shape[0], self.shape[1],
                                              self.shape[2], self.filename)
        data = '.data(%i,%i,%i)' % (self.shape[0], self.shape[1],
                                    self.shape[2])
        if self.data is None:
            data = 'no data'
        noise = '.var(%i,%i,%i)' % (self.shape[0], self.shape[1],
                                    self.shape[2])
        if self.var == -1:
            noise = 'no noise'
        if self.unit is None:
            unit = 'no unit'
        else:
            unit = self.unit
        print '%s (%s) fscale=%g, %s' % (data, unit, self.fscale, noise)
        if self.wcs is None:
            print 'no world coordinates for spatial direction'
        else:
            self.wcs.info()
        if self.wave is None:
            print 'no world coordinates for spectral direction'
        else:
            self.wave.info()
        print ".ima: ",
        for k in self.ima.keys():
            print k,
        print '\n'

    def __getitem__(self, item):
        """Returns the corresponding object:

        cube[k,p,k] = value

        cube[k,:,:] = spectrum

        cube[:,p,q] = image

        cube[:,:,:] = sub-cube
        """
        if isinstance(item, tuple) and len(item) == 3:
            f = pyfits.open(self.filename, memmap=True)
            data = f[self.data].data[item]
            if self.var != -1:
                var = f[self.var].data[item]
            else:
                var = None
            f.close()
            if is_int(item[0]):
                if is_int(item[1]) and is_int(item[2]):
                    # return a float
                    return data * self.fscale
                else:
                    # return an image
                    from image import Image
                    if is_int(item[1]):
                        shape = (1, data.shape[0])
                    elif is_int(item[2]):
                        shape = (data.shape[0], 1)
                    else:
                        shape = data.shape
                    try:
                        wcs = self.wcs[item[1], item[2]]
                    except:
                        wcs = None
                    res = Image(shape=shape, wcs=wcs, unit=self.unit,
                                fscale=self.fscale, data=data, var=var)
                    return res
            elif is_int(item[1]) and is_int(item[2]):
                # return a spectrum
                from spectrum import Spectrum
                shape = data.shape[0]
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Spectrum(shape=shape, wave=wave, unit=self.unit,
                               fscale=self.fscale, data=data, var=var)
                return res
            else:
                # return a cube
                if is_int(item[1]):
                    shape = (data.shape[0], 1, data.shape[1])
                elif is_int(item[2]):
                    shape = (data.shape[0], data.shape[1], 1)
                else:
                    shape = data.shape
                try:
                    wcs = self.wcs[item[1], item[2]]
                except:
                    wcs = None
                try:
                    wave = self.wave[item[0]]
                except:
                    wave = None
                res = Cube(shape=shape, wcs=wcs, wave=wave, unit=self.unit,
                           fscale=self.fscale, data=data, var=var)
                return res
        else:
            raise ValueError('Operation forbidden')

    def truncate(self, lmin, lmax, y_min, y_max, x_min, x_max, mask=True):
        """ Truncates the cube and return a sub-cube.

          Parameters
          ----------
          lmin  : float
                  Minimum wavelength.
          lmax  : float
                  Maximum wavelength.
          y_min : float
                  Minimum value of y in degrees.
          y_max : float
                  Maximum value of y in degrees.
          x_min : float
                  Minimum value of x in degrees.
          x_max : float
                  Maximum value of x in degrees.
          mask  : boolean
                  if True, pixels outside [y_min,y_max]
                  and [x_min,x_max] are masked.
        """
        skycrd = [[y_min, x_min], [y_min, x_max], [y_max, x_min], [y_max, x_max]]
        pixcrd = self.wcs.sky2pix(skycrd)

        imin = int(np.min(pixcrd[:, 0]))
        if imin < 0:
            imin = 0
        imax = int(np.max(pixcrd[:, 0])) + 1
        if imax > self.shape[1]:
            imax = self.shape[1]
        jmin = int(np.min(pixcrd[:, 1]))
        if jmin < 0:
            jmin = 0
        jmax = int(np.max(pixcrd[:, 1])) + 1
        if jmax > self.shape[2]:
            jmax = self.shape[2]

        kmin = max(0, self.wave.pixel(lmin, nearest=True))
        kmax = min(self.shape[0], self.wave.pixel(lmax, nearest=True) + 1)

        if kmin == kmax:
            raise ValueError('Minimum and maximum wavelengths are equal')

        if kmax == kmin + 1:
            raise ValueError('Minimum and maximum wavelengths '
                             'are outside the spectrum range')

        f = pyfits.open(self.filename, memmap=True)
        data = f[self.data].data[kmin:kmax, imin:imax, jmin:jmax]
        if self.var != -1:
            var = f[self.var].data[kmin:kmax, imin:imax, jmin:jmax]
        else:
            self.var = None
        f.close()

        shape = np.array((data.shape[0], data.shape[1], data.shape[2]))

        try:
            wcs = self.wcs[imin:imax, jmin:jmax]
        except:
            wcs = None
        try:
            wave = self.wave[kmin:kmax]
        except:
            wave = None

        res = Cube(shape=shape, wcs=wcs, wave=wave, unit=self.unit,
                   fscale=self.fscale, data=data, var=var)

        if mask:
            # mask outside pixels
            m = np.ma.make_mask_none(res.data.shape)
            for j in range(res.shape[1]):
                pixcrd = np.array([np.ones(shape[2])
                                   * j, np.arange(shape[2])]).T
                skycrd = res.wcs.pix2sky(pixcrd)
                test_ra_min = np.array(skycrd[:, 1]) < x_min
                test_ra_max = np.array(skycrd[:, 1]) > x_max
                test_dec_min = np.array(skycrd[:, 0]) < y_min
                test_dec_max = np.array(skycrd[:, 0]) > y_max
                m[:, j, :] = test_ra_min + test_ra_max \
                    + test_dec_min + test_dec_max
            try:
                m = np.ma.mask_or(m, np.ma.getmask(res.data))
                res.data = np.ma.MaskedArray(res.data, mask=m)
            except:
                pass
        return res

    def get_white_image(self):
        """Performs a sum over the wavelength dimension and returns an image.
        """
        f = pyfits.open(self.filename, memmap=True)
        from image import Image
        loop = True
        k = self.shape[0]
        while loop:
            try:
                data = np.sum(f[self.data].data[0:k], axis=0)
                loop = False
            except:
                k = k / 2
        kmin = k
        kmax = 2 * k
        while kmax < self.shape[0]:
            data += np.sum(f[self.data].data[kmin:kmax], axis=0)
            kmin = kmax
            kmax += k

        if self.var != -1:
            kmin = 0
            kmax = k
            var = np.zeros((self.shape[1], self.shape[2]))
            while kmax < self.shape[0]:
                var += np.sum(f[self.var].data[kmin:kmax], axis=0)
                kmin = kmax
                kmax += k
        else:
            var = None

        f.close()

        res = Image(shape=data.shape, wcs=self.wcs, unit=self.unit,
                    fscale=self.fscale, data=data, var=var)
        return res
