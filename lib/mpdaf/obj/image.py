"""image.py manages image objects."""

import datetime
import logging
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

from astropy.io import fits as pyfits
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate, ndimage, signal, special
from scipy.optimize import leastsq

from . import plt_norm, plt_zscale
from .coords import WCS, WaveCoord
from .objs import is_float, is_int


class ImageClicks(object):  # Object used to save click on image plot.

    def __init__(self, binding_id, filename=None):
        self.logger = logging.getLogger('mpdaf corelib')
        self.filename = filename  # Name of the table fits file
        #                           where are saved the clicks values.
        self.binding_id = binding_id  # Connection id.
        self.p = []  # Nearest pixel of the cursor position along the y-axis.
        self.q = []  # Nearest pixel of the cursor position along the x-axis.
        self.x = []  # Corresponding nearest position along the x-axis
        #              (world coordinates)
        self.y = []  # Corresponding nearest position along the y-axis
        #              (world coordinates)
        self.data = []  # Corresponding image data value.
        self.id_lines = []  # Plot id (cross for cursor positions).

    def remove(self, ic, jc):
        # removes a cursor position
        d2 = (self.i - ic) * (self.i - ic) + (self.j - jc) * (self.j - jc)
        i = np.argmin(d2)
        line = self.id_lines[i]
        del plt.gca().lines[line]
        self.p.pop(i)
        self.q.pop(i)
        self.x.pop(i)
        self.y.pop(i)
        self.data.pop(i)
        self.id_lines.pop(i)
        for j in range(i, len(self.id_lines)):
            self.id_lines[j] -= 1
        plt.draw()

    def add(self, i, j, x, y, data):
        plt.plot(j, i, 'r+')
        self.p.append(i)
        self.q.append(j)
        self.x.append(x)
        self.y.append(y)
        self.data.append(data)
        self.id_lines.append(len(plt.gca().lines) - 1)

    def iprint(self, i, fscale):
        # prints a cursor positions
        d = {'class': 'ImageClicks', 'method': 'iprint'}
        if fscale == 1:
            msg = 'y=%g\tx=%g\tp=%d\tq=%d\tdata=%g' % (self.y[i], self.x[i],
                                                       self.p[i], self.q[i],
                                                       self.data[i])
            self.logger.info(msg, extra=d)
        else:
            msg = 'y=%g\tx=%g\tp=%d\tq=%d\tdata=%g\t[scaled=%g]' \
                % (self.y[i], self.x[i], self.p[i], self.q[i], self.data[i],
                   self.data[i] / fscale)
            self.logger.info(msg, extra=d)

    def write_fits(self):
        # prints coordinates in fits table.
        d = {'class': 'ImageClicks', 'method': 'write_fits'}
        if self.filename != 'None':
            c1 = pyfits.Column(name='p', format='I', array=self.p)
            c2 = pyfits.Column(name='q', format='I', array=self.q)
            c3 = pyfits.Column(name='x', format='E', array=self.x)
            c4 = pyfits.Column(name='y', format='E', array=self.y)
            c5 = pyfits.Column(name='data', format='E', array=self.data)
            # tbhdu = pyfits.new_table(pyfits.ColDefs([c1, c2, c3, c4, c5]))
            coltab = pyfits.ColDefs([c1, c2, c3, c4, c5])
            tbhdu = pyfits.TableHDU(pyfits.FITS_rec.from_columns(coltab))
            tbhdu.writeto(self.filename, clobber=True, output_verify='fix')
            msg = 'printing coordinates in fits table %s' % self.filename
            self.logger.info(msg, extra=d)

    def clear(self):
        # disconnects and clears
        d = {'class': 'ImageClicks', 'method': 'clear'}
        msg = "disconnecting console coordinate printout..."
        self.logger.info(msg, extra=d)
        plt.disconnect(self.binding_id)
        nlines = len(self.id_lines)
        for i in range(nlines):
            line = self.id_lines[nlines - i - 1]
            del plt.gca().lines[line]
        plt.draw()


class Gauss2D(object):

    """This class stores 2D gaussian parameters.

    Attributes
    ----------
    center     : (float,float)
                Gaussian center (y,x).
    flux       : float
                Gaussian integrated flux.
    fwhm       : (float,float)
                Gaussian fwhm (fhwm_y,fwhm_x).
    cont       : float
                Continuum value.
    rot        : float
                Rotation in degrees.
    peak       : float
                Gaussian peak value.
    err_center : (float,float)
                Estimated error on Gaussian center.
    err_flux   : float
                Estimated error on Gaussian integrated flux.
    err_fwhm   : (float,float)
                Estimated error on Gaussian fwhm.
    err_cont   : float
                Estimated error on continuum value.
    err_rot    : float
                Estimated error on rotation.
    err_peak   : float
                Estimated error on Gaussian peak value.
    ima        : :class:`mpdaf.obj.Image`)
                Gaussian image
    """

    def __init__(self, center, flux, fwhm, cont, rot, peak, err_center,
                 err_flux, err_fwhm, err_cont, err_rot, err_peak, ima=None):
        self.logger = logging.getLogger('mpdaf corelib')
        self.center = center
        self.flux = flux
        self.fwhm = fwhm
        self.cont = cont
        self.rot = rot
        self.peak = peak
        self.err_center = err_center
        self.err_flux = err_flux
        self.err_fwhm = err_fwhm
        self.err_cont = err_cont
        self.err_rot = err_rot
        self.err_peak = err_peak
        self.ima = ima

    def copy(self):
        """Copies Gauss2D object in a new one and returns it."""
        return Gauss2D(self.center, self.flux, self.fwhm, self.cont,
                       self.rot, self.peak, self.err_center, self.err_flux,
                       self.err_fwhm, self.err_cont, self.err_rot,
                       self.err_peak)

    def print_param(self):
        """Prints Gaussian parameters."""
        d = {'class': 'Gauss2D', 'method': 'print_param'}
        msg = 'Gaussian center = (%g,%g) (error:(%g,%g))' \
            % (self.center[0], self.center[1],
               self.err_center[0], self.err_center[1])
        self.logger.info(msg, extra=d)
        msg = 'Gaussian integrated flux = %g (error:%g)' \
            % (self.flux, self.err_flux)
        self.logger.info(msg, extra=d)
        msg = 'Gaussian peak value = %g (error:%g)' \
            % (self.peak, self.err_peak)
        self.logger.info(msg, extra=d)
        msg = 'Gaussian fwhm = (%g,%g) (error:(%g,%g))' \
            % (self.fwhm[0], self.fwhm[1], self.err_fwhm[0], self.err_fwhm[1])
        self.logger.info(msg, extra=d)
        msg = 'Rotation in degree: %g (error:%g)' % (self.rot, self.err_rot)
        self.logger.info(msg, extra=d)
        msg = 'Gaussian continuum = %g (error:%g)' \
            % (self.cont, self.err_cont)
        self.logger.info(msg, extra=d)


class Moffat2D(object):

    """This class stores 2D moffat parameters.

    Attributes
    ----------
    center     : (float,float)
                peak center (y,x).
    flux       : float
                integrated flux.
    fwhm       : (float,float)
                fwhm (fhwm_y,fwhm_x).
    cont       : float
                Continuum value.
    n          : integer
                Atmospheric scattering coefficient.
    rot        : float
                Rotation in degrees.
    peak       : float
                intensity peak value.
    err_center : (float,float)
                Estimated error on center.
    err_flux   : float
                Estimated error on integrated flux.
    err_fwhm   : (float,float)
                Estimated error on fwhm.
    err_cont   : float
                Estimated error on continuum value.
    err_n      : float
                Estimated error on n coefficient.
    err_rot    : float
                Estimated error on rotation.
    err_peak   : float
                Estimated error on peak value.
    ima        : (:class:`mpdaf.obj.Image`)
                Moffat image
    """

    def __init__(self, center, flux, fwhm, cont, n, rot, peak, err_center,
                 err_flux, err_fwhm, err_cont, err_n, err_rot, err_peak,
                 ima=None):
        self.logger = logging.getLogger('mpdaf corelib')
        self.center = center
        self.flux = flux
        self.fwhm = fwhm
        self.cont = cont
        self.rot = rot
        self.peak = peak
        self.n = n
        self.err_center = err_center
        self.err_flux = err_flux
        self.err_fwhm = err_fwhm
        self.err_cont = err_cont
        self.err_rot = err_rot
        self.err_peak = err_peak
        self.err_n = err_n
        self.ima = ima

    def copy(self):
        """Returns a copy of a Moffat2D object."""
        return Moffat2D(self.center, self.flux, self.fwhm, self.cont,
                        self.n, self.rot, self.peak, self.err_center,
                        self.err_flux, self.err_fwhm, self.err_cont,
                        self.err_n, self.err_rot, self.err_peak)

    def print_param(self):
        """Prints Moffat parameters."""
        d = {'class': 'Moffat2D', 'method': 'print_param'}
        msg = 'center = (%g,%g) (error:(%g,%g))' \
            % (self.center[0], self.center[1],
               self.err_center[0], self.err_center[1])
        self.logger.info(msg, extra=d)
        msg = 'integrated flux = %g (error:%g)' % (self.flux, self.err_flux)
        self.logger.info(msg, extra=d)
        msg = 'peak value = %g (error:%g)' % (self.peak, self.err_peak)
        self.logger.info(msg, extra=d)
        msg = 'fwhm = (%g,%g) (error:(%g,%g))' \
            % (self.fwhm[0], self.fwhm[1], self.err_fwhm[0], self.err_fwhm[1])
        self.logger.info(msg, extra=d)
        msg = 'n = %g (error:%g)' % (self.n, self.err_n)
        self.logger.info(msg, extra=d)
        msg = 'rotation in degree: %g (error:%g)' % (self.rot, self.err_rot)
        self.logger.info(msg, extra=d)
        msg = 'continuum = %g (error:%g)' % (self.cont, self.err_cont)
        self.logger.info(msg, extra=d)


class Image(object):

    """Image class manages image, optionally including a variance and a bad
    pixel mask.

    Parameters
    ----------
    filename : string
            Possible filename (.fits, .png or .bmp).
    ext      : integer or (integer,integer) or string or (string,string)
            Number/name of the data extension or numbers/names
            of the data and variance extensions.
    notnoise : boolean
            True if the noise Variance image is not read (if it exists).
            Use notnoise=True to create image without variance extension.
    shape    : integer or (integer,integer)
            Lengths of data in Y and X.
            Python notation is used: (ny,nx). (101,101) by default.
    wcs      : :class:`mpdaf.obj.WCS`
            World coordinates.
    unit     : string
            Possible data unit type. None by default.
    data     : float array
            Array containing the pixel values of the image.
            None by default.
    var      : float array
            Array containing the variance. None by default.
    fscale   : float
            Flux scaling factor (1 by default).

    Attributes
    ----------
    filename       : string
                    Possible FITS filename.
    unit           : string
                    Possible data unit type.
    primary_header : pyfits.Header
                    Possible FITS primary header instance.
    data_header    : pyfits.Header
                    Possible FITS data header instance.
    data           : array or masked array)
                    Array containing the pixel values of the image.
    shape          : array of 2 integers
                    Lengths of data in Y and X
                    (python notation: (ny,nx)).
    var            : array
                    Array containing the variance.
    fscale         : float
                    Flux scaling factor (1 by default).
    wcs            : :class:`mpdaf.obj.WCS`
                    World coordinates.
    """

    def __init__(self, filename=None, ext=None, notnoise=False,
                 shape=(101, 101), wcs=None, unit=None, data=None,
                 var=None, fscale=1.0):
        """Creates a Image object.

        Parameters
        ----------
        filename : string
                Possible filename (.fits, .png or .bmp).
        ext      : integer or (integer,integer) or string or (string,string)
                Number/name of the data extension or numbers/names
                of the data and variance extensions.
        notnoise : boolean
                True if the noise Variance image is not read (if it exists).
                Use notnoise=True to create image without variance extension.
        shape    : integer or (integer,integer)
                Lengths of data in Y and X.
                Python notation is used: (ny,nx). (101,101) by default.
        wcs      : :class:`mpdaf.obj.WCS`
                World coordinates.
        unit     : string
                Possible data unit type. None by default.
        data     : float array
                Array containing the pixel values of the image.
                None by default.
        var      : float array
                Array containing the variance. None by default.
        fscale   : float
                Flux scaling factor (1 by default).
        """
        d = {'class': 'Image', 'method': '__init__'}
        self.logger = logging.getLogger('mpdaf corelib')
        self.image = True
        self._clicks = None
        self._selector = None
        # possible FITS filename
        self.filename = filename
        if filename is not None:
            if not os.path.isfile(filename):
                raise IOError('No such file or directory: %s' % filename)
            if filename[-4:] == "fits" or filename[-7:] == "fits.gz":
                f = pyfits.open(filename)
                # primary header
                hdr = f[0].header
                if len(f) == 1:
                    # if the number of extension is 1,
                    # we just read the data from the primary header
                    # test if image
                    if hdr['NAXIS'] != 2:
                        raise IOError('not an image')
                    self.unit = hdr.get('BUNIT', None)
                    self.primary_header = pyfits.Header()
                    self.data_header = hdr
                    self.shape = np.array([hdr['NAXIS2'], hdr['NAXIS1']])
                    self.data = np.array(f[0].data, dtype=float)
                    self.var = None
                    self.fscale = hdr.get('FSCALE', 1.0)
                    if wcs is None:
                        try:
                            self.wcs = WCS(hdr)  # WCS object from data header
                        except pyfits.VerifyError as e:
                            # Workaround for
                            # https://github.com/astropy/astropy/issues/887
                            self.logger.warning(e, extra=d)
                            self.wcs = WCS(hdr)
                    else:
                        self.wcs = wcs
                        self.wcs.set_naxis1(self.shape[1])
                        self.wcs.set_naxis2(self.shape[0])
                        if wcs.naxis1 != 0 and wcs.naxis2 != 0 \
                            and (wcs.naxis1 != self.shape[1]
                                 or wcs.naxis2 != self.shape[0]):
                            self.logger.warning('world coordinates and data have '
                                           'not the same dimensions: %s',
                                           "shape of WCS object is modified", extra=d)
                else:
                    if ext is None:
                        try:
                            h = f['DATA'].header
                            d = np.array(f['DATA'].data, dtype=float)
                        except:
                            try:
                                h = f['SCI'].header
                                d = np.array(f['SCI'].data, dtype=float)
                            except:
                                raise IOError('no DATA or SCI extension')
                    else:
                        if is_int(ext) or isinstance(ext, str):
                            n = ext
                        else:
                            n = ext[0]
                        h = f[n].header
                        d = np.array(f[n].data, dtype=float)

                    if h['NAXIS'] != 2:
                        raise IOError('Wrong dimension number '
                                      'in DATA extension')
                    self.unit = h.get('BUNIT', None)
                    self.primary_header = hdr
                    self.data_header = h
                    self.shape = np.array([h['NAXIS2'], h['NAXIS1']])
                    self.data = d
                    self.fscale = h.get('FSCALE', 1.0)
                    if wcs is None:
                        self.wcs = WCS(h)  # WCS object from data header
                    else:
                        self.wcs = wcs
                        self.wcs.set_naxis1(self.shape[1])
                        self.wcs.set_naxis2(self.shape[0])
                        if wcs.naxis1 != 0 and wcs.naxis2 != 0 \
                            and (wcs.naxis1 != self.shape[1]
                                 or wcs.naxis2 != self.shape[0]):
                            self.logger.warning('world coordinates and data have '
                                           'not the same dimensions: %s',
                                           "shape of WCS object is modified",
                                           extra=d)
                    self.var = None
                    if not notnoise:
                        if ext is None:
                            try:
                                fstat = f['STAT']
                            except:
                                try:
                                    fstat = f['WHT']
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
                            if fstat.header['NAXIS'] != 2:
                                raise IOError('Wrong dimension number '
                                              'in STAT extension')
                                if fstat.header['NAXIS1'] != self.shape[1] \
                                        and fstat.header['NAXIS2'] != self.shape[0]:
                                    raise IOError('Number of points in STAT '
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
                from PIL import Image as PILima
                im = PILima.open(filename)
                self.data = np.array(im.getdata(), dtype=float)\
                    .reshape(im.size[1], im.size[0])
                self.var = None
                self.shape = np.array(self.data.shape)
                self.fscale = np.float(fscale)
                self.unit = unit
                self.primary_header = pyfits.Header()
                self.data_header = pyfits.Header()
                if wcs is None:
                    self.wcs = WCS()
                    self.wcs.set_naxis1(self.shape[1])
                    self.wcs.set_naxis2(self.shape[0])
                else:
                    self.wcs = wcs
                    self.wcs.set_naxis1(self.shape[1])
                    self.wcs.set_naxis2(self.shape[0])
                    if wcs.naxis1 != 0 and wcs.naxis2 != 0 \
                        and (wcs.naxis1 != self.shape[1]
                             or wcs.naxis2 != self.shape[0]):
                        self.logger.warning('world coordinates and data have '
                                       'not the same dimensions: %s',
                                       "shape of WCS object is modified",
                                       extra=d)
        else:
            # possible data unit type
            self.unit = unit
            # possible FITS header instance
            self.data_header = pyfits.Header()
            self.primary_header = pyfits.Header()
            # data
            if is_int(shape):
                shape = (shape, shape)
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
                    self.wcs.set_naxis1(self.shape[1])
                    self.wcs.set_naxis2(self.shape[0])
                    if wcs.naxis1 != 0 and wcs.naxis2 != 0 \
                        and (wcs.naxis1 != self.shape[1]
                             or wcs.naxis2 != self.shape[0]):
                        self.logger.warning("world coordinates and data have not "
                                       "the same dimensions: %s",
                                       "shape of WCS object is modified", extra=d)
            except:
                self.wcs = None
                self.logger.warning("wcs not copied", extra=d)
        # Mask an array where invalid values occur (NaNs or infs).
        if self.data is not None:
            self.data = np.ma.masked_invalid(self.data)

    def copy(self):
        """Returns a new copy of an Image object."""
        ima = Image()
        ima.filename = self.filename
        ima.unit = self.unit
        ima.primary_header = pyfits.Header(self.primary_header)
        ima.data_header = pyfits.Header(self.data_header)
        ima.shape = self.shape.__copy__()
        try:
            ima.data = self.data.copy()
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

    def clone(self, var=False):
        """Returns a new image of the same shape and coordinates, filled with
        zeros.

        Parameters
        ----------
        var : boolean
            Presence of the variance extension.
        """
        try:
            wcs = self.wcs.copy()
        except:
            wcs = None
        if var is False:
            ima = Image(wcs=wcs, data=np.zeros(shape=self.shape),
                        unit=self.unit)
        else:
            ima = Image(wcs=wcs, data=np.zeros(shape=self.shape),
                        var=np.zeros(shape=self.shape), unit=self.unit)
        return ima

    def write(self, filename, fscale=None, savemask=True):
        """Saves the object in a FITS file.

        Parameters
        ----------
        filename : string
                The FITS filename.
        fscale   : float
                Flux scaling factor.
        savemask : boolean
                If True, Image mask is saved in DQ extension
        """
        # update fscale
        if fscale is None:
            fscale = self.fscale
        # create primary header
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
                            s = card.value[0: n]
                            prihdu.header['hierarch %s' % card.keyword] = \
                                (s, card.comment)
                        else:
                            prihdu.header['hierarch %s' % card.keyword] = \
                                (card.value, card.comment)
                    except:
                        d = {'class': 'Image', 'method': 'write'}
                        self.logger.warning("%s not copied in primary header",
                                       card.keyword, extra=d)
                        pass
        prihdu.header['date'] = \
            (str(datetime.datetime.now()), 'creation date')
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        hdulist = [prihdu]

        # world coordinates
        wcs_cards = self.wcs.to_header().cards

        # create spectrum DATA extension
        tbhdu = pyfits.ImageHDU(name='DATA', data=(self.data.data
                                                   * np.double(self.fscale / fscale))
                                .astype(np.float32))
        for card in self.data_header.cards:
            try:
                if card.keyword != 'CD1_1' and card.keyword != 'CD1_2' \
                        and card.keyword != 'CD2_1' and card.keyword != 'CD2_2' \
                        and card.keyword != 'CDELT1' and card.keyword != 'CDELT2' \
                        and tbhdu.header.keys().count(card.keyword) == 0:
                    tbhdu.header[card.keyword] = (card.value, card.comment)
            except:
                try:
                    card.verify('fix')
                    if card.keyword != 'CD1_1' and card.keyword != 'CD1_2' \
                            and card.keyword != 'CD2_1' and card.keyword != 'CD2_2' \
                            and card.keyword != 'CDELT1' and card.keyword != 'CDELT2'\
                            and tbhdu.header.keys().count(card.keyword) == 0:
                        prihdu.header[card.keyword] = \
                            (card.value, card.comment)
                except:
                    d = {'class': 'Image', 'method': 'write'}
                    self.logger.warning("%s not copied in data header",
                                   card.keyword, extra=d)
                    pass

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

        if self.unit is not None:
            tbhdu.header['BUNIT'] = (self.unit, 'data unit type')
        tbhdu.header['FSCALE'] = (fscale, 'Flux scaling factor')
        hdulist.append(tbhdu)

        self.wcs = WCS(tbhdu.header)

        # create image STAT extension
        if self.var is not None:
            nbhdu = pyfits.ImageHDU(name='STAT', data=(
                self.var * np.double(self.fscale * self.fscale
                                     / fscale / fscale)).astype(np.float32))
            for card in wcs_cards:
                nbhdu.header[card.keyword] = (card.value, card.comment)
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
        """Prints information."""
        d = {'class': 'Image', 'method': 'info'}
        if self.filename is None:
            msg = '%i X %i image (no name)' % (self.shape[0], self.shape[1])
        else:
            msg = '%i X %i image (%s)' % (self.shape[0], self.shape[1],
                                          self.filename)
        self.logger.info(msg, extra=d)
        data = '.data(%i,%i)' % (self.shape[0], self.shape[1])
        if self.data is None:
            data = 'no data'
        noise = '.var(%i,%i)' % (self.shape[0], self.shape[1])
        if self.var is None:
            noise = 'no noise'
        if self.unit is None:
            unit = 'no unit'
        else:
            unit = self.unit
        msg = '%s (%s) fscale=%g, %s' % (data, unit, self.fscale, noise)
        self.logger.info(msg, extra=d)
        if self.wcs is None:
            msg = 'no world coordinates'
            self.logger.info(msg, extra=d)
        else:
            self.wcs.info()

    def __le__(self, item):
        """Masks data array where greater than a given value (operator <=).

        Parameters
        ----------
        item : float
            minimum value.

        Returns
        -------
        out : Image object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater(self.data, item / self.fscale)
        return result

    def __lt__(self, item):
        """Masks data array where greater or equal than a given value
        (operator.

        <).

        Parameters
        ----------
        item : float
            minimum value.

        Returns
        -------
        out : Image object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data,
                                                     item / self.fscale)
        return result

    def __ge__(self, item):
        """Masks data array where less than a given value (operator >=).

        Parameters
        ----------
        item : float
            maximum value.

        Returns
        -------
        out : Image object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item / self.fscale)
        return result

    def __gt__(self, item):
        """Masks data array where less or equal than a given value (operator.

        >).

        Parameters
        ----------
        item : float
            maximum value.

        Returns
        -------
        out : Image object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data,
                                                  item / self.fscale)
        return result

    def resize(self):
        """Resizes the image to have a minimum number of masked values."""
        if self.data is not None:
            ksel = np.where(self.data.mask == False)
            try:
                item = (slice(ksel[0][0], ksel[0][-1] + 1, None),
                        slice(ksel[1][0], ksel[1][-1] + 1, None))
                self.data = self.data[item]
                if is_int(item[0]):
                    self.shape = np.array((1, self.data.shape[0]))
                elif is_int(item[1]):
                    self.shape = np.array((self.data.shape[0], 1))
                else:
                    self.shape = np.array((self.data.shape[0],
                                           self.data.shape[1]))
                if self.var is not None:
                    self.var = self.var[item]
                try:
                    self.wcs = self.wcs[item[0], item[1]]
                except:
                    self.wcs = None
                    d = {'class': 'Image', 'method': 'resize'}
                    self.logger.warning("wcs not copied", extra=d)
            except:
                pass

    def __add__(self, other):
        """Operator +.

        image1 + number = image2 (image2[p,q] = image1[p,q] + number)

        image1 + image2 = image3 (image3[p,q] = image1[p,q] + image2[p,q])

        image + cube1 = cube2 (cube2[k,p,q] = cube1[k,p,q] + image[p,q])

        Parameters
        ----------
        other : number or Image or Cube object.
                x is Image: Dimensions and world coordinates must be the same.

                x is Cube: The last two dimensions of the cube must be equal
                to the image dimensions.
                World coordinates in spatial directions must be the same.

        Returns
        -------
        out : Image or Cube object
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # image1 + number = image2 (image2[j,i]=image1[j,i]+number)
            res = self.copy()
            res.data = self.data + (other / np.double(self.fscale))
            return res
        try:
            # image1 + image2 = image3 (image3[j,i]=image1[j,i]+image2[j,i])
            # Dimensions must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.image:
                if other.data is None or self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                else:
                    res = Image(shape=self.shape, fscale=self.fscale)
                    # coordinates
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        raise IOError('Operation forbidden for images '
                                      'with different world coordinates')
                    # var
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var * \
                            np.double(other.fscale * other.fscale
                                      / self.fscale / self.fscale)
                    elif other.var is None:
                        res.var = self.var
                    else:
                        res.var = self.var + other.var * \
                            np.double(other.fscale * other.fscale
                                      / self.fscale / self.fscale)
                    # data
                    res.data = self.data + \
                        (other.data * np.double(other.fscale / self.fscale))
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # image + cube1 = cube2 (cube2[k,j,i]=cube1[k,j,i]+image[j,i])
                # The first two dimensions of cube1 must be equal
                # to the image dimensions.
                # If not equal to None, world coordinates
                # in spatial directions must be the same.
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
        """Operator -.

        image1 - number = image2 (image2[p,q] = image1[p,q] - number)

        image1 - image2 = image3 (image3[p,q] = image1[p,q] - image2[p,q])

        image - cube1 = cube2 (cube2[k,p,q] = image[p,q] - cube1[k,p,q])

        Parameters
        ----------
        other : number or Image or Cube object.
                x is Image: Dimensions and world coordinates must be the same.

                x is Cube: The last two dimensions of the cube must be equal
                to the image dimensions.
                World coordinates in spatial directions must be the same.

        Returns
        -------
        out : Image or Cube object.
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # image1 - number = image2 (image2[j,i]=image1[j,i]-number)
            res = self.copy()
            res.data = self.data - (other / np.double(self.fscale))
            return res
        try:
            # image1 - image2 = image3 (image3[j,i]=image1[j,i]-image2[j,i])
            # Dimensions must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.image:
                if other.data is None or self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                else:
                    res = Image(shape=self.shape, fscale=self.fscale)
                    # wcs
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        raise IOError('Operation forbidden for images with '
                                      'different world coordinates')
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var \
                            * np.double(other.fscale * other.fscale
                                        / self.fscale / self.fscale)
                    elif other.var is None:
                        res.var = self.var
                    else:
                        res.var = self.var + other.var \
                            * np.double(other.fscale * other.fscale
                                        / self.fscale / self.fscale)
                    # data
                    res.data = self.data - \
                        (other.data * np.double(other.fscale / self.fscale))
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # image - cube1 = cube2
                # (cube2[k,j,i]=image[j,i] - cube1[k,j,i])
                # The first two dimensions of cube1 must be equal
                # to the image dimensions.
                # If not equal to None, world coordinates
                # in spatial directions must be the same.
                if other.cube:
                    if other.data is None or self.shape[0] != other.shape[1] \
                            or self.shape[1] != other.shape[2]:
                        raise IOError('Operation forbidden for images '
                                      'with different sizes')
                    else:
                        from cube import Cube
                        res = Cube(shape=other.shape, wave=other.wave,
                                   fscale=self.fscale)
                        # coordinates
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            raise IOError('Operation forbidden for objects '
                                          'with different world coordinates')
                        # variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = other.var \
                                * np.double(other.fscale * other.fscale
                                            / self.fscale / self.fscale)
                        elif other.var is None:
                            res.var = np.ones(res.shape) \
                                * self.var[np.newaxis, :,:]
                        else:
                            res.var = self.var[np.newaxis, :,:] + other.var \
                                * np.double(other.fscale * other.fscale
                                            / self.fscale / self.fscale)
                        # data
                        res.data = self.data[np.newaxis, :,:] \
                            - (other.data * np.double(other.fscale / self.fscale))
                        # unit
                        if self.unit == other.unit:
                            res.unit = self.unit
                        return res
            except IOError as e:
                raise e
            except:
                raise IOError('Operation forbidden')
                return None

    def __rsub__(self, other):
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            res = self.copy()
            res.data = (other / np.double(self.fscale)) - self.data
            return res
        try:
            if other.image:
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
        """Operator \*.

        image1 \* number = image2 (image2[p,q] = image1[p,q] \* number)

        image1 \* image2 = image3 (image3[p,q] = image1[p,q] \* image2[p,q])

        image \* cube1 = cube2 (cube2[k,p,q] = image[p,q] \* cube1[k,p,q])

        image \* spectrum = cube (cube[k,p,q] = image[p,q] \* spectrum[k]

        Parameters
        ----------
        other : number or Spectrum or Image or Cube object.
                x is Image: Dimensions and world coordinates must be the same.

                x is Cube: The last two dimensions of the cube must be equal
                to the image dimensions.
                World coordinates in spatial directions must be the same.

        Returns
        -------
        out : Spectrum or Image or Cube object.
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # image1 * number = image2 (image2[j,i]=image1[j,i]*number)
            res = self.copy()
            res.data *= other
            return res
        try:
            # image1 * image2 = image3 (image3[j,i]=image1[j,i]*image2[j,i])
            # Dimensions must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.image:
                if other.data is None or self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                else:
                    res = Image(shape=self.shape,
                                fscale=self.fscale)
                    # coordinates
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        raise IOError('Operation forbidden for images '
                                      'with different world coordinates')
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
                        res.var = (other.var * self.data * self.data +
                                   self.var * other.data * other.data) \
                            * other.fscale * other.fscale
                    # data
                    res.data = self.data * other.data * other.fscale
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # image * cube1 = cube2
                # (cube2[k,j,i]=image[j,i] * cube1[k,j,i])
                # The first two dimensions of cube1 must be equal
                # to the image dimensions.
                # If not equal to None, world coordinates
                # in spatial directions must be the same.
                if other.cube:
                    res = other.__mul__(self)
                    return res
            except IOError as e:
                raise e
            except:
                try:
                    # image * spectrum = cube
                    # (cube[k,j,i]=image[j,i]*spectrum[k]
                    if other.spectrum:
                        if other.data is None:
                            raise IOError('Operation forbidden '
                                          'for empty data')
                        else:
                            from cube import Cube
                            shape = (other.shape, self.shape[0], self.shape[1])
                            res = Cube(shape=shape, wave=other.wave,
                                       wcs=self.wcs,
                                       fscale=self.fscale)
                            # data
                            res.data = self.data[np.newaxis, :,:] \
                                * other.data[:, np.newaxis, np.newaxis] \
                                * other.fscale
                            # variance
                            if self.var is None and other.var is None:
                                res.var = None
                            elif self.var is None:
                                res.var = np.ones(res.shape) \
                                    * other.var[:, np.newaxis, np.newaxis] \
                                    * self.data * self.data \
                                    * other.fscale * other.fscale
                            elif other.var is None:
                                res.var = np.ones(res.shape) \
                                    * self.var[np.newaxis, :,:] \
                                    * other.data * other.data \
                                    * other.fscale * other.fscale
                            else:
                                res.var = (np.ones(res.shape)
                                           * other.var[:, np.newaxis, np.newaxis]
                                           * self.data * self.data
                                           + np.ones(res.shape)
                                           * self.var[np.newaxis, :,:]
                                           * other.data * other.data) \
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
        """Operator /.

        image1 / number = image2 (image2[p,q] = image1[p,q] / number)

        image1 / image2 = image3 (image3[p,q] = image1[p,q] / image2[p,q])

        image / cube1 = cube2 (cube2[k,p,q] = image[p,q] / cube1[k,p,q])

        Parameters
        ----------
        other : number or Image or Cube object.
                x is Image: Dimensions and world coordinates must be the same.

                x is Cube: The last two dimensions of the cube must
                be equal to the image dimensions.
                World coordinates in spatial directions must be the same.

        Returns
        -------
        out : Image or Cube object.
        """
        if self.data is None:
            raise ValueError('empty data array')
        if is_float(other) or is_int(other):
            # image1 / number = image2 (image2[j,i]=image1[j,i]/number
            res = self.copy()
            res.data /= other
            return res
        try:
            # image1 / image2 = image3 (image3[j,i]=image1[j,i]/image2[j,i])
            # Dimensions must be the same.
            # If not equal to None, world coordinates must be the same.
            if other.image:
                if other.data is None or self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1]:
                    raise IOError('Operation forbidden '
                                  'for images with different sizes')
                else:
                    res = Image(shape=self.shape,
                                fscale=self.fscale)
                    # coordinates
                    if self.wcs is None or other.wcs is None:
                        res.wcs = None
                    elif self.wcs.isEqual(other.wcs):
                        res.wcs = self.wcs
                    else:
                        raise IOError('Operation forbidden for images '
                                      'with different world coordinates')
                    # variance
                    if self.var is None and other.var is None:
                        res.var = None
                    elif self.var is None:
                        res.var = other.var * self.data * self.data \
                            / (other.data ** 4) / other.fscale / other.fscale
                    elif other.var is None:
                        res.var = self.var * other.data * other.data / \
                            (other.data ** 4) / other.fscale / other.fscale
                    else:
                        res.var = (other.var * self.data * self.data
                                   + self.var * other.data * other.data) \
                            / (other.data ** 4) / (other.fscale ** 2)
                    # data
                    res.data = self.data / other.data / other.fscale
                    # unit
                    if self.unit == other.unit:
                        res.unit = self.unit
                    return res
        except IOError as e:
            raise e
        except:
            try:
                # image / cube1 = cube2
                # (cube2[k,j,i]=image[j,i] / cube1[k,j,i])
                # The first two dimensions of cube1 must be equal
                # to the image dimensions.
                # If not equal to None, world coordinates in spatial
                # directions must be the same.
                if other.cube:
                    if other.data is None or self.shape[0] != other.shape[1] \
                            or self.shape[1] != other.shape[2]:
                        raise ValueError('Operation forbidden for images '
                                         'with different sizes')
                    else:
                        from cube import Cube
                        res = Cube(shape=other.shape, wave=other.wave,
                                   fscale=self.fscale)
                        # coordinates
                        if self.wcs is None or other.wcs is None:
                            res.wcs = None
                        elif self.wcs.isEqual(other.wcs):
                            res.wcs = self.wcs
                        else:
                            raise ValueError('Operation forbidden '
                                             'for objects with different'
                                             ' world coordinates')
                        # variance
                        if self.var is None and other.var is None:
                            res.var = None
                        elif self.var is None:
                            res.var = other.var * self.data[np.newaxis, :,:]\
                                * self.data[np.newaxis, :,:] \
                                / (other.data ** 4) / (other.fscale ** 2)
                        elif other.var is None:
                            res.var = self.var[np.newaxis, :,:] \
                                * other.data * other.data \
                                / (other.data ** 4) / (other.fscale ** 2)
                        else:
                            res.var = \
                                (other.var * self.data[np.newaxis, :,:]
                                 * self.data[np.newaxis, :,:]
                                 + self.var[np.newaxis, :,:]
                                 * other.data * other.data) \
                                / (other.data ** 4) / (other.fscale ** 2)
                            # data
                        res.data = self.data[np.newaxis, :,:] / other.data \
                            / other.fscale
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
            # image1 / number = image2 (image2[j,i]=image1[j,i]/number
            res = self.copy()
            res.fscale = other / res.fscale
            return res
        try:
            if other.image:
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

    def __pow__(self, other):
        """Computes the power exponent of data extensions (operator \*\*).
        """
        if self.data is None:
            raise ValueError('empty data array')
        res = self.copy()
        if is_float(other) or is_int(other):
            res.data = self.data ** other * (self.fscale ** (other - 1))
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
        """Returns an image containing the positive square-root
        of data extension.
        """
        res = self.copy()
        res._sqrt()
        return res

    def _abs(self):
        """Computes the absolute value of data extension."""
        if self.data is None:
            raise ValueError('empty data array')
        self.data = np.ma.abs(self.data)
        self.var = None

    def abs(self):
        """Returns an image containing the absolute value of data extension."""
        res = self.copy()
        res._abs()
        return res

    def __getitem__(self, item):
        """Returns the corresponding value or sub-image.
        """
        if isinstance(item, tuple) and len(item) == 2:
            if is_int(item[0]) and is_int(item[1]):
                return self.data[item] * self.fscale
            else:
                data = self.data[item]
                var = None
                if self.var is not None:
                    var = self.var[item]
                try:
                    wcs = self.wcs[item]
                except:
                    wcs = None
                if is_int(item[0]):
                    from spectrum import Spectrum
                    if self.wcs.is_deg():
                        cunit = 'deg'
                    else:
                        cunit = 'pixel'
                    wave = WaveCoord(crpix=1.0, cdelt=self.get_step()[1],
                                     crval=self.get_start()[1], cunit=cunit,
                                     shape=data.shape[0])
                    res = Spectrum(shape=data.shape[0], wave=wave,
                                   unit=self.unit, data=data, var=var,
                                   fscale=self.fscale)
                    res.data = data
                    res.var = var
                    return res
                elif is_int(item[1]):
                    from spectrum import Spectrum
                    if self.wcs.is_deg():
                        cunit = 'deg'
                    else:
                        cunit = 'pixel'
                    wave = WaveCoord(crpix=1.0, cdelt=self.get_step()[0],
                                     crval=self.get_start()[0], cunit=cunit,
                                     shape=data.shape[0])
                    res = Spectrum(shape=data.shape[0], wave=wave,
                                   unit=self.unit, data=data, var=var,
                                   fscale=self.fscale)
                    res.data = data
                    res.var = var
                    return res
                else:
                    res = Image(shape=data.shape, wcs=wcs,
                                unit=self.unit, fscale=self.fscale)
                    res.data = data
                    res.var = var
                    return res
        else:
            if self.shape[0] == 1 or self.shape[1] == 1:
                if isinstance(item, int):
                    return self.data[item] * self.fscale
                else:
                    data = self.data[item]
                    var = None
                    if self.var is not None:
                        var = self.var[item]
                    from spectrum import Spectrum
                    if self.wcs.is_deg():
                        cunit = 'deg'
                    else:
                        cunit = 'pixel'
                    if self.shape[0] == 1:
                        wave = WaveCoord(crpix=1.0, cdelt=self.get_step()[1],
                                         crval=self.get_start()[1],
                                         cunit=cunit, shape=data.shape[0])
                    else:
                        wave = WaveCoord(crpix=1.0, cdelt=self.get_step()[0],
                                         crval=self.get_start()[0],
                                         cunit=cunit, shape=data.shape[0])
                    res = Spectrum(shape=data.shape[0], wave=wave,
                                   unit=self.unit, data=data,
                                   var=var, fscale=self.fscale)
                    res.data = data
                    res.var = var
                    return res
            else:
                raise ValueError('Operation forbidden')

    def get_step(self):
        """Returns the image steps [dy, dx].

        Returns
        -------
        out : float array
        """
        return self.wcs.get_step()

    def get_range(self):
        """Returns [ [y_min,x_min], [y_max,x_max] ]

        Returns
        -------
        out : float array
        """
        return self.wcs.get_range()

    def get_start(self):
        """Returns [y,x] corresponding to pixel (0,0).

        Returns
        -------
        out : float array
        """
        return self.wcs.get_start()

    def get_end(self):
        """Returns [y,x] corresponding to pixel (-1,-1).

        Returns
        -------
        out : float array
        """
        return self.wcs.get_end()

    def get_rot(self):
        """Returns the angle of rotation.

        Returns
        -------
        out : float
        """
        return self.wcs.get_rot()

    def get_np_data(self):
        """Returns numpy masked array containing the flux multiplied by scaling
        factor."""
        return self.data * self.fscale

    def __setitem__(self, key, other):
        """Sets the corresponding part of data."""
        # self.data[key] = other
        if self.data is None:
            raise ValueError('empty data array')
        try:
            self.data[key] = other / np.double(self.fscale)
        except:
            try:
                # other is an image
                if other.image:
                    if self.wcs is not None and other.wcs is not None \
                            and (self.wcs.get_step() != other.wcs.get_step()).any():
                        d = {'class': 'Image', 'method': '__setitem__'}
                        self.logger.warning("images with different steps", extra=d)
                    self.data[key] = other.data \
                        * np.double(other.fscale / self.fscale)
            except:
                raise IOError('Operation forbidden')

    def set_wcs(self, wcs):
        """Sets the world coordinates.

        Parameters
        ----------
        wcs : :class:`mpdaf.obj.WCS`
              World coordinates.
        """
        self.wcs = wcs
        self.wcs.set_naxis1(self.shape[1])
        self.wcs.set_naxis2(self.shape[0])
        if wcs.naxis1 != 0 and wcs.naxis2 != 0 \
            and (wcs.naxis1 != self.shape[1]
                 or wcs.naxis2 != self.shape[0]):
            d = {'class': 'Image', 'method': 'set_wcs'}
            self.logger.warning('world coordinates and data have not '
                           'the same dimensions', extra=d)

    def set_var(self, var):
        """Sets the variance array.

        Parameters
        ----------
        var : float array
              Input variance array.
              If None, variance is set with zeros.
        """
        if var is None:
            self.var = np.zeros((self.shape[0], self.shape[1]))
        else:
            if self.shape[0] == np.shape(var)[0] \
                    and self.shape[1] == np.shape(var)[1]:
                self.var = var
            else:
                raise ValueError('var and data have not the same dimensions.')

    def mask(self, center, radius, pix=False, inside=True):
        """Masks values inside/outside the described region.

        Parameters
        ----------
        center : (float,float)
                 Center of the explored region.
                 If pix is False, center = (y,x) is in degrees.
                 If pix is True, center = (p,q) is in pixels.
        radius : float or (float,float)
                 Radius defined the explored region.
                 If radius is float, it defined a circular region.
                 If radius is (float,float), it defined a rectangular region.
                 If pix is False, radius = (dy/2, dx/2) is in arcsecs.
                 If pix is True, radius = (dp,dq) is in pixels.
        pix    : boolean
                 If pix is False, center and radius are in degrees and arcsecs.
                 If pix is True, center and radius are in pixels.
        inside : boolean
                 If inside is True, pixels inside the described region are masked.
                 If inside is False, pixels outside the described region are masked.
        """
        if is_int(radius) or is_float(radius):
            circular = True
            radius2 = radius * radius
            radius = (radius, radius)
        else:
            circular = False

        if not pix:
            center = self.wcs.sky2pix(center)[0]
            radius = radius / np.abs(self.wcs.get_step()) / 3600.
            radius2 = radius[0] * radius[1]

        imin = max(0, center[0] - radius[0])
        imax = min(center[0] + radius[0] + 1, self.shape[0])
        jmin = max(0, center[1] - radius[1])
        jmax = min(center[1] + radius[1] + 1, self.shape[1])

        if inside and not circular:
            self.data.mask[imin:imax, jmin:jmax] = 1
        elif inside and circular:
        
            grid = np.meshgrid(np.arange(imin,imax)-center[0], \
                               np.arange(jmin,jmax)-center[1], indexing='ij')
            self.data.mask[imin:imax, jmin:jmax] = \
            np.logical_or(self.data.mask[imin:imax, jmin:jmax],\
                          (grid[0]**2 + grid[1]**2) < radius2)
        elif not inside and circular:
            self.data.mask[0:imin, :] = 1
            self.data.mask[imax:, :] = 1
            self.data.mask[imin:imax, 0:jmin] = 1
            self.data.mask[imin:imax:, jmax:] = 1
            grid = np.meshgrid(np.arange(imin,imax)-center[0], \
                               np.arange(jmin,jmax)-center[1], indexing='ij')
            self.data.mask[imin:imax, jmin:jmax] = \
            np.logical_or(self.data.mask[imin:imax, jmin:jmax],\
                          (grid[0]**2 + grid[1]**2) > radius2)
        else:
            self.data.mask[0:imin, :] = 1
            self.data.mask[imax:, :] = 1
            self.data.mask[imin:imax, 0:jmin] = 1
            self.data.mask[imin:imax:, jmax:] = 1

    def mask_ellipse(self, center, radius, posangle, pix=False, inside=True):
        """Masks values inside/outside the described region. Uses an elliptical
        shape.

        Parameters
        ----------
        center : (float,float)
                 Center of the explored region.
                 If pix is False, center = (y,x) is in degrees.
                 If pix is True, center = (p,q) is in pixels.
        radius : (float,float)
                 Radius defined the explored region.
                 radius is (float,float), it defines an elliptical region with semi-major and semi-minor axes.
                 If pix is False, radius = (da, db) is in arcsecs.
                 If pix is True, radius = (dp,dq) is in pixels.
        posangle : float
                 Position angle of the first axis. It is defined in degrees against the horizontal (q) axis of the image, counted counterclockwise.
        pix    : boolean
                 If pix is False, center and radius are in degrees and arcsecs.
                 If pix is True, center and radius are in pixels.
        inside : boolean
                 If inside is True, pixels inside the described region are masked.
        """
        if not pix:
            center = self.wcs.sky2pix(center)[0]
            radius = radius / np.abs(self.wcs.get_step()) / 3600.

        maxradius = max(radius[0], radius[1])

        imin = max(0, center[0] - maxradius)
        imax = min(center[0] + maxradius + 1, self.shape[0])
        jmin = max(0, center[1] - maxradius)
        jmax = min(center[1] + maxradius + 1, self.shape[0])

        cospa = np.cos(np.radians(posangle))
        sinpa = np.sin(np.radians(posangle))

        if inside:
            grid = np.meshgrid(np.arange(imin,imax)-center[0], \
                               np.arange(jmin,jmax)-center[1], indexing='ij')
            self.data.mask[imin:imax, jmin:jmax] = \
            np.logical_or(self.data.mask[imin:imax, jmin:jmax],\
                          ((grid[1]*cospa+grid[0]*sinpa)/radius[0])**2 \
                          + ((grid[0] * cospa - grid[1] * sinpa) \
                             / radius[1]) ** 2 < 1)
        if not inside:
            self.data.mask[0:imin, :] = 1
            self.data.mask[imax:, :] = 1
            self.data.mask[imin:imax, 0:jmin] = 1
            self.data.mask[imin:imax:, jmax:] = 1
            grid = np.meshgrid(np.arange(imin,imax)-center[0], \
                               np.arange(jmin,jmax)-center[1], indexing='ij')
            self.data.mask[imin:imax, jmin:jmax] = \
            np.logical_or(self.data.mask[imin:imax, jmin:jmax],\
                          ((grid[1]*cospa+grid[0]*sinpa)/radius[0])**2 \
                          + ((grid[0] * cospa - grid[1] * sinpa) \
                             / radius[1]) ** 2 > 1)

    def unmask(self):
        """Unmasks the image (just invalid data (nan,inf) are masked)."""
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
            raise ValueError('Operation forbidden without '
                             'variance extension.')
        else:
            ksel = np.where(self.var > threshold)
            self.data[ksel] = np.ma.masked

    def mask_selection(self, ksel):
        """Masks pixels corresponding to a selection.

        Parameters
        ----------
        ksel : output of np.where
               elements depending on a condition.
        """
        self.data[ksel] = np.ma.masked

    def _truncate(self, y_min, y_max, x_min, x_max, mask=True):
        """Truncates the image.

        Parameters
        ----------
        y_min : float
                Minimum value of y in degrees.
        y_max : float
                Maximum value of y in degrees.
        x_min : float
                Minimum value of x in degrees.
        x_max : float
                Maximum value of x in degrees.
        mask  : boolean
                if True, pixels outside [dec_min,dec_max]
                and [ra_min,ra_max] are masked.
        """
        skycrd = [[y_min, x_min], [y_min, x_max],
                  [y_max, x_min], [y_max, x_max]]
        pixcrd = self.wcs.sky2pix(skycrd)

        imin = int(np.min(pixcrd[:, 0]))
        if imin < 0:
            imin = 0
        imax = int(np.max(pixcrd[:, 0])) + 1
        if imax > self.shape[0]:
            imax = self.shape[0]
        jmin = int(np.min(pixcrd[:, 1]))
        if jmin < 0:
            jmin = 0
        jmax = int(np.max(pixcrd[:, 1])) + 1
        if jmax > self.shape[1]:
            jmax = self.shape[1]
            
        self.data = self.data[imin:imax, jmin:jmax]
        self.shape = np.array((self.data.shape[0], self.data.shape[1]))
        if self.var is not None:
            self.var = self.var[imin:imax, jmin:jmax]
        try:
            self.wcs = self.wcs[imin:imax, jmin:jmax]
        except:
            self.wcs = None

        if mask:
            # mask outside pixels 
            grid = np.meshgrid(np.arange(0,self.shape[0]), \
                               np.arange(0,self.shape[1]), indexing='ij')
            shape = grid[1].shape
            pixcrd = np.array([[p, q] for p,q in zip(np.ravel(grid[0]), np.ravel(grid[1]))])
            skycrd = np.array(self.wcs.pix2sky(pixcrd))
            x = skycrd[:, 1].reshape(shape)
            y = skycrd[:, 0].reshape(shape)
            test_x = np.logical_or(x < x_min, x > x_max)
            test_y = np.logical_or(y < y_min, y > y_max)
            test = np.logical_or(test_x, test_y)
            self.data.mask = np.logical_or(self.data.mask, test)
            self.resize()

    def truncate(self, y_min, y_max, x_min, x_max, mask=True):
        """Returns truncated image.

        Parameters
        ----------
        y_min : float
                Minimum value of y in degrees.
        y_max : float
                Maximum value of y in degrees.
        x_min : float
                Minimum value of x in degrees.
        x_max : float
                Maximum value of x in degrees.
        mask  : boolean

        Returns
        -------
        out : Image
        """
        res = self.copy()
        res._truncate(y_min, y_max, x_min, x_max, mask)
        return res

    def rotate_wcs(self, theta):
        """Rotates WCS coordinates to new orientation given by theta (in
        place).

        Parameters
        ----------
        theta : float
                Rotation in degree.
        """
        self.wcs.rotate(theta)

    def _rotate(self, theta, interp='no', reshape=False):
        """ Rotates the image using spline interpolation
        (uses `scipy.ndimage.rotate <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.interpolation.rotate.html>`_)

        Parameters
        ----------
        theta   : float
                Rotation in degrees.
        interp  : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        reshape : boolean
                if reshape is true, the output image is adapted
                so that the input image is contained completely in the output.
                Default is False.
        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        mask = np.array(1 - self.data.mask, dtype=bool)
        mask_rot = ndimage.rotate(mask, -theta, reshape=reshape, order=0)
        data_rot = ndimage.rotate(data, -theta, reshape=reshape)
        mask_ma = np.ma.make_mask(1 - mask_rot)
        self.data = np.ma.array(data_rot, mask=mask_ma)

        try:
            center_pix = (np.array([self.shape[0], self.shape[1]]) + 1) / 2.
            center_coord = self.wcs.pix2sky([center_pix - 1])
            old_crpix = self.wcs.wcs.wcs.crpix.copy()
            self.wcs.set_crpix1(center_pix[1])
            self.wcs.set_crpix2(center_pix[0])
            self.wcs.set_crval1(center_coord[0][1])
            self.wcs.set_crval2(center_coord[0][0])
            # rotate the wcs
            self.wcs.rotate(-theta)
            # translate the new wcs
            self.shape = np.array(data_rot.shape)
            self.wcs.set_naxis1(self.shape[1])
            self.wcs.set_naxis2(self.shape[0])
            self.wcs.set_crpix1((self.shape[1] + 1) / 2.)
            self.wcs.set_crpix2((self.shape[0] + 1) / 2.)
            # compute the new value of the old crpix
            new_crval = self.wcs.pix2sky([old_crpix[1] - 1, old_crpix[0] - 1])
            self.wcs.set_crpix1(old_crpix[0])
            self.wcs.set_crpix2(old_crpix[1])
            self.wcs.set_crval1(new_crval[0][1])
            self.wcs.set_crval2(new_crval[0][0])
        except:
            self.shape = np.array(data_rot.shape)
            self.wcs = None

    def rotate(self, theta, interp='no', reshape=False):
        """ Returns rotated image
        (uses `scipy.ndimage.rotate <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.interpolation.rotate.html>`_)

        Parameters
        ----------
        theta   : float
                Rotation in degrees.
        interp  : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        reshape : boolean
                if reshape is true, the output image is adapted
                so that the input image is contained completely in the output.
                Default is False.

        Returns
        -------
        out : Image
        """
        res = self.copy()
        res._rotate(theta, interp, reshape)
        return res

    def sum(self, axis=None):
        """Returns the sum over the given axis.

        Parameters
        ----------
        axis : None, 0 or 1
               axis = None returns a float
               axis=0 or 1 returns a Spectrum object corresponding to a line or a column,
               other cases return None.

        Returns
        -------
        out : float or Image
        """
        if axis is None:
            return self.data.sum() * self.fscale
        elif axis == 0 or axis == 1:
            # return a spectrum
            data = np.ma.sum(self.data, axis)
            var = None
            if self.var is not None:
                var = np.sum(self.var, axis)
            if axis == 0:
                #wcs = self.wcs[0, :]
                #shape = (1, data.shape[0])
                step = self.wcs.get_step()[1]
                start = self.wcs.get_start()[1]
            else:
                #wcs = self.wcs[:, 0]
                #shape = (data.shape[0], 1)
                step = self.wcs.get_step()[0]
                start = self.wcs.get_start()[0]

            from spectrum import Spectrum
            if self.wcs.is_deg():
                cunit = 'deg'
            else:
                cunit = 'pixel'
            wave = WaveCoord(crpix=1.0, cdelt=step, crval=start,
                             cunit=cunit, shape=data.shape[0])
            res = Spectrum(shape=data.shape[0], wave=wave, unit=self.unit,
                           data=data, var=var, fscale=self.fscale)

            res.data = data
            res.var = var
            return res
        else:
            return None

    def norm(self, type='flux', value=1.0):
        """Normalizes in place total flux to value (default 1).

        Parameters
        ----------
        type  : 'flux' | 'sum' | 'max'
                If 'flux',the flux is normalized and
                the pixel area is taken into account.

                If 'sum', the flux is normalized to the sum
                of flux independantly of pixel size.

                If 'max', the flux is normalized so that
                the maximum of intensity will be 'value'.
        value : float
                Normalized value (default 1).
        """
        if type == 'flux':
            norm = value / (self.get_step().prod()
                            * self.fscale * self.data.sum())
        elif type == 'sum':
            norm = value / (self.fscale * self.data.sum())
        elif type == 'max':
            norm = value / (self.fscale * self.data.max())
        else:
            raise ValueError('Error in type: only flux,sum,max permitted')
        self.data *= norm
        if self.var is not None:
            self.var *= (norm * norm)

    def background(self, niter=3):
        """Computes the image background. Returns the background value and its
        standard deviation.

        Parameters
        ----------
        niter: integer
               Number of iterations.

        Returns
        -------
        out : 2-dim float array
        """
        tab = self.data
        for n in range(niter + 1):
            ksel = np.where(tab <= (np.ma.mean(tab) + 3 * np.ma.std(tab)))
            tab = tab[ksel]
        return np.array([np.ma.mean(tab) * self.fscale,
                         np.ma.std(tab) * self.fscale])

    def _struct(self, n):
        struct = np.zeros([n, n])
        for i in range(0, n):
            dist = abs(i - (n / 2))
            struct[i][dist: abs(n - dist)] = 1
        return struct

    def peak_detection(self, nstruct, niter, threshold=None):
        """Returns a list of peak locations.

        Parameters
        ----------
        nstruct   : integer
                    Size of the structuring element used for the erosion.
        niter     : integer
                    number of iterations used for the erosion and the dilatation.
        threshold : float
                    threshold value.
                    If None, it is initialized with background value

        Returns
        -------
        out : np.array
        """
        # threshold value
        (background, std) = self.background()
        if threshold is None:
            threshold = background + 10 * std
        else:
            threshold /= self.fscale

        selec = self.data > threshold
        selec.bill_value = False
        selec = \
            ndimage.morphology.binary_erosion(selec,
                                              structure=self._struct(nstruct),
                                              iterations=niter)
        selec = \
            ndimage.morphology.binary_dilation(selec,
                                               structure=self._struct(nstruct),
                                               iterations=niter)
        selec = ndimage.binary_fill_holes(selec)
        structure = ndimage.morphology.generate_binary_structure(2, 2)
        label = ndimage.measurements.label(selec, structure)
        pos = ndimage.measurements.center_of_mass(self.data, label[0],
                                                  np.arange(label[1]) + 1)
        return np.array(pos)

    def peak(self, center=None, radius=0, pix=False, dpix=2,
             background=None, plot=False):
        """Finds image peak location.
        Used `scipy.ndimage.measurements.maximum_position <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.measurements.maximum_position.html>`_ and `scipy.ndimage.measurements.center_of_mass <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.measurements.center_of_mass.html>`_.

        Parameters
        ----------
        center     : (float,float)
                    Center of the explored region.
                    If pix is False, center = (y, x) is in degrees.
                    If pix is True, center = (p,q) is in pixels.
                    If center is None, the full image is explored.
        radius     : float or (float,float)
                    Radius defined the explored region.
                    If pix is False, radius = (dy/2, dx/2) is in arcsecs.
                    If pix is True, radius = (dp,dq) is in pixels.
        pix        : boolean
                    If pix is False, center and radius are in degrees and arcsecs.
                    If pix is True, center and radius are in pixels.
        dpix       : integer
                    Half size of the window to compute the center of gravity.
        background : float
                    background value. If None, it is computed.
        plot       : boolean
                    If True, the peak center is overplotted on the image.

        Returns
        -------
        out : dictionary {'y', 'x', 'p', 'q', 'data'}
            containing the peak position and the peak intensity.
        """
        if center is None or radius == 0:
            d = self.data
            imin = 0
            jmin = 0
        else:
            if is_int(radius) or is_float(radius):
                radius = (radius, radius)

            if not pix:
                center = self.wcs.sky2pix(center)[0]
                radius = radius / np.abs(self.wcs.get_step()) / 3600.

            imin = center[0] - radius[0]
            if imin < 0:
                imin = 0
            imax = center[0] + radius[0] + 1
            jmin = center[1] - radius[1]
            if jmin < 0:
                jmin = 0
            jmax = center[1] + radius[1] + 1

            d = self.data[imin:imax, jmin:jmax]
            if np.shape(d)[0] == 0 or np.shape(d)[1] == 0:
                raise ValueError('Coord area outside image limits')

        ic, jc = ndimage.measurements.maximum_position(d)
        if dpix == 0:
            di = 0
            dj = 0
        else:
            if background is None:
                background = self.background()[0] / self.fscale
            else:
                background /= self.fscale
            di, dj = ndimage.measurements.center_of_mass(d[max(0, ic - dpix):ic + dpix + 1,
                                                           max(0, jc - dpix):jc + dpix + 1]
                                                         - background)
        ic = imin + max(0, ic - dpix) + di
        jc = jmin + max(0, jc - dpix) + dj
        [[dec, ra]] = self.wcs.pix2sky([[ic, jc]])
        maxv = self.fscale * self.data[int(round(ic)), int(round(jc))]
        if plot:
            plt.plot(jc, ic, 'r+')
            try:
                _str = 'center (%g,%g) radius (%g,%g) dpix %i peak: %g %g' % \
                    (center[0], center[1], radius[0], radius[1], dpix, jc, ic)
            except:
                _str = 'dpix %i peak: %g %g' % (dpix, ic, jc)
            plt.title(_str)

        return {'x': ra, 'y': dec, 'p': ic, 'q': jc, 'data': maxv}

    def fwhm(self, center=None, radius=0, pix=False):
        """Computes the fwhm center.

        Parameters
        ----------
        center : (float,float)
                 Center of the explored region.
                 If pix is False, center = (y,x) is in degrees.
                 If pix is True, center = (p,q) is in pixels.
                 If center is None, the full image is explored.
        radius : float or (float,float)
                 Radius defined the explored region.
                 If pix is False, radius = (dy/2, dx/2) is in arcsecs.
                 If pix is True, radius = (dp,dq) is in pixels.
        pix    : boolean
                 If pix is False, center and radius are
                 in degrees and arcsecs. fwhm is returned in arcseconds.
                 If pix is True, center and radius are in pixels.
                 fwhm is returned in pixels.

        Returns
        -------
        out : array of float
              [fwhm_y,fwhm_x].
        """
        if center is None or radius == 0:
            sigma = self.moments(pix=pix)
        else:
            if is_int(radius) or is_float(radius):
                radius = (radius, radius)

            if not pix:
                center = self.wcs.sky2pix(center)[0]
                radius = radius / np.abs(self.wcs.get_step()) / 3600.

            imin = max(0, center[0] - radius[0])
            imax = min(center[0] + radius[0] + 1, self.shape[0])
            jmin = max(0, center[1] - radius[1])
            jmax = min(center[1] + radius[1] + 1, self.shape[1])

            sigma = self[imin:imax, jmin:jmax].moments(pix=pix)

        return sigma * 2. * np.sqrt(2. * np.log(2.0))

    def ee(self, center=None, radius=0, pix=False, frac=False, cont=0):
        """Computes ensquared energy.

        Parameters
        ----------
        center : (float,float)
                 Center of the explored region.
                 If pix is False, center = (y,x) is in degrees.
                 If pix is True, center = (p,q) is in pixels.
                 If center is None, the full image is explored.
        radius : float or (float,float)
                 Radius defined the explored region.
                 If radius is float, it defined a circular region.
                 If radius is (float,float), it defined a rectangular region.
                 If pix is False, radius = (dy/2, dx/2) is in arcsecs.
                 If pix is True, radius = (dp,dq) is in pixels.
        pix    : boolean
                 If pix is False, center and radius are in degrees and arcsecs.
                 If pix is True, center and radius are in pixels.
        frac   : boolean
                 If frac is True, result is given relative to the total energy.
        cont   : float
                 continuum value.

        Returns
        -------
        out : float
        """
        cont /= self.fscale
        if center is None or radius == 0:
            if frac:
                return 1.
            else:
                return (self.data - cont).sum() * self.fscale
        else:
            if is_int(radius) or is_float(radius):
                circular = True
                radius2 = radius * radius
                radius = (radius, radius)
            else:
                circular = False

            if not pix:
                center = self.wcs.sky2pix(center)[0]
                radius = radius / np.abs(self.wcs.get_step()) / 3600.
                radius2 = radius[0] * radius[1]

            imin = max(0, center[0] - radius[0])
            imax = min(center[0] + radius[0] + 1, self.shape[0])
            jmin = max(0, center[1] - radius[1])
            jmax = min(center[1] + radius[1] + 1, self.shape[1])
            ima = self[imin:imax, jmin:jmax]

            if circular:
                xaxis = np.arange(ima.shape[0], dtype=np.float) \
                    - ima.shape[0] / 2.
                yaxis = np.arange(ima.shape[1], dtype=np.float) \
                    - ima.shape[1] / 2.
                gridx = np.empty(ima.shape, dtype=np.float)
                gridy = np.empty(ima.shape, dtype=np.float)
                for j in range(ima.shape[1]):
                    gridx[:, j] = xaxis
                for i in range(ima.shape[0]):
                    gridy[i, :] = yaxis
                r2 = gridx * gridx + gridy * gridy
                ksel = np.where(r2 < radius2)
                if frac:
                    return (ima.data[ksel] - cont).sum() \
                        / (self.data - cont).sum()
                else:
                    return (ima.data[ksel] - cont).sum() * self.fscale
            else:
                if frac:
                    return (ima.data - cont).sum() / (self.data - cont).sum()
                else:
                    return (ima.data - cont).sum() * self.fscale

    def ee_curve(self, center=None, pix=False, etot=None, cont=0):
        """Returns Spectrum object containing enclosed energy as function of
        radius.

        Parameters
        ----------
        center : (float,float)
                 Center of the explored region.
                 If pix is False, center = (y,x) is in degrees.
                 If pix is True, center = (p,q) is in pixels.
                 If center is None, center of the image is used.
        pix    : boolean
                 If pix is False, center is in degrees.
                 If pix is True, center is in pixels.
        etot   : float
                 Total energy. If etot is not set
                 it is computed from the full image.
        cont   : float
                 continuum value

        Returns
        -------
        out : :class:`mpdaf.obj.Spectrum`
        """
        cont /= self.fscale
        from spectrum import Spectrum
        if center is None:
            i = self.shape[0] / 2
            j = self.shape[1] / 2
        else:
            if pix:
                i = center[0]
                j = center[1]
            else:
                pixcrd = self.wcs.sky2pix([[center[0], center[1]]])
                i = int(pixcrd[0][0] + 0.5)
                j = int(pixcrd[0][1] + 0.5)
        nmax = min(self.shape[0] - i, self.shape[1] - j, i, j)
        if etot is None:
            etot = self.fscale * (self.data - cont).sum()
        step = self.get_step()
        if nmax <= 1:
            raise ValueError('Coord area outside image limits')
        ee = np.empty(nmax)
        for d in range(0, nmax):
            ee[d] = self.fscale * \
                (self.data[i - d:i + d + 1, j - d:j + d + 1] - cont).sum() / etot
        #plt.plot(range(0, nmax), ee)
        wave = WaveCoord(cdelt=np.sqrt(step[0] ** 2 + step[1] ** 2),
                         crval=0.0, cunit='')
        return Spectrum(wave=wave, data=ee)

    def ee_size(self, center=None, pix=False, ee=None, frac=0.9, cont=0):
        """Computes the size of the square centered on (y,x) containing the
        fraction of the energy.

        Parameters
        ----------
        center : (float,float)
                 Center of the explored region.
                 If pix is False, center = (y,x) is in degrees.
                 If pix is True, center = (p,q) is in pixels.
                 If center is None, center of the image is used.
        pix    : boolean
                 If pix is False, center is in degrees.
                 If pix is True, center is in pixels.
        ee     : float
                 Enclosed energy. If ee is not set it is computed from
                 the full image that contain the fraction (frac) of the total energy.
        frac   : float in ]0,1]
                 Fraction of energy.
        cont   : float
                 continuum value

        Returns
        -------
        out : float array
        """
        cont /= self.fscale
        if center is None:
            i = self.shape[0] / 2
            j = self.shape[1] / 2
        else:
            if pix:
                i = center[0]
                j = center[1]
            else:
                pixcrd = self.wcs.sky2pix([[center[0], center[1]]])
                i = int(pixcrd[0][0] + 0.5)
                j = int(pixcrd[0][1] + 0.5)
        nmax = min(self.shape[0] - i, self.shape[1] - j, i, j)
        if ee is None:
            ee = self.fscale * (self.data - cont).sum()
        step = self.get_step()

        if nmax <= 1:
            return np.array([step[0], step[1], 0, 0])
        for d in range(1, nmax):
            ee2 = self.fscale * (self.data[i - d:i + d + 1, j - d:j + d + 1]
                                 - cont).sum() / ee
            if ee2 > frac:
                break
        d -= 1
        ee1 = self.fscale \
            * (self.data[i - d:i + d + 1, i - d:i + d + 1] - cont).sum() / ee
        d += (frac - ee1) / (ee2 - ee1)  # interpolate
        dx = d * step[0] * 2
        dy = d * step[1] * 2
        return np.array([dx, dy])

    def _interp(self, grid, spline=False):
        """returns the interpolated values corresponding to the grid points.

        Parameters
        ----------
        grid :
        pixel values

        spline : bool
        False: linear interpolation
        (uses `scipy.interpolate.griddata <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_), True: spline interpolation (uses `scipy.interpolate.bisplrep <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplrep.html>`_ and `scipy.interpolate.bisplev <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplev.html>`_)
        """
        ksel = np.where(self.data.mask == False)
        x = ksel[0]
        y = ksel[1]
        data = self.data.data[ksel]
        npoints = np.shape(data)[0]

        grid = np.array(grid)
        n = np.shape(grid)[0]

        if spline:
            if self.var is not None:
                weight = np.empty(n, dtype=float)
                for i in range(npoints):
                    weight[i] = 1. / self.var[x[i], y[i]]
                np.ma.fix_invalid(weight, copy=False, fill_value=0)
            else:
                weight = None

            tck = interpolate.bisplrep(x, y, data, w=weight)
            res = interpolate.bisplev(grid[:, 0], grid[:, 1], tck)
            # res = np.zeros(n,dtype=float)
            # for i in range(n):
            #     res[i] = interpolate.bisplev(grid[i,0],grid[i,1],tck)
            return res
        else:
            # scipy 0.9 griddata
            # interpolate.interp2d segfaults when there are too many data points
            # f = interpolate.interp2d(x, y, data)
            points = np.empty((npoints, 2), dtype=float)
            points[:, 0] = ksel[0]
            points[:, 1] = ksel[1]
            res = interpolate.griddata(points, data,
                                       (grid[:, 0], grid[:, 1]),
                                       method='linear')
            # res = np.zeros(n,dtype=float)
            # for i in range(n):
            #     res[i] = interpolate.griddata(points, data, (grid[i,0],grid[i,1]), method='linear')
            return res

    def _interp_data(self, spline=False):
        """returns data array with interpolated values for masked pixels.

        Parameters
        ----------
        spline : bool
                False: bilinear interpolation
                (it uses `scipy.interpolate.griddata <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_), True: spline interpolation (it uses `scipy.interpolate.bisplrep <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplrep.html>`_ and `scipy.interpolate.bisplev <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplev.html>`_)
        """
        if np.ma.count_masked(self.data) == 0:
            return self.data.data
        else:
            ksel = np.where(self.data.mask == True)
            data = self.data.data.__copy__()
            data[ksel] = self._interp(ksel, spline)
            return data

    def moments(self, pix=False):
        """Returns [width_y, width_x] first moments of the 2D gaussian.

        Parameters
        ----------
        pix : boolean
              if pix is True, returned moments are in pixels.
              if pix is False, returned moments are in arcseconds.

        Returns
        -------
        out : float array
        """
        cdelt = self.wcs.get_step()
        total = np.abs(self.data).sum()
        P, Q = np.indices(self.data.shape)
        # python convention: reverse x,y numpy.indices
        p = np.argmax((Q * np.abs(self.data)).sum(axis=1) / total)
        q = np.argmax((P * np.abs(self.data)).sum(axis=0) / total)
        col = self.data[int(p), :]
        width_q = np.sqrt(np.abs((np.arange(col.size) - p) * col).sum()
                          / np.abs(col).sum())
        row = self.data[:, int(q)]
        width_p = np.sqrt(np.abs((np.arange(row.size) - q) * row).sum()
                          / np.abs(row).sum())
        mom = np.array([width_p, width_q])
        if pix is True:
            return mom
        else:
            cdelt = np.abs(self.wcs.get_step())
            return mom * cdelt

    def gauss_fit(self, pos_min=None, pos_max=None, center=None, flux=None,
                  fwhm=None, circular=False, cont=0, fit_back=True, rot=0,
                  peak=False, factor=1, weight=True, plot=False, pix=False,
                  verbose=True, full_output=0):
        """Performs Gaussian fit on image.

        Parameters
        ----------
        pos_min     : (float,float)
                      Minimum y and x values in degrees (pix=False)
                      or in pixels (pix=True).
        pos_max     : (float,float)
                      Maximum y and x values in degrees (pix=False)
                      or in pixels (pix=True).
        center      : (float,float)
                      Initial gaussian center (y_peak,x_peak)
                      in degrees (pix=False) or in pixels (pix=True).
                      If None it is estimated.
        flux        : float
                      Initial integrated gaussian flux
                      or gaussian peak value if peak is True.
                      If None, peak value is estimated.
        fwhm        : (float,float)
                      Initial gaussian fwhm (fwhm_y,fwhm_x) in arcseconds
                      (pix=False) or in pixels (pix=True).
                      If None, they are estimated.
        circular    : boolean
                      True: circular gaussian, False: elliptical gaussian
        cont        : float
                      continuum value, 0 by default.
        fit_back    : boolean
                      False: continuum value is fixed,
                      True: continuum value is a fit parameter.
        rot         : float
                      Initial rotation in degree.
                      If None, rotation is fixed to 0.
        peak        : boolean
                      If true, flux contains a gaussian peak value.
        factor      : integer
                      If factor<=1, gaussian value is computed in the center of each pixel.
                      If factor>1, for each pixel,
                      gaussian value is the sum of the gaussian values
                      on the factor*factor pixels divided by the pixel area.
        weight      : boolean
                      If weight is True, the weight is computed
                      as the inverse of variance.
        plot        : boolean
                      If True, the gaussian is plotted.
        pix         : boolean
                      If pix is False, center and fwhm are in degrees
                      and arcsecs (input and output).
                      If pix is True, center and fwhm are in pixels.
        verbose     : boolean
                      If True, the Gaussian parameters are printed
                      at the end of the method.
        full_output : integer
                      non-zero to return a
                      :class:`mpdaf.obj.Gauss2D` object containing the gauss image

        Returns
        -------
        out : :class:`mpdaf.obj.Gauss2D`
        """
        if pix:
            if pos_min is None:
                pmin = 0
                qmin = 0
            else:
                pmin = pos_min[0]
                qmin = pos_min[1]
            if pos_max is None:
                pmax = self.shape[0]
                qmax = self.shape[1]
            else:
                pmax = pos_max[0]
                qmax = pos_max[1]
        else:
            if pos_min is None:
                pmin = 0
                qmin = 0
            else:
                pixcrd = self.wcs.sky2pix(pos_min)
                pmin = pixcrd[0][0]
                qmin = pixcrd[0][1]
            if pos_max is None:
                pmax = self.shape[0]
                qmax = self.shape[1]
            else:
                pixcrd = self.wcs.sky2pix(pos_max)
                pmax = pixcrd[0][0]
                qmax = pixcrd[0][1]
            if pmin > pmax:
                a = pmax
                pmax = pmin
                pmin = a
            if qmin > qmax:
                a = qmax
                qmax = qmin
                qmin = a

        pmin = max(0, pmin)
        qmin = max(0, qmin)
        ima = self[pmin:pmax, qmin:qmax]

        ksel = np.where(ima.data.mask == False)
        N = np.shape(ksel[0])[0]
        if N == 0:
            raise ValueError('empty sub-image')
        pixcrd = np.empty((np.shape(ksel[0])[0], 2))
        p = ksel[0]
        q = ksel[1]
        data = ima.data.data[ksel]

        # weight
        if ima.var is not None and weight:
            wght = 1.0 / ima.var[ksel]
            np.ma.fix_invalid(wght, copy=False, fill_value=0)
        else:
            wght = np.ones(np.shape(ksel[0])[0])

        # initial gaussian peak position
        if center is None:
            imax = data.argmax()
            center = np.array([p[imax], q[imax]])
        else:
            if not pix:
                center = ima.wcs.sky2pix(center)[0]
            else:
                center = np.array(center)
                center[0] -= pmin
                center[1] -= qmin

        # continuum value
        cont = cont / self.fscale

        # initial moment value
        if fwhm is None:
            width = ima.moments(pix=True)
            fwhm = width * 2. * np.sqrt(2. * np.log(2.0))
        else:
            if not pix:
                fwhm = np.array(fwhm) / np.abs(ima.wcs.get_step())
                if self.wcs.is_deg():
                    fwhm /= 3600.
            width = np.array(fwhm) / (2. * np.sqrt(2. * np.log(2.0)))

        # initial gaussian integrated flux
        if flux is None:
            peak = ima.data.data[center[0], center[1]] - cont
            flux = peak * np.sqrt(2 * np.pi * (width[0] ** 2)) \
                * np.sqrt(2 * np.pi * (width[1] ** 2))
        elif peak is True:
            flux /= self.fscale
            peak = flux - cont
            flux = peak * np.sqrt(2 * np.pi * (width[0] ** 2)) \
                * np.sqrt(2 * np.pi * (width[1] ** 2))
        else:
            flux /= self.fscale

        if circular:
            rot = None
            if not fit_back:
                # 2d gaussian function
                gaussfit = lambda v, p, q: \
                    cont + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                    * np.exp(-(p - v[1]) ** 2 / (2 * v[2] ** 2)) \
                    * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                    * np.exp(-(q - v[3]) ** 2 / (2 * v[2] ** 2))
                # inital guesses for Gaussian Fit
                v0 = [flux, center[0], width[0], center[1]]
            else:
                # 2d gaussian function
                gaussfit = lambda v, p, q: \
                    v[4] + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                    * np.exp(-(p - v[1]) ** 2 / (2 * v[2] ** 2)) \
                    * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                    * np.exp(-(q - v[3]) ** 2 / (2 * v[2] ** 2))
                # inital guesses for Gaussian Fit
                v0 = [flux, center[0], width[0], center[1], cont]
        else:
            if not fit_back:
                if rot is None:
                    # 2d gaussian function
                    gaussfit = lambda v, p, q: \
                        cont + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                        * np.exp(-(p - v[1]) ** 2 / (2 * v[2] ** 2)) \
                        * (1 / np.sqrt(2 * np.pi * (v[4] ** 2))) \
                        * np.exp(-(q - v[3]) ** 2 / (2 * v[4] ** 2))
                    # inital guesses for Gaussian Fit
                    v0 = [flux, center[0], width[0], center[1], width[1]]
                else:
                    # rotation angle in rad
                    rot = np.pi * rot / 180.0
                    # 2d gaussian function
                    gaussfit = lambda v, p, q: \
                        cont + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                        * np.exp(-((p - v[1]) * np.cos(v[5])
                                   - (q - v[3]) * np.sin(v[5])) ** 2
                                 / (2 * v[2] ** 2)) \
                        * (1 / np.sqrt(2 * np.pi * (v[4] ** 2))) \
                        * np.exp(-((p - v[1]) * np.sin(v[5])
                                   + (q - v[3]) * np.cos(v[5])) ** 2
                                 / (2 * v[4] ** 2))
                    # inital guesses for Gaussian Fit
                    v0 = [flux, center[0], width[0], center[1], width[1], rot]
            else:
                if rot is None:
                    # 2d gaussian function
                    gaussfit = lambda v, p, q: \
                        v[5] + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                        * np.exp(-(p - v[1]) ** 2 / (2 * v[2] ** 2)) \
                        * (1 / np.sqrt(2 * np.pi * (v[4] ** 2))) \
                        * np.exp(-(q - v[3]) ** 2 / (2 * v[4] ** 2))
                    # inital guesses for Gaussian Fit
                    v0 = [flux, center[0], width[0], center[1],
                          width[1], cont]
                else:
                    # r otation angle in rad
                    rot = np.pi * rot / 180.0
                    # 2d gaussian function
                    gaussfit = lambda v, p, q: \
                        v[6] + v[0] * (1 / np.sqrt(2 * np.pi * (v[2] ** 2))) \
                        * np.exp(-((p - v[1]) * np.cos(v[5])
                                   - (q - v[3]) * np.sin(v[5])) ** 2
                                 / (2 * v[2] ** 2)) \
                        * (1 / np.sqrt(2 * np.pi * (v[4] ** 2))) \
                        * np.exp(-((p - v[1]) * np.sin(v[5])
                                   + (q - v[3]) * np.cos(v[5])) ** 2
                                 / (2 * v[4] ** 2))
                    # inital guesses for Gaussian Fit
                    v0 = [flux, center[0], width[0], center[1],
                          width[1], rot, cont]

        # Minimize the sum of squares
        if factor > 1:
            factor = int(factor)
            deci = np.ones((factor, factor)) \
                * np.arange(factor)[:, np.newaxis] \
                / float(factor) + 1. / float(factor * 2) - 0.5
            fp = (p[:, np.newaxis] + deci.ravel()[np.newaxis, :]).ravel()
            fq = (q[:, np.newaxis] + deci.T.ravel()[np.newaxis, :]).ravel()
            pixcrd = np.array(zip(fp, fq))

            e_gauss_fit = lambda v, p, q, data, w: \
                w * (((gaussfit(v, p, q)).reshape(N, factor * factor).sum(1)
                      / factor / factor).T.ravel() - data)
            v, covar, info, mesg, success = \
                leastsq(e_gauss_fit, v0[:],
                        args=(pixcrd[:, 0], pixcrd[:, 1], data, wght),
                        maxfev=100, full_output=1)
        else:
            e_gauss_fit = lambda v, p, q, data, w : \
                w * (gaussfit(v, p, q) - data)
            v, covar, info, mesg, success = \
                leastsq(e_gauss_fit, v0[:], args=(p, q, data, wght),
                        maxfev=100, full_output=1)

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i]))
                            * np.sqrt(np.abs(chisq / dof))
                            for i in range(len(v))])
        else:
            err = None

        # center in pixel in the input image
        v[1] += int(pmin)
        v[3] += int(qmin)

        # plot
        if plot:
            pp = np.arange(pmin, pmax, float(pmax - pmin) / 100)
            qq = np.arange(qmin, qmax, float(qmax - qmin) / 100)
            ff = np.empty((np.shape(pp)[0], np.shape(qq)[0]))
            for i in range(np.shape(pp)[0]):
                ff[i, :] = gaussfit(v, pp[i], qq[:]) * self.fscale
            plt.contour(qq, pp, ff, 5)

        # Gauss2D object in pixels
        flux = v[0] * self.fscale
        p_peak = v[1]
        q_peak = v[3]
        if circular:
            if fit_back:
                cont = v[4]
            p_width = np.abs(v[2])
            q_width = p_width
            rot = 0
        else:
            if fit_back:
                if rot is None:
                    cont = v[5]
                else:
                    cont = v[6]
            if rot is None:
                p_width = np.abs(v[2])
                q_width = np.abs(v[4])
                rot = 0
            else:
                if np.abs(v[2]) > np.abs(v[4]):
                    p_width = np.abs(v[2])
                    q_width = np.abs(v[4])
                    rot = (v[5] * 180.0 / np.pi) % 180
                else:
                    p_width = np.abs(v[4])
                    q_width = np.abs(v[2])
                    rot = (v[5] * 180.0 / np.pi + 90) % 180
        p_fwhm = p_width * 2 * np.sqrt(2 * np.log(2))
        q_fwhm = q_width * 2 * np.sqrt(2 * np.log(2))
        peak = flux / np.sqrt(2 * np.pi * (p_width ** 2)) \
            / np.sqrt(2 * np.pi * (q_width ** 2))
        # error
        if err is not None:
            err_flux = err[0] * self.fscale
            err_p_peak = err[1]
            err_q_peak = err[3]
            if circular:
                if fit_back:
                    err_cont = err[4]
                else:
                    err_cont = 0
                err_p_width = np.abs(err[2])
                err_q_width = err_p_width
                err_rot = 0
            else:
                if fit_back:
                    try:
                        err_cont = err[6]
                    except:
                        err_cont = err[5]
                else:
                    err_cont = 0

                if np.abs(v[2]) > np.abs(v[4]) or rot == 0:
                    err_p_width = np.abs(err[2])
                    err_q_width = np.abs(err[4])
                else:
                    err_p_width = np.abs(err[4])
                    err_q_width = np.abs(err[2])

                try:
                    err_rot = err[4] * 180.0 / np.pi
                except:
                    err_rot = 0
            err_p_fwhm = err_p_width * 2 * np.sqrt(2 * np.log(2))
            err_q_fwhm = err_q_width * 2 * np.sqrt(2 * np.log(2))
            err_peak = (err_flux * p_width * q_width - flux
                        * (err_p_width * q_width + err_q_width * p_width)) \
                / (2 * np.pi * p_width * p_width * q_width * q_width)
        else:
            err_flux = np.NAN
            err_p_peak = np.NAN
            err_p_width = np.NAN
            err_p_fwhm = np.NAN
            err_q_peak = np.NAN
            err_q_width = np.NAN
            err_q_fwhm = np.NAN
            err_rot = np.NAN
            err_peak = np.NAN
            err_cont = np.NAN

        if not pix:
            # Gauss2D object in degrees/arcseconds
            center = self.wcs.pix2sky([p_peak, q_peak])[0]
            err_center = np.array([err_p_peak, err_q_peak]) \
                * np.abs(self.wcs.get_step())
            fwhm = np.array([p_fwhm, q_fwhm]) \
                * np.abs(self.wcs.get_step())
            err_fwhm = np.array([err_p_fwhm, err_q_fwhm]) \
                * np.abs(self.wcs.get_step())
            if self.wcs.is_deg():
                fwhm *= 3600.0
                err_fwhm *= 3600.0
            gauss = Gauss2D(center, flux, fwhm, cont * self.fscale, rot,
                            peak, err_center, err_flux, err_fwhm,
                            err_cont * self.fscale, err_rot, err_peak)
        else:
            gauss = Gauss2D((p_peak, q_peak), flux, (p_fwhm, q_fwhm),
                            cont * self.fscale, rot, peak,
                            (err_p_peak, err_q_peak), err_flux,
                            (err_p_fwhm, err_q_fwhm), err_cont * self.fscale,
                            err_rot, err_peak)
        if verbose:
            gauss.print_param()
        if full_output != 0:
            ima = gauss_image(shape=self.shape, wcs=self.wcs, gauss=gauss)
            gauss.ima = ima
        return gauss

    def moffat_fit(self, pos_min=None, pos_max=None, center=None, fwhm=None,
                   flux=None, n=2.0, circular=False, cont=0, fit_back=True,
                   rot=0, peak=False, factor=1, weight=True, plot=False,
                   pix=False, verbose=True, full_output=0):
        """Performs moffat fit on image.

        Parameters
        ----------

        pos_min     : (float,float)
                      Minimum y and x values in degrees (pix=False)
                      or in pixels (pix=True).
        pos_max     : (float,float)
                      Maximum y and x values in degrees (pix=False)
                      or in pixels (pix=True).
        center      : (float,float)
                      Initial moffat center (y_peak,x_peak)
                      in degrees (pix=False) or in pixels (pix=True).
                      If None it is estimated.
        flux        : float
                      Initial integrated gaussian flux
                      or gaussian peak value if peak is True.
                      If None, peak value is estimated.
        fwhm        : (float,float)
                      Initial gaussian fwhm (fwhm_y,fwhm_x) in arcseconds
                      (pix=False) or in pixels (pix=True).
                      If None, they are estimated.
        n           : integer
                      Initial atmospheric scattering coefficient.
        circular    : boolean
                      True: circular moffat, False: elliptical moffat
        cont        : float
                      continuum value, 0 by default.
        fit_back    : boolean
                      False: continuum value is fixed,
                      True: continuum value is a fit parameter.
        rot         : float
                      Initial angle position in degree.
        peak        : boolean
                      If true, flux contains a gaussian peak value.
        factor      : integer
                      If factor<=1,
                      gaussian is computed in the center of each pixel.
                      If factor>1, for each pixel,
                      gaussian value is the sum of the gaussian values
                      on the factor*factor pixels divided by the pixel area.
        weight      : boolean
                      If weight is True, the weight is computed
                      as the inverse of variance.
        plot        : boolean
                      If True, the gaussian is plotted.
        pix         : boolean
                      If pix is False, center and fwhm are in degrees
                      and arcsecs (input and output).
                      If pix is True, center and fwhm are in pixels.
        full_output : integer
                      non-zero to return a :class:`mpdaf.obj.Moffat2D`
                      object containing the moffat image

        Returns
        -------
        out : :class:`mpdaf.obj.Moffat2D`
        """
        if pix:
            if pos_min is None:
                pmin = 0
                qmin = 0
            else:
                pmin = pos_min[0]
                qmin = pos_min[1]
            if pos_max is None:
                pmax = self.shape[0]
                qmax = self.shape[1]
            else:
                pmax = pos_max[0]
                qmax = pos_max[1]
        else:
            if pos_min is None:
                pmin = 0
                qmin = 0
            else:
                pixcrd = self.wcs.sky2pix(pos_min)
                pmin = pixcrd[0][0]
                qmin = pixcrd[0][1]
            if pos_max is None:
                pmax = self.shape[0]
                qmax = self.shape[1]
            else:
                pixcrd = self.wcs.sky2pix(pos_max)
                pmax = pixcrd[0][0]
                qmax = pixcrd[0][1]
            if pmin > pmax:
                a = pmax
                pmax = pmin
                pmin = a
            if qmin > qmax:
                a = qmax
                qmax = qmin
                qmin = a

        pmin = max(0, pmin)
        qmin = max(0, qmin)
        ima = self[pmin:pmax, qmin:qmax]

        ksel = np.where(ima.data.mask == False)
        N = np.shape(ksel[0])[0]
        if N == 0:
            raise ValueError('empty sub-image')
        pixcrd = np.empty((np.shape(ksel[0])[0], 2))
        p = ksel[0]
        q = ksel[1]
        data = ima.data.data[ksel]

        # weight
        if ima.var is not None and weight:
            wght = 1.0 / ima.var[ksel]
            np.ma.fix_invalid(wght, copy=False, fill_value=0)
        else:
            wght = np.ones(np.shape(ksel[0])[0])

        # initial peak position
        if center is None:
            imax = data.argmax()
            center = np.array([p[imax], q[imax]])
        else:
            if not pix:
                center = ima.wcs.sky2pix(center)[0]
            else:
                center = np.array(center)
                center[0] -= pmin
                center[1] -= qmin

        # continuum value
        cont /= self.fscale

        # initial width value
        if fwhm is None:
            width = ima.moments(pix=True)
            fwhm = width * 2. * np.sqrt(2. * np.log(2.0))
        else:
            if not pix:
                fwhm = np.array(fwhm) / np.abs(ima.wcs.get_step())
                if self.wcs.is_deg():
                    fwhm /= 3600.
            else:
                fwhm = np.array(fwhm)

        a = fwhm[0] / (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
        e = fwhm[0] / fwhm[1]

        # initial gaussian integrated flux
        if flux is None:
            I = ima.data.data[center[0], center[1]] - cont
        elif peak is True:
            flux /= self.fscale
            I = flux - cont
        else:
            flux /= self.fscale
            I = flux * (n - 1) / (np.pi * a * a * e)

        if circular:
            rot = None
            if not fit_back:
                # 2d moffat function
                moffatfit = lambda v, p, q: \
                    cont + v[0] * (1 + ((p - v[1]) / v[3]) ** 2
                                   + ((q - v[2]) / v[3]) ** 2) ** (-v[4])
                # inital guesses
                v0 = [I, center[0], center[1], a, n]
            else:
                # 2d moffat function
                moffatfit = lambda v, p, q: \
                    v[5] + v[0] * (1 + ((p - v[1]) / v[3]) ** 2
                                   + ((q - v[2]) / v[3]) ** 2) ** (-v[4])
                # inital guesses
                v0 = [I, center[0], center[1], a, n, cont]
        else:
            if not fit_back:
                if rot is None:
                    # 2d moffat function
                    moffatfit = lambda v, p, q: \
                        cont + v[0] * (1 + ((p - v[1]) / v[3]) ** 2
                                       + ((q - v[2]) / v[3] / v[5]) ** 2) \
                        ** (-v[4])
                    # inital guesses
                    v0 = [I, center[0], center[1], a, n, e]
                else:
                    # rotation angle in rad
                    rot = np.pi * rot / 180.0
                    # 2d moffat function
                    moffatfit = lambda v, p, q: cont + v[0] \
                        * (1 + (((p - v[1]) * np.cos(v[6]) - (q - v[2])
                                 * np.sin(v[6])) / v[3]) ** 2
                           + (((p - v[1]) * np.sin(v[6]) + (q - v[2])
                               * np.cos(v[6])) / v[3] / v[5]) ** 2) ** (-v[4])
                    # inital guesses
                    v0 = [I, center[0], center[1], a, n, e, rot]
            else:
                if rot is None:
                    # 2d moffat function
                    moffatfit = lambda v, p, q: v[6] + v[0] \
                        * (1 + ((p - v[1]) / v[3]) ** 2
                           + ((q - v[2]) / v[3] / v[5]) ** 2) ** (-v[4])
                    # inital guesses
                    v0 = [I, center[0], center[1], a, n, e, cont]
                else:
                    # rotation angle in rad
                    rot = np.pi * rot / 180.0
                    # 2d moffat function
                    moffatfit = lambda v, p, q: v[7] + v[0] \
                        * (1 + (((p - v[1]) * np.cos(v[6])
                                 - (q - v[2]) * np.sin(v[6])) / v[3]) ** 2
                           + (((p - v[1]) * np.sin(v[6])
                               + (q - v[2]) * np.cos(v[6])) / v[3] / v[5]) ** 2) \
                        ** (-v[4])
                    # inital guesses
                    v0 = [I, center[0], center[1], a, n, e, rot, cont]

        # Minimize the sum of squares
        if factor > 1:
            factor = int(factor)
            deci = np.ones((factor, factor)) \
                * np.arange(factor)[:, np.newaxis] / float(factor) \
                + 1 / float(factor * 2)
            fp = (p[:, np.newaxis] + deci.ravel()[np.newaxis, :]).ravel()
            fq = (q[:, np.newaxis] + deci.T.ravel()[np.newaxis, :]).ravel()
            pixcrd = np.array(zip(fp, fq))

            e_moffat_fit = lambda v, p, q, data, w: \
                w * (((moffatfit(v, p, q)).reshape(N, factor * factor).sum(1)
                      / factor / factor).T.ravel() - data)
            v, covar, info, mesg, success = \
                leastsq(e_moffat_fit, v0[:], args=(pixcrd[:, 0], pixcrd[:, 1],
                                                   data, wght),
                        maxfev=100, full_output=1)
            while np.abs(v[1] - v0[1]) > 0.1 or np.abs(v[2] - v0[2]) > 0.1 \
                    or np.abs(v[3] - v0[3]) > 0.1:
                v0 = v
                v, covar, info, mesg, success = \
                    leastsq(e_moffat_fit, v0[:],
                            args=(pixcrd[:, 0], pixcrd[:, 1],
                                  data, wght), maxfev=100, full_output=1)
        else:
            e_moffat_fit = lambda v, p, q, data, w: \
                w * (moffatfit(v, p, q) - data)
            v, covar, info, mesg, success = \
                leastsq(e_moffat_fit, v0[:],
                        args=(p, q, data, wght),
                        maxfev=100, full_output=1)
            while np.abs(v[1] - v0[1]) > 0.1 or np.abs(v[2] - v0[2]) > 0.1 \
                    or np.abs(v[3] - v0[3]) > 0.1:
                v0 = v
                v, covar, info, mesg, success = \
                    leastsq(e_moffat_fit, v0[:],
                            args=(p, q, data, wght),
                            maxfev=100, full_output=1)

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i]))
                            * np.sqrt(np.abs(chisq / dof))
                            for i in range(len(v))])
        else:
            err = np.zeros_like(v)
            err[:] = np.abs(v[:] - v0[:])

        # center in pixel in the input image
        v[1] += int(pmin)
        v[2] += int(qmin)

        # plot
        if plot:
            pp = np.arange(pmin, pmax, float(pmax - pmin) / 100)
            qq = np.arange(qmin, qmax, float(qmax - qmin) / 100)
            ff = np.empty((np.shape(pp)[0], np.shape(qq)[0]))
            for i in range(np.shape(pp)[0]):
                ff[i, :] = moffatfit(v, pp[i], qq[:]) * self.fscale
            plt.contour(qq, pp, ff, 5)

        # Gauss2D object in pixels
        I = v[0] * self.fscale
        p_peak = v[1]
        q_peak = v[2]
        a = np.abs(v[3])
        n = v[4]
        if circular:
            e = 1
            rot = 0
            if fit_back:
                cont = v[5]
        else:
            e = np.abs(v[5])
            if rot is None:
                rot = 0
                if fit_back:
                    cont = v[6]
            else:
                rot = (v[6] * 180.0 / np.pi) % 180
                if fit_back:
                    cont = v[7]

        fwhm[0] = a * (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
        fwhm[1] = fwhm[0] / e

        flux = I / (n - 1) * (np.pi * a * a * e)

        if err is not None:
            err_I = err[0] * self.fscale
            err_p_peak = err[1]
            err_q_peak = err[2]
            err_a = err[3]
            err_n = err[4]
            err_fwhm = err_a * err_n
            if circular:
                err_e = 0
                err_rot = 0
                err_fwhm = np.array([err_fwhm, err_fwhm])
                if fit_back:
                    err_cont = err[5]
                err_flux = err_I * err_n * err_a * err_a
            else:
                err_e = err[5]
                if err_e != 0:
                    err_fwhm = np.array([err_fwhm, err_fwhm / err_e])
                else:
                    err_fwhm = np.array([err_fwhm, err_fwhm])
                if rot is None:
                    err_rot = 0
                    if fit_back:
                        err_cont = err[6]
                    else:
                        err_cont = 0
                else:
                    err_rot = err[6] * 180.0 / np.pi
                    if fit_back:
                        err_cont = err[7]
                    else:
                        err_cont = 0
                err_flux = err_I * err_n * err_a * err_a * err_e
        else:
            err_I = np.NAN
            err_p_peak = np.NAN
            err_q_peak = np.NAN
            err_a = np.NAN
            err_n = np.NAN
            err_e = np.NAN
            err_rot = np.NAN
            err_cont = np.NAN
            err_fwhm = (np.NAN, np.NAN)
            err_flux = np.NAN

        if not pix:
            # Gauss2D object in degrees/arcseconds
            center = self.wcs.pix2sky([p_peak, q_peak])[0]
            err_center = np.array([err_p_peak, err_q_peak]) \
                * np.abs(self.wcs.get_step()[0])
            a = a * np.abs(self.wcs.get_step()[0])
            err_a = err_a * np.abs(self.wcs.get_step()[0])
            if self.wcs.is_deg():
                a *= 3600.0
                err_a *= 3600.0
            moffat = Moffat2D(center, flux, fwhm, cont * self.fscale, n,
                              rot, I, err_center, err_flux, err_fwhm,
                              err_cont * self.fscale, err_n, err_rot, err_I)
        else:
            moffat = Moffat2D((p_peak, q_peak), flux, fwhm,
                              cont * self.fscale, n, rot, I,
                              (err_p_peak, err_q_peak), err_flux,
                              err_fwhm, err_cont * self.fscale,
                              err_n, err_rot, err_I)
        if verbose:
            moffat.print_param()
        if full_output != 0:
            ima = moffat_image(shape=self.shape, wcs=self.wcs, moffat=moffat)
            moffat.ima = ima
        return moffat

    def _rebin_factor_(self, factor):
        """Shrinks the size of the image by factor. New size is an integer
        multiple of the original size.

        Parameters
        ----------
        factor : (integer,integer)
                 Factor in y and x.
                 Python notation: (ny,nx)
        """
        assert not np.sometrue(np.mod(self.shape[0], factor[0]))
        assert not np.sometrue(np.mod(self.shape[1], factor[1]))
        # new size is an integer multiple of the original size
        self.shape = np.array((self.shape[0] / factor[0],
                               self.shape[1] / factor[1]))
        self.data = self.data.reshape(self.shape[0], factor[0],
                                      self.shape[1], factor[1]).sum(1).sum(2)\
            / factor[0] / factor[1]
        if self.var is not None:
            self.var = self.var.reshape(self.shape[0], factor[0],
                                        self.shape[1], factor[1])\
                .sum(1).sum(2) / factor[0] \
                / factor[1] / factor[0] / factor[1]
        #cdelt = self.wcs.get_step()
        self.wcs = self.wcs.rebin_factor(factor)

    def _rebin_factor(self, factor, margin='center'):
        """Shrinks the size of the image by factor.

        Parameters
        ----------
        factor : integer or (integer,integer)
                 Factor in y and x. Python notation: (ny,nx).
        margin : 'center' or 'origin'
                 This parameters is used if new size
                 is not an integer multiple of the original size.
                 In 'center' case, pixels will be added on the left
                 and on the right, on the bottom and of the top of the image.
                In 'origin'case, pixels will be added on (n+1) line/column.
        """
        if is_int(factor):
            factor = (factor, factor)
        if factor[0] <= 1 or factor[0] >= self.shape[0] \
                or factor[1] <= 1 or factor[1] >= self.shape[1]:
            raise ValueError('factor must be in ]1,shape[')
        if not np.sometrue(np.mod(self.shape[0], factor[0])) \
                and not np.sometrue(np.mod(self.shape[1], factor[1])):
            # new size is an integer multiple of the original size
            self._rebin_factor_(factor)
            return None
        elif not np.sometrue(np.mod(self.shape[0], factor[0])):
            newshape1 = self.shape[1] / factor[1]
            n1 = self.shape[1] - newshape1 * factor[1]
            if margin == 'origin' or n1 == 1:
                ima = self[:, :-n1]
                ima._rebin_factor_(factor)
                newshape = (ima.shape[0], ima.shape[1] + 1)
                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[:, 0:-1] = ima.data
                mask[:, 0:-1] = ima.data.mask
                d = self.data[:, -n1:].sum(axis=1)\
                    .reshape(ima.shape[0], factor[0]).sum(1) \
                    / factor[0] / factor[1]
                data[:, -1] = d.data
                mask[:, -1] = d.mask
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[:, 0:-1] = ima.var
                    var[:, -1] = self.var[:, -n1:].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1)\
                        / factor[0] / factor[0] / factor[1] / factor[1]
                wcs = ima.wcs
                wcs.naxis1 = wcs.naxis1 + 1
            else:
                n_left = n1 / 2
                n_right = self.shape[1] - n1 + n_left
                ima = self[:, n_left:n_right]
                ima._rebin_factor_(factor)
                newshape = (ima.shape[0], ima.shape[1] + 2)
                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[:, 1:-1] = ima.data
                mask[:, 1:-1] = ima.data.mask
                d = self.data[:, 0:n_left].sum(axis=1)\
                    .reshape(ima.shape[0], factor[0]).sum(1) \
                    / factor[0] / factor[1]
                data[:, 0] = d.data
                mask[:, 0] = d.mask
                d = self.data[:, n_right:].sum(axis=1)\
                    .reshape(ima.shape[0], factor[0]).sum(1)\
                    / factor[0] / factor[1]
                data[:, -1] = d.data
                mask[:, -1] = d.mask
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[:, 1:-1] = ima.var
                    var[:, 0] = self.var[:, 0:n_left].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                    var[:, -1] = self.var[:, n_right:].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                wcs = ima.wcs
                wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                wcs.set_naxis1(wcs.naxis1 + 2)
        elif not np.sometrue(np.mod(self.shape[1], factor[1])):
            newshape0 = self.shape[0] / factor[0]
            n0 = self.shape[0] - newshape0 * factor[0]
            if margin == 'origin' or n0 == 1:
                ima = self[:-n0, :]
                ima._rebin_factor_(factor)
                newshape = (ima.shape[0] + 1, ima.shape[1])
                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[0:-1, :] = ima.data
                mask[0:-1, :] = ima.data.mask
                d = self.data[-n0:, :].sum(axis=0)\
                    .reshape(ima.shape[1], factor[1]).sum(1) \
                    / factor[0] / factor[1]
                data[-1, :] = d.data
                mask[-1, :] = d.mask
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[0:-1, :] = ima.var
                    var[-1, :] = self.var[-n0:,:].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1]
                wcs = ima.wcs
                wcs.naxis2 = wcs.naxis2 + 1
            else:
                n_left = n0 / 2
                n_right = self.shape[0] - n0 + n_left
                ima = self[n_left:n_right, :]
                ima._rebin_factor_(factor)
                newshape = (ima.shape[0] + 2, ima.shape[1])
                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[1:-1, :] = ima.data
                mask[1:-1, :] = ima.data.mask
                d = self.data[0:n_left, :].sum(axis=0)\
                    .reshape(ima.shape[1], factor[1]).sum(1)\
                    / factor[0] / factor[1]
                data[0, :] = d.data
                mask[0, :] = d.mask
                d = self.data[n_right:, :].sum(axis=0)\
                    .reshape(ima.shape[1], factor[1]).sum(1) \
                    / factor[0] / factor[1]
                data[-1, :] = d.data
                mask[-1, :] = d.mask
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[1:-1, :] = ima.var
                    var[0, :] = self.var[0:n_left,:].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                    var[-1, :] = self.var[n_right:,:].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                wcs = ima.wcs
                wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                wcs.set_naxis2(wcs.naxis2 + 2)
        else:
            factor = np.array(factor)
            newshape = self.shape / factor
            n = self.shape - newshape * factor
            if n[0] == 1 and n[1] == 1:
                margin = 'origin'
            if margin == 'center':
                n_left = n / 2
                n_right = self.shape - n + n_left
                ima = self[n_left[0]:n_right[0], n_left[1]:n_right[1]]
                ima._rebin_factor_(factor)
                if n_left[0] != 0 and n_left[1] != 0:
                    newshape = (ima.shape[0] + 2, ima.shape[1] + 2)
                    data = np.empty(newshape)
                    mask = np.empty(newshape, dtype=bool)
                    data[1:-1, 1:-1] = ima.data
                    mask[1:-1, 1:-1] = ima.data.mask
                    data[0, 0] = self.data[0:n_left[0], 0:n_left[1]].sum() \
                        / factor[0] / factor[1]
                    mask[0, 0] = \
                        self.data.mask[0:n_left[0], 0:n_left[1]].any()
                    data[0, -1] = self.data[0:n_left[0], n_right[1]:].sum() \
                        / factor[0] / factor[1]
                    mask[0, -1] = \
                        self.data.mask[0:n_left[0], n_right[1]:].any()
                    data[-1, 0] = self.data[n_right[0]:, 0:n_left[1]].sum() \
                        / factor[0] / factor[1]
                    mask[-1, 0] = \
                        self.data.mask[n_right[0]:, 0:n_left[1]].any()
                    data[-1, -1] = self.data[n_right[0]:, n_right[1]:].sum() \
                        / factor[0] / factor[1]
                    mask[-1, -1] = \
                        self.data.mask[n_right[0]:, n_right[1]:].any()
                    d = self.data[0:n_left[0], n_left[1]:n_right[1]]\
                        .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1]
                    data[0, 1:-1] = d.data
                    mask[0, 1:-1] = d.mask
                    d = self.data[n_right[0]:, n_left[1]:n_right[1]]\
                        .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1]
                    data[-1, 1:-1] = d.data
                    mask[-1, 1:-1] = d.mask
                    d = self.data[n_left[0]:n_right[0], 0:n_left[1]]\
                        .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1]
                    data[1:-1, 0] = d.data
                    mask[1:-1, 0] = d.mask
                    d = self.data[n_left[0]:n_right[0], n_right[1]:]\
                        .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1]
                    data[1:-1, -1] = d.data
                    mask[1:-1, -1] = d.mask
                    var = None
                    if self.var is not None:
                        var = np.empty(newshape)
                        var[1:-1, 1:-1] = ima.var
                        var[0, 0] = self.var[0:n_left[0], 0:n_left[1]].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[0, -1] = self.var[0:n_left[0], n_right[1]:].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, 0] = self.var[n_right[0]:, 0:n_left[1]].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, -1] = self.var[n_right[0]:, n_right[1]:].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[0, 1:-1] = \
                            self.var[0:n_left[0], n_left[1]:n_right[1]]\
                            .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, 1:-1] = \
                            self.var[n_right[0]:, n_left[1]:n_right[1]]\
                            .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[1:-1, 0] = \
                            self.var[n_left[0]:n_right[0], 0:n_left[1]]\
                            .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[1:-1, -1] = \
                            self.var[n_left[0]:n_right[0], n_right[1]:]\
                            .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                    wcs = ima.wcs
                    #step = wcs.get_step()
                    wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                    wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                    wcs.set_naxis1(wcs.naxis1 + 2)
                    wcs.set_naxis2(wcs.naxis2 + 2)
                elif n_left[0] == 0:
                    newshape = (ima.shape[0] + 1, ima.shape[1] + 2)
                    data = np.empty(newshape)
                    mask = np.empty(newshape, dtype=bool)
                    data[0:-1, 1:-1] = ima.data
                    mask[0:-1, 1:-1] = ima.data.mask

                    data[0, 0] = self.data[0, 0:n_left[1]].sum() \
                        / factor[0] / factor[1]
                    mask[0, 0] = self.data.mask[0, 0:n_left[1]].any()
                    data[0, -1] = self.data[0, n_right[1]:].sum()\
                        / factor[0] / factor[1]
                    mask[0, -1] = self.data.mask[0, n_right[1]:].any()
                    data[-1, 0] = self.data[n_right[0]:, 0:n_left[1]].sum()\
                        / factor[0] / factor[1]
                    mask[-1, 0] = self.data.mask[n_right[0]:, 0:n_left[1]].any()
                    data[-1, -1] = self.data[n_right[0]:, n_right[1]:].sum()\
                        / factor[0] / factor[1]
                    mask[-1, -1] = \
                        self.data.mask[n_right[0]:, n_right[1]:].any()
                    d = self.data[n_right[0]:, n_left[1]:n_right[1]]\
                        .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1)\
                        / factor[0] / factor[1]
                    data[-1, 1:-1] = d.data
                    mask[-1, 1:-1] = d.mask
                    d = self.data[0:n_right[0], 0:n_left[1]].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1)\
                        / factor[0] / factor[1]
                    data[0:-1, 0] = d.data
                    mask[0:-1, 0] = d.mask
                    d = self.data[0:n_right[0], n_right[1]:].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1]
                    data[0:-1, -1] = d.data
                    mask[0:-1, -1] = d.mask
                    var = None
                    if self.var is not None:
                        var = np.empty(newshape)
                        var[0:-1, 1:-1] = ima.var
                        var[0, 0] = self.var[0, 0:n_left[1]].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[0, -1] = self.var[0, n_right[1]:].sum() \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, 0] = self.var[n_right[0]:, 0:n_left[1]].sum()\
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, -1] = self.var[n_right[0]:, n_right[1]:].sum()\
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[-1, 1:-1] = \
                            self.var[n_right[0]:, n_left[1]:n_right[1]]\
                            .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[0:-1, 0] = \
                            self.var[0:n_right[0], 0:n_left[1]]\
                            .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]
                        var[0:-1, -1] = \
                            self.var[0:n_right[0], n_right[1]:]\
                            .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                            / factor[0] / factor[1] / factor[0] / factor[1]

                    wcs = ima.wcs
                    wcs.set_crpix1(wcs.wcs.wcs.crpix[0] + 1)
                    wcs.set_naxis1(wcs.naxis1 + 2)
                    wcs.set_naxis2(wcs.naxis2 + 1)
                else:
                    newshape = (ima.shape[0] + 2, ima.shape[1] + 1)
                    data = np.empty(newshape)
                    mask = np.empty(newshape, dtype=bool)
                    data[1:-1, 0:-1] = ima.data
                    mask[1:-1, 0:-1] = ima.data.mask

                    data[0, 0] = self.data[0:n_left[0], 0].sum() \
                        / factor[0] / factor[1]
                    mask[0, 0] = self.data.mask[0:n_left[0], 0].any()
                    data[0, -1] = self.data[0:n_left[0], n_right[1]:].sum() \
                        / factor[0] / factor[1]
                    mask[0, -1] = \
                        self.data.mask[0:n_left[0], n_right[1]:].any()
                    data[-1, 0] = self.data[n_right[0]:, 0].sum() \
                        / factor[0] / factor[1]
                    mask[-1, 0] = self.data.mask[n_right[0]:, 0].any()
                    data[-1, -1] = self.data[n_right[0]:, n_right[1]:].sum() \
                        / factor[0] / factor[1]
                    mask[-1, -1] = \
                        self.data.mask[n_right[0]:, n_right[1]:].any()
                    d = self.data[0:n_left[0], 0:n_right[1]].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1]
                    data[0, 0:-1] = d.data
                    mask[0, 0:-1] = d.mask
                    d = self.data[n_right[0]:, 0:n_right[1]].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1]
                    data[-1, 0:-1] = d.data
                    mask[-1, 0:-1] = d.mask
                    d = self.data[n_left[0]:n_right[0], n_right[1]:]\
                        .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1]
                    data[1:-1, -1] = d.data
                    mask[1:-1, -1] = d.mask

                    var = None
                    if self.var is not None:
                        var = np.empty(newshape)
                        var[1:-1, 0:-1] = ima.var
                        var[0, 0] = self.var[0:n_left[0], 0].sum() \
                            / factor[0] / factor[1]
                        var[0, -1] = \
                            self.var[0:n_left[0], n_right[1]:].sum() \
                            / factor[0] / factor[1]
                        var[-1, 0] = self.var[n_right[0]:, 0].sum() \
                            / factor[0] / factor[1]
                        var[-1, -1] = \
                            self.var[n_right[0]:, n_right[1]:].sum() \
                            / factor[0] / factor[1]
                        var[0, 0:-1] = \
                            self.var[0:n_left[0], 0:n_right[1]].sum(axis=0)\
                            .reshape(ima.shape[1], factor[1]).sum(1) \
                            / factor[0] / factor[1]
                        var[-1, 0:-1] = self.var[n_right[0]:, 0:n_right[1]]\
                            .sum(axis=0).reshape(ima.shape[1], factor[1]).sum(1) \
                            / factor[0] / factor[1]
                        var[1:-1, -1] = \
                            self.var[n_left[0]:n_right[0], n_right[1]:]\
                            .sum(axis=1).reshape(ima.shape[0], factor[0]).sum(1) \
                            / factor[0] / factor[1]
                    wcs = ima.wcs
                    wcs.set_crpix2(wcs.wcs.wcs.crpix[1] + 1)
                    wcs.set_naxis1(wcs.naxis1 + 1)
                    wcs.set_naxis2(wcs.naxis2 + 2)
            elif margin == 'origin':
                n_right = self.shape - n
                ima = self[0:n_right[0], 0:n_right[1]]
                ima._rebin_factor_(factor)
                newshape = (ima.shape[0] + 1, ima.shape[1] + 1)
                data = np.empty(newshape)
                mask = np.empty(newshape, dtype=bool)
                data[0:-1, 0:-1] = ima.data
                mask[0:-1, 0:-1] = ima.data.mask
                d = self.data[n_right[0]:, 0:n_right[1]].sum(axis=0)\
                    .reshape(ima.shape[1], factor[1]).sum(1) \
                    / factor[0] / factor[1]
                data[-1, 0:-1] = d.data
                mask[-1, 0:-1] = d.mask
                d = self.data[0:n_right[0], n_right[1]:].sum(axis=1)\
                    .reshape(ima.shape[0], factor[0]).sum(1) \
                    / factor[0] / factor[1]
                data[0:-1, -1] = d.data
                mask[0:-1, -1] = d.mask
                data[-1, -1] = self.data[n_right[0]:, n_right[1]:].sum() \
                    / factor[0] / factor[1]
                mask[-1, -1] = self.data.mask[n_right[0]:, n_right[1]:].any()
                var = None
                if self.var is not None:
                    var = np.empty(newshape)
                    var[0:-1, 0:-1] = ima.var
                    var[-1, 0:-1] = \
                        self.var[n_right[0]:, 0:n_right[1]].sum(axis=0)\
                        .reshape(ima.shape[1], factor[1]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                    var[0:-1, -1] = \
                        self.var[0:n_right[0], n_right[1]:].sum(axis=1)\
                        .reshape(ima.shape[0], factor[0]).sum(1) \
                        / factor[0] / factor[1] / factor[0] / factor[1]
                    var[-1, -1] = self.var[n_right[0]:, n_right[1]:]\
                        .sum() / factor[0] / factor[1] / factor[0] / factor[1]
                wcs = ima.wcs
                wcs.naxis1 = wcs.naxis1 + 1
                wcs.naxis2 = wcs.naxis2 + 1
            else:
                raise ValueError('margin must be center|origin')
        self.shape = np.array(newshape)
        self.wcs = wcs
        self.data = np.ma.array(data, mask=mask)
        self.var = var

    def rebin_factor(self, factor, margin='center'):
        """Returns an image that shrinks the size of the current image by
        factor.

        Parameters
        ----------
        factor : integer or (integer,integer)
                 Factor in y and x. Python notation: (ny,nx).
        margin : 'center' or 'origin'
                 This parameters is used if new size is not
                 an integer multiple of the original size.
                 In 'center' case, pixels will be added on the left
                 and on the right, on the bottom and of the top of the image.
                 In 'origin'case, pixels will be added on (n+1) line/column.

        Returns
        -------
        out : Image
        """
        res = self.copy()
        res._rebin_factor(factor, margin)
        return res

    def _med_(self, p, q, pfactor, qfactor):
        return np.ma.median(self.data[p * pfactor:(p + 1) * pfactor,
                                      q * qfactor:(q + 1) * qfactor])

    def _rebin_median_(self, factor):
        """Shrinks the size of the image by factor. New size is an integer
        multiple of the original size.

        Parameters
        ----------
        factor : (integer,integer)
                 Factor in y and x.
                 Python notation: (ny,nx)
        """
        assert not np.sometrue(np.mod(self.shape[0], factor[0]))
        assert not np.sometrue(np.mod(self.shape[1], factor[1]))
        # new size is an integer multiple of the original size
        self.shape = np.array((self.shape[0] / factor[0],
                               self.shape[1] / factor[1]))
        Nq, Np = np.meshgrid(xrange(self.shape[1]), xrange(self.shape[0]))
        vfunc = np.vectorize(self._med_)
        data = vfunc(Np, Nq, factor[0], factor[1])
        mask = self.data.mask.reshape(self.shape[0], factor[0],
                                      self.shape[1], factor[1])\
            .sum(1).sum(2) / factor[0] / factor[1]
        self.data = np.ma.array(data, mask=mask)
        self.var = None
        self.wcs = self.wcs.rebin_factor(factor)

    def rebin_median(self, factor, margin='center'):
        """Shrinks the size of the image by factor. Median values are used.

        Parameters
        ----------
        factor : integer or (integer,integer)
                 Factor in y and x. Python notation: (ny,nx).
        margin : 'center' or 'origin'
                  This parameters is used
                  if new size is not an integer multiple of the original size.
                  In 'center' case, image is truncated on the left
                  and on the right, on the bottom and of the top of the image.
                  In 'origin'case, image is truncated
                  at the end along each direction.

        Returns
        -------
        out : :class:`mpdaf.obj.Image`
        """
        if is_int(factor):
            factor = (factor, factor)
        if factor[0] <= 1 or factor[0] >= self.shape[0] \
                or factor[1] <= 1 or factor[1] >= self.shape[1]:
            raise ValueError('factor must be in ]1,shape[')
            return None
        if not np.sometrue(np.mod(self.shape[0], factor[0])) \
                and not np.sometrue(np.mod(self.shape[1], factor[1])):
            # new size is an integer multiple of the original size
            res = self.copy()
        elif not np.sometrue(np.mod(self.shape[0], factor[0])):
            newshape1 = self.shape[1] / factor[1]
            n1 = self.shape[1] - newshape1 * factor[1]
            if margin == 'origin' or n1 == 1:
                res = self[:, :-n1]
            else:
                n_left = n1 / 2
                n_right = self.shape[1] - n1 + n_left
                res = self[:, n_left:n_right]
        elif not np.sometrue(np.mod(self.shape[1], factor[1])):
            newshape0 = self.shape[0] / factor[0]
            n0 = self.shape[0] - newshape0 * factor[0]
            if margin == 'origin' or n0 == 1:
                res = self[:-n0, :]
            else:
                n_left = n0 / 2
                n_right = self.shape[0] - n0 + n_left
                res = self[n_left:n_right, :]
        else:
            factor = np.array(factor)
            newshape = self.shape / factor
            n = self.shape - newshape * factor
            if n[0] == 1 and n[1] == 1:
                margin = 'origin'
            if margin == 'center':
                n_left = n / 2
                n_right = self.shape - n + n_left
                res = self[n_left[0]:n_right[0], n_left[1]:n_right[1]]
            elif margin == 'origin':
                n_right = self.shape - n
                res = self[0:n_right[0], 0:n_right[1]]
            else:
                raise ValueError('margin must be center|origin')
        res._rebin_median_(factor)
        return res

    def _rebin(self, newdim, newstart, newstep,
               flux=False, order=3, interp='no'):
        """Rebins the image to a new coordinate system.
        Uses `scipy.ndimage.affine_transform <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_.

        Parameters
        ----------
        newdim   : integer or (integer,integer)
                New dimensions. Python notation: (ny,nx)
        newstart : float or (float, float)
                New positions (y,x) for the pixel (0,0).
                If None, old position is used.
        newstep  : float or (float, float)
                New step (dy,dx).
        flux     : boolean
                if flux is True, the flux is conserved.
        order    : integer
                The order of the spline interpolation, default is 3.
                The order has to be in the range 0-5.
        interp   : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        """
        if is_int(newdim):
            newdim = (newdim, newdim)
        if newstart is None:
            newstart = self.wcs.get_start()
        elif is_int(newstart) or is_float(newstart):
            newstart = (newstart, newstart)
        else:
            pass
        if is_int(newstep) or is_float(newstep):
            newstep = (newstep, newstep)
        newdim = np.array(newdim)
        newstart = np.array(newstart)
        newstep = np.array(newstep)

        wcs = WCS(crpix=[1, 1], crval=newstart, cdelt=newstep, deg=self.wcs.is_deg(), rot=self.wcs.get_rot(), shape=newdim)
        pstep = newstep / self.wcs.get_step()

        poffset = self.wcs.sky2pix(newstart)[0] / pstep  # ok without rotation

        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        data = ndimage.affine_transform(data, pstep, poffset,
                                        output_shape=newdim, order=order)
        mask = np.array(1 - self.data.mask, dtype=bool)
        newmask = ndimage.affine_transform(mask, pstep, poffset,
                                           output_shape=newdim, order=0)
        mask = np.ma.make_mask(1 - newmask)

        if flux:
            rflux = self.wcs.get_step().prod() / newstep.prod()
            data *= rflux

        self.shape = newdim
        self.wcs = wcs
        self.data = np.ma.array(data, mask=mask)
        self.var = None

    def rebin(self, newdim, newstart, newstep, flux=False,
              order=3, interp='no'):
        """Returns rebinned image to a new coordinate system.
        Uses `scipy.ndimage.affine_transform <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`_.

        Parameters
        ----------
        newdim   : integer or (integer,integer)
                New dimensions. Python notation: (ny,nx)
        newstart : float or (float, float)
                New positions (y,x) for the pixel (0,0).
                If None, old position is used.
        newstep  : float or (float, float)
                New step (dy,dx).
        flux     : boolean
                if flux is True, the flux is conserved.
        order    : integer
                The order of the spline interpolation, default is 3.
                The order has to be in the range 0-5.
        interp   : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : Image
        """
        res = self.copy()
        res._rebin(newdim, newstart, newstep, flux, order, interp)
        return res

    def _gaussian_filter(self, sigma=3, interp='no'):
        """Applies Gaussian filter to the image.
        Uses `scipy.ndimage.gaussian_filter <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.filters.gaussian_filter.html>`_.

        Parameters
        ----------
        sigma  : float
                Standard deviation for Gaussian kernel
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(ndimage.gaussian_filter(data, sigma),
                                mask=self.data.mask)
        if self.var is not None:
            self.var = ndimage.gaussian_filter(self.var, sigma)

    def gaussian_filter(self, sigma=3, interp='no'):
        """Returns an image containing Gaussian filter
        applied to the current image.
        Uses `scipy.ndimage.gaussian_filter <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.filters.gaussian_filter.html>`_.

        Parameters
        ----------
        sigma  : float
                Standard deviation for Gaussian kernel
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : Image
        """
        res = self.copy()
        res._gaussian_filter(sigma, interp)
        return res

    def _median_filter(self, size=3, interp='no'):
        """Applies median filter to the image.
        Uses `scipy.ndimage.median_filter <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.filters.median_filter.html>`_.

        Parameters
        ----------
        size   : float
                Shape that is taken from the input array,
                at every element position, to define the input to
                the filter function. Default is 3.
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(ndimage.median_filter(data, size),
                                mask=self.data.mask)
        if self.var is not None:
            self.var = ndimage.median_filter(self.var, size)

    def median_filter(self, size=3, interp='no'):
        """Returns an image containing median filter
        applied to the current image.
        Uses `scipy.ndimage.median_filter <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.filters.median_filter.html>`_.

        Parameters
        ----------
        size   : float
                Shape that is taken from the input array,
                at every element position, to define the input to
                the filter function. Default is 3.
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : Image
        """
        res = self.copy()
        res._median_filter(size, interp)
        return res

    def _maximum_filter(self, size=3, interp='no'):
        """Applies maximum filter to the image.
        Uses `scipy.ndimage.maximum_filter <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.filters.maximum_filter.html>`_.

        Parameters
        ----------
        size   : float
                Shape that is taken from the input array,
                at every element position, to define the input to
                the filter function. Default is 3.
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(ndimage.maximum_filter(data, size),
                                mask=self.data.mask)

    def maximum_filter(self, size=3, interp='no'):
        """Returns an image containing maximum filter
        applied to the current image.
        Uses `scipy.ndimage.maximum_filter <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.filters.maximum_filter.html>`_.

        Parameters
        ----------
        size   : float
                Shape that is taken from the input array,
                at every element position, to define the input to
                the filter function. Default is 3.
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : Image
        """
        res = self.copy()
        res._maximum_filter(size, interp)
        return res

    def _minimum_filter(self, size=3, interp='no'):
        """Applies minimum filter to the image.
        Uses `scipy.ndimage.minimum_filter <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.filters.minimum_filter.html>`_.

        Parameters
        ----------
        size   : float
                Shape that is taken from the input array,
                at every element position, to define the input to
                the filter function. Default is 3.
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(ndimage.minimum_filter(data, size),
                                mask=self.data.mask)

    def minimum_filter(self, size=3, interp='no'):
        """Returns an image containing minimum filter
        applied to the current image.
        Uses `scipy.ndimage.minimum_filter <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.filters.minimum_filter.html>`_.

        Parameters
        ----------
        size   : float
                Shape that is taken from the input array,
                at every element position, to define the input to
                the filter function. Default is 3.
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : Image
        """
        res = self.copy()
        res._minimum_filter(size, interp)
        return res

    def add(self, other):
        """Adds the image other to the current image in place. The coordinate
        are taken into account.

        Parameters
        ----------
        other : Image
                Second image to add.
        """
        try:
            if other.image:
                ima = other.copy()
                self_rot = self.wcs.get_rot()
                ima_rot = ima.wcs.get_rot()
                if self_rot != ima_rot:
                    ima2 = ima.rotate(-self_rot + ima_rot, reshape=True)
                    if ima.wcs.get_cd()[0, 0] * self.wcs.get_cd()[0, 0] < 0:
                        ima = ima.rotate(180 - self_rot + ima_rot,
                                         reshape=True)
                    else:
                        ima = ima2

                self_cdelt = self.wcs.get_step()
                ima_cdelt = ima.wcs.get_step()
                if (self_cdelt != ima_cdelt).all():
                    try:
                        factor = self_cdelt / ima_cdelt
                        if not np.sometrue(np.mod(self_cdelt[0],
                                                  ima_cdelt[0])) \
                            and not np.sometrue(np.mod(self_cdelt[1],
                                                       ima_cdelt[1])):
                            # ima.step is an integer multiple of the self.step
                            ima = ima.rebin_factor(factor)
                        else:
                            raise ValueError('steps are not integer multiple')
                    except:
                        newdim = ima.shape / factor
                        [[k1, l1]] = \
                            self.wcs.sky2pix(ima.wcs.pix2sky([[0, 0]]))
                        l1 = int(l1 + 0.5)
                        k1 = int(k1 + 0.5)
                        newstart = self.wcs.pix2sky([[k1, l1]])[0]
                        ima = ima.rebin(newdim, newstart,
                                        self_cdelt, flux=True)

                # here ima and self have the same step
                [[k1, l1]] = self.wcs.sky2pix(ima.wcs.pix2sky([[0, 0]]))
                l1 = int(l1 + 0.5)
                k1 = int(k1 + 0.5)
                k2 = k1 + ima.shape[0]
                if k1 < 0:
                    nk1 = -k1
                    k1 = 0
                else:
                    nk1 = 0

                if k2 > self.shape[0]:
                    nk2 = ima.shape[0] - (k2 - self.shape[0])
                    k2 = self.shape[0]
                else:
                    nk2 = ima.shape[0]

                l2 = l1 + ima.shape[1]
                if l1 < 0:
                    nl1 = -l1
                    l1 = 0
                else:
                    nl1 = 0

                if l2 > self.shape[1]:
                    nl2 = ima.shape[1] - (l2 - self.shape[1])
                    l2 = self.shape[1]
                else:
                    nl2 = ima.shape[1]

                mask = self.data.mask.__copy__()
                self.data[k1:k2, l1:l2] += (ima.data[nk1:nk2, nl1:nl2]
                                            * ima.fscale / self.fscale)
                self.data.mask = mask
        except ValueError as e:
            raise e
        except:
            raise IOError('Operation forbidden')

    def segment(self, shape=(2, 2), minsize=20, minpts=None,
                background=20, interp='no', median=None):
        """Segments the image in a number of smaller images.

        Returns a list of images.

        Uses `scipy.ndimage.morphology.generate_binary_structure <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.generate_binary_structure.html>`_, `scipy.ndimage.morphology.grey_dilation <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.grey_dilation.html>`_, `scipy.ndimage.measurements.label <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.measurements.label.html>`_, and `scipy.ndimage.measurements.find_objects <http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.measurements.find_objects.html>`_.

        Parameters
        ----------
        shape      : (integer,integer)
                    Shape used for connectivity.
        minsize    : integer
                    Minimmum size of the images.
        minpts     : integer
                    Minimmum number of points in the object.
        background : float
                    Under this value,
                    flux is considered as background.
        interp     : 'no' | 'linear' | 'spline'
                    if 'no', data median value replaced masked values.
                    if 'linear', linear interpolation of the masked values.
                    if 'spline', spline interpolation of the masked values.
        median     : (integer,integer) or None
                    Size of the median filter

        Returns
        -------
        out : List of Image objects.
        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        structure = \
            ndimage.morphology.generate_binary_structure(shape[0], shape[1])
        if median is not None:
            data = np.ma.array(ndimage.median_filter(data, median),
                               mask=self.data.mask)
        expanded = ndimage.morphology.grey_dilation(data, (minsize, minsize))
        ksel = np.where(expanded < background)
        expanded[ksel] = 0

        lab = ndimage.measurements.label(expanded, structure)
        slices = ndimage.measurements.find_objects(lab[0])

        imalist = []
        for i in range(lab[1]):
            if minpts is not None:
                if (data[slices[i]].ravel() > background)\
                        .sum() < minpts:
                    continue
            [[starty, startx]] = \
                self.wcs.pix2sky(self.wcs.pix2sky([[slices[i][0].start,
                                                    slices[i][1].start]]))
            wcs = WCS(crpix=(1.0, 1.0), crval=(starty, startx),
                      cdelt=self.wcs.get_step(), deg=self.wcs.is_deg(),
                      rot=self.wcs.get_rot())
            if self.var is not None:
                res = Image(data=self.data[slices[i]], wcs=wcs,
                            fscale=self.fscale, unit=self.unit, var=self.var[slices[i]])
            else:
                res = Image(data=self.data[slices[i]], wcs=wcs,
                            fscale=self.fscale, unit=self.unit)
            imalist.append(res)
        return imalist

    def add_gaussian_noise(self, sigma, interp='no'):
        """Adds Gaussian noise to image in place.

        Parameters
        ----------
        sigma  : float
                 Standard deviation.
        interp : 'no' | 'linear' | 'spline'
                 if 'no', data median value replaced masked values.
                 if 'linear', linear interpolation of the masked values.
                 if 'spline', spline interpolation of the masked values.
        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(np.random.normal(data, sigma),
                                mask=self.data.mask)
        if self.var is None:
            self.var = np.ones((self.shape)) * sigma * sigma
        else:
            self.var *= (sigma * sigma)

    def add_poisson_noise(self, interp='no'):
        """Adds Poisson noise to image in place.

        Parameters
        ----------
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        """
        if interp == 'linear':
            data = self._interp_data(spline=False)
        elif interp == 'spline':
            data = self._interp_data(spline=True)
        else:
            data = np.ma.filled(self.data, np.ma.median(self.data))

        self.data = np.ma.array(np.random.poisson(data * self.fscale)
                                .astype(float), mask=self.data.mask)
        self.data /= self.fscale
        if self.var is None:
            self.var = self.data.data.__copy__()
        else:
            self.var += self.data.data

    def inside(self, coord):
        """Returns True if coord is inside image.

        Parameters
        ----------
        coord : (float,float)
                coordinates (y,x) in degrees.

        Returns
        -------
        out : boolean
        """
        pixcrd = self.wcs.sky2pix([coord[0], coord[1]])
        if pixcrd[0][0] >= 0 and pixcrd[0][0] < self.shape[0] \
                and pixcrd[0][1] >= 0 and pixcrd[0][1] < self.shape[1]:
            return True
        else:
            return False

    def _fftconvolve(self, other, interp='no'):
        """Convolves image with other using fft.
        Uses `scipy.signal.fftconvolve <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html>`_.

        Parameters
        ----------
        other  : 2d-array or Image
                Second Image or 2d-array.
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        """
        if self.data is None:
            raise ValueError('empty data array')

        if type(other) is np.array:
            if interp == 'linear':
                data = self._interp_data(spline=False)
            elif interp == 'spline':
                data = self._interp_data(spline=True)
            else:
                data = np.ma.filled(self.data, np.ma.median(self.data))

            if self.shape[0] != other.shape[0] \
                    or self.shape[1] != other.shape[1]:
                raise IOError('Operation forbidden for images '
                              'with different sizes')
            else:
                self.data = np.ma.array(signal.fftconvolve(data, other,
                                                           mode='same'),
                                        mask=self.data.mask)
                if self.var is not None:
                    self.var = signal.fftconvolve(self.var, other,
                                                  mode='same')
        try:
            if other.image:
                if interp == 'linear':
                    data = self._interp_data(spline=False)
                    other_data = other._interp_data(spline=False)
                elif interp == 'spline':
                    data = self._interp_data(spline=True)
                    other_data = other._interp_data(spline=True)
                else:
                    data = np.ma.filled(self.data, np.ma.median(self.data))
                    other_data = other.data.filled(np.ma.median(other.data))

                if other.data is None or self.shape[0] != other.shape[0] \
                        or self.shape[1] != other.shape[1]:
                    raise IOError('Operation forbidden for images '
                                  'with different sizes')
                else:
                    self.data = np.ma.array(signal.fftconvolve(data,
                                                               other_data * other.fscale, mode='same'),
                                            mask=self.data.mask)
                    if self.var is not None:
                        self.var = signal.fftconvolve(self.var,
                                                      other_data * other.fscale, mode='same')
        except IOError as e:
            raise e
        except:
            raise IOError('Operation forbidden')

    def fftconvolve(self, other, interp='no'):
        """Returns the convolution of the image with other using fft.
        Uses `scipy.signal.fftconvolve <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html>`_.

        Parameters
        ----------
        other  : 2d-array or Image
                Second Image or 2d-array.
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.

        Returns
        -------
        out : Image
        """
        res = self.copy()
        res._fftconvolve(other, interp)
        return res

    def fftconvolve_gauss(self, center=None, flux=1., fwhm=(1., 1.),
                          peak=False, rot=0., factor=1, pix=False):
        """Returns the convolution of the image with a 2D gaussian.

        Parameters
        ----------
        center : (float,float)
                Gaussian center (y_peak, x_peak) in degrees (pix=False)
                or in pixels (pix=True).
                If None the center of the image is used.
        flux   : float
                Integrated gaussian flux or gaussian peak
                value if peak is True.
        fwhm   : (float,float)
                Gaussian fwhm (fwhm_y,fwhm_x) in arcseconds (pix=False)
                or in pixels (pix=True).
        peak   : boolean
                If true, flux contains a gaussian peak value.
        rot    : float
                Angle position in degree.
        factor : integer
                If factor<=1, gaussian value is computed
                in the center of each pixel.
                If factor>1, for each pixel, gaussian value
                is the sum of the gaussian values on the
                factor*factor pixels divided by the pixel area.
        pix    : boolean
                If pix is False, center and fwhm are
                in degrees and arcsecs.

        Returns
        -------
        out : Image
        """
        ima = gauss_image(self.shape, wcs=self.wcs, center=center,
                          flux=flux, fwhm=fwhm, peak=peak, rot=rot,
                          factor=factor, gauss=None, pix=pix, cont=0)
        ima.norm(type='sum')
        return self.fftconvolve(ima)

    def fftconvolve_moffat(self, center=None, I=1., a=1.0, q=1.0,
                           n=2, rot=0., factor=1, pix=False):
        """Returns the convolution of the image with a 2D moffat.

        Parameters
        ----------
        center : (float,float)
                Gaussian center (y_peak, x_peak)
                in degrees (pix=False) or pixels (pix=True).
                If None the center of the image is used.
        I      : float
                Intensity at image center. 1 by default.
        a      : float
                Half width at half maximum of the image
                in the absence of atmospheric scattering in arcseconds
                (pix=False) or in pixels (pix=True). 1 by default.
        q      : float
                Axis ratio, 1 by default.
        n      : integer
                Atmospheric scattering coefficient. 2 by default.
        rot    : float
                Angle position in degree.
        factor : integer
                If factor<=1, moffat value is computed
                in the center of each pixel.
                If factor>1, for each pixel, moffat value is the sum
                of the moffat values on the factor*factor pixels
                divided by the pixel area.
        pix    : boolean
                If pix is False, center and fwhm are
                in degrees and arcsecs.
                If pix is True, center and fwhm are in pixels.

        Returns
        -------
        out : Image
        """
        fwhmy = a * (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
        fwhmx = fwhmy / q

        ima = moffat_image(self.shape, wcs=self.wcs, factor=factor,
                           center=center, flux=I, fwhm=(fwhmy, fwhmx), n=n,
                           rot=rot, pix=pix)

        ima.norm(type='sum')
        return self.fftconvolve(ima)

    def correlate2d(self, other, interp='no'):
        """Returns the cross-correlation of the image with an array/image
        Uses `scipy.signal.correlate2d <http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html>`_.

        Parameters
        ----------
        other : 2d-array or Image
                Second Image or 2d-array.
        interp : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.
        """
        if self.data is None:
            raise ValueError('empty data array')

        if type(other) is np.array:
            if interp == 'linear':
                data = self._interp_data(spline=False)
            elif interp == 'spline':
                data = self._interp_data(spline=True)
            else:
                data = np.ma.filled(self.data, np.ma.median(self.data))

            res = self.copy()
            res.data = np.ma.array(signal.correlate2d(data, other, mode='same'),
                                   mask=res.data.mask)
            if res.var is not None:
                res.var = signal.correlate2d(res.var, other, mode='same')
            return res
        try:
            if other.image:
                if interp == 'linear':
                    data = self._interp_data(spline=False)
                    other_data = other._interp_data(spline=False)
                elif interp == 'spline':
                    data = self._interp_data(spline=True)
                    other_data = other._interp_data(spline=True)
                else:
                    data = np.ma.filled(self.data, np.ma.median(self.data))
                    other_data = other.data.filled(np.ma.median(other.data))

                res = self.copy()
                res.data = np.ma.array(signal.correlate2d(data,
                                                          other_data * other.fscale, mode='same'),
                                       mask=res.data.mask)
                res.fscale = self.fscale
                if res.var is not None:
                    res.var = signal.correlate2d(res.var, other_data * other.fscale,
                                                 mode='same')
                return res
        except:
            raise IOError('Operation forbidden')

    def plot(self, title=None, scale='linear', vmin=None, vmax=None,
             zscale=False, colorbar=None, var=False, **kargs):
        """Plots the image.

        Parameters
        ----------
        title    : string
                Figure title (None by default).
        scale    : 'linear' | 'log' | 'sqrt' | 'arcsinh' | 'power'
                The stretch function to use for the scaling
                (default is 'linear').
        vmin     : float
                Minimum pixel value to use for the scaling.
                If None, vmin is set to min of data.
        vmax     : float
                Maximum pixel value to use for the scaling.
                If None, vmax is set to max of data.
        zscale   : boolean
                If true, vmin and vmax are computed
                using the IRAF zscale algorithm.
        colorbar : boolean
                If 'h'/'v', a horizontal/vertical colorbar is added.
        var      : boolean
                If var is True, the inverse of variance
                is overplotted.
        kargs    : matplotlib.artist.Artist
                kargs can be used to set additional Artist properties.

        Returns
        -------
        out : matplotlib AxesImage
        """

        f = self.data * self.fscale
        xaxis = np.arange(self.shape[1], dtype=np.float)
        yaxis = np.arange(self.shape[0], dtype=np.float)
        xunit = 'pixel'
        yunit = 'pixel'

        if np.shape(xaxis)[0] == 1:
            # plot a  column
            plt.plot(yaxis, f)
            plt.xlabel('p (%s)' % yunit)
            plt.ylabel(self.unit)
        elif np.shape(yaxis)[0] == 1:
            # plot a line
            plt.plot(xaxis, f)
            plt.xlabel('q (%s)' % xunit)
            plt.ylabel(self.unit)
        else:
            if zscale:
                vmin, vmax = plt_zscale.zscale(self.data.filled(0))
            if scale == 'log':
                from matplotlib.colors import LogNorm
                norm = LogNorm(vmin=vmin, vmax=vmax)
            elif scale == 'arcsinh':
                norm = plt_norm.ArcsinhNorm(vmin=vmin, vmax=vmax)
            elif scale == 'power':
                norm = plt_norm.PowerNorm(vmin=vmin, vmax=vmax)
            elif scale == 'sqrt':
                norm = plt_norm.SqrtNorm(vmin=vmin, vmax=vmax)
            else:
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

            if self.var is not None and var:
                wght = 1.0 / (self.var * self.fscale * self.fscale)
                np.ma.fix_invalid(wght, copy=False, fill_value=0)

                normalpha = matplotlib.colors.Normalize(wght.min(), wght.max())

                img_array = plt.get_cmap('jet')(norm(f))
                img_array[:, :, 3] = 1 - normalpha(wght)/2
                cax = plt.imshow(img_array, interpolation='nearest', origin='lower',
                                 norm=norm, **kargs)
            else:
                cax = plt.imshow(f, interpolation='nearest', origin='lower',
                                 norm=norm, **kargs)

            plt.xlabel('q (%s)' % xunit)
            plt.ylabel('p (%s)' % yunit)

            # create colorbar
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            if colorbar == "h":
                cax2 = divider.append_axes("top", size="5%", pad=0.2)
                cbar = plt.colorbar(cax, cax=cax2, orientation='horizontal')
                for t in cbar.ax.xaxis.get_major_ticks():
                    t.tick1On = True
                    t.tick2On = True
                    t.label1On = False
                    t.label2On = True
            elif colorbar == "v":
                cax2 = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cax, cax=cax2)
            else:
                pass

            self._ax = cax

        if title is not None:
            plt.title(title)

        self._fig = plt.get_current_fig_manager()
        plt.connect('motion_notify_event', self._on_move)
        return cax

    def _on_move(self, event):
        """prints y,x,p,q and data in the figure toolbar."""
        if event.inaxes is not None:
            j, i = event.xdata, event.ydata
            try:
                pixsky = self.wcs.pix2sky([i, j])
                yc = pixsky[0][0]
                xc = pixsky[0][1]
                val = self.data.data[i, j] * self.fscale
                s = 'y= %g x=%g p=%i q=%i data=%g' % (yc, xc, i, j, val)
                self._fig.toolbar.set_message(s)
            except:
                pass

    def ipos(self, filename='None'):
        """Prints cursor position in interactive mode (p and q define the
        nearest pixel, x and y are the position, data contains the image data
        value (data[p,q]) ).

          To read cursor position, click on the left mouse button.

          To remove a cursor position, click on the left mouse button + <d>

          To quit the interactive mode, click on the right mouse button.

          At the end, clicks are saved in self.clicks as dictionary
          {'y','x','p','q','data'}.

        Parameters
        ----------
        filename : string
                If filename is not None, the cursor values are
                saved as a fits table with columns labeled
                'I'|'J'|'RA'|'DEC'|'DATA'.
        """
        d = {'class': 'Image', 'method': 'ipos'}
        msg = 'To read cursor position, click on the left mouse button'
        self.logger.info(msg, extra=d)
        msg = 'To remove a cursor position,'\
            ' click on the left mouse button + <d>'
        self.logger.info(msg, extra=d)
        msg = 'To quit the interactive mode, click on the right mouse button.'
        self.logger.info(msg, extra=d)
        msg = 'After quit, clicks are saved '\
            'in self.clicks as dictionary {y,x,p,q,data}.'
        self.logger.info(msg, extra=d)

        if self._clicks is None:
            binding_id = plt.connect('button_press_event', self._on_click)
            self._clicks = ImageClicks(binding_id, filename)

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")
        else:
            self._clicks.filename = filename

    def _on_click(self, event):
        """prints dec,ra,i,j and data corresponding to the cursor position."""
        d = {'class': 'Image', 'method': '_on_click'}
        if event.key == 'd':
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        j, i = event.xdata, event.ydata
                        self._clicks.remove(i, j)
                        msg = "new selection:"
                        self.logger.info(msg, extra=d)
                        for i in range(len(self._clicks.x)):
                            self._clicks.iprint(i, self.fscale)
                    except:
                        pass
        else:
            if event.button == 1:
                if event.inaxes is not None:
                    j, i = event.xdata, event.ydata
                    try:
                        i = int(i)
                        j = int(j)
                        [[y, x]] = self.wcs.pix2sky([i, j])
                        val = self.data[i, j] * self.fscale
                        if len(self._clicks.x) == 0:
                            print ''
                        self._clicks.add(i, j, x, y, val)
                        self._clicks.iprint(len(self._clicks.x) - 1, self.fscale)
                    except:
                        pass
            else:
                self._clicks.write_fits()
                # save clicks in a dictionary {'i','j','x','y','data'}
                d = {'p': self._clicks.p, 'q': self._clicks.q,
                     'x': self._clicks.x, 'y': self._clicks.y,
                     'data': self._clicks.data}
                self.clicks = d
                # clear
                self._clicks.clear()
                self._clicks = None
                fig = plt.gcf()
                fig.canvas.stop_event_loop_default()

    def idist(self):
        """Gets distance and center from 2 cursor positions (interactive mode).

        To quit the interactive mode, click on the right mouse button.
        """
        d = {'class': 'Image', 'method': 'idist'}
        msg = 'Use left mouse button to define the line.'
        self.logger.info(msg, extra=d)
        msg = 'To quit the interactive mode, click on the right mouse button.'
        self.logger.info(msg, extra=d)
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_dist,
                                               drawtype='line')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_dist(self, eclick, erelease):
        """Prints distance and center between 2 cursor positions."""
        d = {'class': 'Image', 'method': '_on_select_dist'}
        if eclick.button == 1:
            try:
                j1, i1 = int(eclick.xdata), int(eclick.ydata)
                [[y1, x1]] = self.wcs.pix2sky([i1, j1])
                j2, i2 = int(erelease.xdata), int(erelease.ydata)
                [[y2, x2]] = self.wcs.pix2sky([i2, j2])
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                msg = 'Center: (%g,%g)\tDistance: %g' % (xc, yc, dist)
                self.logger.info(msg, extra=d)
            except:
                pass
        else:
            msg = 'idist deactivated.'
            self.logger.info(msg, extra=d)
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def istat(self):
        """Computes image statistics from windows defined with left mouse
        button (mean is the mean value, median the median value, std is the rms
        standard deviation, sum the sum, peak the peak value, npts is the total
        number of points).

        To quit the interactive mode, click on the right mouse button.
        """
        d = {'class': 'Image', 'method': 'istat'}
        msg = 'Use left mouse button to define the box.'
        self.logger.info(msg, extra=d)
        msg = 'To quit the interactive mode, click on the right mouse button.'
        self.logger.info(msg, extra=d)
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_stat,
                                               drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_stat(self, eclick, erelease):
        """Prints image statistics from windows defined by 2 cursor
        positions."""
        d = {'class': 'Image', 'method': '_on_select_stat'}
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata, erelease.xdata))
                j2 = int(max(eclick.xdata, erelease.xdata))
                i1 = int(min(eclick.ydata, erelease.ydata))
                i2 = int(max(eclick.ydata, erelease.ydata))
                d = self.data[i1:i2, j1:j2]
                mean = self.fscale * np.ma.mean(d)
                median = self.fscale * np.ma.median(np.ma.ravel(d))
                vsum = self.fscale * d.sum()
                std = self.fscale * np.ma.std(d)
                npts = d.shape[0] * d.shape[1]
                peak = self.fscale * d.max()
                msg = 'mean=%g\tmedian=%g\tstd=%g\tsum=%g\tpeak=%g\tnpts=%d' \
                    % (mean, median, std, vsum, peak, npts)
                self.logger.info(msg, extra=d)
            except:
                pass
        else:
            msg = 'istat deactivated.'
            self.logger.info(msg, extra=d)
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def ipeak(self):
        """Finds peak location in windows defined with left mouse button.

        To quit the interactive mode, click on the right mouse button.
        """
        d = {'class': 'Image', 'method': 'ipeak'}
        msg = 'Use left mouse button to define the box.'
        self.logger.info(msg, extra=d)
        msg = 'To quit the interactive mode, click on the right mouse button.'
        self.logger.info(msg, extra=d)
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_peak,
                                               drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_peak(self, eclick, erelease):
        """Prints image peak location in windows defined by 2 cursor
        positions."""
        d = {'class': 'Image', 'method': '_on_select_peak'}
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata, erelease.xdata))
                j2 = int(max(eclick.xdata, erelease.xdata))
                i1 = int(min(eclick.ydata, erelease.ydata))
                i2 = int(max(eclick.ydata, erelease.ydata))
                center = ((i2 + i1) / 2, (j2 + j1) / 2)
                radius = (np.abs(i2 - i1) / 2, np.abs(j2 - j1) / 2)
                peak = self.peak(center, radius, True)
                msg = 'peak: y=%g\tx=%g\tp=%d\tq=%d\tdata=%g' \
                    % (peak['y'], peak['x'], peak['p'], peak['q'], peak['data'])
                self.logger.info(msg, extra=d)
            except:
                pass
        else:
            msg = 'ipeak deactivated.'
            self.logger.info(msg, extra=d)
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def ifwhm(self):
        """Computes fwhm in windows defined with left mouse button.

        To quit the interactive mode, click on the right mouse button.
        """
        d = {'class': 'Image', 'method': 'ifwhm'}
        msg = 'Use left mouse button to define the box.'
        self.logger.info(msg, extra=d)
        msg = 'To quit the interactive mode, click on the right mouse button.'
        self.logger.info(msg, extra=d)
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_fwhm,
                                               drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_fwhm(self, eclick, erelease):
        """Prints image peak location in windows defined'\ 'by 2 cursor
        positions."""
        d = {'class': 'Image', 'method': '_on_select_fwhm'}
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata, erelease.xdata))
                j2 = int(max(eclick.xdata, erelease.xdata))
                i1 = int(min(eclick.ydata, erelease.ydata))
                i2 = int(max(eclick.ydata, erelease.ydata))
                center = ((i2 + i1) / 2, (j2 + j1) / 2)
                radius = (np.abs(i2 - i1) / 2, np.abs(j2 - j1) / 2)
                fwhm = self.fwhm(center, radius, True)
                msg = 'fwhm_y=%g\tfwhm_x=%g' % (fwhm[0], fwhm[1])
                self.logger.info(msg, extra=d)
            except:
                pass
        else:
            msg = 'ifwhm deactivated.'
            self.logger.info(msg, extra=d)
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def iee(self):
        """Computes enclosed energy in windows defined with left mouse button.

        To quit the interactive mode, click on the right mouse button.
        """
        d = {'class': 'Image', 'method': 'iee'}
        msg = 'Use left mouse button to define the box.'
        self.logger.info(msg, extra=d)
        msg = 'To quit the interactive mode, click on the right mouse button.'
        self.logger.info(msg, extra=d)
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = RectangleSelector(ax, self._on_select_ee,
                                               drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_ee(self, eclick, erelease):
        """Prints image peak location in windows defined by 2 cursor
        positions."""
        d = {'class': 'Image', 'method': '_on_select_ee'}
        if eclick.button == 1:
            try:
                j1 = int(min(eclick.xdata, erelease.xdata))
                j2 = int(max(eclick.xdata, erelease.xdata))
                i1 = int(min(eclick.ydata, erelease.ydata))
                i2 = int(max(eclick.ydata, erelease.ydata))
                center = ((i2 + i1) / 2, (j2 + j1) / 2)
                radius = (np.abs(i2 - i1) / 2, np.abs(j2 - j1) / 2)
                ee = self.ee(center, radius, True)
                msg = 'ee=%g' % ee
                self.logger.info(msg, extra=d)
            except:
                pass
        else:
            msg = 'iee deactivated.'
            self.logger.info(msg, extra=d)
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def imask(self):
        """Over-plots masked values (interactive mode).
        """
        try:
            try:
                self._plot_mask_id.remove()
                # plt.draw()
            except:
                pass
            xaxis = np.arange(self.shape[1], dtype=np.float)
            yaxis = np.arange(self.shape[0], dtype=np.float)

            if np.shape(xaxis)[0] == 1:
                # plot a  column
                plt.plot(yaxis, self.data.data, alpha=0.3)
            elif np.shape(yaxis)[0] == 1:
                # plot a line
                plt.plot(xaxis, self.data.data, alpha=0.3)
            else:
                mask = np.array(1 - self.data.mask, dtype=bool)
                data = np.ma.MaskedArray(self.data.data * self.fscale,
                                         mask=mask)
                self._plot_mask_id = \
                    plt.imshow(data, interpolation='nearest', origin='lower',
                               extent=(0, self.shape[1] - 1, 0,
                                       self.shape[0] - 1),
                               vmin=self.data.min(), vmax=self.data.max(),
                               alpha=0.9)
        except:
            pass

    def igauss_fit(self):
        """Performs Gaussian fit in windows defined with left mouse button.

        To quit the interactive mode, click on the right mouse button.
        """
        d = {'class': 'Image', 'method': 'igauss_fit'}
        msg = 'Use left mouse button to define the box.'
        self.logger.info(msg, extra=d)
        msg ='To quit the interactive mode, click on the right mouse button.'
        self.logger.info(msg, extra=d)
        msg = 'The parameters of the last gaussian are saved in self.gauss.'
        self.logger.info(msg, extra=d)
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = \
                RectangleSelector(ax, self._on_select_gauss_fit, drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_gauss_fit(self, eclick, erelease):
        d = {'class': 'Image', 'method': '_on_select_gauss_fit'}
        if eclick.button == 1:
            try:
                q1 = int(min(eclick.xdata, erelease.xdata))
                q2 = int(max(eclick.xdata, erelease.xdata))
                p1 = int(min(eclick.ydata, erelease.ydata))
                p2 = int(max(eclick.ydata, erelease.ydata))
                pos_min = self.wcs.pix2sky([p1, q1])[0]
                pos_max = self.wcs.pix2sky([p2, q2])[0]
                self.gauss = self.gauss_fit(pos_min, pos_max, plot=True)
                self.gauss.print_param()
            except:
                pass
        else:
            msg = 'igauss_fit deactivated.'
            self.logger.info(msg, extra=d)
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def imoffat_fit(self):
        """Performs Moffat fit in windows defined with left mouse button.

        To quit the interactive mode, click on the right mouse button.
        """
        d = {'class': 'Image', 'method': 'imoffat_fit'}
        msg = 'Use left mouse button to define the box.'
        self.logger.info(msg, extra=d)
        msg = 'To quit the interactive mode, click on the right mouse button.'
        self.logger.info(msg, extra=d)
        msg = 'The parameters of the last moffat fit are '\
            'saved in self.moffat.'
        self.logger.info(msg, extra=d)
        if self._clicks is None and self._selector is None:
            ax = plt.subplot(111)
            self._selector = \
                RectangleSelector(ax, self._on_select_moffat_fit, drawtype='box')

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_select_moffat_fit(self, eclick, erelease):
        d = {'class': 'Spectrum', 'method': 'info'}
        if eclick.button == 1:
            try:
                q1 = int(min(eclick.xdata, erelease.xdata))
                q2 = int(max(eclick.xdata, erelease.xdata))
                p1 = int(min(eclick.ydata, erelease.ydata))
                p2 = int(max(eclick.ydata, erelease.ydata))
                pos_min = self.wcs.pix2sky([p1, q1])[0]
                pos_max = self.wcs.pix2sky([p2, q2])[0]
                self.moffat = self.moffat_fit(pos_min, pos_max, plot=True)
                self.moffat.print_param()
            except:
                pass
        else:
            msg = 'imoffat_fit deactivated.'
            self.logger.info(msg, extra=d)
            self._selector.set_active(False)
            self._selector = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()


def gauss_image(shape=(101, 101), wcs=WCS(), factor=1, gauss=None,
                center=None, flux=1., fwhm=(1., 1.), peak=False,
                rot=0., cont=0, pix=False):
    """Creates a new image from a 2D gaussian.

    Parameters
    ----------
    shape  :  integer or (integer,integer)
            Lengths of the image in Y and X
            with python notation: (ny,nx). (101,101) by default.
            If wcs object contains dimensions,
            shape is ignored and wcs dimensions are used.
    wcs    : :class:`mpdaf.obj.WCS`
            World coordinates.
    factor : integer
            If factor<=1, gaussian value is computed
            in the center of each pixel.
            If factor>1, for each pixel, gaussian value is the sum
            of the gaussian values on the factor*factor pixels divided
            by the pixel area.
    gauss  : :class:`mpdaf.obj.Gauss2D`
            object that contains all Gaussian parameters.
            If it is present, following parameters are not used.
    center : (float,float)
            Gaussian center (y_peak, x_peak)
            in degrees (pix=False) or pixel (pix=True).
            If None the center of the image is used.
    flux   : float
            Integrated gaussian flux or gaussian peak value if peak is True.
    fwhm   : (float,float)
            Gaussian fwhm (fwhm_y,fwhm_x) in arcsecondes (pix=False)
            or pixel (pix=True).
    peak   : boolean
            If true, flux contains a gaussian peak value.
    rot    : float
            Angle position in degree.
    cont   : float
            Continuum value. 0 by default.
    pix    : boolean
            If pix is False, center and fwhm are in degrees and arcsecs.
            If pix is True, center and fwhm are in pixels.

    Returns
    -------
    out : obj.Image object (`Image class`_)
    """
    if is_int(shape):
        shape = (shape, shape)
    shape = np.array(shape)

    if wcs.naxis1 == 1. and wcs.naxis2 == 1.:
        wcs.naxis1 = shape[1]
        wcs.naxis2 = shape[0]
    else:
        if wcs.naxis1 != 0. or wcs.naxis2 != 0.:
            shape[1] = wcs.naxis1
            shape[0] = wcs.naxis2

    if gauss is not None:
        center = gauss.center
        flux = gauss.flux
        fwhm = gauss.fwhm
        peak = False
        rot = gauss.rot
        cont = gauss.cont

    if center is None:
        center = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])
    else:
        if pix is False:
            center = wcs.sky2pix(center)[0]

    if pix is False:
        fwhm = np.array(fwhm) / wcs.get_step()
        if wcs.is_deg():
            fwhm /= 3600.

    data = np.empty(shape=shape, dtype=float)

    if fwhm[1] == 0 or fwhm[0] == 0:
        raise ValueError('fwhm equal to 0')
    p_width = fwhm[0] / 2.0 / np.sqrt(2 * np.log(2))
    q_width = fwhm[1] / 2.0 / np.sqrt(2 * np.log(2))

    # rotation angle in rad
    theta = np.pi * rot / 180.0

    if peak is True:
        I = flux * np.sqrt(2 * np.pi * (p_width ** 2)) \
            * np.sqrt(2 * np.pi * (q_width ** 2))
    else:
        I = flux

    gauss = lambda p, q: I * (1 / np.sqrt(2 * np.pi * (p_width ** 2))) \
        * np.exp(-((p - center[0]) * np.cos(theta)
                   - (q - center[1]) * np.sin(theta)) ** 2
                 / (2 * p_width ** 2)) \
        * (1 / np.sqrt(2 * np.pi * (q_width ** 2))) \
        * np.exp(-((p - center[0]) * np.sin(theta)
                   + (q - center[1]) * np.cos(theta)) ** 2
                 / (2 * q_width ** 2))

    if factor > 1:
        if rot == 0:
            X, Y = np.meshgrid(xrange(shape[0]), xrange(shape[1]))
            pixcrd_min = np.array(zip(X.ravel(), Y.ravel())) - 0.5
            # pixsky_min = wcs.pix2sky(pixcrd)
            xmin = (pixcrd_min[:, 1] - center[1]) / np.sqrt(2.0) / q_width
            ymin = (pixcrd_min[:, 0] - center[0]) / np.sqrt(2.0) / p_width

            pixcrd_max = np.array(zip(X.ravel(), Y.ravel())) + 0.5
            # pixsky_max = wcs.pix2sky(pixcrd)
            xmax = (pixcrd_max[:, 1] - center[1]) / np.sqrt(2.0) / q_width
            ymax = (pixcrd_max[:, 0] - center[0]) / np.sqrt(2.0) / p_width

            dx = pixcrd_max[:, 1] - pixcrd_min[:, 1]
            dy = pixcrd_max[:, 0] - pixcrd_min[:, 0]
            data = I * 0.25 / dx / dy \
                * (special.erf(xmax) - special.erf(xmin)) \
                * (special.erf(ymax) - special.erf(ymin))
            data = np.reshape(data, (shape[1], shape[0])).T
        else:
            X, Y = np.meshgrid(xrange(shape[0] * factor),
                               xrange(shape[1] * factor))
            factor = float(factor)
            pixcrd = np.array(zip(X.ravel() / factor, Y.ravel() / factor))
            # pixsky = wcs.pix2sky(pixcrd)
            data = gauss(pixcrd[:, 0], pixcrd[:, 1])
            data = (data.reshape(shape[1], factor, shape[0], factor)
                    .sum(1).sum(2) / factor / factor).T
    else:
        X, Y = np.meshgrid(xrange(shape[0]), xrange(shape[1]))
        pixcrd = np.array(zip(X.ravel(), Y.ravel()))
        # data = gauss(pixcrd[:,1],pixcrd[:,0])
        data = gauss(pixcrd[:, 0], pixcrd[:, 1])
        data = np.reshape(data, (shape[1], shape[0])).T

    return Image(data=data + cont, wcs=wcs)


def moffat_image(shape=(101, 101), wcs=WCS(), factor=1, moffat=None,
                 center=None, flux=1., fwhm=(1., 1.), peak=False, n=2,
                 rot=0., cont=0, pix=False):
    """Creates a new image from a 2D Moffat function.

    Parameters
    ----------
    shape  : integer or (integer,integer)
            Lengths of the image in Y and X
            with python notation: (ny,nx). (101,101) by default.
            If wcs object contains dimensions,
            shape is ignored and wcs dimensions are used.
    wcs    : :class:`mpdaf.obj.WCS`
            World coordinates.
    factor : integer
            If factor<=1, moffat value is computed
            in the center of each pixel.
            If factor>1, for each pixel, moffat value is the sum
            of the moffat values on the factor*factor pixels divided
            by the pixel area.
    moffat : :class:`mpdaf.obj.Moffat2D`
            object that contains all moffat parameters.
            If it is present, following parameters are not used.
    center : (float,float)
            Peak center (x_peak, y_peak) in degrees (pix=False)
            or pixel (pix=True).
            If None the center of the image is used.
    flux   : float
            Integrated gaussian flux or gaussian peak value
            if peak is True.
    fwhm   : (float,float)
            Gaussian fwhm (fwhm_y,fwhm_x) in arcsecondes (pix=False)
            or pixel (pix=True).
    peak   : boolean
            If true, flux contains a gaussian peak value.
    n      : integer
            Atmospheric scattering coefficient. 2 by default.
    rot    : float
            Angle position in degree.
    cont   : float
            Continuum value. 0 by default.
    pix    : boolean
            If pix is False, center and fwhm are in degrees and arcsecs.
            If pix is True, center and fwhm are in pixels.

    Returns
    -------
    out : obj.Image object (`Image class`_)
    """
    n = float(n)
    if is_int(shape):
        shape = (shape, shape)
    shape = np.array(shape)

    if wcs.naxis1 == 1. and wcs.naxis2 == 1.:
        wcs.naxis1 = shape[1]
        wcs.naxis2 = shape[0]
    else:
        if wcs.naxis1 != 0. or wcs.naxis2 != 0.:
            shape[1] = wcs.naxis1
            shape[0] = wcs.naxis2

    if moffat is not None:
        center = moffat.center
        flux = moffat.flux
        fwhm = moffat.fwhm
        peak = False
        n = moffat.n
        rot = moffat.rot
        cont = moffat.cont

    fwhm = np.array(fwhm)
    a = fwhm[0] / (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
    e = fwhm[0] / fwhm[1]

    if peak:
        I = flux
    else:
        I = flux * (n - 1) / (np.pi * a * a * e)

    if center is None:
        center = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])
    else:
        if pix is False:
            center = wcs.sky2pix(center)[0]

    if pix is False:
        a = a / wcs.get_step()[0]
        if wcs.is_deg():
            a /= 3600.

    data = np.empty(shape=shape, dtype=float)

    # rotation angle in rad
    theta = np.pi * rot / 180.0

    moffat = lambda p, q: \
        I * (1 + (((p - center[0]) * np.cos(theta)
                   - (q - center[1]) * np.sin(theta)) / a) ** 2
             + (((p - center[0]) * np.sin(theta)
                 + (q - center[1]) * np.cos(theta)) / a / e) ** 2) ** (-n)

    if factor > 1:
        X, Y = np.meshgrid(xrange(shape[0] * factor),
                           xrange(shape[1] * factor))
        factor = float(factor)
        pixcrd = np.array(zip(X.ravel() / factor, Y.ravel() / factor))
        data = moffat(pixcrd[:, 0], pixcrd[:, 1])
        data = (data.reshape(shape[1], factor, shape[0], factor)
                .sum(1).sum(2) / factor / factor).T
    else:
        X, Y = np.meshgrid(xrange(shape[0]), xrange(shape[1]))
        pixcrd = np.array(zip(X.ravel(), Y.ravel()))
        data = moffat(pixcrd[:, 0], pixcrd[:, 1])
        data = np.reshape(data, (shape[1], shape[0])).T

    return Image(data=data + cont, wcs=wcs)


def make_image(x, y, z, steps, deg=True, limits=None,
               spline=False, order=3, smooth=0):
    """Interpolates z(x,y) and returns an image.

    Parameters
    ----------
    x      : float array
            Coordinate array corresponding to the declinaison.
    y      : float array
            Coordinate array corresponding to the right ascension.
    z      : float array
            Input data.
    steps  : (float,float)
            Steps of the output image (dy,dRx).
    deg    : boolean
            If True, world coordinates are in decimal degrees
            (CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CUNIT1=CUNIT2='deg').
            If False (by default), world coordinates are linear
            (CTYPE1=CTYPE2='LINEAR').
    limits : (float,float,float,float)
            Limits of the image (y_min,x_min,y_max,x_max).
            If None, minum and maximum values of x,y arrays are used.
    spline : boolean
            False: bilinear interpolation
            (it uses `scipy.interpolate.griddata <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`).
            True: spline interpolation
            (it uses `scipy.interpolate.bisplrep <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplrep.html>`_ and `scipy.interpolate.bisplev <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplev.html>`).
    order  : integer
            Polynomial order for spline interpolation (default 3)
    smooth : float
            Smoothing parameter for spline interpolation (default 0: no smoothing)

    Results
    -------
    out : obj.Image object (`Image class`_)
    """
    if limits == None:
        x1 = x.min()
        x2 = x.max()
        y1 = y.min()
        y2 = y.max()
    else:
        x1, x2, y1, y2 = limits
    dx, dy = steps
    nx = int((x2 - x1) / dx + 1.5)
    ny = int((y2 - y1) / dy + 1.5)

    wcs = WCS(crpix=(1, 1), crval=(x1, y1),
              cdelt=(dx, dy), deg=deg, shape=(nx, ny))

    xi = np.arange(nx) * dx + x1
    yi = np.arange(ny) * dy + y1

    Y, X = np.meshgrid(y, x)

    if spline:
        tck = interpolate.bisplrep(X, Y, z, s=smooth, kx=order, ky=order)
        data = interpolate.bisplev(xi, yi, tck)
    else:
        n = np.shape(x)[0] * np.shape(y)[0]
        points = np.empty((n, 2), dtype=float)
        points[:, 0] = X.ravel()[:]
        points[:, 1] = Y.ravel()[:]
        Yi, Xi = np.meshgrid(yi, xi)
        data = interpolate.griddata(points, z.ravel(),
                                    (Xi, Yi), method='linear')

    return Image(data=data, wcs=wcs)


def composite_image(ImaColList, mode='lin', cuts=(10, 90),
                    bar=False, interp='no'):
    """Builds composite image from a list of image and colors.

    Parameters
    ----------
    ImaColList : list of tuple (Image,float,float)
                List of images and colors [(Image, hue, saturation)].
    mode       : 'lin' or 'sqrt'
                Intensity mode. Use 'lin' for linear and 'sqrt'
                for root square.
    cuts       : (float,float)
                Minimum and maximum in percent.
    bar        : boolean
                If bar is True a color bar image is created.
    interp     : 'no' | 'linear' | 'spline'
                if 'no', data median value replaced masked values.
                if 'linear', linear interpolation of the masked values.
                if 'spline', spline interpolation of the masked values.

    Returns
    -------
    out : Returns a PIL RGB image (or 2 PIL images if bar is True).
    """
    from PIL import Image as PILima
    from PIL import ImageColor
    from PIL import ImageChops

    # compute statistic of intensity and derive cuts
    first = True
    for ImaCol in ImaColList:
        ima, col, sat = ImaCol

        if interp == 'linear':
            data = ima._interp_data(spline=False)
        elif interp == 'spline':
            data = ima._interp_data(spline=True)
        else:
            data = np.ma.filled(ima.data, np.ma.median(ima.data))

        if mode == 'lin':
            f = data
        elif mode == 'sqrt':
            f = np.sqrt(np.clip(data, 0, 1.e99))
        else:
            raise ValueError('Wrong cut mode')
        if first:
            d = f.ravel()
            first = False
        else:
            d = np.concatenate([d, f.ravel()])
    d.sort()
    k1, k2 = cuts
    d1 = d[max(int(0.01 * k1 * len(d) + 0.5), 0)]
    d2 = d[min(int(0.01 * k2 * len(d) + 0.5), len(d) - 1)]

    # first image
    ima, col, sat = ImaColList[0]
    p1 = PILima.new('RGB', (ima.shape[0], ima.shape[1]))
    if interp == 'linear':
        data = ima._interp_data(spline=False)
    elif interp == 'spline':
        data = ima._interp_data(spline=True)
    else:
        data = np.ma.filled(ima.data, np.ma.median(ima.data))
    if mode == 'lin':
        f = data
    elif mode == 'sqrt':
        f = np.sqrt(np.clip(data, 0, 1.e99))
    lum = np.clip((f - d1) * 100 / (d2 - d1), 0, 100)
    for i in range(ima.shape[0]):
        for j in range(ima.shape[1]):
            p1.putpixel((i, j),
                        ImageColor.getrgb('hsl(%d,%d%%,%d%%)'
                                          % (int(col), int(sat), int(lum[i, j]))))

    for ImaCol in ImaColList[1:]:
        ima, col, sat = ImaCol
        p2 = PILima.new('RGB', (ima.shape[0], ima.shape[1]))
        if interp == 'linear':
            data = ima._interp_data(spline=False)
        elif interp == 'spline':
            data = ima._interp_data(spline=True)
        else:
            data = np.ma.filled(ima.data, np.ma.median(ima.data))
        if mode == 'lin':
            f = data
        elif mode == 'sqrt':
            f = np.sqrt(np.clip(data, 0, 1.e99))
        lum = np.clip((f - d1) * 100 / (d2 - d1), 0, 100)
        for i in range(ima.shape[0]):
            for j in range(ima.shape[1]):
                p2.putpixel((i, j), ImageColor.getrgb('hsl(%d,%d%%,%d%%)'
                                                      % (int(col), int(sat), int(lum[i, j]))))
        p1 = ImageChops.add(p1, p2)

    if bar:
        nxb = ima.shape[0]
        nyb = 50
        dx = nxb / len(ImaColList)
        p3 = PILima.new('RGB', (nxb, nyb))
        i1 = 0
        for ImaCol in ImaColList:
            ima, col, sat = ImaCol
            for i in range(i1, i1 + dx):
                for j in range(nyb):
                    p3.putpixel((i, j), ImageColor.getrgb('hsl(%d,%d%%,%d%%)'
                                                          % (int(col), int(sat), 50)))
            i1 += dx

    if bar:
        return p1, p3
    else:
        return p1

def mask_image(shape=(101, 101), wcs=WCS(), objects=[]):
    """Creates a new image from a table of apertures.
    
    ra(deg), dec(deg) and radius(arcsec).

    Parameters
    ----------
    shape  : integer or (integer,integer)
            Lengths of the image in Y and X
            with python notation: (ny,nx). (101,101) by default.
            If wcs object contains dimensions,
            shape is ignored and wcs dimensions are used.
    wcs    : :class:`mpdaf.obj.WCS`
            World coordinates.
    sky      : list of (float, float, float)
               (y, x, size) describes an aperture on the sky,
                   defined by a center (y, x) in degrees,
                   and size (radius) in arcsec.
                 
    Returns
    -------
    out : obj.Image object (`Image class`_)
    center, radius, pix=False, inside=True):
    """
    
    if is_int(shape):
        shape = (shape, shape)
    shape = np.array(shape)

    if wcs.naxis1 == 1. and wcs.naxis2 == 1.:
        wcs.naxis1 = shape[1]
        wcs.naxis2 = shape[0]
    else:
        if wcs.naxis1 != 0. or wcs.naxis2 != 0.:
            shape[1] = wcs.naxis1
            shape[0] = wcs.naxis2
            
    data = np.zeros(shape)
    
    for y ,x , r in objects:        
        center = wcs.sky2pix([y,x])[0]
        r = r / np.abs(wcs.get_step()) / 3600.
        r2 = r[0] * r[1]

        imin = max(0, center[0] - r[0])
        imax = min(center[0] + r[0] + 1, shape[0])
        jmin = max(0, center[1] - r[1])
        jmax = min(center[1] + r[1] + 1, shape[1])
        
        grid = np.meshgrid(np.arange(imin,imax)-center[0], np.arange(jmin,jmax)-center[1], indexing='ij')
        data[imin:imax, jmin:jmax] = np.array((grid[0]**2 + grid[1]**2) < r2, dtype =int)
        
    return Image(data=data, wcs=wcs)
