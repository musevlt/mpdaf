
import logging
import numpy as np
import os
import warnings

from astropy import units as u
from astropy.io import fits as pyfits
from functools import partial
from numpy import ma

from .coords import WCS, WaveCoord
from ..tools import MpdafWarning, deprecated

# __all__ = ['iter_spe', 'iter_ima', 'Cube', 'CubeDisk']


def is_valid_fits_file(filename):
    return os.path.isfile(filename) and filename.endswith(("fits", "fits.gz"))


def read_slice_from_fits(filename, item=None, ext='DATA', mask_ext=None,
                         dtype=None):
    hdulist = pyfits.open(filename)
    if item is None:
        data = np.asarray(hdulist[ext].data, dtype=dtype)
    else:
        data = np.asarray(hdulist[ext].data[item], dtype=dtype)

    # mask extension
    if mask_ext is not None and mask_ext in hdulist:
        mask = ma.make_mask(hdulist[mask_ext].data[item])
        data = ma.MaskedArray(data, mask=mask)

    hdulist.close()
    return data


class DataArray(object):

    """Base class to handle arrays.

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
                Lengths of data in Z, Y and X. Python notation is used
                (nz,ny,nx). If data is not None, its shape is used instead.
    wcs      : :class:`mpdaf.obj.WCS`
                World coordinates.
    wave     : :class:`mpdaf.obj.WaveCoord`
                Wavelength coordinates.
    unit     : string
                Possible data unit type. None by default.
    data     : float array
                Array containing the pixel values of the cube. None by
                default.
    var      : float array
                Array containing the variance. None by default.

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
    wcs            : :class:`mpdaf.obj.WCS`
                     World coordinates.
    wave           : :class:`mpdaf.obj.WaveCoord`
                     Wavelength coordinates
    """

    _ndim = None
    _has_wcs = False
    _has_wave = False

    def __init__(self, filename=None, ext=None, notnoise=False, unit=u.count,
                 data=None, var=None, shape=None, copy=True, dtype=float,
                 **kwargs):
        d = {'class': self.__class__.__name__, 'method': '__init__'}
        self.logger = logging.getLogger('mpdaf corelib')
        self.filename = filename
        self._data = None
        self._data_ext = None
        self._var = None
        self._var_ext = None
        self._shape = (shape, ) if np.isscalar(shape) else shape
        self.wcs = None
        self.wave = None
        self.dtype = dtype
        self.unit = unit
        self.data_header = pyfits.Header()
        self.primary_header = pyfits.Header()

        # FIXME: Deprecate and remove shape, fscale and notnoise
        if shape is not None:
            warnings.warn('The shape parameter is no more used, it is derived '
                          'from the data instead', MpdafWarning)

        if filename is not None:
            if not is_valid_fits_file(filename):
                raise IOError('Invalid file: %s' % filename)

            hdulist = pyfits.open(filename)
            # primary header
            self.primary_header = hdulist[0].header

            if len(hdulist) == 1:
                # if the number of extension is 1,
                # we just read the data from the primary header
                self._data_ext = 0
            elif ext is None:
                if 'DATA' in hdulist:
                    self._data_ext = 'DATA'
                elif 'SCI' in hdulist:
                    self._data_ext = 'SCI'
                else:
                    raise IOError('no DATA or SCI extension')

                if 'STAT' in hdulist:
                    self._var_ext = 'STAT'
            elif isinstance(ext, (list, tuple, np.ndarray)):
                self._data_ext = ext[0]
                self._var_ext = ext[1]
            elif isinstance(ext, int):
                self._data_ext = ext
                self._var_ext = None

            if notnoise:
                self._var_ext = None

            self.data_header = hdr = hdulist[self._data_ext].header
            self.unit = u.Unit(hdr.get('BUNIT', 'count'))
            self._shape = hdulist[self._data_ext].data.shape
            # self.shape = np.array([hdr['NAXIS3'], hdr['NAXIS2'],
            #                        hdr['NAXIS1']])

            self.ndim = hdr['NAXIS']
            if self._ndim is not None and hdr['NAXIS'] != self._ndim:
                raise IOError('Wrong dimension number, should be %s'
                              % self._ndim)

            if self._has_wcs:
                try:
                    self.wcs = WCS(hdr)  # WCS object from data header
                except pyfits.VerifyError as e:
                    # Workaround for
                    # https://github.com/astropy/astropy/issues/887
                    self.logger.warning(e, extra=d)
                    self.wcs = WCS(hdr)

            # Wavelength coordinates
            wave_ext = 1 if self._ndim == 1 else 3
            crpix = 'CRPIX{}'.format(wave_ext)
            crval = 'CRVAL{}'.format(wave_ext)
            if self._has_wave and crpix in hdr and crval in hdr:
                # if 'CDELT{}'.format(wave_ext) in hdr:
                #     cdelt = hdr.get('CDELT{}'.format(wave_ext))
                # elif 'CD{0}_{0}'.format(wave_ext) in hdr:
                #     cdelt = hdr.get('CD{0}_{0}'.format(wave_ext))
                # else:
                #     cdelt = 1.0
                # cunit = hdr.get('CUNIT{}'.format(wave_ext), '')
                # ctype = hdr.get('CTYPE{}'.format(wave_ext), 'LINEAR')
                # self.wave = WaveCoord(hdr[crpix], cdelt, hdr[crval],
                #                       cunit, ctype, self._shape[0])
                self.wave = WaveCoord(hdr)

            hdulist.close()
        else:
            if data is not None:
                # set mask=False to force the expansion of the mask array with
                # the same dimension as the data
                self._data = ma.MaskedArray(data, mask=False, dtype=dtype,
                                            copy=copy)
                self._shape = self._data.shape

            if not notnoise and var is not None:
                self._var = np.array(var, dtype=dtype, copy=copy)

        wcs = kwargs.pop('wcs', None)
        if wcs is not None:
            try:
                self.wcs = wcs.copy()
                if wcs.naxis1 != 0 and wcs.naxis2 != 0 and \
                    (wcs.naxis1 != self._shape[-1] or
                        wcs.naxis2 != self._shape[-2]):
                    self.logger.warning(
                        'world coordinates and data have not the same '
                        'dimensions: shape of WCS object is modified', extra=d)
                self.wcs.naxis1 = self._shape[-1]
                self.wcs.naxis2 = self._shape[-2]
            except:
                self.logger.warning('world coordinates not copied',
                                    exc_info=True, extra=d)

        wave = kwargs.pop('wave', None)
        if wave is not None:
            try:
                self.wave = wave.copy()
                if wave.shape is not None and wave.shape != self._shape[0]:
                    self.logger.warning(
                        'wavelength coordinates and data have not the same '
                        'dimensions: shape of WaveCoord object is modified',
                        extra=d)
                self.wave.shape = self._shape[0]
            except:
                self.logger.warning('wavelength solution not copied',
                                    exc_info=True, extra=d)

    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        else:
            return self.data.shape

    @property
    def data(self):
        if self._data is None and self.filename is not None:
            self._data = read_slice_from_fits(
                self.filename, ext=self._data_ext, mask_ext='DQ',
                dtype=self.dtype)

            # Mask an array where invalid values occur (NaNs or infs).
            if ma.is_masked(self._data):
                self._data.mask |= ~(np.isfinite(self._data.data))
            else:
                self._data = ma.masked_invalid(self._data)
            self._shape = self._data.shape

        return self._data

    @data.setter
    def data(self, value):
        self._data = ma.MaskedArray(value)
        self._shape = self._data.shape

    @property
    def var(self):
        if self._var is None and self._var_ext is not None and \
                self.filename is not None:
            var = read_slice_from_fits(
                self.filename, ext=self._var_ext, dtype=self.dtype)
            if var.ndim != self.data.ndim:
                raise IOError('Wrong dimension number in STAT extension')
            if not np.array_equal(var.shape, self.data.shape):
                raise IOError('Number of points in STAT not equal to DATA')
            self._var = var

        return self._var

    @var.setter
    def var(self, value):
        value = np.asarray(value)
        if not np.array_equal(self.shape, value.shape):
            raise ValueError('var and data have not the same dimensions.')
        self._var = value

    @deprecated('Variance should now be set with the `.var` attribute')
    def set_var(self, var):
        self.var = var

    def copy(self):
        """Returns a copy of the object."""
        obj = self.__class__(
            data=self.data.copy(),
            unit=self.unit,
            fscale=self.fscale,
            var=None if self.var is None else self.var.copy(),
            wcs=None if self.wcs is None else self.wcs.copy(),
            wave=None if self.wave is None else self.wave.copy()
        )
        obj.filename = self.filename
        obj.data_header = pyfits.Header(self.data_header)
        obj.primary_header = pyfits.Header(self.primary_header)
        return obj

    def clone(self, var=False):
        """Returns a shallow copy with the same shape and coordinates, filled
        with zeros.

        Parameters
        ----------
        var : bool
        Presence of the variance extension.
        """
        obj = self.__class__(
            data=np.zeros(shape=self.shape),
            unit=self.unit,
            fscale=self.fscale,
            var=None if var is False else np.zeros(shape=self.shape),
            wcs=None if self.wcs is None else self.wcs.copy(),
            wave=None if self.wave is None else self.wave.copy()
        )
        obj.data_header = pyfits.Header(self.data_header)
        obj.primary_header = pyfits.Header(self.primary_header)
        return obj

    def info(self):
        """Prints information."""
        d = {'class': self.__class__.__name__, 'method': 'info'}
        log_info = partial(self.logger.info, extra=d)

        shape_str = [str(x) for x in self.shape]
        log_info('%s %s (%s)', ' x '.join(shape_str),
                 self.__class__.__name__, self.filename or 'no name')

        data = ('no data' if self._data is None and self._data_ext is None
                else '.data({})'.format(','.join(shape_str)))
        noise = ('no noise' if self._var is None and self._var_ext is None
                 else '.var({})'.format(','.join(shape_str)))
        unit = 'no unit' if self.unit is None else self.unit
        log_info('%s (%s) fscale=%g, %s', data, unit, self.fscale, noise)

        if self._has_wcs:
            if self.wcs is None:
                log_info('no world coordinates for spatial direction')
            else:
                self.wcs.info()

        if self._has_wave:
            if self.wave is None:
                log_info('no world coordinates for spectral direction')
            else:
                self.wave.info()
