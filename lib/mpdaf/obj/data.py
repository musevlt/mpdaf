
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
from .objs import fix_unit

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
    hdulist  : pyfits.hdulist
               HDU list class.
    ext      : integer or (integer,integer) or string or (string,string)
                Number/name of the data extension
                or numbers/names of the data and variance extensions.
    unit     : astropy.units
               Physical units of the data values.
               u.dimensionless_unscaled by default.
    copy     : boolean
               If true (default), then the data and variance arrays are copied.
    dtype    : numpy.dtype
               Type of the data (integer, float)

    Attributes
    ----------
    filename       : string
                     Possible FITS filename.
    primary_header : pyfits.Header
                     FITS primary header instance.
    wcs            : :class:`mpdaf.obj.WCS`
                     World coordinates.
    wave           : :class:`mpdaf.obj.WaveCoord`
                     Wavelength coordinates
    ndim           : integer
                     Number of dimensions.
    shape          : tuple
                     Lengths of data (python notation (nz,ny,nx)).
    data           : masked array numpy.ma
                     Masked array containing the cube pixel values.
    data_header    : pyfits.Header
                     FITS data header instance.
    unit           : astropy.units
                     Physical units of the data values.
    dtype          : numpy.dtype
                     Type of the data (integer, float)
    var            : float array
                     Array containing the variance.
    """

    _ndim_required = None
    _has_wcs = False
    _has_wave = False

    def __init__(self, filename=None, hdulist=None, ext=None, data=None,
                 var=None, unit=u.dimensionless_unscaled, copy=True,
                 dtype=float, **kwargs):
        self._logger = logging.getLogger(__name__)
        self.filename = filename
        self._data = None
        self._data_ext = None
        self._var = None
        self._var_ext = None
        self._ndim = None
        # self._shape = (shape, ) if np.isscalar(shape) else shape
        self.wcs = None
        self.wave = None
        self.dtype = dtype
        self.unit = unit
        self.data_header = pyfits.Header()
        self.primary_header = pyfits.Header()

        if kwargs.pop('shape', None) is not None:
            warnings.warn('The shape parameter is no more used, it is derived '
                          'from the data instead', MpdafWarning)

        if kwargs.pop('notnoise', None) is not None:
            warnings.warn('The notnoise parameter is no more used, the '
                          'variance wll be read if necessary', MpdafWarning)

        if filename is not None:
            if not is_valid_fits_file(filename):
                raise IOError('Invalid file: %s' % filename)

            if hdulist is None:
                hdulist = pyfits.open(filename)
                close_hdu = True
            else:
                close_hdu = False

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
            elif isinstance(ext, (int, str, unicode)):
                self._data_ext = ext
                self._var_ext = None

            self.data_header = hdr = hdulist[self._data_ext].header

            self.unit = u.dimensionless_unscaled
            try:
                self.unit = u.Unit(fix_unit(hdr['BUNIT']))
            except KeyError:
                self._logger.warning('The physical unit of the data is not '
                                     'loaded from the FITS header.\n'
                                     'No BUNIT in the DATA.')
            except Exception as e:
                self._logger.warning('The physical unit of the data is not '
                                     'loaded from the FITS header.\n %s', e)

            if 'FSCALE' in hdr:
                self.unit *= u.Unit(hdr['FSCALE'])

            self._shape = hdulist[self._data_ext].data.shape
            # self.shape = np.array([hdr['NAXIS3'], hdr['NAXIS2'],
            #                        hdr['NAXIS1']])

            self._ndim = hdr['NAXIS']
            if self._ndim_required is not None and \
                    hdr['NAXIS'] != self._ndim_required:
                raise IOError('Wrong dimension number, should be %s'
                              % self._ndim_required)

            if self._has_wcs:
                try:
                    self.wcs = WCS(hdr)  # WCS object from data header
                except pyfits.VerifyError as e:
                    # Workaround for
                    # https://github.com/astropy/astropy/issues/887
                    self._logger.warning(e)
                    self.wcs = WCS(hdr)

            # Wavelength coordinates
            wave_ext = 1 if self._ndim_required == 1 else 3
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

            if close_hdu:
                hdulist.close()
        else:
            if data is not None:
                # set mask=False to force the expansion of the mask array with
                # the same dimension as the data
                self._data = ma.MaskedArray(data, mask=False, dtype=dtype,
                                            copy=copy)
                self._shape = self._data.shape

            if var is not None:
                self._var = np.array(var, dtype=dtype, copy=copy)

        wcs = kwargs.pop('wcs', None)
        if wcs is not None:
            try:
                self.wcs = wcs.copy()
                if wcs.naxis1 != 0 and wcs.naxis2 != 0 and \
                    (wcs.naxis1 != self._shape[-1] or
                        wcs.naxis2 != self._shape[-2]):
                    self._logger.warning(
                        'world coordinates and data have not the same '
                        'dimensions: shape of WCS object is modified')
                self.wcs.naxis1 = self._shape[-1]
                self.wcs.naxis2 = self._shape[-2]
            except:
                self._logger.warning('world coordinates not copied',
                                     exc_info=True)

        wave = kwargs.pop('wave', None)
        if wave is not None:
            try:
                self.wave = wave.copy()
                if wave.shape is not None and wave.shape != self._shape[0]:
                    self._logger.warning(
                        'wavelength coordinates and data have not the same '
                        'dimensions: shape of WaveCoord object is modified')
                self.wave.shape = self._shape[0]
            except:
                self._logger.warning('wavelength solution not copied',
                                     exc_info=True)

    @property
    def ndim(self):
        if self._ndim is not None:
            return self._ndim
        elif self.data is not None:
            return self.data.ndim
        else:
            raise AttributeError('No ndim attribute')

    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        elif self.data is not None:
            return self.data.shape
        else:
            raise AttributeError('No shape attribute')

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
        if value is not None:
            value = np.asarray(value)
            if not np.array_equal(self.shape, value.shape):
                raise ValueError('var and data have not the same dimensions.')
        self._var = value

    @deprecated('Variance should now be set with the `.var` attribute')
    def set_var(self, var):
        self.var = var

    def copy(self):
        """Returns a copy of the object."""
        obj = self.__class__(data=self.data, unit=self.unit, var=self.var,
                             wcs=self.wcs, wave=self.wave, copy=True)
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
            var=None if var is False else np.zeros(shape=self.shape),
            wcs=None if self.wcs is None else self.wcs.copy(),
            wave=None if self.wave is None else self.wave.copy()
        )
        obj.data_header = pyfits.Header(self.data_header)
        obj.primary_header = pyfits.Header(self.primary_header)
        return obj

    def info(self):
        """Prints information."""
        log_info = partial(self._logger.info)

        shape = (self.shape, ) if np.isscalar(self.shape) else self.shape
        shape_str = [str(x) for x in shape]
        log_info('%s %s (%s)', ' x '.join(shape_str),
                 self.__class__.__name__, self.filename or 'no name')

        data = ('no data' if self._data is None and self._data_ext is None
                else '.data({})'.format(','.join(shape_str)))
        noise = ('no noise' if self._var is None and self._var_ext is None
                 else '.var({})'.format(','.join(shape_str)))
        unit = str(self.unit) or 'no unit'
        log_info('%s (%s), %s', data, unit, noise)

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

    @deprecated('Data should now be set with the `.data` attribute')
    def get_np_data(self):
        return self.data

    def __le__(self, item):
        """Mask data array where greater than a given value (<=).

        Parameters
        ----------
        item : float
            minimum value.

        Returns
        -------
        out : New object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater(self.data, item)
        return result

    def __lt__(self, item):
        """Mask data array where greater or equal than a given value (<).

        Parameters
        ----------
        item : float
               minimum value.

        Returns
        -------
        out : New object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_greater_equal(self.data, item)
        return result

    def __ge__(self, item):
        """Mask data array where less than a given value (>=).

        Parameters
        ----------
        item : float
            maximum value.

        Returns
        -------
        out : New object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less(self.data, item)
        return result

    def __gt__(self, item):
        """Mask data array where less or equal than a given value (>).

        Parameters
        ----------
        item : float
               maximum value.

        Returns
        -------
        out : New object.
        """
        result = self.copy()
        if self.data is not None:
            result.data = np.ma.masked_less_equal(self.data, item)
        return result

    def _sqrt(self):
        if self.data is None:
            raise ValueError('empty data array')
        if self.var is not None:
            self.var = 3 * self.var / self.data.data ** 4
        self.data = np.ma.sqrt(self.data)
        self.unit /= np.sqrt(self.unit.scale)

    def sqrt(self):
        """Return a new object with the positive square-root of the data."""
        res = self.copy()
        res._sqrt()
        return res

    def _abs(self):
        if self.data is None:
            raise ValueError('empty data array')
        self.data = np.ma.abs(self.data)

    def abs(self):
        """Return a new object with the absolute value of the data."""
        res = self.copy()
        res._abs()
        return res

    def unmask(self):
        """Unmask the data (just invalid data (nan,inf) are masked)."""
        self.data.mask = False
        self.data = np.ma.masked_invalid(self.data)

    def mask_variance(self, threshold):
        """Mask pixels with a variance upper than threshold value.

        Parameters
        ----------
        threshold : float
                    Threshold value.
        """
        if self.var is None:
            raise ValueError('Operation forbidden without variance extension.')
        else:
            self.data[self.var > threshold] = np.ma.masked

    def mask_selection(self, ksel):
        """Masks pixels corresponding to the selection.

        Parameters
        ----------
        ksel : output of np.where
               elements depending on a condition
        """
        self.data[ksel] = np.ma.masked
