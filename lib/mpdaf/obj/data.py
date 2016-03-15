# Import the recommended python 2 -> 3 compatibility modules.

from __future__ import absolute_import
from __future__ import division

import logging
import numpy as np
import warnings

from astropy import units as u
from astropy.io import fits
from datetime import datetime
from numpy import ma

from .coords import WCS, WaveCoord
from .objs import UnitMaskedArray
from ..tools import (MpdafWarning, MpdafUnitsWarning, deprecated,
                     fix_unit_read, is_valid_fits_file, read_slice_from_fits,
                     copy_header)


class DataArray(object):

    """The DataArray class is the parent of the mpdaf.obj.Cube,
    mpdaf.obj.Image and mpdaf.obj.Spectrum classes. Its primary
    purpose is to store pixel values in a masked numpy array. For
    Cube objects this is a 3D array indexed in the order
    [wavelength,image_y,image_x]. For Image objects it is a 2D
    array indexed in the order [image_y,image_x]. For Spectrum
    objects it is a 1D spectrum.

    Image arrays hold flat 2D map-projections of the sky. The X and Y
    axes of the image arrays are orthogonal on the sky at the tangent
    point of the projection. When the rotation angle of the projection
    on the sky is zero, the Y axis of the image arrays is along the
    declination axis, and the X axis is perpendicular to this, with
    the positive X axis pointing east.

    The DataArray class has a number of optional features. There is
    a .var member which can optionally hold an array of variances
    for each value in the data array. For cubes and spectra, the
    wavelengths of the spectral pixels can be specified in the
    .wave member. For cubes and images, the world-coordinates of
    the image pixels can be specified in the .wcs member.

    When a DataArray object is constructed from a FITS file, the
    name of the file and the file's primary header are recorded. If
    the data are read from a FITS extension, the header of this
    extension is also recorded. Alternatively, the primary header
    and data header can be passed to the DataArray constructor.
    Where FITS headers are neither provided, nor available in a
    provided FITS file, generic headers are substituted.

    Methods are provided for masking and unmasking pixels, and
    performing basic arithmetic operations on pixels. Operations
    that are specific to cubes or spectra or images are provided
    elsewhere by derived classes.

    Parameters
    ----------
    filename : str
        FITS file name, default to None.
    hdulist : astropy.fits.HDUList
        HDU list class, used instead of fits.open(filename) if not None,
        to avoid opening the FITS file.
    ext : int or (int,int) or str or (str,str)
        Number/name of the data extension or numbers/names of the data and
        variance extensions.
    unit : astropy.units.Unit
        Physical units of the data values, default to
        u.dimensionless_unscaled.
    copy : bool
        If True (default), then the data and variance arrays are copied.
        Passed to numpy.ma.MaskedArray.
    dtype : numpy.dtype
        Type of the data, default to float.
        Passed to numpy.ma.MaskedArray.
    data : numpy.ndarray or list
        Data array, passed to numpy.ma.MaskedArray.
    var : numpy.ndarray or list
        Variance array, passed to numpy.array().
    mask : bool or numpy.ma.nomask or numpy.ndarray
        Mask used for the creation of the .data MaskedArray. If mask is
        False (default value), a mask array of the same size of the data array
        is created. To avoid creating an array, it is possible to use
        numpy.ma.nomask, but in this case several methods will break if
        they use the mask.

    Attributes
    ----------
    filename : str
        FITS filename.
    primary_header : astropy.io.fits.Header
        FITS primary header instance.
    wcs : mpdaf.obj.WCS
        World coordinates.
    wave : mpdaf.obj.WaveCoord
        Wavelength coordinates
    ndim : int
        Number of dimensions.
    shape : sequence
        Lengths of the data axes (python notation (nz,ny,nx)).
    data : numpy.ma.MaskedArray
        Masked array containing the cube of pixel values.
    data_header : astropy.io.fits.Header
        FITS data header instance.
    unit : astropy.units.Unit
        Physical units of the data values.
    dtype : numpy.dtype
        Type of the data (int, float, ...).
    var : numpy.ndarray
        Array containing the variance.

    """

    _ndim_required = None
    _has_wcs = False
    _has_wave = False

    def __init__(self, filename=None, hdulist=None, data=None, mask=False,
                 var=None, ext=None, unit=u.dimensionless_unscaled, copy=True,
                 dtype=float, primary_header=None, data_header=None, **kwargs):
        self._logger = logging.getLogger(__name__)
        self.filename = filename
        self._data = None
        self._data_ext = None
        self._var = None
        self._var_ext = None
        self._ndim = None
        self._shape = None
        self.wcs = None
        self.wave = None
        self.dtype = dtype
        self.unit = unit
        self.data_header = data_header or fits.Header()
        self.primary_header = primary_header or fits.Header()

        if kwargs.pop('shape', None) is not None:
            warnings.warn('The shape parameter is no longer used. It is '
                          'derived from the data instead', MpdafWarning)

        if kwargs.pop('notnoise', None) is not None:
            warnings.warn('The notnoise parameter is no longer used. The '
                          'variances wll be read if necessary.', MpdafWarning)

        # Read the data from a FITS file?

        if filename is not None and data is None:
            if not is_valid_fits_file(filename):
                raise IOError('Invalid file: %s' % filename)

            if hdulist is None:
                hdulist = fits.open(filename)
                close_hdu = True
            else:
                close_hdu = False

            # primary header
            self.primary_header = hdulist[0].header

            # Find the hdu of the data. This is either the primary HDU
            # or a DATA or SCI extension. Also see if there is an extension
            # that contains variances. This is either a STAT extension,
            # or the second of a tuple of extensions passed via the ext[]
            # parameter.

            if len(hdulist) == 1:
                # if the number of extension is 1,
                # we just read the data from the primary header
                self._data_ext = 0
            elif ext is None:
                if 'DATA' in hdulist:
                    self._data_ext = 'DATA'
                elif 'SCI' in hdulist:
                    self._data_ext = 'SCI'
                else:   # Use primary data array if no DATA or SCI extension
                    raise IOError('no DATA or SCI extension\n',
                                  ' Please use ext parameter to open others extensions.')

                if 'STAT' in hdulist:
                    self._var_ext = 'STAT'
            elif isinstance(ext, (list, tuple, np.ndarray)):
                self._data_ext = ext[0]
                self._var_ext = ext[1]
            elif isinstance(ext, (int, str, unicode)):
                self._data_ext = ext
                self._var_ext = None

            # Record the data header.

            self.data_header = hdr = hdulist[self._data_ext].header

            try:
                self.unit = u.Unit(fix_unit_read(hdr['BUNIT']))
            except KeyError:
                warnings.warn('No physical unit in the FITS header: missing '
                              'BUNIT keyword.', MpdafUnitsWarning)
            except Exception as e:
                warnings.warn('Error parsing the BUNIT: ' + str(e),
                              MpdafUnitsWarning)

            if 'FSCALE' in hdr:
                self.unit *= u.Unit(hdr['FSCALE'])

            self._shape = hdulist[self._data_ext].data.shape
            self._ndim = hdr['NAXIS']
            if self._ndim_required is not None and \
                    hdr['NAXIS'] != self._ndim_required:
                raise IOError('Wrong dimension number, should be %s'
                              % self._ndim_required)

            # Is this a derived class like Cube and Image that require
            # WCS information?

            if self._has_wcs:
                try:
                    self.wcs = WCS(hdr)  # WCS object from data header
                except fits.VerifyError as e:
                    # Workaround for
                    # https://github.com/astropy/astropy/issues/887
                    self._logger.warning(e)
                    self.wcs = WCS(hdr)

            # Get the wavelength coordinates.

            wave_ext = 1 if self._ndim_required == 1 else 3
            crpix = 'CRPIX{}'.format(wave_ext)
            crval = 'CRVAL{}'.format(wave_ext)
            if self._has_wave and crpix in hdr and crval in hdr:
                self.wave = WaveCoord(hdr)

            if close_hdu:
                hdulist.close()

        # There is no FITS file to read the data from.

        else:

            # Use a specified numpy data array?

            if data is not None:
                # By default, if mask=False create a mask array with False
                # values. numpy.ma does it but with a np.resize/np.concatenate
                # which cause a huge memory peak, so a workaround is to create
                # the mask here.
                if mask is False:
                    mask = np.zeros(data.shape, dtype=bool)
                self._data = ma.MaskedArray(data, mask=mask, dtype=dtype,
                                            copy=copy)
                self._shape = self._data.shape

            # Use a specified variance array?

            if var is not None:
                self._var = np.array(var, dtype=dtype, copy=copy)

        # If a WCS object was specified as an optional parameter, install it.
        wcs = kwargs.pop('wcs', None)
        if self._has_wcs and wcs is not None and wcs.naxis1 != 1 and \
                wcs.naxis2 != 1:
            try:
                self.wcs = wcs.copy()
                if self._shape is not None:
                    if (wcs.naxis1 != 0 and wcs.naxis2 != 0 and
                        (wcs.naxis1 != self._shape[-1] or
                         wcs.naxis2 != self._shape[-2])):
                        self._logger.warning(
                            'The world coordinates and data have different '
                            'dimensions: Modifying the shape of the WCS '
                            'object')
                    self.wcs.naxis1 = self._shape[-1]
                    self.wcs.naxis2 = self._shape[-2]
            except:
                self._logger.warning('world coordinates not copied',
                                     exc_info=True)

        # If a wavelength coordinate object was specified as an
        # optional parameter, install it.
        wave = kwargs.pop('wave', None)
        if self._has_wave and wave is not None and wave.shape != 1:
            try:
                self.wave = wave.copy()
                if self._shape is not None:
                    if wave.shape is not None and \
                            wave.shape != self._shape[0]:
                        self._logger.warning(
                            'wavelength coordinates and data have different '
                            'dimensions: Modifying the shape of the WaveCoord '
                            'object')
                    self.wave.shape = self._shape[0]
            except:
                self._logger.warning('wavelength solution not copied',
                                     exc_info=True)

    @classmethod
    def new_from_obj(cls, obj, data=None, var=None, copy=False):
        """Create a new object from another one, copying its attributes."""
        data = obj.data if data is None else data
        var = obj.var if var is None else var
        kwargs = dict(filename=obj.filename, data=data, unit=obj.unit, var=var,
                      dtype=obj.dtype, copy=copy, data_header=obj.data_header,
                      primary_header=obj.primary_header)
        if cls._has_wcs:
            kwargs['wcs'] = obj.wcs
        if cls._has_wave:
            kwargs['wave'] = obj.wave
        return cls(**kwargs)

    @property
    def ndim(self):
        """ The number of dimensions in the data and variance arrays : int """
        if self._ndim is not None:
            return self._ndim
        elif self.data is not None:
            return self.data.ndim

    @property
    def shape(self):
        """ The lengths of each of the .ndim data axes. """
        if self._shape is not None:
            return self._shape
        elif self.data is not None:
            return self.data.shape

    @property
    def data(self):
        """ A masked array of data values : numpy.ma.MaskedArray. """

        # The DataArray constructor postpones reading data from FITS files
        # until they are first used. Read the data array here if not already
        # read.

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
        self._ndim = self._data.ndim

    @property
    def var(self):
        """ Either None, or a numpy.ndarray containing the variances
        of each data value. This has the same shape as the data array. """

        # The DataArray constructor postpones reading data from FITS
        # files until they are first used. On the first call to this
        # function, read the variances if a suitable extension exists.

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
                raise ValueError('var and data have different dimensions.')
        else:
            self._var_ext = None
        self._var = value

    @deprecated('Variance should now be set with the .var attribute')
    def set_var(self, var):
        """Deprecated: The variance array can simply be assigned to
        the .var attribute"""

        self.var = var

    @property
    def masked_var(self):
        """Return a MaskedArray for the variance with the data mask."""
        var = ma.masked_invalid(self.var, copy=False)
        var[self.data.mask] = ma.masked
        return var

    def copy(self):
        """Return a copy of the object."""
        return self.__class__(
            filename=self.filename, data=self.data, unit=self.unit,
            var=self.var, wcs=self.wcs, wave=self.wave, copy=True,
            data_header=fits.Header(self.data_header),
            primary_header=fits.Header(self.primary_header))

    def clone(self, var=None, data_init=None, var_init=None):
        """Return a shallow copy with the same header and coordinates.

        Parameters
        ----------
        var : bool
            **Deprecated**, replaced by var_init.
        data_init : function
            Function used to create the data array (takes the shape as
            parameter). For example np.zeros or np.empty. Default to
            None which means that the data attribute is None.
        var_init : function
            Function used to create the data array, same as data_init.
            Default to None which set the var attribute to None.

        """
        if var is not None:
            warnings.warn('The "var" parameter is no longer used. Use '
                          '"var_init"instead.', MpdafWarning)

        return self.__class__(
            unit=self.unit, dtype=None, copy=False,
            data=None if data_init is None else data_init(self.shape,
                                                          dtype=self.dtype),
            var=None if var_init is None else var_init(self.shape,
                                                       dtype=self.dtype),
            wcs=None if self.wcs is None else self.wcs.copy(),
            wave=None if self.wave is None else self.wave.copy(),
            data_header=fits.Header(self.data_header),
            primary_header=fits.Header(self.primary_header))

    def info(self):
        """Print information."""
        log = self._logger.info
        shape_str = (' x '.join(str(x) for x in self.shape)
                     if self.shape is not None else 'no shape')
        log('%s %s (%s)', shape_str, self.__class__.__name__,
            self.filename or 'no name')

        data = ('no data' if self._data is None and self._data_ext is None
                else '.data({})'.format(shape_str))
        noise = ('no noise' if self._var is None and self._var_ext is None
                 else '.var({})'.format(shape_str))
        unit = str(self.unit) or 'no unit'
        log('%s (%s), %s', data, unit, noise)

        if self._has_wcs:
            if self.wcs is None:
                log('no world coordinates for spatial direction')
            else:
                self.wcs.info()

        if self._has_wave:
            if self.wave is None:
                log('no world coordinates for spectral direction')
            else:
                self.wave.info()

    @deprecated('Data should now be set with the .data attribute')
    def get_np_data(self):
        """Deprecated: Set the .data attribute instead of calling
        this function."""

        return self.data

    def __le__(self, item):
        """Mask data elements whose values are greater than a
           given value (<=).

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
        """Mask data elements whose values are greater than or equal
        to a given value (<).

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
        """Mask data elements whose values are less than a given value (>=).

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
        """Mask data elements whose values are less than or equal to a
        given value (>).

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

    def __getitem__(self, item):
        """Return a sliced object.

        cube[k,p,k] = value
        cube[k,:,:] = spectrum
        cube[:,p,q] = image
        cube[:,:,:] = sub-cube

        """
        # The DataArray constructor postpones reading data from FITS files
        # until they are first used. Read the slice from the FITS file if
        # the data array hasn't been read yet.

        if self._data is None and self.filename is not None:
            data = read_slice_from_fits(self.filename, item=item,
                                        ext=self._data_ext, mask_ext='DQ',
                                        dtype=self.dtype)

            # Mask an array where invalid values occur (NaNs or infs).
            if ma.is_masked(data):
                data.mask |= ~(np.isfinite(data.data))
            else:
                data = ma.masked_invalid(data)
        else:
            data = self._data[item]  # data = self.data[item].copy()

        # The DataArray constructor postpones reading variances from
        # FITS files until they are first used. Read the slice of
        # variances from the FITS file if the variance array hasn't
        # been read yet.

        if self._var is None:
            if self.filename is not None:
                if self._var_ext is None:
                    var = None
                else:
                    var = read_slice_from_fits(
                        self.filename, item=item, ext=self._var_ext,
                        dtype=self.dtype)
                    if var.ndim != data.ndim:
                        raise IOError('Wrong dimension number in STAT '
                                      'extension')
                    if not np.array_equal(var.shape, data.shape):
                        raise IOError('Number of points in STAT not equal to '
                                      'DATA')
            else:
                var = None
        else:
            var = self._var[item]  # copy

        # Construct new WCS and wavelength coordinate information for
        # the slice.

        wave = None
        wcs = None
        if self.ndim == 3 and isinstance(item, (list, tuple)) and \
                len(item) == 3:
            try:
                wcs = self.wcs[item[1], item[2]]
            except:
                wcs = None
            try:
                wave = self.wave[item[0]]
            except:
                wave = None
        elif self.ndim == 2 and isinstance(item, (list, tuple)) and \
                len(item) == 2:
            try:
                wcs = self.wcs[item]
            except:
                wcs = None
        elif self.ndim == 1 and isinstance(item, slice):
            try:
                wave = self.wave[item]
            except:
                wave = None

        if data.shape == ():
            return data
        else:
            return self.__class__(
                data=data, unit=self.unit, var=var, wcs=wcs, wave=wave,  # copy
                filename=self.filename,
                data_header=fits.Header(self.data_header),
                primary_header=fits.Header(self.primary_header))

    def __setitem__(self, item, other):
        """Set the corresponding part of data."""
        if self.data is None:
            raise ValueError('empty data array')

        if isinstance(other, DataArray):
            # FIXME: check only step or full wcs ?
            if self._has_wave and other._has_wave \
                    and not self.wave.isEqual(other.wave):
                raise ValueError('Operation forbidden for cubes with different'
                                 ' world coordinates in spectral direction')
            if self._has_wcs and other._has_wcs \
                    and not self.wcs.isEqual(other.wcs):
                raise ValueError('Operation forbidden for cubes with different'
                                 ' world coordinates in spatial directions')

            if self.unit == other.unit:
                self.data[item] = other.data
            else:
                self.data[item] = UnitMaskedArray(other.data, other.unit,
                                                  self.unit)
        else:
            self.data[item] = other

    def get_wcs_header(self):
        """Return a FITS header containing coordinate descriptions."""
        if self.ndim == 1 and self.wave is not None:
            return self.wave.to_header()
        elif self.ndim == 2 and self.wcs is not None:
            return self.wcs.to_header()
        elif self.ndim == 3 and self.wcs is not None:
            return self.wcs.to_cube_header(self.wave)

    def get_data_hdu(self, name='DATA', savemask='dq'):
        """Return an ImageHDU corresponding to the DATA extension.

        Parameters
        ----------
        name : str
            Extension name, DATA by default.
        savemask : str
            If 'dq', the mask array is saved in a DQ extension.
            If 'nan', masked data are replaced by nan in a DATA extension.
            If 'none', masked array is not saved.

        Returns
        -------
        out : astropy.io.fits.ImageHDU

        """
        if self.data.dtype == np.float64:
            # Force data to be stored in float instead of double
            data = self.data.astype(np.float32)
        else:
            data = self.data

        # create DATA extension
        if savemask == 'nan' and ma.count_masked(data) > 0:
            # NaNs can be used only for float arrays, so we raise an exception
            # if there are masked values in a non-float array.
            if not np.issubdtype(data.dtype, np.float):
                raise ValueError('The .data array contains masked values but '
                                 'its type does not allow replacement with '
                                 'NaNs. You can either fill the array with '
                                 'another value or use another option for '
                                 'savemask.')
            data = data.filled(fill_value=np.nan)
        else:
            data = data.data

        hdr = copy_header(self.data_header, self.get_wcs_header(),
                          exclude=('CD*', 'PC*', 'CDELT*', 'CRPIX*', 'CRVAL*',
                                   'CSYER*', 'CTYPE*', 'CUNIT*', 'NAXIS*',
                                   'RADESYS', 'LATPOLE', 'LONPOLE'),
                          unit=self.unit)
        return fits.ImageHDU(name=name, data=data, header=hdr)

    def get_stat_hdu(self, name='STAT', header=None):
        """Return an ImageHDU corresponding to the STAT extension.

        Parameters
        ----------
        name : str
            Extension name, STAT by default.

        Returns
        -------
        out : astropy.io.fits.ImageHDU

        """
        if self.var is None:
            return None

        if self.var.dtype == np.float64:
            # Force var to be stored in float instead of double
            var = self.var.astype(np.float32)
        else:
            var = self.var

        # world coordinates
        if header is None:
            header = self.get_wcs_header()

        header = copy_header(self.data_header, header,
                             exclude=('CD*', 'PC*'), unit=self.unit**2)
        return fits.ImageHDU(name=name, data=var, header=header)

    def write(self, filename, savemask='dq'):
        """Save the data to a FITS file.

        Parameters
        ----------
        filename : str
            The FITS filename.
        savemask : str
            If 'dq', the mask array is saved in DQ extension
            If 'nan', masked data are replaced by nan in DATA extension.
            If 'none', masked array is not saved.

        """
        warnings.simplefilter('ignore')
        header = copy_header(self.primary_header)
        header['date'] = (str(datetime.now()), 'creation date')
        header['author'] = ('MPDAF', 'origin of the file')
        hdulist = [fits.PrimaryHDU(header=header)]
        warnings.simplefilter('default')

        # create cube DATA extension
        datahdu = self.get_data_hdu(savemask=savemask)
        hdulist.append(datahdu)

        # create spectrum STAT extension
        if self.var is not None:
            hdulist.append(self.get_stat_hdu(header=datahdu.header.copy()))

        # create DQ extension
        if savemask == 'dq' and np.ma.count_masked(self.data) != 0:
            hdulist.append(fits.ImageHDU(
                name='DQ', header=datahdu.header.copy(),
                data=np.uint8(self.data.mask)))

        # save to disk
        hdu = fits.HDUList(hdulist)
        warnings.simplefilter('ignore')
        hdu.writeto(filename, clobber=True, output_verify='silentfix')
        warnings.simplefilter('default')

        self.filename = filename

    def sqrt(self, out=None):
        """Return a new object with positive data square-rooted, and
        negative data masked.

        Parameters
        ----------
        out : mpdaf.obj.DataArray, optional
            Array of the same shape as input, into which the output is placed.
            By default, a new array is created.

        """
        if self.data is None:
            raise ValueError('empty data array')

        if out is None:
            out = self.clone()

        out.data = np.ma.sqrt(self.data)
        out.unit = self.unit / u.Unit(np.sqrt(self.unit.scale))

        # Modify the variances to account for the effect of the square root.

        if self.var is not None:
            # For a value x, picked from a distribution of
            # variance, vx, the expected variance of sqrt(x), is:
            #
            #  vs = (d[sqrt(x)]/dx)**2 * vx
            #     = (0.5 / sqrt(x))**2 * vx
            #     = 0.25 / x * vx.
            out.var = 0.25 * self.var / self.data.data
        return out

    def abs(self, out=None):
        """Return a new object with the absolute value of the data.

        Parameters
        ----------
        out : mpdaf.obj.DataArray, optional
            Array of the same shape as input, into which the output is placed.
            By default, a new array is created.

        """
        if self.data is None:
            raise ValueError('empty data array')

        if out is None:
            out = self.clone()

        out.data = np.ma.abs(self.data)
        if self.var is not None:
            out.var = self.var.copy()
        return out

    def unmask(self):
        """Unmask the data (just invalid data (nan,inf) are masked)."""
        self.data.mask = False
        self.data = np.ma.masked_invalid(self.data)

    def mask_variance(self, threshold):
        """Mask pixels with a variance above a threshold value.

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
        """Mask selected pixels.

        Parameters
        ----------
        ksel : output of np.where
            Elements depending on a condition

        """
        self.data[ksel] = np.ma.masked
