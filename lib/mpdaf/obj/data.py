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
from .objs import UnitMaskedArray, UnitArray
from ..tools import (MpdafWarning, MpdafUnitsWarning, deprecated,
                     fix_unit_read, is_valid_fits_file, read_slice_from_fits,
                     copy_header)

__all__ = ('DataArray', )

# class SharedMaskArray(ma.MaskedArray):
#     """This is a subclass of numpy.ma.MaskedArray() which prevents
#     the mask of the array from being replaced when data are assigned
#     to slices of it. In the standard MaskedArray(), a simple
#     assignment like v[2] = np.ma.masked, causes the whole mask array
#     of v to be replaced with a new mask. This behavior is innefficient
#     and problematic when one is trying to share a mask between two
#     masked arrays, so the SharedMaskArray subclass overrides the
#     __setitem__() method of MaskedArray to directly assign to the
#     specified elements of the existing data and mask arrays.
# 
#     Parameters
#     ----------
#     data     : The initial data to be stored in the array.
#     **kwargs : The remaining key-value arguments are forwarded to
#          the MaskedArray constructor.
# 
#     """
# 
#     def __init__(self, data, **kwargs):
#         super(ma.MaskedArray, self).__init__(data, **kwargs)
# 
#     def __setitem__(self, indx, value):
#         if isinstance(value, ma.MaskedArray):
#             self.data[indx] = value.data
#             self.mask[indx] = value.mask
#         else:
#             self.data[indx] = value
#             self.mask[indx] = False
# 
#     def __setslice__(self, i, j, value):
#             self.__setitem__(slice(i,j), value)

class DataArray(object):

    """The DataArray class is the parent of the `~mpdaf.obj.Cube`,
    `~mpdaf.obj.Image` and `~mpdaf.obj.Spectrum` classes. Its primary
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
    hdulist : `astropy.fits.HDUList`
        HDU list class, used instead of fits.open(filename) if not None,
        to avoid opening the FITS file.
    ext : int or (int,int) or str or (str,str)
        Number/name of the data extension or numbers/names of the data and
        variance extensions.
    unit : `astropy.units.Unit`
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
    mask : numpy.ma.nomask or numpy.ndarray
    
    
        Mask used for the creation of the .data MaskedArray. If mask is
        False (default value), a mask array of the same size of the data array
        is created. To avoid creating an array, it is possible to use
        numpy.ma.nomask, but in this case several methods will break if
        they use the mask.
        
        

    Attributes
    ----------
    filename : str
        FITS filename.
    primary_header : `astropy.io.fits.Header`
        FITS primary header instance.
    wcs : `mpdaf.obj.WCS`
        World coordinates.
    wave : `mpdaf.obj.WaveCoord`
        Wavelength coordinates
    ndim : int
        Number of dimensions.
    shape : sequence
        Lengths of the data axes (python notation (nz,ny,nx)).
    data : numpy.ma.MaskedArray
        Masked array containing the cube of pixel values.
    data_header : `astropy.io.fits.Header`
        FITS data header instance.
    unit : `astropy.units.Unit`
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
        
        # used for properties
        self._pdata = None
        self._pvar = None
        self._pmask = None
        
        #self._data = None
        self._data_ext = None
        #self._var = None
        self._var_ext = None
        if mask is ma.nomask:
            self._mask = ma.nomask
        else:
            self._mask = False
        self._ndim = None
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
                    raise IOError('No DATA or SCI extension found.\n'
                                  'Please use the `ext` parameter to specify '
                                  'which extension must be loaded.')

                if 'STAT' in hdulist:
                    self._var_ext = 'STAT'
            elif isinstance(ext, (list, tuple, np.ndarray)):
                self._data_ext = ext[0]
                self._var_ext = ext[1]
            elif isinstance(ext, (int, str, unicode)):
                self._data_ext = ext
                self._var_ext = None

            self.primary_header = hdulist[0].header
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

        else:
            # Use a specified numpy data array?
            if data is not None:
                if isinstance(data, ma.MaskedArray):
                    self._data = np.array(data.data, dtype=dtype, copy=copy)
                    self._mask = data.mask
                else:
                    self._data = np.array(data, dtype=dtype, copy=copy)
                    if mask is ma.nomask:
                        self._mask = ma.nomask
                    elif mask is None:
                        self._mask = ~(np.isfinite(data))
                    else:
                        self._mask = np.resize(np.array(mask, dtype=bool, copy=copy),
                                               self._data.shape)
                        
                self._ndim = self._data.ndim

            # Use a specified variance array?
            if var is not None:
                if isinstance(var, ma.MaskedArray):
                    self._var = np.array(var.data, dtype=dtype, copy=copy)
                    self._mask |= var.mask
                else:
                    self._var = np.array(var, dtype=dtype, copy=copy)

        # If a WCS object was specified as an optional parameter, install it.
        wcs = kwargs.pop('wcs', None)
        if self._has_wcs and wcs is not None and wcs.naxis1 != 1 and \
                wcs.naxis2 != 1:
            try:
                self.wcs = wcs.copy()
                if self.shape is not None:
                    if (wcs.naxis1 != 0 and wcs.naxis2 != 0 and
                        (wcs.naxis1 != self.shape[-1] or
                         wcs.naxis2 != self.shape[-2])):
                        self._logger.warning(
                            'The world coordinates and data have different '
                            'dimensions: Modifying the shape of the WCS '
                            'object')
                    self.wcs.naxis1 = self.shape[-1]
                    self.wcs.naxis2 = self.shape[-2]
            except:
                self._logger.warning('world coordinates not copied',
                                     exc_info=True)

        # If a wavelength coordinate object was specified as an
        # optional parameter, install it.
        wave = kwargs.pop('wave', None)
        if self._has_wave and wave is not None and wave.shape != 1:
            try:
                self.wave = wave.copy()
                if self.shape is not None:
                    if wave.shape is not None and \
                            wave.shape != self.shape[0]:
                        self._logger.warning(
                            'wavelength coordinates and data have different '
                            'dimensions: Modifying the shape of the WaveCoord '
                            'object')
                    self.wave.shape = self.shape[0]
            except:
                self._logger.warning('wavelength solution not copied',
                                     exc_info=True)
                
    def _read_from_file(self, item=None, data=True, var=False):
        hdulist = fits.open(self.filename)
        if item:
            if data:
                data = np.asarray(hdulist[self._data_ext].data[item], dtype=self.dtype)
                if self._pmask is not ma.nomask:
                    if 'DQ' in hdulist:
                        mask = np.asarray(hdulist['DQ'].data[item], dtype=np.bool)
                    else:
                        mask = ~(np.isfinite(data))
                else:
                    mask = ma.nomask
            else:
                data = self._pdata
                mask = self._pmask
            if var and self._var_ext is not None:
                var = np.asarray(hdulist[self._var_ext].data[item], dtype=self.dtype)
            else:
                var = None
        else:
            if data:
                data = np.asarray(hdulist[self._data_ext].data, dtype=self.dtype)
                if self._pmask is not ma.nomask:
                    if 'DQ' in hdulist:
                        mask = np.asarray(hdulist['DQ'].data, dtype=np.bool)
                    else:
                        mask = ~(np.isfinite(data))
                else:
                    mask = ma.nomask
            else:
                data = self._pdata
                mask = self._pmask
            if var and self._var_ext is not None:
                var = np.asarray(hdulist[self._var_ext].data, dtype=self.dtype) 
            else:
                var = None
        hdulist.close()
        if hasattr(self, '_shape'):
            del self._shape
        return data, mask, var
                
    @property
    def _data(self):
        """ A array of data and mask values
        """
        if self._pdata is None and self.filename is not None:
            self._pdata, self._pmask, self._pvar = \
            self._read_from_file(var=False)
            self._ndim = self._pdata.ndim
        if self._pdata is None:
            raise ValueError('empty data array')
        return self._pdata
           
    @_data.setter
    def _data(self, value):
        self._pdata = value
      
    @property
    def _var(self):
        """ A array of data, var and mask values
        """
        if self._pvar is None and self._var_ext is not None and \
                self.filename is not None:
            self._pdata, self._pmask, self._pvar = \
            self._read_from_file(data=(self._pdata is None), var=True)
            self._ndim = self._pdata.ndim
        return self._pvar
    
    @_var.setter
    def _var(self, value):
        self._pvar = value
     
    @property
    def _mask(self):
        """ A array of data and mask values
        """
        if self._pdata is None and self.filename is not None:
            self._pdata, self._pmask, self._var = \
            self._read_from_file(var=False)
            self._ndim = self._pdata.ndim
        return self._pmask
           
    @_mask.setter
    def _mask(self, value):
        self._pmask = value

    @classmethod
    def new_from_obj(cls, obj, data=None, var=None, copy=False):
        """Create a new object from another one, copying its attributes.

        Parameters
        ----------
        data : ndarray-like
            Optional data array, otherwise ``obj.data`` is used.
        var : ndarray-like
            Optional variance array, otherwise ``obj.var`` is used.
        copy : bool
            Copy the data and variance if True (default False).

        """
        data = obj.data if data is None else data
        var = obj._var if var is None else var
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
        elif self._pdata is not None:
            return self._pdata.ndim

    @ndim.setter
    def ndim(self, value):
        if self._ndim_required is not None and value != self._ndim_required:
            raise ValueError('Wrong dimension number, should be {}, got {}'
                             .format(self._ndim_required, value))
        self._ndim = value

    @property
    def shape(self):
        """ The lengths of each of the .ndim data axes. """
        if self._pdata is not None:
            return self._pdata.shape
        elif hasattr(self, '_shape'):
            return self._shape
        else:
            return None
   
    @property
    def data(self):
        """ A masked array of data values : numpy.ma.MaskedArray.

        The DataArray constructor postpones reading data from FITS files until
        they are first used. Read the data array here if not already read.

        """
        #res = SharedMaskArray(self._data, mask=self._mask, copy=False)
        res = ma.MaskedArray(self._data, mask=self._mask, copy=False)
        res._sharedmask = False
        return res
    

    @data.setter
    def data(self, value):
        if self.shape is not None and not np.array_equal(value.shape, self.shape):
            raise ValueError('try to set data with an array with a different shape')
        if isinstance(value, ma.MaskedArray):
            self._data = value.data
            self._mask = value.mask
        else:
            self._data = value
            if self._mask is not ma.nomask:
                self._mask = ~(np.isfinite(value))

    @property
    def var(self):
        """ Either None, or a numpy.ndarray containing the variances
        of each data value. This has the same shape as the data array.
        """
        if self._var is None:
            return None
        else:
            return ma.MaskedArray(self._var, mask=self._mask, copy=False)

    @var.setter
    def var(self, value):
        if value is not None:
            if self.shape is not None and  not np.array_equal(value.shape, self.shape):
                raise ValueError('try to set var with an array with a different shape')
            if isinstance(value, ma.MaskedArray):
                self._var = value.data
                self._mask |= value.mask
            else:
                value = np.asarray(value)
                self._var = value
        else:
            self._var_ext = None
            self._var = value

    @deprecated('Variance should now be set with the .var attribute')
    def set_var(self, var):
        """Deprecated: The variance array can simply be assigned to
        the .var attribute"""
        self.var = var

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        # By default, if mask=False create a mask array with False values.
        # numpy.ma does it but with a np.resize/np.concatenate which cause a
        # huge memory peak, so a workaround is to create the mask here.
        # Also we force the creation of a mask array because currently many
        # method in MPDAF expect that the mask is an array and will not work
        # with np.ma.nomask. But nomask can still be used explicitly for
        # specific cases.
        if self.shape is not None and not np.array_equal(value.shape, self.shape):
            raise ValueError('try to set mask with an array with a different shape')
        if value is ma.nomask:
            self._mask = value
        else:
            self._mask = np.asarray(value, dtype=bool)

    def copy(self):
        """Return a copy of the object."""
        return self.__class__(
            filename=self.filename, data=self._data, mask=self._mask,
            var=self._var, unit=self.unit,
             wcs=self.wcs, wave=self.wave, copy=True,
            data_header=fits.Header(self.data_header),
            primary_header=fits.Header(self.primary_header),
            ext= (self._data_ext, self._var_ext), dtype=self.dtype)

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
            data, mask, var = self._read_from_file(item)
            
        else:
            data = self._data[item]
            if self._mask is ma.nomask:
                mask = ma.nomask
            else:
                mask = self._mask[item]
            if self._var is None:
                var = None
            else:
                var = self._var[item]

        if data.ndim == 0:
            return data

        # Construct new WCS and wavelength coordinate information for the slice
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

        return self.__class__(
            data=data, unit=self.unit, var=var, mask=mask, wcs=wcs, wave=wave,
            filename=self.filename, data_header=fits.Header(self.data_header),
            primary_header=fits.Header(self.primary_header))

    def __setitem__(self, item, other):
        """Set the corresponding part of data."""
        if self._data is None:
            raise ValueError('empty data array')

        if isinstance(other, DataArray):
            # FIXME: check only step
            
            if self._has_wave and other._has_wave \
                    and not np.allclose(self.wave.get_step(),
                                    other.wave.get_step(unit=self.wave.unit),
                                    atol=1E-2, rtol=0):
                raise ValueError('Operation forbidden for cubes with different'
                                 ' world coordinates in spectral direction')
            if self._has_wcs and other._has_wcs \
                    and not np.allclose(self.wcs.get_step(),
                                    other.wcs.get_step(unit=self.wcs.unit),
                                    atol=1E-3, rtol=0):
                raise ValueError('Operation forbidden for cubes with different'
                                 ' world coordinates in spatial directions')

            if self.unit == other.unit:
                if self._var is not None and other._var is not None:
                    self._var[item] = other._var
                other = other.data
            else:
                if self._var is not None and other._var is not None:
                    self._var[item] = UnitArray(other._var,
                                                other.unit**2, self.unit**2)
                other = UnitMaskedArray(other.data, other.unit, self.unit)
                
        if isinstance(other, ma.MaskedArray):
            self._data[item] = other.data
            if self._mask is ma.nomask:
                self._mask = np.zeros(self.shape, dtype=bool)
            self._mask[item] = other.mask
        else:
            self._data[item] = other
            if self._mask is not ma.nomask:
                self._mask[item] = ~(np.isfinite(other))
            
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
        out : `astropy.io.fits.ImageHDU`

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
        out : `astropy.io.fits.ImageHDU`

        """
        if self._var is None:
            return None

        if self._var.dtype == np.float64:
            # Force var to be stored in float instead of double
            var = self._var.astype(np.float32)
        else:
            var = self._var

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
        if self._var is not None:
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
        out : `mpdaf.obj.DataArray`, optional
            Array of the same shape as input, into which the output is placed.
            By default, a new array is created.

        """
        if self._data is None:
            raise ValueError('empty data array')

        if out is None:
            out = self.clone()

        out.data = np.ma.sqrt(self.data)
        out.unit = self.unit / u.Unit(np.sqrt(self.unit.scale))

        # Modify the variances to account for the effect of the square root.

        if self._var is not None:
            # For a value x, picked from a distribution of
            # variance, vx, the expected variance of sqrt(x), is:
            #
            #  vs = (d[sqrt(x)]/dx)**2 * vx
            #     = (0.5 / sqrt(x))**2 * vx
            #     = 0.25 / x * vx.
            out._var = 0.25 * self._var / self._data
        return out

    def abs(self, out=None):
        """Return a new object with the absolute value of the data.

        Parameters
        ----------
        out : `mpdaf.obj.DataArray`, optional
            Array of the same shape as input, into which the output is placed.
            By default, a new array is created.

        """
        if out is None:
            out = self.clone()

        out.data = np.ma.abs(self.data)
        if self._var is not None:
            out._var = self._var.copy()
        return out

    def unmask(self):
        """Unmask the data (just invalid data (nan,inf) are masked)."""
        self._mask = ~np.isfinite(self._data)

    def mask_variance(self, threshold):
        """Mask pixels with a variance above a threshold value.

        Parameters
        ----------
        threshold : float
            Threshold value.

        """
        if self._var is None:
            raise ValueError('Operation forbidden without variance extension.')
        self.data[self._var > threshold] = ma.masked

    def mask_selection(self, ksel):
        """Mask selected pixels.

        Parameters
        ----------
        ksel : output of np.where
            Elements depending on a condition

        """
        self.data[ksel] = ma.masked

    def crop(self):
        """Reduce the size of the array to the smallest sub-array that
        keeps all unmasked pixels.

        This removes any margins around the array that only contain masked
        pixels. If all pixels are masked in the input cube, the data and
        variance arrays are deleted.

        Returns
        -------
        item : list of slices
            The slices that were used to extract the sub-array.

        """
        if self._data is None:
            return

        nmasked = ma.count_masked(self.data)
        if nmasked == 0:
            return
        elif nmasked == np.prod(self.shape):
            # If all pixels are masked, simply delete data and variance
            self._data = None
            self._var = None
            return

        # Determine the ranges of indexes along each axis that encompass all of
        # the unmasked pixels, and convert this to slice prescriptions for
        # selecting the corresponding sub-array.
        dimensions = list(range(self.ndim))
        item = []
        for dim in dimensions:
            other_dims = dimensions[:]
            other_dims.remove(dim)
            mask = np.apply_over_axes(np.logical_and.reduce, self.data.mask,
                                      other_dims).ravel()
            ksel = np.where(~mask)[0]
            item.append(slice(ksel[0], ksel[-1] + 1, None))

        self._data = self._data[item]
        if self._var is not None:
            self._var = self._var[item]
        self._mask = self._mask[item]

        # Adjust the world-coordinates to match the image slice.
        if self._has_wcs:
            try:
                if self.ndim == 2:
                    self.wcs = self.wcs[item]
                else:
                    self.wcs = self.wcs[item[1:]]
            except:
                self.wcs = None
                self._logger.warning('wcs not copied, attribute set to None',
                                     exc_info=True)

        # Adjust the wavelength coordinates to match the spectral slice.
        if self._has_wave:
            try:
                self.wave = self.wave[item[0]]
            except:
                self.wave = None
                self._logger.warning('wavelength solution not copied: '
                                     'attribute set to None', exc_info=True)

        return item
