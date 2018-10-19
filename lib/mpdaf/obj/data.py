"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c) 2015-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import logging
import numpy as np
import warnings

from astropy import units as u
from astropy.io import fits
from datetime import datetime
from numpy import ma

from .coords import WCS, WaveCoord, determine_refframe
from .objs import UnitMaskedArray, UnitArray, is_int
from ..tools import (MpdafUnitsWarning, fix_unit_read, is_valid_fits_file,
                     copy_header, read_slice_from_fits)

__all__ = ('DataArray', )


class LazyData(object):

    def __init__(self, label):
        self.label = label

    def read_data(self, obj):
        obj_dict = obj.__dict__
        data, mask = read_slice_from_fits(obj.filename, ext=obj._data_ext,
                                          mask_ext='DQ', dtype=obj.dtype,
                                          convert_float64=obj._convert_float64)
        if mask is None:
            mask = ~(np.isfinite(data))
        obj_dict['_data'] = data
        obj_dict['_mask'] = mask
        obj._loaded_data = True
        return mask if self.label == '_mask' else data

    def __get__(self, obj, owner=None):
        try:
            return obj.__dict__[self.label]
        except KeyError:
            if obj.filename is None:
                return

            if self.label in ('_data', '_mask'):
                val = self.read_data(obj)

            if self.label == '_var':
                if obj._var_ext is None:
                    return None
                # Make sure that data is read because the mask may be needed
                if not obj._loaded_data:
                    self.read_data(obj)
                val, _ = read_slice_from_fits(
                    obj.filename, ext=obj._var_ext, dtype=obj._var_dtype,
                    convert_float64=obj._convert_float64)
                obj.__dict__[self.label] = val
            return val

    def __set__(self, obj, val):
        label = self.label
        if label == '_data':
            obj._loaded_data = True
        elif (val is not None and val is not np.ma.nomask and
                obj.shape is not None and
                not np.array_equal(val.shape, obj.shape)):
            raise ValueError("Can't set %s with a different shape" % label)
        obj.__dict__[label] = val


class DataArray(object):

    """Parent class for `~mpdaf.obj.Cube`, `~mpdaf.obj.Image` and
    `~mpdaf.obj.Spectrum`.

    Its primary purpose is to store pixel values and, optionally, also
    variances in masked Numpy arrays. For Cube objects these are 3D arrays
    indexed in the order ``[wavelength, image_y, image_x]``. For Image objects
    they are 2D arrays indexed in the order ``[image_y, image_x]``. For
    Spectrum objects they are 1D arrays.

    Image arrays hold flat 2D map-projections of the sky. The X and Y axes of
    the image arrays are orthogonal on the sky at the tangent point of the
    projection. When the rotation angle of the projection on the sky is zero,
    the Y axis of the image arrays is along the declination axis, and the
    X axis is perpendicular to this, with the positive X axis pointing east.

    Given a DataArray object ``obj``, the data and variance arrays are
    accessible via the properties ``obj.data`` and ``obj.var``. These two
    masked arrays share a single array of boolean masking elements, which is
    also accessible as a simple boolean array via the ``obj.mask property``.
    The shared mask can be modified through any of the three properties::

        obj.data[20:22] = numpy.ma.masked

    Is equivalent to::

        obj.var[20:22] = numpy.ma.masked

    And is also equivalent to::

        obj.mask[20:22] = True

    All three of the above statements mask pixels 20 and 21 of the data and
    variance arrays, without changing their values.

    Similarly, if one performs an operation on either ``obj.data`` or
    ``obj.var`` that modifies the mask, this is reflected in the shared mask of
    all three properties. For example, the following statement multiplies
    elements 20 and 21 of the data by 2.0, while changing the shared mask of
    these pixels to True. In this way the data and the variances of these
    pixels are consistently masked::

        obj.data[20:22] *= numpy.ma.array([2.0,2.0], mask=[True,True])

    The data and variance arrays can be completely replaced by assigning new
    arrays to the ``obj.data`` and ``obj.var`` properties, but these must have
    the same shape as before (ie. obj.shape).  New arrays that are assigned to
    ``obj.data`` or ``obj.var`` can be simple numpy arrays, or a numpy masked
    arrays.

    When a normal numpy array is assigned to ``obj.data``, the ``obj.mask``
    array is also assigned a mask array, whose elements are True wherever NaN
    or Inf values are found in the data array. An exception to this rule is if
    the mask has previously been disabled by assigning ``numpy.ma.nomask`` to
    ``obj.mask``. In this case a mask array is not constructed.

    When a numpy masked array is assigned to ``obj.data``, then its mask is
    also assigned, unchanged, to ``obj.mask``.

    Assigning a normal numpy array to the ``obj.var`` attribute, has no effect
    on the contents of ``obj.mask``. On the other hand, when a numpy masked
    array is assigned to ``obj.var`` the ``obj.mask`` array becomes the union
    of its current value and the mask of the provided array.

    The ability to record variances for each element is optional.  When no
    variances are stored, ``obj.var`` has the value None. To discard an
    unwanted variance array, None can be subsequently assigned to ``obj.var``.

    For cubes and spectra, the wavelengths of the spectral pixels are specified
    in the ``.wave`` member. For cubes and images, the world-coordinates of the
    image pixels are specified in the ``.wcs`` member.

    When a DataArray object is constructed from a FITS file, the name of the
    file and the file's primary header are recorded. If the data are read from
    a FITS extension, the header of this extension is also recorded.
    Alternatively, the primary header and data header can be passed to the
    DataArray constructor.  Where FITS headers are neither provided, nor
    available in a provided FITS file, generic headers are substituted.

    Methods are provided for masking and unmasking pixels, and performing basic
    arithmetic operations on pixels. Operations that are specific to cubes or
    spectra or images are provided elsewhere by derived classes.


    Parameters
    ----------
    filename : str
        FITS file name, default to None.
    hdulist : `astropy.fits.HDUList`
        HDUList object, can be used instead of ``filename`` to avoid opening
        the FITS file multiple times.
    data : numpy.ndarray or list
        Data array, passed to `numpy.ma.MaskedArray`.
    mask : bool or numpy.ma.nomask or numpy.ndarray
        Mask used for the creation of the ``.data`` MaskedArray. If mask is
        False (default value), a mask array of the same size of the data array
        is created. To avoid creating an array, it is possible to use
        numpy.ma.nomask, but in this case several methods will break if
        they use the mask.
    var : numpy.ndarray or list
        Variance array, passed to `numpy.ma.MaskedArray`.
    ext : int or (int,int) or str or (str,str)
        Number/name of the data extension or numbers/names of the data and
        variance extensions.
    unit : `astropy.units.Unit`
        Physical units of the data values, default to u.dimensionless_unscaled.
    copy : bool
        If True (default), then the data and variance arrays are copied.
        Passed to `numpy.ma.MaskedArray`.
    dtype : numpy.dtype
        Type of the data. Passed to `numpy.ma.MaskedArray`.
    primary_header : `astropy.io.fits.Header`
        FITS primary header instance.
    data_header : `astropy.io.fits.Header`
        FITS data header instance.
    fits_kwargs : dict
        Additional arguments that can be passed to `astropy.io.fits.open`.
    convert_float64: bool
        By default input arrays or FITS data are converted to float64, in
        order to increase precision to the detriment of memory usage.

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

    _data = LazyData('_data')
    _mask = LazyData('_mask')
    _var = LazyData('_var')

    def __init__(self, filename=None, hdulist=None, data=None, mask=False,
                 var=None, ext=None, unit=u.dimensionless_unscaled, copy=True,
                 dtype=None, primary_header=None, data_header=None,
                 convert_float64=True, **kwargs):
        self._logger = logging.getLogger(__name__)

        self._loaded_data = False
        self._data_ext = None
        self._var_ext = None
        self._convert_float64 = convert_float64

        self.filename = filename
        self.wcs = None
        self.wave = None
        self._dtype = dtype
        self._var_dtype = np.float64 if convert_float64 else None
        self.unit = unit
        self.data_header = data_header or fits.Header()
        self.primary_header = primary_header or fits.Header()

        if (filename is not None or hdulist is not None) and data is None:
            # Read the data from a FITS file
            if not hdulist and not is_valid_fits_file(filename):
                raise IOError('Invalid file: %s' % filename)

            if hdulist is None:
                fits_kwargs = kwargs.pop('fits_kwargs', {})
                hdulist = fits.open(filename, **fits_kwargs)
                close_hdu = True
            else:
                if filename is None:
                    self.filename = hdulist.filename()
                close_hdu = False

            # Find the hdu of the data. This is either the primary HDU (if the
            # number of extension is 1) or a DATA or SCI extension. Also see if
            # there is an extension that contains variances. This is either
            # a STAT extension, or the second of a tuple of extensions passed
            # via the ext[] parameter.
            if len(hdulist) == 1:
                self._data_ext = 0
            elif ext is None:
                if 'DATA' in hdulist:
                    self._data_ext = 'DATA'
                elif 'SCI' in hdulist:
                    self._data_ext = 'SCI'
                else:
                    raise IOError('No DATA or SCI extension found.\n'
                                  'Please use the `ext` parameter to specify '
                                  'which extension must be loaded.')

                if 'STAT' in hdulist:
                    self._var_ext = 'STAT'
            elif isinstance(ext, (list, tuple, np.ndarray)):
                self._data_ext, self._var_ext = ext
            elif isinstance(ext, (int, str)):
                self._data_ext = ext
                self._var_ext = None

            self.primary_header = hdulist[0].header
            self.data_header = hdr = hdulist[self._data_ext].header

            if (self._ndim_required is not None and
                    self._ndim_required != self.data_header['NAXIS']):
                raise ValueError('{} class cannot manage data with {} axes'
                                 .format(self.__class__.__name__,
                                         self.data_header['NAXIS']))

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

            self._compute_wcs_from_header()

            if close_hdu:
                hdulist.close()

        else:
            if mask is ma.nomask:
                self._mask = mask

            # Use a specified numpy data array?
            if data is not None:
                # Force data to be in double instead of float
                if (self._dtype is None and data.dtype.type == np.float32 and
                        convert_float64):
                    self._dtype = np.float64

                if isinstance(data, ma.MaskedArray):
                    self._data = np.array(data.data, dtype=self.dtype,
                                          copy=copy)
                    if data.mask is ma.nomask:
                        self._mask = data.mask
                    else:
                        self._mask = np.array(data.mask, dtype=bool, copy=copy)
                else:
                    self._data = np.array(data, dtype=self.dtype, copy=copy)
                    if mask is None or mask is False:
                        self._mask = ~(np.isfinite(data))
                    elif mask is True:
                        self._mask = np.ones(shape=data.shape, dtype=bool)
                    elif mask is not ma.nomask:
                        self._mask = np.array(mask, dtype=bool, copy=copy)

            # Use a specified variance array?
            if var is not None:
                if isinstance(var, ma.MaskedArray):
                    self._var = np.array(var.data, dtype=self._var_dtype,
                                         copy=copy)
                    self._mask |= var.mask
                else:
                    self._var = np.array(var, dtype=self._var_dtype, copy=copy)

        # Where WCS and/or wavelength objects are specified as optional
        # parameters, install them.
        self.set_wcs(wcs=kwargs.pop('wcs', None),
                     wave=kwargs.pop('wave', None))

    def __getstate__(self):
        state = self.__dict__.copy()

        # remove un-pickable objects
        state['_logger'] = None
        state['wcs'] = None
        state['wave'] = None
        if '_spflims' in state and state['_spflims'] is not None:
            state['_spflims'] = None

        # Image.plot stores the matplotlib axes on the object, but it is not
        # pickable. Delete it!
        if '_ax' in state:
            del state['_ax']

        # Try to determine if the object has some wcs/wave information but not
        # available in the FITS header. In this case we add the wcs info in the
        # data header.
        if (self.data_header.get('WCSAXES') is None and
                ((self._has_wcs and self.wcs is not None) or
                 (self._has_wave and self.wave is not None))):
            hdu = self.get_data_hdu(convert_float32=False)
            state['data_header'] = hdu.header

        return state

    def __setstate__(self, state):
        # set attributes on the object, making sure that _data, _mask and _var
        # are set last as these are descriptors and need the other attributes.
        # Also these attributes may bot exists yet if the data is not loaded.
        for slot, value in state.items():
            if slot not in ('_data', '_mask', '_var'):
                setattr(self, slot, value)
        for slot in ('_data', '_mask', '_var'):
            if slot in state:
                setattr(self, slot, state[slot])

        self._logger = logging.getLogger(__name__)
        # Recreate the wcs/wave objects from the fits headers
        self._compute_wcs_from_header()
        # and to get the naxis1/naxis2 right
        self.set_wcs(wcs=self.wcs, wave=self.wave)

    def _compute_wcs_from_header(self):
        frame, equinox = determine_refframe(self.primary_header)
        hdr = self.data_header

        # Is this a derived class like Cube/Image that require WCS information?
        if self._has_wcs:
            try:
                self.wcs = WCS(hdr, frame=frame, equinox=equinox)
            except fits.VerifyError as e:
                # Workaround for
                # https://github.com/astropy/astropy/issues/887
                self._logger.warning(e)
                if 'IRAF-B/P' in hdr:
                    hdr.remove('IRAF-B/P')
                self.wcs = WCS(hdr, frame=frame, equinox=equinox)

        # Get the wavelength coordinates.
        wave_ext = 1 if self._ndim_required == 1 else 3
        crpix = 'CRPIX{}'.format(wave_ext)
        crval = 'CRVAL{}'.format(wave_ext)
        if self._has_wave and crpix in hdr and crval in hdr:
            self.wave = WaveCoord(hdr)

    @classmethod
    def new_from_obj(cls, obj, data=None, var=None, copy=False):
        """Create a new object from another one, copying its attributes.

        Parameters
        ----------
        obj  : `mpdaf.obj.DataArray`
            The object to use as the template for the new object. This
            should be an object based on DataArray, such as an Image,
            Cube or Spectrum.
        data : ndarray-like
            An optional data array, or None to indicate that
            ``obj.data`` should be used. The default is None.
        var : ndarray-like
            An optional variance array, or None to indicate that
            ``obj.var`` should be used, or False to indicate that the
            new object should not have any variances. The default
            is None.
        copy : bool
            Copy the data and variance arrays if True (default False).

        """
        data = obj.data if data is None else data
        if var is None:
            var = obj._var
        elif var is False:
            var = None
        kwargs = dict(filename=obj.filename, data=data, unit=obj.unit, var=var,
                      dtype=obj.dtype, copy=copy,
                      ext=(obj._data_ext, obj._var_ext),
                      data_header=obj.data_header.copy(),
                      primary_header=obj.primary_header.copy())
        if cls._has_wcs:
            kwargs['wcs'] = obj.wcs
        if cls._has_wave:
            kwargs['wave'] = obj.wave
        return cls(**kwargs)

    @property
    def ndim(self):
        """ The number of dimensions in the data and variance arrays : int """
        if self._loaded_data:
            return self._data.ndim
        try:
            return self.data_header['NAXIS']
        except KeyError:
            return None

    @property
    def shape(self):
        """The lengths of each of the data axes."""
        if self._loaded_data:
            return self._data.shape
        try:
            return tuple(self.data_header['NAXIS%d' % i]
                         for i in range(self.ndim, 0, -1))
        except (KeyError, TypeError):
            return None

    @property
    def dtype(self):
        """The type of the data."""
        if self._loaded_data:
            return self._data.dtype
        else:
            return self._dtype

    @property
    def data(self):
        """Return data as a `numpy.ma.MaskedArray`.

        The DataArray constructor postpones reading data from FITS files until
        they are first used. Read the data array here if not already read.

        Changes can be made to individual elements of the data property. When
        simple numeric values or Numpy array elements are assigned to elements
        of the data property, the values of these elements are updated and
        become unmasked.

        When masked Numpy values or masked-array elements are assigned to
        elements of the data property, then these change both the values of the
        data property and the shared mask of the data and var properties.

        Completely new arrays can also be assigned to the data property,
        provided that they have the same shape as before.

        """
        res = ma.MaskedArray(self._data, mask=self._mask, copy=False)
        res._sharedmask = False
        return res

    @data.setter
    def data(self, value):
        # Handle this case specifically for .data, since it is already done for
        # ._var and ._mask, but ._data can be used to change the shape
        if self.shape is not None and \
                not np.array_equal(value.shape, self.shape):
            raise ValueError('try to set data with an array with a different '
                             'shape')

        if isinstance(value, ma.MaskedArray):
            self._data = value.data
            self._mask = value.mask
        else:
            self._data = value
            self._mask = ~(np.isfinite(value))

    @property
    def var(self):
        """Return variance as a `numpy.ma.MaskedArray`.

        If variances have been provided for each data pixel, then this property
        can be used to record those variances. Normally this is a masked array
        which shares the mask of the data property. However if no variances
        have been provided, then this property is None.

        Variances are typically provided along with the data values in the
        originating FITS file. Alternatively a variance array can be assigned
        to this property after the data have been read.

        Note that any function that modifies the contents of the data array may
        need to update the array of variances accordingly.  For example, after
        scaling pixel values by a constant factor c, the variances should be
        scaled by c**2.

        When masked-array values are assigned to elements of the var property,
        the mask of the new values is assigned to the shared mask of the data
        and variance properties.

        Completely new arrays can also be assigned to the var property. When
        a masked array is assigned to the var property, its mask is combined
        with the existing shared mask, rather than replacing it.

        """
        if self._var is None:
            return None
        else:
            res = ma.MaskedArray(self._var, mask=self._mask, copy=False)
            res._sharedmask = False
            return res

    @var.setter
    def var(self, value):
        if value is not None:
            if isinstance(value, ma.MaskedArray):
                self._var = value.data
                self._mask |= value.mask
            else:
                self._var = np.asarray(value)
        else:
            self._var_ext = None
            self._var = value

    @property
    def mask(self):
        """The shared masking array of the data and variance arrays.

        This is a bool array which has the same shape as the data and variance
        arrays. Setting an element of this array to True, flags the
        corresponding pixels of the data and variance arrays, so that they
        don't take part in subsequent calculations. Reverting this element to
        False, unmasks the pixel again.

        This array can be modified either directly by assignments to elements
        of this property or by modifying the masks of the .data and .var
        arrays. An entirely new mask array can also be assigned to this
        property, provided that it has the same shape as the data array.

        """
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
        if value is ma.nomask:
            self._mask = value
        else:
            self._mask = np.asarray(value, dtype=bool)

    def copy(self):
        """Return a copy of the object."""
        return self.__class__.new_from_obj(self, copy=True)

    def clone(self, data_init=None, var_init=None):
        """Return a shallow copy with the same header and coordinates.

        Optionally fill the cloned array using values returned by provided
        functions.

        Parameters
        ----------
        data_init : function
            An optional function to use to create the data array
            (it takes the shape as parameter). For example ``np.zeros``
            or ``np.empty`` can be used. It defaults to None, which results
            in the data attribute being None.
        var_init : function
            An optional function to use to create the variance array,
            with the same specifics as data_init. This default to None,
            which results in the var attribute being assigned None.

        """
        # Create a new data_header with correct NAXIS keywords because an
        # object without data relies on this to get the shape
        hdr = copy_header(self.data_header, self.get_wcs_header(),
                          exclude=('CD*', 'PC*', 'CDELT*', 'CRPIX*', 'CRVAL*',
                                   'CSYER*', 'CTYPE*', 'CUNIT*', 'NAXIS*',
                                   'RADESYS', 'LATPOLE', 'LONPOLE'),
                          unit=self.unit)
        hdr['NAXIS'] = self.ndim
        for i in range(1, self.ndim + 1):
            hdr['NAXIS%d' % i] = self.shape[-i]

        return self.__class__(
            unit=self.unit, dtype=None, copy=False,
            data=None if data_init is None else data_init(self.shape,
                                                          dtype=self.dtype),
            var=None if var_init is None else var_init(self.shape,
                                                       dtype=self.dtype),
            wcs=None if self.wcs is None else self.wcs,
            wave=None if self.wave is None else self.wave,
            data_header=hdr,
            primary_header=self.primary_header.copy()
        )

    def __repr__(self):
        fmt = """<{}(shape={}, unit='{}', dtype='{}')>"""
        return fmt.format(self.__class__.__name__, self.shape,
                          self.unit.to_string(), self.dtype)

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
        var = None
        if self._loaded_data:
            data = self._data[item]
            mask = self._mask
            if mask is not ma.nomask:
                mask = mask[item]
            if self._var is not None:
                var = self._var[item]
        elif self.filename is not None:
            with fits.open(self.filename) as hdu:
                data, mask = read_slice_from_fits(
                    hdu, ext=self._data_ext, mask_ext='DQ', dtype=self.dtype,
                    item=item, convert_float64=self._convert_float64)
                if self._var_ext is not None:
                    var = read_slice_from_fits(
                        hdu, ext=self._var_ext, dtype=self._var_dtype,
                        convert_float64=self._convert_float64, item=item)[0]
                if mask is None:
                    mask = ~(np.isfinite(data))
        else:
            raise ValueError('empty data array')

        if data.ndim == 0:
            return data

        # If the data, mask and variance arrays need to be reshaped to
        # reintroduce a single-pixel dimension, the following variable will be
        # assigned the required shape.
        reshape = None

        # Construct new WCS and wavelength coordinate information for the slice
        wave = None
        wcs = None

        # Slice a Cube?
        if self.ndim == 3:

            # Handle cube[ii,jj,kk], where ii, jj, kk can be int or slice
            # objects.
            if isinstance(item, (list, tuple)) and len(item) == 3:
                try:
                    wcs = self.wcs[item[1], item[2]]
                except Exception:
                    wcs = None
                try:
                    wave = self.wave[item[0]]
                except Exception:
                    wave = None

                # If x's slice is selected with an integer, and y's slice
                # is selected with a slice object (or vice versa), then
                # numpy will have removed one dimension of the resulting
                # image or cube. Compute the shape that is needed to
                # reinstate this dimension as an axis with one element.
                if isinstance(item[1], int) != isinstance(item[2], int):
                    if isinstance(item[0], int):  # An Image has been selected
                        if isinstance(item[1], int):
                            reshape = (1, data.shape[0])
                        else:
                            reshape = (data.shape[0], 1)
                    else:                        # A Cube has been selected
                        if isinstance(item[1], int):
                            reshape = (data.shape[0], 1, data.shape[1])
                        else:
                            reshape = (data.shape[0], data.shape[1], 1)

            # Handle cube[ii,jj], where ii and jj can be int or slice objects.
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    wcs = self.wcs[item[1], slice(None)]
                except Exception:
                    wcs = None
                try:
                    wave = self.wave[item[0]]
                except Exception:
                    wave = None

                # If the Y-axis slice has been specified as an int, then
                # the Y-axis dimension will have been removed by numpy.
                # Compute the shape that is needed to reinstante it as
                # an axis with one pixel.
                if isinstance(item[1], int):
                    if isinstance(item[0], int):  # An Image has been selected
                        reshape = (1, data.shape[0])
                    else:                        # A Cube has been selected
                        reshape = (data.shape[0], 1, data.shape[1])

            # Handle cube[ii] where ii can be an int or a slice.
            elif isinstance(item, (int, slice)):
                try:
                    wcs = self.wcs.copy()
                except Exception:
                    wcs = None
                try:
                    wave = self.wave[item]
                except Exception:
                    wave = None

            # Handle cube[]
            elif item is None or item is ():
                try:
                    wcs = self.wcs.copy()
                except Exception:
                    wcs = None
                try:
                    wave = self.wave.copy()
                except Exception:
                    wave = None

        # Slice an Image?
        elif self.ndim == 2:

            # Handle image[ii,jj], where ii and jj can be int or slice objects.
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    wcs = self.wcs[item]
                except Exception:
                    wcs = None

                # If x's slice is selected with an integer, and y's slice
                # is selected with a slice object (or vice versa), then
                # numpy will have removed one dimension of the resulting
                # image or cube. Compute the shape that is needed to
                # reinstate this dimension as an axis with one element.
                if isinstance(item[0], int) != isinstance(item[1], int):
                    if isinstance(item[0], int):
                        reshape = (1, data.shape[0])
                    else:
                        reshape = (data.shape[0], 1)

            # Handle image[ii], where ii be an int or a slice object.
            elif isinstance(item, (int, slice)):
                try:
                    wcs = self.wcs[item, slice(None)]
                except Exception:
                    wcs = None

                # If item is an int, then numpy will have removed
                # the x-axis dimension from data[]. Compute the
                # shape that is needed to reinstate it.
                if isinstance(item, int):
                    reshape = (1, data.shape[0])

            # Handle image[]
            elif item is None or item is ():
                try:
                    wcs = self.wcs.copy()
                except Exception:
                    wcs = None

        # Slice a Spectrum?
        elif self.ndim == 1:

            # Handle spectrum[item]
            if isinstance(item, slice):
                try:
                    wave = self.wave[item]
                except Exception:
                    wave = None

            # Handle spectrum[]
            elif item is None or item is ():
                try:
                    wave = self.wave.copy()
                except Exception:
                    wave = None

        # Reshape the data, variance and mask arrays, if necessary.

        if reshape is not None:
            data = data.reshape(reshape)
            if mask is not ma.nomask:
                mask = mask.reshape(reshape)
            if var is not None:
                var = var.reshape(reshape)

        return self.__class__(
            data=data, unit=self.unit, var=var, mask=mask, wcs=wcs, wave=wave,
            filename=self.filename, data_header=self.data_header.copy(),
            primary_header=self.primary_header.copy(), copy=False)

    def __setitem__(self, item, other):
        """Set the corresponding part of data."""
        if self._data is None:
            raise ValueError('empty data array')

        if isinstance(other, DataArray):
            # FIXME: check only step

            if self._has_wave and other._has_wave and \
                    not np.allclose(self.wave.get_step(),
                                    other.wave.get_step(unit=self.wave.unit),
                                    atol=1E-2, rtol=0):
                raise ValueError('Operation forbidden for cubes with different'
                                 ' world coordinates in spectral direction')
            if self._has_wcs and other._has_wcs and \
                    not np.allclose(self.wcs.get_step(),
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

        self.data[item] = other

    def get_wcs_header(self):
        """Return a FITS header containing coordinate descriptions."""
        if self.ndim == 1 and self.wave is not None:
            return self.wave.to_header()
        elif self.ndim == 2 and self.wcs is not None:
            return self.wcs.to_header()
        elif self.ndim == 3 and self.wcs is not None:
            return self.wcs.to_cube_header(self.wave)

    def get_data_hdu(self, name='DATA', savemask='dq', convert_float32=True):
        """Return an ImageHDU corresponding to the DATA extension.

        Parameters
        ----------
        name : str
            Extension name, DATA by default.
        savemask : str
            If 'dq', the mask array is saved in a DQ extension.
            If 'nan', masked data are replaced by nan in a DATA extension.
            If 'none', masked array is not saved.
        convert_float32: bool
            By default float64 arrays are converted to float32, in order to
            produce smaller files.

        Returns
        -------
        out : `astropy.io.fits.ImageHDU`

        """
        if convert_float32 and self._data.dtype == np.float64:
            # Force data to be stored in float instead of double
            data = self.data.astype(np.float32)
        else:
            data = self.data

        # create DATA extension
        if savemask == 'nan' and ma.count_masked(data) > 0:
            # NaNs can be used only for float arrays, so we raise an exception
            # if there are masked values in a non-float array.
            if not np.issubdtype(data.dtype, np.floating):
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

    def get_stat_hdu(self, name='STAT', header=None, convert_float32=True):
        """Return an ImageHDU corresponding to the STAT extension.

        Parameters
        ----------
        name : str
            Extension name, STAT by default.
        header: fits.Header
            Fits Header to put in the extension, typically to reuse the same as
            in the DATA extension. Otherwise it is created with the wcs.
        convert_float32: bool
            By default float64 arrays are converted to float32, in order to
            produce smaller files.

        Returns
        -------
        out : `astropy.io.fits.ImageHDU`

        """
        if self._var is None:
            return None

        if convert_float32 and self._var.dtype == np.float64:
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

    def write(self, filename, savemask='dq', checksum=False,
              convert_float32=True):
        """Save the data to a FITS file.

        Overwrite the file if it exists.

        Parameters
        ----------
        filename : str
            The FITS filename.
        savemask : str
            If 'dq', the mask array is saved in a ``DQ`` extension
            If 'nan', masked data are replaced by nan in the ``DATA`` extension
            If 'none', masked array is not saved
        checksum : bool
            If ``True``, adds both ``DATASUM`` and ``CHECKSUM`` cards to the
            headers of all HDU's written to the file.
        convert_float32: bool
            By default float64 arrays are converted to float32, in order to
            produce smaller files.

        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            header = copy_header(self.primary_header)

        header['date'] = (str(datetime.now()), 'creation date')
        header['author'] = ('MPDAF', 'origin of the file')
        hdulist = fits.HDUList([fits.PrimaryHDU(header=header)])

        # create cube DATA extension
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            datahdu = self.get_data_hdu(savemask=savemask,
                                        convert_float32=convert_float32)
        hdulist.append(datahdu)

        # create spectrum STAT extension
        if self._var is not None:
            hdulist.append(self.get_stat_hdu(header=datahdu.header.copy(),
                                             convert_float32=convert_float32))

        # create DQ extension
        if savemask == 'dq' and np.ma.count_masked(self.data) != 0:
            hdulist.append(fits.ImageHDU(
                name='DQ', header=datahdu.header.copy(),
                data=np.uint8(self.data.mask)))

        hdulist.writeto(filename, overwrite=True,
                        output_verify='silentfix', checksum=checksum)
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
        if self._mask is not ma.nomask:
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

        # numpy 1.15: indexing needs a tuple instead of list
        item = tuple(item)

        self._data = self._data[item]
        if self._var is not None:
            self._var = self._var[item]
        if self._mask is not ma.nomask:
            self._mask = self._mask[item]

        # Adjust the world-coordinates to match the image slice.
        if self._has_wcs:
            try:
                if self.ndim == 2:
                    self.wcs = self.wcs[item]
                else:
                    self.wcs = self.wcs[item[1:]]
            except Exception:
                self.wcs = None
                self._logger.warning('wcs not copied, attribute set to None',
                                     exc_info=True)

        # Adjust the wavelength coordinates to match the spectral slice.
        if self._has_wave:
            try:
                self.wave = self.wave[item[0]]
            except Exception:
                self.wave = None
                self._logger.warning('wavelength solution not copied: '
                                     'attribute set to None', exc_info=True)

        return item

    def set_wcs(self, wcs=None, wave=None):
        """Set the world coordinates (spatial and/or spectral where pertinent).

        Parameters
        ----------
        wcs : `mpdaf.obj.WCS`
            Spatial world coordinates. This argument is ignored when
            self is a Spectrum.
        wave : `mpdaf.obj.WaveCoord`
            Spectral wavelength coordinates. This argument is ignored when
            self is an Image.

        """

        # Install spatial world-corrdinates?
        # Note that we have to test the length of self.shape in
        # addition to _has_wcs, because of functions like
        # Cube.__getitem__(), which creates a temporary cube to hold a
        # spectrum before converting this to a Spectrum object.
        if self._has_wcs and wcs is not None and len(self.shape) > 1:
            try:
                self.wcs = wcs.copy()
                if self.shape is not None:
                    if (wcs.naxis1 != 0 and wcs.naxis2 != 0 and
                        (wcs.naxis1 != self.shape[-1] or
                         wcs.naxis2 != self.shape[-2])):
                        self._logger.warning(
                            'The world coordinates and data have different '
                            'dimensions. Modifying the shape of the WCS '
                            'object')
                    self.wcs.naxis1 = self.shape[-1]
                    self.wcs.naxis2 = self.shape[-2]
            except Exception:
                self._logger.warning('Unable to install world coordinates',
                                     exc_info=True)

        # Install spectral world coordinates?
        # Note that we have to test the length of self.shape in
        # addition to _has_wave, because of functions like
        # Cube.__getitem__(), which creates a temporary cube to hold a
        # image before converting this to an Image object.
        if self._has_wave and wave is not None and len(self.shape) != 2:
            try:
                self.wave = wave.copy()
                if self.shape is not None:
                    if wave.shape is not None and wave.shape != self.shape[0]:
                        self._logger.warning(
                            'The wavelength coordinates and data have '
                            'different dimensions. Modifying the shape of '
                            'the WaveCoord object')
                    self.wave.shape = self.shape[0]
            except Exception:
                self._logger.warning('Unable to install wavelength '
                                     'coordinates', exc_info=True)

    def _rebin(self, factor, margin='center', inplace=False):
        """Combine neighboring pixels to reduce the size by integer factors
        along each axis.

        This function is designed to be called by the rebin methods of
        Spectrum, Image and Cube.

        Each output pixel is the mean of n pixels, where n is the
        product of the reduction factors in the factor argument.

        Parameters
        ----------
        factor : int or (int,int,int)
            The integer reduction factors along the wavelength, z
            array axis, and the image y and x array axes,
            respectively. Python notation: (nz,ny,nx).
        margin : 'center', 'origin', 'left' or 'right'
            When the dimensions of the input array are not integer
            multiples of the reduction factor, the array is truncated
            to remove just enough pixels that its dimensions are
            multiples of the reduction factor. This subarray is then
            rebinned in place of the original array. The margin
            parameter determines which pixels of the input array are
            truncated, and which remain.

            The options are:
              'origin' or 'center':
                 The starts of the axes of the output array are
                 coincident with the starts of the axes of the input
                 array.
              'center':
                 The center of the output array is aligned with the
                 center of the input array, within one pixel along
                 each axis.
              'right':
                 The ends of the axes of the output array are
                 coincident with the ends of the axes of the input
                 array.
        inplace : bool
            If False, return a rebinned copy of the DataArray (the default).
            If True, rebin the original DataArray in-place, and return that.

        """

        # Change the input cube or change a copy of it?
        res = self if inplace else self.copy()

        # Reject unsupported margin modes.
        if margin not in ('center', 'origin', 'left', 'right'):
            raise ValueError('Unsuported margin parameter: %s' % margin)

        # Use the same reduction factor for all dimensions?
        if is_int(factor):
            factor = np.ones((res.ndim), dtype=int) * factor
        else:
            factor = np.asarray(factor)

        # The reduction factors must be in the range 1 to shape-1.
        if np.any(factor < 1) or np.any(factor >= res.shape):
            raise ValueError('The reduction factors must be from 1 to shape.')

        # Compute the number of pixels by which each axis dimension
        # exceeds being an integer multiple of its reduction factor.
        n = np.mod(res.shape, factor).astype(int)

        # If necessary, compute the slices needed to truncate the
        # dimensions to be integer multiples of the axis reduction
        # factors.
        if np.any(n != 0):

            # Add a slice for each axis to a list of slices.
            slices = []
            for k in range(res.ndim):
                # Compute the slice of axis k needed to truncate this axis.
                if margin == 'origin' or margin == 'left':
                    nstart = 0
                elif margin == 'center':
                    nstart = n[k] // 2
                elif margin == 'right':
                    nstart = n[k]
                slices.append(slice(nstart, res.shape[k] - n[k] + nstart))

            # If there is only one axis, extract the single slice from slices.
            if len(slices) == 1:
                slices = slices[0]
            else:
                slices = tuple(slices)

            # Get a sliced copy of the input object.
            tmp = res[slices]

            # Copy the sliced data back into res, so that inplace=True works.
            res._data = tmp._data
            res._var = tmp._var
            res._mask = tmp._mask
            res.wcs = tmp.wcs
            res.wave = tmp.wave

        # At this point the dimensions are integer multiples of
        # the reduction factors. What is the shape of the output image?
        newshape = res.shape // factor

        # Create a list of array dimensions that are composed of each
        # of the final dimensions of the array followed by the corresponding
        # axis reduction factor. Reshaping with these dimensions places all
        # of the pixels of each axis that are to be summed on its own axis.
        preshape = np.column_stack((newshape, factor)).ravel()

        # Compute the number of unmasked pixels of the input array
        # that will contribute to each mean pixel in the output array.
        unmasked = res.data.reshape(preshape).count(1)
        for k in range(2, res.ndim + 1):
            unmasked = unmasked.sum(k)

        # Reduce the size of the data array by taking the mean of
        # successive groups of 'factor[0] x factor[1]' pixels. Note
        # that the following uses np.ma.mean(), which takes account of
        # masked pixels.
        newdata = res.data.reshape(preshape)
        for k in range(1, res.ndim + 1):
            newdata = newdata.mean(k)
        res._data = newdata.data

        # The treatment of the variance array is complicated by the
        # possibility of masked pixels in the data array. A sum of N
        # data pixels p[i] of variance v[i] has a variance of
        # sum(v[i] / N^2), where N^2 is the number of unmasked pixels
        # in that particular sum.
        if res._var is not None:
            newvar = res.var.reshape(preshape)
            for k in range(1, res.ndim + 1):
                newvar = newvar.sum(k)
            newvar /= unmasked**2
            res._var = newvar.data

        # Any pixels in the output array that come from zero unmasked
        # pixels of the input array should be masked.
        res._mask = unmasked < 1

        # Update spatial world coordinates.
        if res._has_wcs and res.wcs is not None and res.ndim > 1:
            res.wcs = res.wcs.rebin([factor[-2], factor[-1]])

        # Update the spectral world coordinates.
        if res._has_wave and res.wave is not None and res.ndim != 2:
            res.wave.rebin(factor[0])

        return res

    def _convolve(self, convolution_function, other, inplace=False):
        """Convolve a DataArray with an array of the same number of dimensions
        using a specified convolution function.

        Masked values in self.data and self.var are replaced with
        zeros before the convolution is performed. However masked
        pixels in the input data remain masked in the output.

        Any variances in self.var are propagated correctly.

        If self.var exists, the variances are propagated using the equation::

            result.var = self.var (*) other**2

        where (*) indicates convolution. This equation can be derived
        by applying the usual rules of error-propagation to the
        discrete convolution equation.

        Uses `scipy.signal.convolve`.

        Parameters
        ----------
        fn : function
            The convolution function to use, chosen from:

            - `scipy.signal.fftconvolve`: This exploits the Fourier convolution
            theorem to perform the convolution via multiplication in the
            Fourier plane. It's speed scales as O(Nd x log(Nd)), where
            Nd=self.data.size.
            - `scipy.signal.convolve`: This uses the discrete convolution
            equation. It's speed scales as O(Nd x No), where Nd=self.data.size
            and No=other.data.size.

            In general fftconvolve() is faster than convolve() except when
            other.data only contains a few pixels. However fftconvolve uses
            a lot more memory than convolve(), so convolve() is sometimes the
            only reasonable choice. In particular, fftconvolve allocates two
            arrays whose dimensions are the sum of self.shape and other.shape,
            rounded up to a power of two. These arrays can be impractically
            large for some input data-sets.
        other : DataArray or np.ndarray
          The array with which to convolve the contents of self.  This must
          have the same number of dimensions as self, but it can have fewer
          elements. When this array contains a symmetric filtering function,
          the center of the function should be placed at the center of pixel,
          ``(other.shape - 1)//2``.

          Note that passing a DataArray object is equivalent to just
          passing its DataArray.data member. If it has any variances,
          these are ignored.
        inplace : bool
            If False (the default), return a new object containing the
            convolved array.
            If True, record the convolved array in self and return self.

        Returns
        -------
        out : `~mpdaf.obj.DataArray`

        """
        out = self if inplace else self.copy()

        if out.ndim != other.ndim:
            raise IOError('The other array must have the same rank as self')

        if np.any(np.asarray(other.shape) > np.asarray(self.shape)):
            raise IOError('The other array must be no larger than self')

        kernel = other.data if isinstance(other, DataArray) else other

        # Replace any masked pixels in the convolution kernel with zeros.
        if isinstance(kernel, ma.MaskedArray) and ma.count_masked(kernel) > 0:
            kernel = kernel.filled(0.0)

        # Replace any masked pixels in out._data with zeros
        masked = self._mask is not None and self._mask.sum() > 0
        if masked:
            out._data = out.data.filled(0.0)
        elif out._mask is None and ~np.isfinite(out._data.sum()):
            out._data = out._data.copy()
            out._data[~np.isfinite(out._data)] = 0.0

        out._data = convolution_function(out._data, kernel, mode="same")

        # Are there any variances to be propagated?
        if out._var is not None:
            # Replace any masked pixels in out._var with zeros
            if masked:
                out._var = out.var.filled(0.0)
            elif out._mask is None and ~np.isfinite(out._var.sum()):
                out._var = out._var.copy()
                out._var[~np.isfinite(out._var)] = 0.0

            # Convolve the var array with the square of the kernel
            out._var = convolution_function(out._var, kernel**2, mode="same")

        return out

    def to_ds9(self, ds9id=None, newframe=False, zscale=False, cmap='grey'):
        """Send the data to ds9 (this will create a copy in memory)

        Parameters
        ----------
        ds9id: None or str
            The DS9 session ID.  If 'None', a new one will be created.
            To find your ds9 session ID, open the ds9 menu option
            File:XPA:Information and look for the XPA_METHOD string, e.g.
            ``XPA_METHOD:  86ab2314:60063``.  You would then calll this
            function as ``cube.to_ds9('86ab2314:60063')``
        newframe: bool
            Send the cube to a new frame or to the current frame?

        """
        try:
            import pyds9 as ds9
        except ImportError:
            self._logger.error('pyds9 was not found, please install it')
            return

        if ds9id is None:
            dd = ds9.DS9(start=True)
        else:
            dd = ds9.DS9(target=ds9id, start=False)

        if newframe:
            dd.set('frame new')

        dd.set_pyfits(fits.HDUList(
            fits.PrimaryHDU(data=self.data.filled(np.nan),
                            header=self.get_wcs_header())))
        dd.set('cmap %s' % cmap)
        if zscale:
            dd.set('zscale true')
        return dd
