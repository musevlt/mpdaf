# -*- coding: utf-8 -*-

import astropy.units as u
import numpy as np
from mpdaf.obj import Image, Cube, WCS, WaveCoord, Spectrum
from numpy.testing import assert_array_equal


def assert_image_equal(ima, shape=None, start=None, end=None, step=None):

    """Raise an assertion error if the characteristics of a given image
       don't match the specified parameters.

    Parameters
    ----------
    ima   : :class:`mpdaf.obj.Image`
        The image to be tested.
    shape : tuple
        The shape of the data array of the image.
    start : tuple
        The [y,x] coordinate of pixel [0,0].
    end   : tuple
        The [y,x] coordinate of pixel [-1,-1].
    step  : tuple
        The pixel step size in [x,y].

    """

    if shape is not None:
        assert_array_equal(ima.shape, shape)
    if start is not None:
        assert_array_equal(ima.get_start(), start)
    if end is not None:
        assert_array_equal(ima.get_end(), end)
    if step is not None:
        assert_array_equal(ima.get_step(), step)


def generate_image(data=2.0, var=1.0, mask=None, shape=None,
                   unit=u.ct, wcs=None, copy=True):

    """Generate a simple image for unit tests.

    The data array can either be specified explicitly, or its shape
    can be specified along with a constant value to assign to its
    elements. Similarly for the variance and mask arrays. If one or
    more of the data, var or mask array arguments are provided, their
    shapes must match each other and the optional shape argument.

    Parameters
    ----------
    data : float or numpy.ndarray
        Either a 2D array to assign to the image's data array, or a float
        to assign to each element of the data array.
    var  : float or numpy.ndarray
        Either a 2D array to assign to the image's variance array, a float
        to assign to each element of the variance array, or None if no
        variance array is desired.
    mask : Either a 3D boolean array to use to mask the data array, or
           None, to indicate that all data values should be left unmasked.
    shape : tuple of 2 integers
        Either None, or the shape to give the data and variance arrays.
        If either data or var are arrays, this must match their shape.
        If shape==None and neither data nor var are arrays, (6,5) is used.
    unit  : :class:`astropy.units.Unit`
        The units of the data.
    wcs   : :class:`mpdaf.obj.WCS`
        The world coordinates of image pixels.
    copy  : boolean
        If true (default), the data and variance arrays are copied.

    """

    # Ignore the copy argument until we know we've been given arrays.

    docopy = False

    # Convert the data and var arguments to ndarray's so that we
    # can check their dimensions.

    data = np.asarray(data)
    if var is None:
        var = np.asarray(var)

    # Determine a shape for the data and var arrays. This is either a
    # specified shape, the shape of a specified data or var array, or
    # the default shape.

    shape = (shape or
             (data.shape if data.ndim > 0 else None) or
             (var.shape if var is not None and var.ndim > 0 else None) or
             (mask.shape if mask is not None and mask.ndim > 0 else None) or
             (6, 5))

    # Check the shape denotes a image with at least 1 element.

    if len(shape) != 2:
        raise ValueError('The image must have 2 dimensions.')
    elif np.prod(shape) < 1:
        raise ValueError('The image must have at least one pixel.')

    # Create data and var arrays from scalar values where specified.

    if data.ndim == 0:
        data = data * np.ones(shape)
    else:
        docopy = copy

    # Don't create a variance array?

    if var is None:
        pass

    # Create a variance array filled with a scalar value?

    elif var.ndim == 0:
        var = var * np.ones(shape)

    # Use a specified variance array, and heed the caller's copy argument.

    else:
        docopy = copy

    # Check that the shapes of the data and var arguments are consistent.

    if not np.array_equal(shape, data.shape):
        raise ValueError('Mismatch between shape and data arguments.')
    elif var is not None and not np.array_equal(shape, var.shape):
        raise ValueError('Mismatch between shape and var arguments.')
    elif mask is not None and not np.array_equal(shape, mask.shape):
        raise ValueError('Mismatch between shape and mask arguments.')

    # Substitute default world-coordinates where not specified.

    wcs = wcs or WCS(crval=(0, 0), crpix=1.0, shape=shape)

    # Create the image.

    return Image(data=data, var=var, mask=mask, wcs=wcs, unit=unit,
                 copy=docopy)


def generate_spectrum(data=None, var=1.0, mask=None, shape=None,
                      uwave=u.angstrom, crpix=2.0, cdelt=3.0,
                      crval=0.5, wave=None, unit=u.ct, copy=True):

    """Generate a simple spectrum for unit tests.

    The data array can either be specified explicitly, or its shape
    can be specified along with a constant value to assign to its
    elements. Similarly for the variance and mask arrays. If one or
    more of the data, var or mask array arguments are provided, their
    shapes must match each other and the optional shape argument.

    Parameters
    ----------
    data : float or numpy.ndarray
        Either a 1D array to assign to the spectrum's data array,
        a float to assign to each element of the data array, or
        None to substitute the default spectrum (0.5, 1, 2, 3 ...).
    var  : float or numpy.ndarray
        Either a 1D array to assign to the spectrum's variance array,
        a float to assign to each element of the variance array,
        or None if no variance array is desired.
    mask : Either a 3D boolean array to use to mask the data array, or
           None, to indicate that all data values should be left unmasked.
    shape : int
        Either None, or the size to give the data and variance arrays.
        If either data, var or mask are arrays, this must match their shape.
        If shape==None and neither data, var, nor mask are arrays, 10 is used.
    uwave : :class:`astropy.units.Unit`
        The units to use for wavelengths.
    crpix : float
        The reference pixel of the spectrum.
    cdelt : float
        The step in wavelength between pixels.
    crval : float
        The wavelength of the reference pixel.
    wave  : :class:`mpdaf.obj.WaveCoord`
        The wavelength coordinates of spectral pixels.
    unit  : :class:`astropy.units.Unit`
        The units of the data.
    copy  : boolean
        If true (default), the data and variance arrays are copied.

    """

    # Ignore the copy argument until we know we've been given arrays.

    docopy = False

    # Convert the data and var arguments to ndarray's so that we
    # can check their dimensions.

    if data is not None:
        data = np.asarray(data)
    if var is not None:
        var = np.asarray(var)

    # Determine a shape for the data and var arrays. This is either a
    # specified shape, the shape of a specified data or var array, or
    # the default shape.

    shape = (shape or
             (data.shape if data is not None and data.ndim > 0 else None) or
             (var.shape if var is not None and var.ndim > 0 else None) or
             (mask.shape if mask is not None and mask.ndim > 0 else None) or
             10)

    # To allow comparison with numpy.ndarray shapes, force shape to have
    # at least one dimension.

    if np.asarray(shape).ndim == 0:
        shape = (shape,)

    # Check the shape denotes a spectrum with at least 1 element.

    if len(shape) > 1:
        raise ValueError('The spectrum must have 1 dimension.')
    elif np.prod(shape) < 1:
        raise ValueError('The spectrum must have at least one pixel.')

    # Substitute a default data array?

    if data is None:
        data = np.arange(shape[0], dtype=np.float)
        data[0] = 0.5

    # Create a data array filled with a scalar value?

    elif data.ndim == 0:
        data = data * np.ones(shape)

    # Use a specified data array, and heed the caller's copy argument.

    else:
        docopy = copy

    # Don't create a variance array?

    if var is None:
        pass

    # Create a variance array filled with a scalar value?

    elif var.ndim == 0:
        var = var * np.ones(shape)

    # Use a specified variance array, and heed the caller's copy argument.

    else:
        docopy = copy

    # Check that the shapes of the data and var arguments are consistent.

    if not np.array_equal(shape, data.shape):
        raise ValueError('Mismatch between shape and data arguments.')
    elif var is not None and not np.array_equal(shape, var.shape):
        raise ValueError('Mismatch between shape and var arguments.')
    elif mask is not None and not np.array_equal(shape, mask.shape):
        raise ValueError('Mismatch between shape and mask arguments.')

    # Determine the wavelength coordinates.

    wave = wave or WaveCoord(crpix=crpix, cdelt=cdelt, crval=crval,
                             shape=shape[0], cunit=uwave)
    if wave.shape is None:
        wave.shape = shape[0]

    # Create the spectrum.

    return Spectrum(data=data, var=var, mask=mask, wave=wave,
                    unit=unit, copy=docopy)


def generate_cube(data=2.3, var=1.0, mask=None, shape=None, uwave=u.angstrom,
                  unit=u.ct, wcs=None, wave=None, copy=True):
    """Generate a simple cube for unit tests.

    The data array can either be specified explicitly, or its shape
    can be specified along with a constant value to assign to its
    elements. Similarly for the variance and mask arrays. If one or
    more of the data, var or mask array arguments are provided, their
    shapes must match each other and the optional shape argument.

    Parameters
    ----------
    data : float or numpy.ndarray
        Either a 3D array to assign to the cube's data array, or a float
        to assign to each element of the data array.
    var  : float or numpy.ndarray
        Either a 3D array to assign to the cube's variance array, a float
        to assign to each element of the variance array, or None if no
        variance array is desired.
    mask : Either a 3D boolean array to use to mask the data array, or
           None, to indicate that all data values should be left unmasked.
    shape : tuple of 3 integers
        Either None, or the shape to give the data and variance arrays.
        If either data or var are arrays, this must match their shape.
        If shape==None and neither data nor var are arrays, (10,6,5) is used.
    uwave : :class:`astropy.units.Unit`
        The units to use for wavelengths.
    unit  : :class:`astropy.units.Unit`
        The units of the data.
    wcs   : :class:`mpdaf.obj.WCS`
        The world coordinates of image pixels.
    wave  : :class:`mpdaf.obj.WaveCoord`
        The wavelength coordinates of spectral pixels.
    copy  : boolean
        If true (default), the data and variance arrays are copied.

    """

    # Ignore the copy argument until we know we've been given arrays.

    docopy = False

    # Convert the data and var arguments to ndarray's so that we
    # can check their dimensions.

    data = np.asarray(data)
    if var is None:
        var = np.asarray(var)

    # Determine a shape for the data and var arrays. This is either a
    # specified shape, the shape of a specified data or var array, or
    # the default shape.

    shape = (shape or
             (data.shape if data.ndim > 0 else None) or
             (var.shape if var is not None and var.ndim > 0 else None) or
             (mask.shape if mask is not None and mask.ndim > 0 else None) or
             (10, 6, 5))

    # Check the shape denotes a cube with at least 1 element.

    if len(shape) != 3:
        raise ValueError('The cube must have 3 dimensions.')
    elif np.prod(shape) < 1:
        raise ValueError('The cube must have at least one pixel.')

    # Create data and var arrays from scalar values where specified.

    if data.ndim == 0:
        data = data * np.ones(shape)
    else:
        docopy = copy

    # Don't create a variance array?

    if var is None:
        pass

    # Create a variance array filled with a scalar value?

    elif var.ndim == 0:
        var = var * np.ones(shape)

    # Use a specified variance array, and heed the caller's copy argument.

    else:
        docopy = copy

    # Check that the shapes of the data, var and mask arguments are consistent.

    if not np.array_equal(shape, data.shape):
        raise ValueError('Mismatch between shape and data arguments.')
    elif var is not None and not np.array_equal(shape, var.shape):
        raise ValueError('Mismatch between shape and var arguments.')
    elif mask is not None and not np.array_equal(shape, mask.shape):
        raise ValueError('Mismatch between shape and mask arguments.')

    # Substitute default world-coordinates where not specified.

    wcs = wcs or WCS(crval=(0, 0), crpix=1.0, shape=shape[1:])

    # Substitute default wavelength-coordinates where not specified.

    wave = wave or WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=shape[0],
                             cunit=uwave)
    if wave.shape is None:
        wave.shape = shape[0]

    # Create the cube.

    return Cube(data=data, var=var, mask=mask, wave=wave, wcs=wcs, unit=unit,
                copy=docopy)
