# -*- coding: utf-8 -*-

from __future__ import absolute_import, division

import astropy.units as u
import numpy as np
from mpdaf.obj import Image, Cube, WCS, WaveCoord, Spectrum
from numpy.testing import assert_array_equal, assert_allclose
from nose.tools import assert_is

DEFAULT_SHAPE = (10, 6, 5)

def astronomical_image():
    """Return a test image from a real observation """

    ima = Image("data/obj/a370II.fits")

    # The CD matrix of the above image includes a small shear term
    # which means that the image can't be displayed accurately with
    # rectangular pixels. All of the functions in MPDAF assume
    # rectangular pixels, so replace the CD matrix with a similar one
    # that doesn't have a shear component.

    ima.wcs.set_cd(np.array([[2.30899476e-5,  -5.22301199e-5],
                             [-5.22871997e-5, -2.30647413e-5]]))

    return ima

def assert_image_equal(ima, shape=None, start=None, end=None, step=None):
    """Raise an assertion error if the characteristics of a given image
       don't match the specified parameters.

    Parameters
    ----------
    ima   : `mpdaf.obj.Image`
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


def assert_masked_allclose(d1, d2):
    """Compare the values of two masked arrays"""

    if d1 is not None or d2 is not None:

        # Check that the two arrays have the same shape.

        assert_array_equal(d1.shape, d2.shape)

        # Check that they have identical masks.

        if d1.mask is np.ma.nomask or d2.mask is np.ma.nomask:
            assert_is(d2.mask, d1.mask)
        else:
            assert_array_equal(d1.mask, d2.mask)

        # Check that unmasked values in the array are approximately equal.

        assert_allclose(np.ma.filled(d1, 0.0), np.ma.filled(d2, 0.0))


def _generate_test_data(data=2.3, var=1.0, mask=None, shape=None, unit=u.ct,
                        uwave=u.angstrom, wcs=None, wave=None, copy=True,
                        ndim=None, crpix=2.0, cdelt=3.0, crval=0.5):

    # Determine a shape for the data and var arrays. This is either a
    # specified shape, the shape of a specified data or var array, or
    # the default shape.

    if shape is None:
        if isinstance(data, np.ndarray):
            shape = data.shape
            ndim = data.ndim
        elif isinstance(var, np.ndarray):
            shape = var.shape
            ndim = var.ndim
        elif isinstance(mask, np.ndarray):
            shape = mask.shape
            ndim = mask.ndim
        elif ndim is not None:
            if ndim == 1:
                shape = DEFAULT_SHAPE[0]
            elif ndim == 2:
                shape = DEFAULT_SHAPE[1:]
            elif ndim == 3:
                shape = DEFAULT_SHAPE
        else:
            raise ValueError('Missing shape/ndim specification')

    if np.isscalar(shape):
        shape = (shape,)

    if len(shape) != ndim:
        raise ValueError('shape does not match the number of dimensions')

    # Convert the data and var arguments to ndarray's
    if data is None:
        if ndim == 1:
            # Special case for spectra ...
            data = np.arange(shape[0], dtype=np.float)
            data[0] = 0.5
    elif np.isscalar(data):
        data = data * np.ones(shape, dtype=type(data))
    elif data is not None:
        data = np.array(data, copy=copy)
        np.testing.assert_equal(shape, data.shape)

    if np.isscalar(var):
        var = var * np.ones(shape, dtype=type(var))
    elif var is not None:
        var = np.array(var, copy=copy)
        np.testing.assert_equal(shape, var.shape)

    if mask is None:
        mask = False

    if not np.isscalar(mask):
        mask = np.array(mask, copy=copy, dtype=bool)
        np.testing.assert_equal(shape, mask.shape)

    # Substitute default world-coordinates where not specified.
    if ndim == 2:
        wcs = wcs or WCS(crval=(0, 0), crpix=1.0, shape=shape)
    elif ndim == 3:
        wcs = wcs or WCS(crval=(0, 0), crpix=1.0, shape=shape[1:])

    # Substitute default wavelength-coordinates where not specified.
    if ndim in (1, 3):
        wave = wave or WaveCoord(crpix=crpix, cdelt=cdelt, crval=crval,
                                 shape=shape[0], cunit=uwave)
        if wave.shape is None:
            wave.shape = shape[0]

    if ndim == 1:
        cls = Spectrum
    elif ndim == 2:
        cls = Image
    elif ndim == 3:
        cls = Cube

    return cls(data=data, var=var, mask=mask, wave=wave, wcs=wcs,
               unit=unit, copy=copy, dtype=None)


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
    mask : Either a 2D boolean array to use to mask the data array, a
           boolean value to assign to each element of the mask array, or
           None, to indicate that all data values should be left unmasked.
    shape : tuple of 2 integers
        Either None, or the shape to give the data and variance arrays.
        If either data or var are arrays, this must match their shape.
        If shape==None and neither data nor var are arrays, (6,5) is used.
    unit  : `astropy.units.Unit`
        The units of the data.
    wcs   : `mpdaf.obj.WCS`
        The world coordinates of image pixels.
    copy  : boolean
        If true (default), the data, variance and mask arrays are copied.

    """

    return _generate_test_data(data=data, var=var, mask=mask, shape=shape,
                               unit=unit, wcs=wcs, copy=copy, ndim=2)


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
    mask : Either a 1D boolean array to use to mask the data array, a
           boolean value to assign to each element of the mask array, or
           None, to indicate that all data values should be left unmasked.
    shape : int
        Either None, or the size to give the data and variance arrays.
        If either data, var or mask are arrays, this must match their shape.
        If shape==None and neither data, var, nor mask are arrays, 10 is used.
    uwave : `astropy.units.Unit`
        The units to use for wavelengths.
    crpix : float
        The reference pixel of the spectrum.
    cdelt : float
        The step in wavelength between pixels.
    crval : float
        The wavelength of the reference pixel.
    wave  : `mpdaf.obj.WaveCoord`
        The wavelength coordinates of spectral pixels.
    unit  : `astropy.units.Unit`
        The units of the data.
    copy  : boolean
        If true (default), the data, variance and mask arrays are copied.

    """
    return _generate_test_data(data=data, var=var, mask=mask, shape=shape,
                               uwave=uwave, wave=wave, copy=copy, ndim=1,
                               crpix=crpix, cdelt=cdelt, crval=crval)


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
    mask : Either a 3D boolean array to use to mask the data array, a
           boolean value to assign to each element of the mask array, or
           None, to indicate that all data values should be left unmasked.
    shape : tuple of 3 integers
        Either None, or the shape to give the data and variance arrays.
        If either data or var are arrays, this must match their shape.
        If shape==None and neither data nor var are arrays, (10,6,5) is used.
    uwave : `astropy.units.Unit`
        The units to use for wavelengths.
    unit  : `astropy.units.Unit`
        The units of the data.
    wcs   : `mpdaf.obj.WCS`
        The world coordinates of image pixels.
    wave  : `mpdaf.obj.WaveCoord`
        The wavelength coordinates of spectral pixels.
    copy  : boolean
        If true (default), the data, variance and mask arrays are copied.

    """
    return _generate_test_data(data=data, var=var, mask=mask, shape=shape,
                               unit=unit, uwave=uwave, wcs=wcs, wave=wave,
                               copy=copy, ndim=3)
