"""Test on DataArray objects."""

# Import the recommended python 2 -> 3 compatibility modules.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import nose.tools
from nose.plugins.attrib import attr

import astropy.units as u
import os
import tempfile
import numpy as np
from astropy.io import fits
from mpdaf.obj import DataArray, WaveCoord, WCS, Cube
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
from os.path import join

from ..utils import generate_image, generate_cube, generate_spectrum

DATADIR = join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
TESTIMG = join(DATADIR, 'data', 'obj', 'a370II.fits')
TESTSPE = join(DATADIR, 'data', 'obj', 'Spectrum_lines.fits')


@attr(speed='fast')
def test_fits_img():
    """DataArray class: Testing FITS image reading"""
    hdu = fits.open(TESTIMG)
    data = DataArray(filename=TESTIMG)
    nose.tools.assert_equal(data.shape, hdu[0].data.shape)
    nose.tools.assert_equal(data.ndim, 2)
    hdu.close()


@attr(speed='fast')
def test_fits_spectrum():
    """DataArray class: Testing FITS spectrum reading"""
    hdu = fits.open(TESTSPE)
    data = DataArray(filename=TESTSPE)
    nose.tools.assert_equal(data.shape, hdu[1].data.shape)
    nose.tools.assert_equal(data.ndim, 1)
    hdu.close()


@attr(speed='fast')
def test_from_ndarray():
    """DataArray class: Testing initialization from a numpy.ndarray"""
    data = np.arange(10)
    d = DataArray(data=data)
    nose.tools.assert_tuple_equal(d.shape, data.shape)
    assert_allclose(d.data, data)


@attr(speed='fast')
def test_copy():
    """DataArray class: Testing the copy method"""
    wcs = WCS(deg=True)
    wave = WaveCoord(cunit=u.angstrom)
    cube1 = DataArray(data=np.arange(5 * 4 * 3).reshape(5, 4, 3), wave=wave,
                      wcs=wcs)
    cube2 = cube1.copy()
    nose.tools.assert_true(cube1.wcs.isEqual(cube2.wcs))
    nose.tools.assert_true(cube1.wave.isEqual(cube2.wave))
    assert_allclose(cube1.data, cube2.data)


@attr(speed='fast')
def test_clone():
    """DataArray class: Testing the clone method"""
    cube1 = generate_cube()
    cube2 = cube1.clone()
    nose.tools.assert_true(cube1.wcs.isEqual(cube2.wcs))
    nose.tools.assert_true(cube1.wave.isEqual(cube2.wave))
    nose.tools.assert_true(cube2.data is None)
    nose.tools.assert_true(cube2.var is None)


@attr(speed='fast')
def test_clone_with_data():
    """DataArray class: Testing the clone method with data"""

    # Define a couple of functions to be used to fill the data and
    # variance arrays of the cloned cube.

    def data_fn(shape, dtype=np.float):
        return np.arange(np.asarray(shape, dtype=int).prod(),
                         dtype=dtype).reshape(shape) * 0.25
    def var_fn(shape, dtype=np.float):
        return np.arange(np.asarray(shape, dtype=int).prod(),
                         dtype=dtype).reshape(shape) * 0.5

    # Create a generic cube of a specified shape.

    shape = (6,3,2)
    cube1 = generate_cube(shape = shape)

    # Create a shallow copy of the cube and fill its data and
    # variance arrays using the functions defined above.

    cube2 = cube1.clone(data_init=data_fn, var_init=var_fn)

    # Check that the data and variance arrays ended up containing
    # the expected ramps of values.

    assert_allclose(data_fn(shape), cube2.data)
    assert_allclose(var_fn(shape), cube2.var)

@attr(speed='fast')
def test_set_var():
    """DataArray class: Testing the set_var method"""

    # Create a cube of a specified shape.

    nz = 3; ny = 4; nx = 5
    shape = (nz, ny, nx)
    cube = generate_cube(var = None, shape=shape)

    # Create a variance array of the same shape as the cube.

    var = np.arange(nz * ny * nx, dtype=float).reshape(shape)

    # Assign the variance array to the cube and check that the
    # variance array in the cube then matches it.

    cube.var = var
    assert_allclose(cube.var, var)

    # Remove the variance array and check that this worked.

    cube.var = None
    nose.tools.assert_true(cube.var is None)

@attr(speed='fast')
def test_le():
    """DataArray class: Testing __le__ method"""

    # Generate a test spectrum initialized with a linear ramp of values.

    n = 10
    spec = generate_spectrum(data=np.arange(n, dtype=float), var=1.0)

    # Apply the less-than-or-equal method.

    s = spec <= n//2

    # Verify that ramp values <= n//2 are not masked,
    # and that values > n//2 are masked.

    nose.tools.assert_equal(s.data[:n//2+1].count(), n//2 + 1)
    nose.tools.assert_equal(s.data[n//2+1:].count(), 0)

@attr(speed='fast')
def test_lt():
    """DataArray class: Testing __lt__ method"""

    # Generate a test spectrum initialized with a linear ramp of values.

    n = 10
    spec = generate_spectrum(data=np.arange(n, dtype=float), var=1.0)

    # Apply the less-than method.

    s = spec < n//2

    # Verify that ramp values < n//2 are not masked,
    # and that values >= n//2 are masked.

    nose.tools.assert_equal(s.data[:n//2+1].count(), n//2)
    nose.tools.assert_equal(s.data[n//2+1:].count(), 0)

@attr(speed='fast')
def test_ge():
    """DataArray class: Testing __ge__ method"""

    # Generate a test spectrum initialized with a linear ramp of values.

    n = 10
    spec = generate_spectrum(data=np.arange(n, dtype=float), var=1.0)

    # Apply the greater-than-or-equal method.

    s = spec >= n//2

    # Verify that ramp values >= n//2 are not masked,
    # and that values < n//2 are masked.

    nose.tools.assert_equal(s.data[n//2:].count(), n//2)
    nose.tools.assert_equal(s.data[:n//2].count(), 0)

@attr(speed='fast')
def test_gt():
    """DataArray class: Testing __gt__ method"""

    # Generate a test spectrum initialized with a linear ramp of values.

    n = 10
    spec = generate_spectrum(data=np.arange(n, dtype=float), var=1.0)

    # Apply the greater-than method.

    s = spec > n//2

    # Verify that ramp values > n//2 are not masked,
    # and that values <= n//2 are masked.

    nose.tools.assert_equal(s.data[n//2+1:].count(), n//2 - 1)
    nose.tools.assert_equal(s.data[:n//2+1].count(), 0)

@attr(speed='fast')
def test_getitem():
    """DataArray class: Testing the __getitem__ method"""

    # Set the dimensions of a test cube.

    nz = 50
    ny = 30
    nx = 20

    # Create a cube, filled with a ramp.

    cube = generate_cube(
        data = np.arange(nx*ny*nz,dtype=float).reshape((nz,ny,nx)),
        var = np.arange(nx*ny*nz,dtype=float).reshape((nz,ny,nx)) / 2.0,
        wcs = WCS(deg=True),
        wave = WaveCoord(cunit=u.angstrom))

    # Extract a pixel from the cube and check that it has the expected value
    # from its position on the 1D ramp that was used to initialize the cube.

    za = 34
    ya = 5
    xa = 14
    pixel_a = xa + ya*nx + za*(nx*ny)
    s = cube[za,ya,xa]
    assert_allclose(s, pixel_a)

    # Extract a spectrum from the cube, and check that it has the expected
    # values.

    zb = za + 10
    yb = ya
    xb = xa
    dz = zb - za
    pixel_b = xb + yb*nx + zb*(nx*ny)
    s = cube[za:zb,ya,xa]
    expected_spec = np.arange(pixel_a, pixel_a + dz*(nx*ny), nx*ny,
                              dtype=float)
    assert_allclose(s.data, expected_spec)

    # Check that the wavelength of the first spectrum pixel matches that
    # of pixel za,ya,xa in the original cube.

    expected_wave = cube.wave.coord(za)
    actual_wave = s.wave.coord(0)
    assert_allclose(actual_wave, expected_wave)

    # Extract an image from the cube, and check that it has the
    # expected values.

    zb = za
    yb = ya + 3
    xb = xa + 5
    dx = xb - xa
    dy = yb - ya
    s = cube[za,ya:yb,xa:xb]
    first_row = np.arange(pixel_a, pixel_a + dx, dtype=float)
    row_offsets = np.arange(0, dy*nx, nx, dtype=float)
    expected_image = np.resize(first_row, (dy,dx)) + np.repeat(row_offsets, dx).reshape((dy,dx))
    assert_allclose(s.data, expected_image)

    # Check that the world coordinates of the first pixel of the
    # image match those of pixel(za,ya,xa) of the original cube.

    expected_sky = cube.wcs.pix2sky((ya,xa))
    actual_sky = s.wcs.pix2sky((0,0))
    assert_allclose(actual_sky, expected_sky)

    # Extract a sub-cube from the cube, and check that it has the
    # expected values.

    zb = za + 2
    yb = ya + 3
    xb = xa + 5
    dx = xb - xa
    dy = yb - ya
    dz = zb - za
    s = cube[za:zb,ya:yb,xa:xb]
    spec_offsets = np.arange(0, dz*ny*nx, ny*nx, dtype=float)
    expected_cube = np.resize(expected_image, (dz,dy,dx)) + np.repeat(spec_offsets, dy*dx).reshape((dz,dy,dx))
    assert_allclose(s.data, expected_cube)

    # Check that the world coordinates of the first pixel of the
    # image match those of pixel(za,ya,xa) of the original cube.

    expected_sky = cube.wcs.pix2sky((ya,xa))
    actual_sky = s.wcs.pix2sky((0,0))
    assert_allclose(actual_sky, expected_sky)

    # Check that the wavelength of the first spectrum pixel matches that
    # of pixel za,ya,xa in the original cube.

    expected_wave = cube.wave.coord(za)
    actual_wave = s.wave.coord(0)
    assert_allclose(actual_wave, expected_wave)

@attr(speed='fast')
def test_get_wcs_header():
    """DataArray class: Testing the get_wcs_header method"""

    # Set up a WCS object with non-default values.

    wcs = WCS(crpix=(15.0,10.0), crval=(12.0,14.0), cdelt=(0.1,0.2), deg=True)

    # Create a test image, passing the constructor the above
    # world-coordinate object.

    im = generate_image(wcs = wcs)

    # Generate a header that describes the world coordinates.

    hdr = im.get_wcs_header()

    # Create a new wcs object from the header.

    hdr_wcs = WCS(hdr)

    # Check that the WCS information taken from the header matches the
    # WCS information that was passed to the image constructor.

    nose.tools.assert_true(wcs.isEqual(hdr_wcs))

@attr(speed='fast')
def test_get_data_hdu():
    """DataArray class: Testing the get_data_hdu method"""

    # Set the dimensions of a test cube.

    nz = 5
    ny = 3
    nx = 2

    # Set up a WCS object with non-default values.

    wcs = WCS(crpix=(15.0,10.0), crval=(12.0,14.0), cdelt=(0.1,0.2), deg=True)

    # Set up a wavelength coordinate object with non-default values.

    wave = WaveCoord(crpix=2.0, cdelt=1.5, crval=12.5)

    # Create a simple test cube.

    cube = generate_cube(shape=(nz,ny,nx), wcs = wcs, wave = wave)

    # Get a data HDU for the cube.

    hdu = cube.get_data_hdu()

    # Get the header of the HDU.

    hdr = hdu.header

    # Create a new wcs object from the header.

    hdr_wcs = WCS(hdr)

    # Check that the WCS information taken from the header matches the
    # WCS information stored with the cube.

    nose.tools.assert_true(cube.wcs.isEqual(hdr_wcs))

    # Create a new WaveCoord object from the header.

    hdr_wave = WaveCoord(hdr)

    # Check that the wavelength information taken from the header
    # matches the wavelength information stored with the cube.

    nose.tools.assert_true(cube.wave.isEqual(hdr_wave))

@attr(speed='fast')
def test_get_stat_hdu():
    """DataArray class: Testing the get_stat_hdu method"""

    # Set the dimensions of a test cube.

    nz = 5
    ny = 20
    nx = 10

    # Create a cube, with a ramp of values assigned to the variance array.

    cube = generate_cube(
        data = 0.1,
        var = np.arange(nx*ny*nz,dtype=float).reshape((nz,ny,nx)),
        wcs = WCS(deg=True),
        wave = WaveCoord(cunit=u.angstrom))

    # Get the STAT HDU.

    hdu = cube.get_stat_hdu()

    # Get the array of variances from the HDU.

    var = np.asarray(hdu.data, dtype=cube.dtype)

    assert_allclose(var, cube.var)

@attr(speed='fast')
def test_write():
    """DataArray class: Testing the write method"""

    # Set the dimensions of a test cube.

    nz = 5
    ny = 20
    nx = 10

    # Create a cube, with a ramp of values assigned to the variance array.

    data = np.arange(nx*ny*nz,dtype=float).reshape((nz,ny,nx))
    var = data / 10.0
    mask = np.asarray(data,dtype=int) % 10 == 0
    cube = generate_cube(
        data = data, var = var, mask = mask,
        wcs = WCS(deg=True),
        wave = WaveCoord(cunit=u.angstrom))

    # To get a temporary filename for the FITS file, create a
    # temporary file using the tempfile module, then close it and use
    # the same name for the FITS file. We do it this way because
    # os.tmpname() outputs a security warning.

    tmpfile = tempfile.NamedTemporaryFile(suffix=".fits")
    filename = tmpfile.name
    tmpfile.close()

    # Create the temporary file with the same name as the temporary file.

    cube.write(filename, savemask='dq')

    # Read the file into a new cube.

    cube2 = Cube(filename)

    # Verify that the contents of the file match the original cube.

    assert_allclose(cube2.data.data, cube.data.data)
    assert_array_equal(cube2.data.mask, cube.data.mask)
    assert_allclose(cube2.var, cube.var)
    nose.tools.assert_true(cube2.wcs.isEqual(cube.wcs))
    nose.tools.assert_true(cube2.wave.isEqual(cube.wave))

    # Delete the temporary file.

    os.remove(filename)

@attr(speed='fast')
def test_sqrt():
    """DataArray class: Testing the sqrt method"""

    # Create a spectrum containing gaussian noise, offset by a mean
    # level that prevents it going negative. The lack of negatives
    # means that we can use the variance of its square rooted pixels
    # to estimate the expected variances of the square rooted data
    # values.

    mean = 100.0  # Mean of spectrum pixels.
    sdev = 2.0    # Standard deviation of pixel values around the mean.
    n = 100000    # The number of spectral pixels.
    spec = generate_spectrum(
        data = np.random.normal(loc=mean, scale=sdev, size=n), var = sdev**2)

    # Get the square root of the spectrum.

    s = spec.sqrt()

    # Determine the mean and variance of the square rooted data.

    s_mean = s.data.mean()
    s_var = s.data.var()

    # Check that the mean is as close as expected to the theoretical
    # mean, noting that the standard deviation of the mean of the
    # square rooted data should be sqrt(0.25*sdev**2/mean/n) [for
    # for sdev=2,mean=100 and n=100000, this gives 0.0003.

    assert_allclose(s_mean, np.sqrt(mean), rtol=0.001)

    # Given a sample of value x, picked from a distribution of variance vx,
    # compute the expected variance, vs, of sqrt(x).
    #
    #  vs = (d(sqrt(x))/dx)**2 * vx
    #     = (0.5/sqrt(x))**2 * vx
    #     = 0.25/x * vx.
    #
    # Sanity check this by seeing if the variance of square root of
    # the random pixels matches it.

    expected_var = 0.25 * sdev**2 / mean
    assert_allclose(s_var, expected_var, rtol=0.05)

    # Check that the recorded variances match the expected variance.

    assert_allclose(np.ones(n)*s_var, expected_var, rtol=0.1)

    # Generate another spectrum, but this time fill it with positive
    # and negative values.

    n = 10        # The number of spectral pixels.
    sdev = 0.5    # The variance to be recorded with each pixel.
    spec = generate_spectrum(
        data = np.hstack((np.ones(n//2)*-1.0, np.ones(n//2)*1.0)),
        var = sdev**2)

    # Get the square root of the spectrum.

    s = spec.sqrt()

    # Check that the square-rooted masked values that were negative
    # and not those that weren't.

    nose.tools.assert_equal(s.data[:n//2].count(), 0)
    nose.tools.assert_equal(s.data[n//2:].count(), n//2)

@attr(speed='fast')
def test_abs():
    """DataArray class: Testing the abs method"""

    # Create a spectrum containing a ramp that goes from negative to
    # positive integral values, stored as floats. Assign a constant
    # variance to all points, and mask a couple of points either side
    # of zero.

    ramp = np.arange(10.0, dtype=float) - 5
    var = 0.5
    mask = np.logical_and(ramp > -2, ramp < 2)
    spec1 = generate_spectrum(ramp, var = var, mask = mask)

    # Get the absolute values of the spectrum.

    spec2 = spec1.abs()

    # Check that the unmasked elements of the data array contain the
    # absolute values of the input array, and that the masked elements
    # are unchanged.

    assert_allclose(np.where(mask, ramp, np.abs(ramp)), spec2.data)

    # Check that the mask and variance arrays are unchanged.

    assert_array_equal(spec2.data.mask, spec1.data.mask)
    assert_allclose(spec2.var, spec1.var)

@attr(speed='fast')
def test_unmask():
    """DataArray class: Testing the unmask method"""

    # Create a spectrum containing a simple ramp from negative to
    # positive values, with negative values masked, and one each of
    # the infinity and nan special values.

    n = 10
    ramp = np.arange(n, dtype=float) - 5
    var = 0.5
    mask = ramp < 0
    ramp[0] = np.inf
    ramp[1] = np.nan
    spec = generate_spectrum(ramp, var = var, mask = mask)

    # Clear the mask, keeping just the inf and nan values masked.

    spec.unmask()

    # Check that all negative values are unmasked, and that the Inf and
    # Nan values remain masked.

    assert_array_equal(spec.data.mask, np.arange(n, dtype=int) < 2)

@attr(speed='fast')
def test_mask_variance():
    """DataArray class: Testing the mask_variance method"""

    # Create a spectrum whose variance array contains a positive ramp
    # of integral values stored as floats, and with a data array where
    # values below a specified threshold, chosen to be half way between
    # data values, are masked.

    n = 10
    var = np.arange(n,dtype=float)
    lower_lim = 1.5
    spec = generate_spectrum(var = var, mask = var < lower_lim)

    # Mask all variances above a second threshold that is again chosen
    # to be half way between two known integral values in the array.

    upper_lim = 6.5
    spec.mask_variance(upper_lim)

    # Check that the originally masked values are still masked, and that
    # the values above the upper threshold are now also masked, and that
    # values in between them are not masked.

    assert_array_equal(spec.data.mask,
                       np.logical_or(var < lower_lim, var > upper_lim))

@attr(speed='fast')
def test_mask_selection():
    """DataArray class: Testing the mask_selection method"""

    # Create a test spectrum with a ramp for the data array
    # and all values below a specified limit masked.

    n = 10
    ramp = np.arange(n, dtype=float)
    lower_lim = 1.5
    spec = generate_spectrum(data=ramp, mask = ramp < lower_lim)

    # Apply the mask_selection method to mask values above a different
    # threshold.

    upper_lim = 6.5
    spec.mask_selection(np.where(ramp > upper_lim))

    # Check that the resulting data mask is the union of the
    # originally mask and the newly selected mask.

    assert_array_equal(spec.data.mask,
                       np.logical_or(ramp < lower_lim, ramp > upper_lim))
