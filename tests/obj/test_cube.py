"""Test on Cube objects."""
from __future__ import absolute_import, division

import nose.tools
from nose.plugins.attrib import attr

import astropy.units as u
import numpy as np
import six

from astropy.io import fits
from mpdaf.obj import Spectrum, Image, Cube, iter_spe, iter_ima, WCS, WaveCoord
from numpy import ma
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)
from nose.tools import assert_equal
from tempfile import NamedTemporaryFile

from ..utils import (generate_cube, generate_image, generate_spectrum,
                     assert_masked_allclose)
if six.PY2:
    from operator import add, sub, mul, div
else:
    from operator import add, sub, mul, truediv as div


@attr(speed='fast')
def test_copy():
    """Cube class: testing copy method."""
    cube1 = generate_cube()
    cube2 = cube1.copy()
    s = cube1.data.sum()
    cube1[0, 0, 0] = 1000
    nose.tools.assert_true(cube1.wcs.isEqual(cube2.wcs))
    nose.tools.assert_true(cube1.wave.isEqual(cube2.wave))
    assert_equal(s, cube2.data.sum())


@attr(speed='fast')
def test_arithmetricOperator_Cube():
    """Cube class: tests arithmetic functions"""
    cube1 = generate_cube(uwave=u.nm)
    image1 = generate_image(wcs=cube1.wcs, unit=2 * u.ct)
    spectrum1 = generate_spectrum(data=2.3, cdelt=30.0, crval=5)
    cube2 = image1 + cube1

    for op in (add, sub, mul, div):
        cube3 = op(cube1, cube2)
        assert_almost_equal(cube3.data, op(cube1.data, cube2.data))

    # with spectrum
    sp1 = spectrum1.data[:, np.newaxis, np.newaxis]
    for op in (add, sub, mul, div):
        cube3 = op(cube1, spectrum1)
        assert_almost_equal(cube3.data, op(cube1.data, sp1))

    # with image
    im1 = image1.data.data[np.newaxis, :, :] * image1.unit
    for op in (add, sub, mul, div):
        cube3 = op(cube1, image1)
        assert_almost_equal((cube3.data.data * cube3.unit).value,
                            op(cube1.data.data * cube1.unit, im1).value)

    cube2 = cube1 / 25.3
    assert_almost_equal(cube2.data, cube1.data / 25.3)


@attr(speed='fast')
def test_get_Cube():
    """Cube class: tests getters"""
    cube1 = generate_cube()
    assert_array_equal(cube1[2, :, :].shape, (6, 5))
    assert_equal(cube1[:, 2, 3].shape[0], 10)
    assert_array_equal(cube1[1:7, 0:2, 0:3].shape, (6, 2, 3))
    assert_array_equal(cube1.select_lambda(1.2, 15.6).shape, (6, 6, 5))
    a = cube1[2:4, 0:2, 1:4]
    assert_array_equal(a.get_start(), (3.5, 0, 1))
    assert_array_equal(a.get_end(), (6.5, 1, 3))


@attr(speed='fast')
def test_iter_ima():
    """Cube class: tests Image iterator"""
    cube1 = generate_cube()
    ones = np.ones(shape=(6, 5))
    for ima, k in iter_ima(cube1, True):
        ima[:, :] = k * ones
    c = np.arange(cube1.shape[0])[:, np.newaxis, np.newaxis]
    assert_array_equal(*np.broadcast_arrays(cube1.data.data, c))


@attr(speed='fast')
def test_iter_spe():
    """Cube class: tests Spectrum iterator"""
    cube1 = generate_cube(data=0.)
    for (spe, (p, q)) in iter_spe(cube1, True):
        spe[:] = spe + p + q

    y, x = np.mgrid[:cube1.shape[1], :cube1.shape[2]]
    assert_array_equal(*np.broadcast_arrays(cube1.data.data, y + x))


@attr(speed='fast')
def test_crop():
    """Cube class: tests the crop method."""
    cube1 = generate_cube()
    cube1.data.mask[0, :, :] = True
    cube1.crop()
    assert_equal(cube1.shape[0], 9)


# A function for testing the multiprocessing function, which takes an
# Image or a Spectrum object as its argument and returns 10 times the
# mean of the data in that object.
def _multiproc_func(obj):
    return obj.data.mean() * 10.0


@attr(speed='fast')
def test_multiprocess():
    """Cube class: tests multiprocess"""
    data_value = 2.2
    cube1 = generate_cube(data=data_value)

    # Test image processing using an Image method.
    spe = cube1.loop_ima_multiprocessing(Image.get_rot, cpu=2, verbose=False)
    assert_allclose(spe.data, cube1.get_rot())

    # Test image processing using a normal function.
    spe = cube1.loop_ima_multiprocessing(_multiproc_func, cpu=2, verbose=False)
    assert_allclose(spe.data, data_value * 10.0)

    # Test spectrum processing using a Spectrum method.
    im = cube1.loop_spe_multiprocessing(Spectrum.mean, cpu=2, verbose=False)
    assert_allclose(im.data[2, 3], cube1[:, 2, 3].mean())

    # Test spectrum processing using a normal function.
    im = cube1.loop_spe_multiprocessing(_multiproc_func, cpu=2, verbose=False)
    assert_allclose(im.data, data_value * 10.0)


@attr(speed='fast')
def test_multiprocess2():
    """Cube class: more tests for multiprocess"""
    cube1 = generate_cube()
    f = Image.ee
    ee = cube1.loop_ima_multiprocessing(f, cpu=2, verbose=False)
    assert_equal(ee[1], cube1[1, :, :].ee())

    f = Image.rotate
    cub2 = cube1.loop_ima_multiprocessing(f, cpu=2, verbose=False, theta=20)
    assert_equal(cub2[4, 3, 2], cube1[4, :, :].rotate(20)[3, 2])

    f = Spectrum.resample
    out = cube1.loop_spe_multiprocessing(f, cpu=2, verbose=False, step=1)
    assert_equal(out[8, 3, 2], cube1[:, 3, 2].resample(step=1)[8])


@attr(speed='fast')
def test_mask():
    """Cube class: testing mask functionalities"""
    cube = generate_cube()

    # A region of half-width=1 and half-height=1 should have a size of
    # 2x2 pixels. A 2x2 region of pixels has a center at the shared
    # corner of the 4 pixels, and the closest corner to the requested
    # center of 2.1,1.8 is 2.5,1.5, so we expect the square of unmasked pixels
    # to be pixels 2,3 along the Y axis, and pixels 1,2 along the X axis.
    cube.mask_region((2.1, 1.8), (1, 1), lmin=2, lmax=5, inside=True,
                     unit_center=None, unit_radius=None, unit_wave=None)

    # The expected mask for the images between lmin and lmax.
    expected_mask = np.array([[False, False, False, False, False],
                              [False, False, False, False, False],
                              [False, True, True, False, False],
                              [False, True, True, False, False],
                              [False, False, False, False, False],
                              [False, False, False, False, False]], dtype=bool)
    assert_array_equal(np.any(cube.mask[:2, :, :], axis=0),
                       np.zeros(cube.shape[1:]))
    assert_array_equal(np.all(cube._mask[2:5, :, :], axis=0), expected_mask)
    assert_array_equal(np.any(cube.mask[5:, :, :], axis=0),
                       np.zeros(cube.shape[1:]))

    cube.unmask()

    # Do the same experiment, but this time mask pixels outside the region
    # instead of within the region.
    cube.mask_region((2.1, 1.8), (1, 1), lmin=2, lmax=5, inside=False,
                     unit_center=None, unit_radius=None, unit_wave=None)
    assert_array_equal(np.any(cube.mask[:2, :, :], axis=0),
                       np.ones(cube.shape[1:]))
    assert_array_equal(np.all(cube._mask[2:5, :, :], axis=0), ~expected_mask)
    assert_array_equal(np.any(cube.mask[5:, :, :], axis=0),
                       np.ones(cube.shape[1:]))
    cube.unmask()

    # Do the same experiment, but this time with the center and size
    # of the region specified using the equivalent world coordinates.
    wcs = WCS(deg=True)
    wave = WaveCoord(cunit=u.angstrom)
    cube = Cube(data=cube.data, wave=wave, wcs=wcs, copy=False)
    cube.mask_region(wcs.pix2sky([2.1, 1.8]), (3600, 3600), lmin=2, lmax=5,
                     inside=False)
    assert_array_equal(np.any(cube.mask[:2, :, :], axis=0),
                       np.ones(cube.shape[1:]))
    assert_array_equal(np.all(cube._mask[2:5, :, :], axis=0), ~expected_mask)
    assert_array_equal(np.any(cube.mask[5:, :, :], axis=0),
                       np.ones(cube.shape[1:]))
    cube.unmask()

    # Mask around a region of half-width and half-height 1.1 pixels,
    # specified in arcseconds, centered close to pixel 2.4,3.8. This
    # ideally corresponds to a region of 2.2x2.2 pixels. The closest
    # possible size is 2x2 pixels. A region of 2x2 pixels has its
    # center at the shared corner of these 4 pixels, and the nearest
    # corner to the desired central index of (2.4,3.8) is (2.5,3.5).
    # So all of the image should be masked, except for a 2x2 area of
    # pixel indexes 2,3 along the Y axis and pixel indexes 3,4 along
    # the X axis.
    cube.mask_region(wcs.pix2sky([2.4, 3.8]), 1.1 * 3600.0, inside=False,
                     lmin=2, lmax=5)

    # The expected mask for the images between lmin and lmax.
    expected_mask = np.array([[True, True, True, True, True],
                              [True, True, True, True, True],
                              [True, True, True, False, False],
                              [True, True, True, False, False],
                              [True, True, True, True, True],
                              [True, True, True, True, True]], dtype=bool)
    assert_array_equal(np.any(cube.mask[:2, :, :], axis=0),
                       np.ones(cube.shape[1:]))
    assert_array_equal(np.all(cube._mask[2:5, :, :], axis=0), expected_mask)
    assert_array_equal(np.any(cube.mask[5:, :, :], axis=0),
                       np.ones(cube.shape[1:]))
    cube.unmask()

    # Mask outside an elliptical region centered at pixel 3.5,3.5.
    # The boolean expected_mask array given below was a verified
    # output of mask_ellipse() for the specified ellipse parameters.
    cube = generate_cube(shape=(10, 8, 8), wcs=wcs, var=None)
    cube.mask_ellipse([3.5, 3.5], (2.5, 3.5), 45.0, unit_radius=None,
                      unit_center=None, inside=False, lmin=2, lmax=5)
    expected_mask = np.array([[True, True, True, True, True, True, True, True],
                              [True, True, True, False, False, False, True, True],
                              [True, True, False, False, False, False, False, True],
                              [True, False, False, False, False, False, False, True],
                              [True, False, False, False, False, False, False, True],
                              [True, False, False, False, False, False, True, True],
                              [True, True, False, False, False, True, True, True],
                              [True, True, True, True, True, True, True, True]],
                             dtype=bool)
    assert_array_equal(np.any(cube.mask[:2, :, :], axis=0),
                       np.ones(cube.shape[1:]))
    assert_array_equal(np.all(cube._mask[2:5, :, :], axis=0), expected_mask)
    assert_array_equal(np.any(cube.mask[5:, :, :], axis=0),
                       np.ones(cube.shape[1:]))

    # Check that we can select the same mask via the output of np.where()
    # passed to mask_selection().
    ksel = np.where(cube.data.mask)
    cube.unmask()
    cube.mask_selection(ksel)
    assert_array_equal(np.any(cube.mask[:2, :, :], axis=0),
                       np.ones(cube.shape[1:]))
    assert_array_equal(np.all(cube._mask[2:5, :, :], axis=0), expected_mask)
    assert_array_equal(np.any(cube.mask[5:, :, :], axis=0),
                       np.ones(cube.shape[1:]))

    # The cube was generated without any variance information.
    # Check that mask_variance() raises an error due to this.
    with nose.tools.assert_raises(ValueError):
        cube.mask_variance(0.1)

    # Add an array of variances to the cube and check that mask_variance()
    # masks all pixels that have variances greater than 0.1.
    cube.unmask()
    cube.var = np.random.randn(*cube.shape)
    mask = cube.var > 0.1
    cube.mask_variance(0.1)
    assert_array_equal(cube.data.mask, mask)


@attr(speed='fast')
def test_truncate():
    """Cube class: testing truncation"""
    cube1 = generate_cube(data=2, wave=WaveCoord(crval=1))
    coord = [2, 0, 1, 5, 1, 3]
    cube2 = cube1.truncate(coord, unit_wcs=cube1.wcs.unit,
                           unit_wave=cube1.wave.unit)
    assert_array_equal(cube2.shape, (4, 2, 3))
    assert_array_equal(cube2.get_start(), (2, 0, 1))
    assert_array_equal(cube2.get_end(), (5, 1, 3))


@attr(speed='fast')
def test_sum():
    """Cube class: testing sum method"""
    cube1 = generate_cube(data=1, wave=WaveCoord(crval=1))
    ind = np.arange(10)
    refsum = ind.sum()
    cube1.data = (ind[:, np.newaxis, np.newaxis] *
                  np.ones((6, 5))[np.newaxis, :, :])
    assert_equal(cube1.sum(), 6 * 5 * refsum)
    assert_array_equal(cube1.sum(axis=0).data, np.full((6, 5), refsum, float))
    weights = np.ones(shape=(10, 6, 5))
    assert_equal(cube1.sum(weights=weights), 6 * 5 * refsum)

    weights = np.ones(shape=(10, 6, 5)) * 2
    assert_equal(cube1.sum(weights=weights), 6 * 5 * refsum)

    assert_array_equal(cube1.sum(axis=(1, 2)).data, ind * 6 * 5)


@attr(speed='fast')
def test_mean():
    """Cube class: testing mean method"""

    # Specify the dimensions of the test cube.
    shape = (2, 3, 4)

    # Create 3D data, variance and mask arrays.
    d = np.arange(np.asarray(shape).prod(), dtype=float).reshape(shape)
    v = d / 10.0
    m = d % 5 == 0

    # Create a cube with the above values.
    cube = generate_cube(data=d, var=v, mask=m)

    # Create masked array versions of the data and variance arrays.
    data = ma.array(d, mask=m)
    var = ma.array(v, mask=m)

    # Test the mean over each of the supported axes.
    for axis in (None, 0, (1, 2)):

        # Test the two weighting options. The loop-variable tuple
        # specifies the weights argument of cube.mean(), and a masked
        # array of the weights that are expected to be used.
        for weights, w in [(None, np.ones(shape, dtype=float)),  # Unweighted
                           (np.sqrt(d), np.sqrt(d))]:           # Weighted

            # Compute the mean.
            mean = cube.mean(axis=axis, weights=weights)

            # Mask the array of the expected weights.
            w = ma.array(w, mask=m)

            # Compute the expected values of the mean and its variance,
            # using a different implementation to cube.mean().
            expected_data = np.sum(data * w, axis=axis) / np.sum(w, axis=axis)
            expected_var = np.sum(var * w**2, axis=axis) / np.sum(w, axis=axis)**2

            # In the case any of the following tests fail, provide an
            # error message that indicates which arguments were passed
            # to cube.mean()
            errmsg = "Failure of cube.mean(axis=%s, weights=%s)" % (axis, weights)

            # See if the mean pixels and variances of these pixels have the
            # expected values.
            if axis is None:
                assert_allclose(mean, expected_data, err_msg=errmsg)
            else:
                assert_allclose(mean.data, expected_data, err_msg=errmsg)
                assert_allclose(mean.var, expected_var, err_msg=errmsg)


@attr(speed='fast')
def test_median():
    """Cube class: testing median methods"""
    cube1 = generate_cube(data=1., wave=WaveCoord(crval=1))
    ind = np.arange(10)
    median = np.median(ind)
    cube1.data = (ind[:, np.newaxis, np.newaxis] *
                  np.ones((6, 5))[np.newaxis, :, :])

    m = cube1.median()
    assert_equal(m, median)
    m = cube1.median(axis=0)
    assert_equal(m[3, 3], median)
    m = cube1.median(axis=(1, 2))
    assert_array_equal(m.data, ind)

    with nose.tools.assert_raises(ValueError):
        m = cube1.median(axis=-1)


@attr(speed='fast')
def test_rebin():
    """Cube class: testing rebin methods"""

    # Create spectral and spatial world coordinates that make each
    # pixel equal its index.

    wcs = WCS(crval=(0, 0), crpix=(1, 1), cdelt=(1.0, 1.0))
    wave = WaveCoord(crval=0.0, crpix=1.0, cdelt=1.0)

    # Create a cube with even valued dimensions, filled with ones.
    data = np.ma.ones((4, 6, 8))       # Start with all pixels 1.0
    data.reshape(4 * 6 * 8)[::2] = 0.0   # Set every second pixel to 0.0
    data.mask = data < -1            # Unmask all pixels.
    cube1 = generate_cube(data=data.data, mask=data.mask, wcs=wcs, wave=wave)

    # Rebin each of the axes such as to leave only 2 pixels along each
    # dimension.
    factor = np.array([2, 3, 4])
    cube2 = cube1.rebin(factor=factor)

    # Compute the expected output cube, given that the input cube is a
    # repeating pattern of 1,0, and we divided the x-axis by a
    # multiple of 2, the output pixels should all be 0.5.
    expected = np.ma.ones((2, 2, 2)) * 0.5
    expected.mask = expected < 0  # All pixels unmasked.
    assert_masked_allclose(cube2.data, expected)

    # Do the same experiment but with the zero valued pixels all masked.
    data = np.ma.ones((4, 6, 8))       # Start with all pixels 1.0
    data.reshape(4 * 6 * 8)[::2] = 0.0   # Set every second pixel to 0.0
    data.mask = data < 0.1           # Mask the pixels that are 0.0
    cube1 = generate_cube(data=data.data, mask=data.mask, wcs=wcs, wave=wave)

    # Rebin each of the axes such as to leave only 2 pixels along each
    # dimension.
    factor = np.array([2, 3, 4])
    cube2 = cube1.rebin(factor=factor)

    # Compute the expected output cube. The zero valued pixels are all
    # masked, leaving just pixels with values of 1, so the mean that is
    # recorded in each output pixel should be 1.0.
    expected = np.ma.ones((2, 2, 2)) * 1.0
    expected.mask = expected < 0  # All output pixels should be unmasked.
    assert_masked_allclose(cube2.data, expected)

    # Check that the world coordinates are correct.  We averaged
    # factor[] pixels whose coordinates were equal to their pixel
    # indexes, so the coordinates of the first pixel of the rebinned
    # cube should be the mean of the first factor indexes along each
    # dimension. The sum from k=0 to factor-1 is
    # ((factor-1)*factor)/2, and dividing this by the number of pixels
    # gives (factor-1)/2.
    assert_allclose(np.asarray(cube2.get_start()), (factor - 1) / 2.0)

    # Create a cube that has a larger number of pixels along the
    # y and x axes of the images, so that we can divide those axes
    # by a number whose remainder is large enough to test selection
    # of the truncated part of the cube.
    shape = np.array([4, 17, 15])
    data = np.ma.ones(shape)                # Start with all pixels 1.0
    data.reshape(shape.prod())[::2] = 0.0   # Set every second pixel to 0.0
    data.mask = data < -1                   # Unmask all pixels.
    cube1 = generate_cube(data=data.data, mask=data.mask, wcs=wcs, wave=wave)

    # Choose the rebinning factors such that there is a significant
    # remainder after dividing the final two dimensions of the cube by
    # the specified factor. We don't do this for the first axis because
    # we want the interleaved pattern of 0s and 1s to remain.
    factor = np.array([2, 7, 9])
    cube2 = cube1.rebin(factor=factor, margin='origin')

    # Compute the expected output cube. Given that the input cube is a
    # repeating pattern of 1,0, and we divided the x-axis by a
    # multiple of 2, the output pixels should all be 0.5.
    expected_shape = cube1.shape // factor
    expected = np.ma.ones(expected_shape) * 0.5
    expected.mask = expected < 0  # All pixels unmasked.
    assert_masked_allclose(cube2.data, expected)

    # We chose a margin value of 'origin', so that the outer corner of
    # pixel 0,0,0 of the input cube would also be the outer corner of
    # pixel 0,0,0 of the rebinned cube. The output world coordinates
    # can be calculated as described for the previous test.
    assert_allclose(np.asarray(cube2.get_start()), (factor - 1) / 2.0)

    # Do the same test, but with margin='center'.
    cube2 = cube1.rebin(factor=factor, margin='center')

    # Compute the expected output cube. The values should be the
    # same as the previous test.
    expected_shape = cube1.shape // factor
    expected = np.ma.ones(expected_shape) * 0.5
    expected.mask = expected < 0  # All pixels unmasked.
    assert_masked_allclose(cube2.data, expected)

    # We chose a margin value of 'center'. We need to know which
    # pixels should have contributed to pixel 0,0,0. First calculate
    # how many pixels remain after dividing the shape by the reduction
    # factors. This is the number of extra pixels that should have been
    # discarded. Divide this by 2 to determine the number of pixels that
    # are removed from the start of each axis.
    cut = np.mod(shape, factor).astype(int) // 2

    # The world coordinates of the input cube were equal to its pixel
    # indexes, and the world coordinates of a pixel of the output cube
    # is thus the mean of the indexes of the pixels that were combined
    # to make that pixel. In this case, we combine pixels
    # data[cut:cut+factor] along each axis. This can be calculated as
    # the sum of pixel indexes from 0 to cut+factor, minus the sum of
    # pixel indexes from 0 to cut, with the result divided by the number
    # of pixels that were averaged. Again we make use of the series sum,
    # sum[n=0..N] = (n*(n-1))/2.
    tmp = cut + factor
    assert_allclose(np.asarray(cube2.get_start()),
                    ((tmp * (tmp - 1)) / 2.0 - (cut * (cut - 1)) / 2.0) / factor)


@attr(speed='fast')
def test_get_image():
    """Cube class: testing get_image method"""
    shape = (2000, 6, 5)
    wave = WaveCoord(crpix=1, cdelt=3.0, crval=2000, cunit=u.angstrom, shape=shape[0])
    wcs = WCS(crval=(0, 0))
    data = np.ones(shape=shape) * 2
    cube1 = Cube(data=data, wave=wave, wcs=wcs)

    # Add a gaussian shaped spectral line at image pixel 2,2.
    cube1[:, 2, 2].add_gaussian(5000, 1200, 20)

    # Specify the range of wavelengths to be combined and the corresponding
    # slice along the wavelength axis. The wavelength range is chosen to
    # include all wavelengths affected by add_gaussian().
    lrange = (4800, 5200)
    lslice = slice(np.rint(cube1.wave.pixel(lrange[0])).astype(int),
                   np.rint(cube1.wave.pixel(lrange[1])).astype(int) + 1)

    # Get an image that is the mean of all images in the above wavelength
    # range, minus a background image estimated from outside this range,
    # where all image values are 2.0.
    ima = cube1.get_image(wave=lrange, is_sum=False, subtract_off=True)

    # In the cube, all spectral pixels of image pixel 0,0 were 2.0,
    # so the mean over the desired wavelength range, minus the mean
    # over background ranges either side of this should be zero.
    assert_equal(ima[0, 0], 0)

    # Spectral pixels of image pixel 2,2 in the original cube were all
    # 2.0 everywhere except where the gaussian was added to 2.0. Hence
    # pixel 2,2 of the mean image should be the mean of the gaussian
    # minus the average 2.0 background.
    nose.tools.assert_almost_equal(ima[2, 2],
                                   cube1[lslice, 2, 2].mean() - 2, 3)

    # Get another mean image, but this time without subtracting off a
    # background.  Image pixel 0,0 should have a mean of 2.0, and
    # pixel 2,2 should equal the mean of the gaussian added to 2.0
    ima = cube1.get_image(wave=lrange, is_sum=False, subtract_off=False)
    assert_equal(ima[0, 0], 2)
    nose.tools.assert_almost_equal(ima[2, 2], cube1[lslice, 2, 2].mean(), 3)

    # For this test, perform a sum over the chosen wavelength range,
    # and subtract off a background image taken from wavelength
    # regions above and below the wavelength range. Pixel 0,0 of the
    # summed image should be zero, since both the summed wavelength
    # range and the background image wavelength ranges have the same
    # pixel values, and the background sum is scaled to have the same
    # units as the output image. Check the background subtraction of
    # pixel 2,2 using an equal number of pixels that were not affected
    # by the addition of the gaussian.
    ima = cube1.get_image(wave=lrange, is_sum=True, subtract_off=True)
    assert_equal(ima[0, 0], 0)
    nose.tools.assert_almost_equal(ima[2, 2], cube1[lslice, 2, 2].sum() -
                                   cube1[lslice, 0, 0].sum(), 3)

    # Finally, perform a sum of the chosen wavelength range without
    # subtracting a background image. This is easy to test by doing
    # equivalent sums through the cube over the chosen wavelength range.
    ima = cube1.get_image(wave=lrange, is_sum=True, subtract_off=False)
    assert_equal(ima[0, 0], cube1[lslice, 0, 0].sum())
    nose.tools.assert_almost_equal(ima[2, 2], cube1[lslice, 2, 2].sum())


@attr(speed='fast')
def test_subcube():
    """Cube class: testing sub-cube extraction methods"""
    wcs = WCS(crval=(0, 0), crpix=1.0, deg=True)
    cube1 = generate_cube(data=1, wave=WaveCoord(crval=1))

    # Extract a sub-cube whose images are centered close to pixel
    # (2.3, 2.8) of the cube and have a width and height of 2 pixels.
    # The center of a 2x2 pixel region is at the shared corner between
    # these pixels, and the closest corner to the requested center
    # of 2.3,2.8 is 2.5,2.5. Thus the sub-images should be from pixels
    # 2,3 along both the X and Y axes.

    cube2 = cube1.subcube(center=(2.3, 2.8), size=2, lbda=(5, 8),
                          unit_center=None, unit_size=None)
    assert_allclose(cube1.data[5:9, 2:4, 2:4], cube2.data)

    # The following should select the same image area as above,
    # followed by masking pixels in this area outside a circle
    # of radius 1.
    cube2 = cube1.subcube_circle_aperture(center=(2.3, 2.8), radius=1,
                                          unit_center=None, unit_radius=None)
    assert_equal(cube2.data.mask[0, 0, 0], True)
    assert_array_equal(cube2.get_start(), (1, 2, 2))
    assert_array_equal(cube2.shape, (10, 2, 2))


@attr(speed='fast')
def test_aperture():
    """Cube class: testing spectrum extraction"""
    cube = generate_cube(data=1, wave=WaveCoord(crval=1))
    spe = cube.aperture(center=(2, 2.8), radius=1,
                        unit_center=None, unit_radius=None)
    assert_equal(spe.shape[0], 10)
    assert_equal(spe.get_start(), 1)


@attr(speed='fast')
def test_write():
    """Cube class: testing write"""
    unit = u.Unit('1e-20 erg/s/cm2/Angstrom')
    cube = generate_cube(data=1, wave=WaveCoord(crval=1, cunit=u.angstrom),
                         unit=unit)
    cube.data[:, 0, 0] = ma.masked
    cube.var = np.ones_like(cube.data)
    fobj = NamedTemporaryFile()
    cube.write(fobj.name)

    hdu = fits.open(fobj)
    assert_array_equal(hdu[1].data.shape, cube.shape)
    assert_array_equal([h.name for h in hdu],
                       ['PRIMARY', 'DATA', 'STAT', 'DQ'])

    hdr = hdu[0].header
    assert_equal(hdr['AUTHOR'], 'MPDAF')

    hdr = hdu[1].header
    assert_equal(hdr['EXTNAME'], 'DATA')
    assert_equal(hdr['NAXIS'], 3)
    assert_equal(u.Unit(hdr['BUNIT']), unit)
    assert_equal(u.Unit(hdr['CUNIT3']), u.angstrom)
    assert_equal(hdr['NAXIS1'], cube.shape[2])
    assert_equal(hdr['NAXIS2'], cube.shape[1])
    assert_equal(hdr['NAXIS3'], cube.shape[0])
    for key in ('CRPIX1', 'CRPIX2'):
        assert_equal(hdr[key], 1.0)


@attr(speed='fast')
def test_get_item():
    """Cube class: testing __getitem__"""
    c = generate_cube(data=1, wave=WaveCoord(crval=1, cunit=u.angstrom))
    c.primary_header['KEY'] = 'primary value'
    c.data_header['KEY'] = 'data value'

    r = c[:, :2, :2]
    assert_array_equal(r.shape, (10, 2, 2))
    assert_equal(r.primary_header['KEY'], c.primary_header['KEY'])
    assert_equal(r.data_header['KEY'], c.data_header['KEY'])
    nose.tools.assert_true(isinstance(r, Cube))
    nose.tools.assert_true(r.wcs.isEqual(c.wcs[:2, :2]))
    nose.tools.assert_true(r.wave.isEqual(c.wave))

    r = c[0, :, :]
    assert_array_equal(r.shape, (6, 5))
    assert_equal(r.primary_header['KEY'], c.primary_header['KEY'])
    assert_equal(r.data_header['KEY'], c.data_header['KEY'])
    nose.tools.assert_true(isinstance(r, Image))
    nose.tools.assert_true(r.wcs.isEqual(c.wcs))
    nose.tools.assert_is_none(r.wave)

    r = c[:, 2, 2]
    assert_array_equal(r.shape, (10, ))
    assert_equal(r.primary_header['KEY'], c.primary_header['KEY'])
    assert_equal(r.data_header['KEY'], c.data_header['KEY'])
    nose.tools.assert_true(isinstance(r, Spectrum))
    nose.tools.assert_true(r.wave.isEqual(c.wave))
    nose.tools.assert_is_none(r.wcs)


def test_bandpass_image():
    """Cube class: testing bandpass_image"""

    shape = (7, 2, 2)

    # Create a rectangular shaped bandpass response whose ends are half
    # way into pixels.

    wavelengths = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sensitivities = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    # Specify a ramp for the values of the pixels in the cube versus
    # wavelength.

    spectral_values = np.arange(shape[0], dtype=float)

    # Specify a ramp for the variances of the pixels in the cube
    # versus wavelength.

    spectral_vars = np.arange(shape[0], dtype=float) * 0.5

    # Calculate the expected weights versus wavelength for each
    # spectral pixels. The weight of each pixel is supposed to be
    # integral of the sensitivity function over the width of the pixel.
    #
    #| 0  |  1  |  2  |  3  |  4  |  5  |  6  |  Pixel indexes
    #         _______________________
    # _______|                       |_________  Sensitivities
    #
    #  0.0  0.5   1.0   1.0   1.0   0.5   0.0    Weights
    #  0.0  1.0   2.0   3.0   4.0   5.0   6.0    Pixel values vs wavelength
    #  0.0  0.5   2.0   3.0   4.0   2.5   0.0    Pixel values * weights

    weights = np.array([0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0])

    # Compute the expected weighted mean of the spectral pixel values,
    # assuming that no pixels are unmasked.

    unmasked_mean = (weights * spectral_values).sum() / weights.sum()

    # Compute the expected weighted mean if pixel 1 is masked.

    masked_pixel = 1
    masked_mean = (((weights * spectral_values).sum() -
                    weights[masked_pixel] * spectral_values[masked_pixel]) /
                   (weights.sum() - weights[masked_pixel]))

    # Compute the expected variances of the unmasked and masked means.

    unmasked_var = (weights**2 * spectral_vars).sum() / weights.sum()**2
    masked_var = (((weights**2 * spectral_vars).sum() -
                   weights[masked_pixel]**2 * spectral_vars[masked_pixel]) /
                  (weights.sum() - weights[masked_pixel])**2)

    # Create the data array of the cube, giving all map pixels the
    # same data and variance spectrums.

    data = spectral_values[:, np.newaxis, np.newaxis] * np.ones(shape)
    var = spectral_vars[:, np.newaxis, np.newaxis] * np.ones(shape)

    # Create a mask with all pixels unmasked.

    mask = np.zeros(shape)

    # Mask spectral pixel 'masked_pixel' of map index 1,1.

    mask[masked_pixel, 1, 1] = True

    # Also mask all pixels of map pixel 0,0.

    mask[:, 0, 0] = True

    # Create a test cube with the above data and mask arrays.

    c = generate_cube(shape=shape, data=data, mask=mask, var=var,
                      wave=WaveCoord(crval=0.0, cdelt=1.0, crpix=1.0,
                                     cunit=u.angstrom))

    # Extract an image that has the above bandpass response.

    im = c.bandpass_image(wavelengths, sensitivities)

    # Only the map pixel in which all spectral pixels are masked should
    # be masked in the output, so just map pixel [0,0] should be masked.

    expected_mask = np.array([[True, False],
                              [False, False]], dtype=bool)

    # What do we expect?

    expected_data = np.ma.array(
        data=[[unmasked_mean, unmasked_mean], [unmasked_mean, masked_mean]],
        mask=expected_mask)

    expected_var = np.ma.array(
        data=[[unmasked_var, unmasked_var], [unmasked_var, masked_var]],
        mask=expected_mask)

    # Are the results consistent with the predicted values?

    assert_masked_allclose(im.data, expected_data)
    assert_masked_allclose(im.var, expected_var)
