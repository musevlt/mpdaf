"""Test on Image objects."""

from __future__ import absolute_import, division

import nose.tools
from nose.plugins.attrib import attr

import astropy.units as u
import numpy as np
import scipy.ndimage as ndi
import six

from mpdaf.obj import Image, WCS, gauss_image, moffat_image
from numpy.testing import assert_array_equal, assert_allclose

from ..utils import (assert_image_equal, generate_image, generate_cube,
                     generate_spectrum, assert_masked_allclose,
                     astronomical_image)

if six.PY2:
    from operator import add, sub, mul, div
else:
    from operator import add, sub, mul, truediv as div


@attr(speed='fast')
def test_copy():
    """Image class: testing copy method."""
    image1 = generate_image()
    image2 = image1.copy()
    s = image1.data.sum()
    image1[0, 0] = 10000
    nose.tools.assert_true(image1.wcs.isEqual(image2.wcs))
    nose.tools.assert_equal(s, image2.data.sum())


@attr(speed='fast')
def test_arithmetic_images():
    image1 = generate_image()
    image2 = generate_image(data=1, unit=u.Unit('2 ct'))

    for op in (add, sub, mul, div):
        image3 = op(image1, image2)
        assert_allclose((image3.data.data * image3.unit).value,
                        op(image1.data.data * image1.unit,
                           (image2.data.data * image2.unit).to(u.ct)).value)


@attr(speed='fast')
def test_arithmetic_scalar():
    image1 = generate_image()
    image1 += 4.2
    assert_allclose(image1.data, 2 + 4.2)
    image1 -= 4.2
    assert_allclose(image1.data, 2)
    image1 *= 4.2
    assert_allclose(image1.data, 2 * 4.2)
    image1 /= 4.2
    assert_allclose(image1.data, 2)

    for op in (add, sub, mul, div):
        assert_allclose(op(image1, 4.2).data, op(2, 4.2))
    for op in (add, sub, mul, div):
        assert_allclose(op(4.2, image1).data, op(4.2, 2))


@attr(speed='fast')
def test_arithmetic_cubes():
    image2 = generate_image(data=1, unit=u.Unit('2 ct'))
    cube1 = generate_cube(data=0.5, unit=u.Unit('2 ct'))

    for op in (add, sub, mul, div):
        assert_allclose(op(image2, cube1).data, op(image2.data, cube1.data))
        assert_allclose(op(cube1, image2).data, op(cube1.data, image2.data))


@attr(speed='fast')
def test_arithmetic_spectra():
    image1 = generate_image()
    spectrum1 = generate_spectrum()

    ref = spectrum1.data[:, np.newaxis, np.newaxis] * image1.data[..., :]
    assert_allclose((image1 * spectrum1).data, ref)
    assert_allclose((spectrum1 * image1).data, ref)

    image2 = (image1 * -2).abs() + (image1 + 4).sqrt() - 2
    assert_allclose(image2.data, np.abs(image1.data * -2) +
                    np.sqrt(image1.data + 4) - 2)


@attr(speed='fast')
def test_get():
    """Image class: testing getters"""
    image1 = generate_image()
    ima = image1[0:2, 1:4]
    assert_image_equal(ima, shape=(2, 3), start=(0, 1), end=(1, 3), step=(1, 1))


@attr(speed='fast')
def test_crop():
    """Image class: testing crop method"""
    # Create an image whose pixels are all masked.

    image1 = generate_image(shape=(9, 7), data=2.0, var=0.5, mask=True)

    # Create a masked array of unmasked values to be assigned to the
    # part of the image, with just a diamond shaped area of pixels
    # unmasked.

    diamond = np.ma.array(data=[[6.0, 2.0, 9.0],
                                [1.0, 4.0, 8.0],
                                [0.0, 5.0, 3.0]],
                          mask=[[True, False, True],
                                [False, False, False],
                                [True, False, True]])

    # Assign the above array to part of the image to clear the mask of
    # an irregular rectangular area of pixels there.

    image1.data[2:5, 1:4] = diamond

    # The following should crop all but the rectangular area that was
    # assigned above.

    image1.crop()

    # Check that the masked data array is as expected.

    assert_masked_allclose(image1.data, diamond)

    # The cropped variance array should look like the following array.

    expected_var = np.ma.array(data=[[0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5]], mask=diamond.mask)

    # Check that the masked variance array is as expected.

    assert_masked_allclose(image1.var, expected_var)

    # Check the WCS characteristics of the cropped image.

    assert_image_equal(image1, shape=(3, 3), start=(2, 1), end=(4, 3))
    nose.tools.assert_equal(image1.get_rot(), 0)


@attr(speed='fast')
def test_truncate():
    """Image class: testing truncation"""
    image1 = generate_image()
    image1 = image1.truncate(0, 1, 1, 3, unit=image1.wcs.unit)
    assert_image_equal(image1, shape=(2, 3), start=(0, 1), end=(1, 3))


@attr(speed='fast')
def test_gauss():
    """Image class: testing Gaussian fit"""
    wcs = WCS(cdelt=(0.2, 0.3), crval=(8.5, 12), shape=(40, 30))
    ima = gauss_image(wcs=wcs, fwhm=(2, 1), factor=1, rot=60, cont=2.0,
                      unit_center=u.pix, unit_fwhm=u.pix)
    # ima2 = gauss_image(wcs=wcs,width=(1,2),factor=2, rot = 60)
    gauss = ima.gauss_fit(cont=2.0, fit_back=False, verbose=False,
                          unit_center=None, unit_fwhm=None)
    nose.tools.assert_almost_equal(gauss.center[0], 19.5)
    nose.tools.assert_almost_equal(gauss.center[1], 14.5)
    nose.tools.assert_almost_equal(gauss.flux, 1)
    ima += 10.3
    gauss2 = ima.gauss_fit(cont=2.0 + 10.3, fit_back=True, verbose=False,
                           unit_center=None, unit_fwhm=None)
    nose.tools.assert_almost_equal(gauss2.center[0], 19.5)
    nose.tools.assert_almost_equal(gauss2.center[1], 14.5)
    nose.tools.assert_almost_equal(gauss2.flux, 1)
    nose.tools.assert_almost_equal(gauss2.cont, 12.3)


@attr(speed='fast')
def test_moffat():
    """Image class: testing Moffat fit"""
    ima = moffat_image(wcs=WCS(crval=(0, 0)), flux=12.3, fwhm=(1.8, 1.8),
                       n=1.6, rot=0., cont=8.24, unit_center=u.pix,
                       unit_fwhm=u.pix)
    moffat = ima.moffat_fit(fit_back=True, verbose=False, unit_center=None,
                            unit_fwhm=None)
    nose.tools.assert_almost_equal(moffat.center[0], 50.)
    nose.tools.assert_almost_equal(moffat.center[1], 50.)
    nose.tools.assert_almost_equal(moffat.flux, 12.3)
    nose.tools.assert_almost_equal(moffat.fwhm[0], 1.8)
    nose.tools.assert_almost_equal(moffat.n, 1.6)
    nose.tools.assert_almost_equal(moffat.cont, 8.24)


@attr(speed='fast')
def test_mask():
    """Image class: testing mask functionalities"""
    wcs = WCS()
    data = np.ones(shape=(6, 5)) * 2

    # A region of half-width=1 and half-height=1 should have a size of
    # 2x2 pixels. A 2x2 region of pixels has a center at the shared
    # corner of the 4 pixels, and the closest corner to the requested
    # center of 2.1,1.8 is 2.5,1.5, so we expect the square of unmasked pixels
    # to be pixels 2,3 along the Y axis, and pixels 1,2 along the X axis.
    image1 = Image(data=data, wcs=wcs)
    image1.mask_region((2.1, 1.8), (1, 1), inside=False, unit_center=None,
                       unit_radius=None)
    expected_mask = np.array([[True,  True,  True,  True,  True],
                              [True,  True,  True,  True,  True],
                              [True,  False, False, True,  True],
                              [True,  False, False, True,  True],
                              [True,  True,  True,  True,  True],
                              [True,  True,  True,  True,  True]], dtype=bool)
    assert_array_equal(image1._mask, expected_mask)

    # Try exactly the same experiment as the above, except that the center
    # and size of the region are specified in world-coordinates instead of
    # pixels.
    wcs = WCS(deg=True)
    image1 = Image(data=data, wcs=wcs)
    image1.mask_region(wcs.pix2sky([2.1, 1.8]), (3600, 3600), inside=False)
    assert_array_equal(image1._mask, expected_mask)

    # Mask around a region of half-width and half-height 1.1 pixels,
    # specified in arcseconds, centered close to pixel 2.4,3.8. This
    # ideally corresponds to a region of 2.2x2.2 pixels. The closest
    # possible size is 2x2 pixels. A region of 2x2 pixels has its
    # center at the shared corner of these 4 pixels, and the nearest
    # corner to the desired central index of (2.4,3.8) is (2.5,3.5).
    # So all of the image should be masked, except for a 2x2 area of
    # pixel indexes 2,3 along the Y axis and pixel indexes 3,4 along
    # the X axis.
    image1.unmask()
    image1.mask_region(wcs.pix2sky([2.4, 3.8]), 1.1*3600.0, inside=False)
    expected_mask = np.array([[True,  True,  True,  True,  True],
                              [True,  True,  True,  True,  True],
                              [True,  True,  True,  False, False],
                              [True,  True,  True,  False, False],
                              [True,  True,  True,  True,  True],
                              [True,  True,  True,  True,  True]], dtype=bool)
    assert_array_equal(image1._mask, expected_mask)

    # Mask outside an elliptical region centered at pixel 3.5,3.5.
    # The boolean expected_mask array given below was a verified
    # output of mask_ellipse() for the specified ellipse parameters.
    data = np.ones(shape=(8, 8))
    image1 = Image(data=data, wcs=wcs)
    image1.mask_ellipse([3.5,3.5], (2.5,3.5), 45.0, unit_radius=None,
                        unit_center=None, inside=False)
    expected_mask = np.array([[True, True, True, True, True, True, True, True],
                              [True, True, True,False,False,False, True, True],
                              [True, True,False,False,False,False,False, True],
                              [True,False,False,False,False,False,False, True],
                              [True,False,False,False,False,False,False, True],
                              [True,False,False,False,False,False, True, True],
                              [True, True,False,False,False, True, True, True],
                              [True, True, True, True, True, True, True, True]],
                             dtype=bool)
    assert_array_equal(image1._mask, expected_mask)

    # Use np.where to select the masked pixels and check that mask_selection()
    # then reproduces the same mask.
    ksel = np.where(image1.data.mask)
    image1.unmask()
    image1.mask_selection(ksel)
    assert_array_equal(image1._mask, expected_mask)


@attr(speed='fast')
def test_background():
    """Image class: testing background value"""
    wcs = WCS()
    data = np.ones(shape=(6, 5)) * 2
    image1 = Image(data=data, wcs=wcs)
    (background, std) = image1.background()
    nose.tools.assert_equal(background, 2)
    nose.tools.assert_equal(std, 0)
    ima = astronomical_image()
    (background, std) = ima[1647:1732, 618:690].background()
    # compare with IRAF results
    nose.tools.assert_true((background - std < 1989) & (background + std > 1989))


@attr(speed='fast')
def test_peak():
    """Image class: testing peak research"""
    wcs = WCS()
    data = np.ones(shape=(6, 5)) * 2
    image1 = Image(data=data, wcs=wcs)
    image1.data[2, 3] = 8
    p = image1.peak()
    nose.tools.assert_equal(p['p'], 2)
    nose.tools.assert_equal(p['q'], 3)
    ima = astronomical_image()
    p = ima.peak(center=(790, 875), radius=20, plot=False, unit_center=None,
                 unit_radius=None)
    nose.tools.assert_almost_equal(p['p'], 793.1, 1)
    nose.tools.assert_almost_equal(p['q'], 875.9, 1)


@attr(speed='fast')
def test_rotate():
    """Image class: testing rotation"""

    # Read a test image.

    before = astronomical_image()

    # Get the maximum of the image.

    peak = before.data.max()

    # Choose a few test pixels at different locations in the input
    # image.

    old_pixels = np.array([
        [before.shape[0] // 4, before.shape[1] // 4],
        [before.shape[0] // 4, (3 * before.shape[1]) // 4],
        [(3 * before.shape[0]) // 4, before.shape[1] // 4],
        [(3 * before.shape[0]) // 4, (3 * before.shape[1]) // 4],
        [before.shape[0] // 2, before.shape[1] // 2]])

    # Get the sky coordinates of the test pixels.

    coords = before.wcs.pix2sky(old_pixels)

    # Set 3x3 squares of pixels around each of the test pixels
    # in the input image to a large value that we can later
    # distinguish from other pixels in the rotated image. We
    # set a square rather than a single pixel to ensure that
    # the output pixel is interpolated from identically valued
    # pixels on either side of it in the input image.

    test_value = 10 * peak
    for pixel in old_pixels:
        py = pixel[0]
        px = pixel[1]
        before.data[py - 1:py + 2, px - 1:px + 2] = test_value

    # Get a rotated copy of the image.

    after = before.rotate(30, reshape=True, regrid=True)

    # See in which pixels the coordinate matrix of the rotated
    # image says that the test sky coordinates should now be found
    # after the rotation.

    new_pixels = np.asarray(np.rint(after.wcs.sky2pix(coords)), dtype=int)

    # Check that the values of the nearest pixels to the rotated
    # locations hold the test value that we wrote to the image before
    # rotation.

    for pixel in new_pixels:
        py = pixel[0]
        px = pixel[1]
        nose.tools.assert_almost_equal(after.data[py, px], test_value, places=6)

    # If both the WCS and the image were rotated wrongly in the same
    # way, then the above test will incrorrectly claim that the
    # rotation worked, so now check that the WCS was rotated correctly.
    assert_allclose(after.wcs.get_rot() - before.wcs.get_rot(), 30.0)


@attr(speed='fast')
def test_resample():
    """Image class: Testing the resample method"""

    # Read a test image.

    before = astronomical_image()

    # What scale factors shall we multiply the input step size by?

    xfactor = 3.5
    yfactor = 4.3

    # Get the maximum of the input image.

    peak = before.data.max()

    # Choose a few test pixels at different locations in the input
    # image.

    old_pixels = np.array([
        [before.shape[0] // 4, before.shape[1] // 4],
        [before.shape[0] // 4, (3 * before.shape[1]) // 4],
        [(3 * before.shape[0]) // 4, before.shape[1] // 4],
        [(3 * before.shape[0]) // 4, (3 * before.shape[1]) // 4],
        [before.shape[0] // 2, before.shape[1] // 2]])

    # Get the sky coordinates of the test pixels.

    coords = before.wcs.pix2sky(old_pixels)

    # Assign a large value to each of the above test pixels, so that
    # we can distinguish the smoothed version of it in the output
    # image.

    test_value = 2 * peak
    for pixel in old_pixels:
        py = pixel[0]
        px = pixel[1]
        before.data[py, px] = test_value

    # Resample the image.

    newstep = np.array(
        [before.get_step(unit=u.arcsec)[0] * yfactor,
         before.get_step(unit=u.arcsec)[1] * xfactor])
    after = before.resample(newdim=before.shape, newstart=None,
                            newstep=newstep, flux=False)

    # See in which pixels the coordinate matrix of the rotated
    # image says that the test sky coordinates should now be found
    # after the rotation.

    new_pixels = np.asarray(np.rint(after.wcs.sky2pix(coords)), dtype=int)

    # Check that the first image moments of the resampled test pixels
    # are at the expected places.

    pad = 10
    for pixel in new_pixels:
        py = pixel[0]
        px = pixel[1]
        offset = ndi.center_of_mass(after.data[py - pad:py + pad + 1,
                                               px - pad:px + pad + 1])
        assert_allclose(offset, np.array([pad, pad]), rtol=0, atol=0.1)


@attr(speed='fast')
def test_inside():
    """Image class: testing inside method."""
    ima = astronomical_image()
    nose.tools.assert_equal(ima.inside((39.951088, -1.4977398), unit=ima.wcs.unit), False)


@attr(speed='fast')
def test_subimage():
    """Image class: testing sub-image extraction."""
    ima = astronomical_image()
    subima = ima.subimage(center=(790, 875), size=40, unit_center=None, unit_size=None)
    nose.tools.assert_equal(subima.peak()['data'], 3035.0)


@attr(speed='fast')
def test_ee():
    """Image class: testing ensquared energy."""
    wcs = WCS()
    data = np.ones(shape=(6, 5)) * 2
    image1 = Image(data=data, wcs=wcs)
    image1.mask_region((2, 2), (1.5, 1.5), inside=False, unit_center=None,
                       unit_radius=None)
    nose.tools.assert_equal(image1.ee(), 9 * 2)
    ee = image1.ee(center=(2, 2), unit_center=None, radius=1, unit_radius=None)
    nose.tools.assert_equal(ee, 4 * 2)
    r, eer = image1.eer_curve(center=(2, 2), unit_center=None,
                              unit_radius=None, cont=0)
    nose.tools.assert_equal(r[1], 1.0)
    nose.tools.assert_equal(eer[1], 1.0)
    size = image1.ee_size(center=(2, 2), unit_center=None, unit_size=None,
                          cont=0)
    nose.tools.assert_almost_equal(size[0], 1.775)


@attr(speed='fast')
def test_rebin():
    """Image class: testing rebin methods."""
    wcs = WCS(crval=(0, 0))
    data = np.arange(30).reshape(6, 5)
    image1 = Image(data=data, wcs=wcs, var=np.ones(data.shape) * 0.5)
    image1.mask_region((2, 2), (1.5, 1.5), inside=False, unit_center=None,
                       unit_radius=None)

    # The test data array looks as follows:
    #
    # ---- ---- ---- ---- ----
    # ----  6.0  7.0  8.0 ----
    # ---- 11.0 12.0 13.0 ----
    # ---- 16.0 17.0 18.0 ----
    # ---- ---- ---- ---- ----
    # ---- ---- ---- ---- ----
    #
    # Where ---- signifies a masked value.
    #
    # After reducing both dimensions by a factor of 2, we should
    # get a data array of the following 6 means of 4 pixels each:
    #
    #  ---- ---- => 6/1         ---- ---- => (7+8)/2
    #  ----  6.0                 7.0  8.0
    #
    #  ---- 11.0 => (11+16)/2   12.0 13.0 => (12+13+17+18)/4
    #  ---- 16.0                17.0 18.0
    #
    #  ---- ---- => ----        ---- ---- => ----
    #  ---- ----                ---- ----

    expected = np.ma.array(
        data=[[6.0, 7.5], [13.5, 15], [0.0, 0.0]],
        mask=[[False, False], [False, False], [True, True]])
    image2 = image1.rebin(2)
    assert_masked_allclose(image2.data, expected)

    # The variances of the original pixels were all 0.5, so taking the
    # mean of N of these should give the mean a variance of 0.5/N.
    # Given the number of pixels averaged in each of the above means,
    # we thus expect the variance array to look as follows.

    expected = np.ma.array(data=[[0.5, 0.25], [0.25, 0.125], [0.0, 0.0]],
                           mask=[[False, False], [False, False], [True, True]])
    assert_masked_allclose(image2.var, expected)

    # Check the WCS information.

    start = image2.get_start()
    nose.tools.assert_equal(start[0], 0.5)
    nose.tools.assert_equal(start[1], 0.5)

@attr(speed='fast')
def test_fftconvolve():
    """Image class: testing convolution methods."""
    wcs = WCS(cdelt=(0.2, 0.3), crval=(8.5, 12), shape=(40, 30), deg=True)
    data = np.zeros((40, 30))
    data[19, 14] = 1
    ima = Image(wcs=wcs, data=data)
    ima2 = ima.fftconvolve_gauss(center=None, flux=1., fwhm=(20000., 10000.),
                                 peak=False, rot=60., factor=1,
                                 unit_center=u.deg, unit_fwhm=u.arcsec)
    g = ima2.gauss_fit(verbose=False)
    nose.tools.assert_almost_equal(g.fwhm[0], 20000, 2)
    nose.tools.assert_almost_equal(g.fwhm[1], 10000, 2)
    nose.tools.assert_almost_equal(g.center[0], 8.5)
    nose.tools.assert_almost_equal(g.center[1], 12)
    ima2 = ima.fftconvolve_moffat(center=None, flux=1., a=10000, q=1, n=2,
                                  peak=False, rot=60., factor=1,
                                  unit_center=u.deg, unit_a=u.arcsec)
    m = ima2.moffat_fit(verbose=False)
    nose.tools.assert_almost_equal(m.center[0], 8.5)
    nose.tools.assert_almost_equal(m.center[1], 12)
    # ima3 = ima.correlate2d(np.ones((40, 30)))
