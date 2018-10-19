"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>
Copyright (c) 2016-2017 Laure Piqueras <laure.piqueras@univ-lyon1.fr>

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

import astropy.units as u
import numpy as np
import pytest
import scipy.ndimage as ndi

from mpdaf.obj import Image, WCS, gauss_image, moffat_image
from numpy.testing import (assert_array_equal, assert_allclose,
                           assert_almost_equal, assert_equal,
                           assert_array_almost_equal)
from operator import add, sub, mul, truediv as div

from ...tests.utils import (assert_image_equal, generate_image, generate_cube,
                            assert_masked_allclose)


def test_copy(image):
    """Image class: testing copy method."""
    image2 = image.copy()
    s = image.data.sum()
    image[0, 0] = 10000
    assert image.wcs.isEqual(image2.wcs)
    assert s == image2.data.sum()


def test_arithmetic_images(image):
    image2 = generate_image(data=1, unit=u.Unit('2 ct'))

    for op in (add, sub, mul, div):
        image3 = op(image, image2)
        assert_allclose((image3.data.data * image3.unit).value,
                        op(image.data.data * image.unit,
                           (image2.data.data * image2.unit).to(u.ct)).value)


def test_arithmetic_scalar(image):
    image += 4.2
    assert_allclose(image.data, 2 + 4.2)
    image -= 4.2
    assert_allclose(image.data, 2)
    image *= 4.2
    assert_allclose(image.data, 2 * 4.2)
    image /= 4.2
    assert_allclose(image.data, 2)

    for op in (add, sub, mul, div):
        assert_allclose(op(image, 4.2).data, op(2, 4.2))
    for op in (add, sub, mul, div):
        assert_allclose(op(4.2, image).data, op(4.2, 2))


def test_arithmetic_cubes():
    image2 = generate_image(data=1, unit=u.Unit('2 ct'))
    cube1 = generate_cube(data=0.5, unit=u.Unit('2 ct'))

    for op in (add, sub, mul, div):
        assert_allclose(op(image2, cube1).data, op(image2.data, cube1.data))
        assert_allclose(op(cube1, image2).data, op(cube1.data, image2.data))


def test_arithmetic_spectra(image, spectrum):
    ref = spectrum.data[:, np.newaxis, np.newaxis] * image.data[..., :]
    assert_allclose((image * spectrum).data, ref)
    assert_allclose((spectrum * image).data, ref)

    image2 = (image * -2).abs() + (image + 4).sqrt() - 2
    assert_allclose(image2.data, np.abs(image.data * -2) +
                    np.sqrt(image.data + 4) - 2)


def test_get(image):
    """Image class: testing getters"""
    ima = image[0:2, 1:4]
    assert_image_equal(ima, shape=(2, 3), start=(0, 1), end=(1, 3),
                       step=(1, 1))


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
    assert image1.get_rot() == 0


def test_truncate(image):
    """Image class: testing truncation"""
    image_orig = image.copy()
    new_image = image.truncate(0, 1, 1, 3, unit=image.wcs.unit)
    new_image[:] = 10
    assert_array_equal(image_orig.data, image.data)
    assert_image_equal(new_image, shape=(2, 3), start=(0, 1), end=(1, 3))

    image.truncate(0, 1, 1, 3, unit=image.wcs.unit, inplace=True)
    assert_image_equal(image, shape=(2, 3), start=(0, 1), end=(1, 3))


@pytest.mark.parametrize('fwhm', (None, (3., 3.)))
@pytest.mark.parametrize('flux', (None, 10))
@pytest.mark.parametrize('factor', (1, 2))
@pytest.mark.parametrize('weight', (True, False))
@pytest.mark.parametrize('fit_back,cont', ((True, 0), (False, 2.0)))
@pytest.mark.parametrize('center,pos_min,pos_max',
                         ((None, None, None),
                          ((18., 15.), (15., 12), (24, 20))))
def test_gauss(fwhm, flux, factor, weight, fit_back, cont, center,
               pos_min, pos_max):
    """Image class: testing Gaussian fit"""
    params = dict(fwhm=(2, 1), rot=60, cont=2.0, flux=5.)
    wcs = WCS(cdelt=(0.2, 0.3), crval=(8.5, 12), shape=(40, 30))
    ima = gauss_image(wcs=wcs, factor=factor, unit_center=u.pix,
                      unit_fwhm=u.pix, **params)
    ima._var = np.ones_like(ima._data)
    gauss = ima.gauss_fit(fit_back=fit_back, cont=cont, verbose=True,
                          center=center, pos_min=pos_min, pos_max=pos_max,
                          factor=factor, fwhm=fwhm, flux=flux, weight=weight,
                          unit_center=None, unit_fwhm=None, circular=False,
                          full_output=True, maxiter=200)
    assert isinstance(gauss.ima, Image)
    if factor == 1:
        assert_array_almost_equal(gauss.center, (19.5, 14.5))
    else:
        # FIXME: This must be fixed, when factor=2 center is wrong
        assert_array_almost_equal(gauss.center, (19.25, 14.25))

    for param, value in params.items():
        if np.isscalar(value):
            assert_almost_equal(getattr(gauss, param), value)
        else:
            assert_array_almost_equal(getattr(gauss, param), value)


@pytest.mark.parametrize('fwhm', (None, (3., 3.)))
@pytest.mark.parametrize('flux', (None, 10))
@pytest.mark.parametrize('factor', (1, 2))
@pytest.mark.parametrize('weight', (True, False))
@pytest.mark.parametrize('fit_back,cont', ((True, 0), (False, 2.0)))
@pytest.mark.parametrize('center,pos_min,pos_max',
                         ((None, None, None),
                          ((18, 15), (15, 12.), (24, 20))))
def test_gauss_circular(fwhm, flux, factor, weight, fit_back, cont, center,
                        pos_min, pos_max):
    """Image class: testing Gaussian fit"""
    params = dict(fwhm=(2, 2), cont=2.0, flux=5.)
    wcs = WCS(cdelt=(0.2, 0.2), crval=(8.5, 12), shape=(40, 30))
    ima = gauss_image(wcs=wcs, factor=factor, unit_center=u.pix,
                      unit_fwhm=u.pix, **params)
    ima._var = np.ones_like(ima._data)
    gauss = ima.gauss_fit(fit_back=fit_back, cont=cont, verbose=True,
                          center=center, pos_min=pos_min, pos_max=pos_max,
                          factor=factor, fwhm=fwhm, flux=flux, weight=weight,
                          unit_center=None, unit_fwhm=None, circular=True,
                          full_output=True, maxiter=200)
    assert isinstance(gauss.ima, Image)
    assert_array_almost_equal(gauss.center, (19.5, 14.5))

    for param, value in params.items():
        if np.isscalar(value):
            assert_almost_equal(getattr(gauss, param), value, 2)
        else:
            assert_array_almost_equal(getattr(gauss, param), value, 2)


@pytest.mark.parametrize('fwhm', (None, (3., 3.)))
@pytest.mark.parametrize('flux', (None,))  # 12.3))
@pytest.mark.parametrize('fit_n,n', ((True, 2.0), (False, 1.6)))
@pytest.mark.parametrize('fit_back,cont', ((True, 0), (False, 8.24)))
@pytest.mark.parametrize('center,pos_min,pos_max',
                         ((None, None, None),
                          ((49, 51), (40., 40.), (60, 60))))
def test_moffat_circular(fwhm, flux, fit_n, n, fit_back, cont, center,
                         pos_min, pos_max):
    """Image class: testing Moffat fit"""
    params = dict(flux=12.3, fwhm=(1.8, 1.8), n=1.6, cont=8.24,
                  center=(50., 50.))
    ima = moffat_image(wcs=WCS(cdelt=(1., 1.), crval=(0, 0)),
                       shape=(101, 101), rot=0., unit_center=u.pix,
                       unit_fwhm=u.pix, **params)

    moffat = ima.moffat_fit(fit_back=fit_back, cont=cont, verbose=True,
                            unit_center=None, unit_fwhm=None, full_output=True,
                            center=center, pos_min=pos_min, pos_max=pos_max,
                            fit_n=fit_n, n=n, circular=True, fwhm=fwhm,
                            flux=flux)
    assert isinstance(moffat.ima, Image)
    for param, value in params.items():
        if np.isscalar(value):
            assert_almost_equal(getattr(moffat, param), value, 2)
        else:
            assert_array_almost_equal(getattr(moffat, param), value, 2)


@pytest.mark.parametrize('fwhm', (None, (3., 3.)))
@pytest.mark.parametrize('flux', (None, ))  # 10))
@pytest.mark.parametrize('fit_n,n', ((True, 2.0), (False, 1.6)))
@pytest.mark.parametrize('fit_back,cont', ((True, 0), (False, 8.24)))
@pytest.mark.parametrize('center,pos_min,pos_max',
                         ((None, None, None),
                          ((49., 51.), (40, 40), (60., 60.))))
def test_moffat(fwhm, flux, fit_n, n, fit_back, cont, center,
                pos_min, pos_max):
    """Image class: testing Moffat fit"""
    params = dict(flux=12.3, fwhm=(2.8, 2.1), n=1.6, cont=8.24,
                  center=(50., 50.), rot=30)
    ima = moffat_image(wcs=WCS(cdelt=(1., 1.), crval=(0, 0)),
                       shape=(101, 101), unit_center=u.pix,
                       unit_fwhm=u.pix, **params)

    moffat = ima.moffat_fit(fit_back=fit_back, cont=cont, verbose=True,
                            unit_center=None, unit_fwhm=None, full_output=True,
                            center=center, pos_min=pos_min, pos_max=pos_max,
                            fit_n=fit_n, n=n, circular=False, fwhm=fwhm,
                            flux=flux)
    assert isinstance(moffat.ima, Image)
    for param, value in params.items():
        if np.isscalar(value):
            assert_almost_equal(getattr(moffat, param), value, 2)
        else:
            assert_array_almost_equal(getattr(moffat, param), value, 2)


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
    expected_mask = np.array([[1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 0, 0, 1, 1],
                              [1, 0, 0, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1]], dtype=bool)
    assert_array_equal(image1._mask, expected_mask)

    # Test that inside=True gives the opposite result
    image1.unmask()
    image1.mask_region((2.1, 1.8), (1, 1), inside=True, unit_center=None,
                       unit_radius=None)
    assert_array_equal(image1._mask, ~expected_mask)

    # And test with a rotation, 90° so should give the same result
    image1.unmask()
    image1.mask_region((2.1, 1.8), (1, 1), inside=True, unit_center=None,
                       unit_radius=None, posangle=90)
    assert_array_equal(image1._mask, ~expected_mask)

    # Try exactly the same experiment as the above, except that the center
    # and size of the region are specified in world-coordinates instead of
    # pixels.
    wcs = WCS(deg=True)
    image1 = Image(data=data, wcs=wcs)
    image1.mask_region(wcs.pix2sky([2.1, 1.8]), (3600, 3600), inside=False)
    assert_array_equal(image1._mask, expected_mask)

    # And same with a rotation
    image1.unmask()
    image1.mask_region(wcs.pix2sky([2.1, 1.8]), (3600, 3600), inside=True,
                       posangle=90)
    assert_array_equal(image1._mask, ~expected_mask)

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
    image1.mask_region(wcs.pix2sky([2.4, 3.8]), 1.1 * 3600.0, inside=False)
    expected_mask = np.array([[1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 0, 0],
                              [1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1]], dtype=bool)
    assert_array_equal(image1._mask, expected_mask)

    # Mask outside an elliptical region centered at pixel 3.5,3.5.
    # The boolean expected_mask array given below was a verified
    # output of mask_ellipse() for the specified ellipse parameters.
    data = np.ones(shape=(8, 8))
    image1 = Image(data=data, wcs=wcs)
    image1.mask_ellipse([3.5, 3.5], (2.5, 3.5), 45.0, unit_radius=None,
                        unit_center=None, inside=False)
    expected_mask = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=bool)
    assert_array_equal(image1._mask, expected_mask)

    # Use np.where to select the masked pixels and check that mask_selection()
    # then reproduces the same mask.
    ksel = np.where(image1.data.mask)
    image1.unmask()
    image1.mask_selection(ksel)
    assert_array_equal(image1._mask, expected_mask)

    # Check inside=True
    image1.unmask()
    image1.mask_ellipse([3.5, 3.5], (2.5, 3.5), 45.0, unit_radius=None,
                        unit_center=None, inside=True)
    assert_array_equal(image1._mask, ~expected_mask)


def test_background(a370II):
    """Image class: testing background value"""
    wcs = WCS()
    data = np.ones(shape=(6, 5)) * 2
    image1 = Image(data=data, wcs=wcs)
    (background, std) = image1.background()
    assert background == 2
    assert std == 0
    (background, std) = a370II[1647:1732, 618:690].background()
    # compare with IRAF results
    assert (background - std < 1989) & (background + std > 1989)


def test_peak(a370II):
    """Image class: testing peak research"""
    wcs = WCS()
    data = np.ones(shape=(6, 5)) * 2
    image1 = Image(data=data, wcs=wcs)
    image1.data[2, 3] = 8
    p = image1.peak()
    assert p['p'] == 2
    assert p['q'] == 3
    p = a370II.peak(center=(790, 875), radius=20, plot=False, unit_center=None,
                    unit_radius=None)
    assert_almost_equal(p['p'], 793.1, 1)
    assert_almost_equal(p['q'], 875.9, 1)


def test_rotate(a370II):
    """Image class: testing rotation"""

    peak = a370II.data.max()

    # Choose a few test pixels at different locations in the input image.
    old_pixels = np.array([
        [a370II.shape[0] // 4, a370II.shape[1] // 4],
        [a370II.shape[0] // 4, (3 * a370II.shape[1]) // 4],
        [(3 * a370II.shape[0]) // 4, a370II.shape[1] // 4],
        [(3 * a370II.shape[0]) // 4, (3 * a370II.shape[1]) // 4],
        [a370II.shape[0] // 2, a370II.shape[1] // 2]])

    # Get the sky coordinates of the test pixels.
    coords = a370II.wcs.pix2sky(old_pixels)

    # Set 3x3 squares of pixels around each of the test pixels
    # in the input image to a large value that we can later
    # distinguish from other pixels in the rotated image. We
    # set a square rather than a single pixel to ensure that
    # the output pixel is interpolated from identically valued
    # pixels on either side of it in the input image.

    test_value = peak
    for pixel in old_pixels:
        py = pixel[0]
        px = pixel[1]
        a370II.data[py - 1:py + 2, px - 1:px + 2] = test_value

    # Get a rotated copy of the image.
    after = a370II.rotate(30, reshape=True, regrid=True)

    # See in which pixels the coordinate matrix of the rotated
    # image says that the test sky coordinates should now be found
    # after the rotation.
    new_pixels = np.asarray(np.rint(after.wcs.sky2pix(coords)), dtype=int)

    # Check that the values of the nearest pixels to the rotated
    # locations hold the test value that we wrote to the image a370II
    # rotation.
    for pixel in new_pixels:
        py = pixel[0]
        px = pixel[1]
        assert_almost_equal(after.data[py, px], test_value, decimal=6)

    # If both the WCS and the image were rotated wrongly in the same
    # way, then the above test will incrorrectly claim that the
    # rotation worked, so now check that the WCS was rotated correctly.
    assert_allclose(after.wcs.get_rot() - a370II.wcs.get_rot(), 30.0)


def test_resample(a370II):
    """Image class: Testing the resample method"""

    # What scale factors shall we multiply the input step size by?
    xfactor = 3.5
    yfactor = 4.3

    # Get the maximum of the input image.
    peak = a370II.data.max()

    # Choose a few test pixels at different locations in the input
    # image.
    old_pixels = np.array([
        [a370II.shape[0] // 4, a370II.shape[1] // 4],
        [a370II.shape[0] // 4, (3 * a370II.shape[1]) // 4],
        [(3 * a370II.shape[0]) // 4, a370II.shape[1] // 4],
        [(3 * a370II.shape[0]) // 4, (3 * a370II.shape[1]) // 4],
        [a370II.shape[0] // 2, a370II.shape[1] // 2]])

    # Get the sky coordinates of the test pixels.
    coords = a370II.wcs.pix2sky(old_pixels)

    # Assign a large value to each of the above test pixels, so that
    # we can distinguish the smoothed version of it in the output
    # image.
    test_value = 2 * peak
    for pixel in old_pixels:
        py = pixel[0]
        px = pixel[1]
        a370II.data[py, px] = test_value

    # Resample the image.
    newstep = np.array(
        [a370II.get_step(unit=u.arcsec)[0] * yfactor,
         a370II.get_step(unit=u.arcsec)[1] * xfactor])
    after = a370II.resample(newdim=a370II.shape, newstart=None,
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


def test_inside(a370II):
    """Image class: testing inside method."""
    assert not a370II.inside((39.951088, -1.4977398), unit=a370II.wcs.unit)


def test_subimage(a370II):
    """Image class: testing sub-image extraction."""
    subima = a370II.subimage(center=(790, 875), size=40, unit_center=None,
                             unit_size=None)
    assert subima.peak()['data'] == 3035.0


def test_ee():
    """Image class: testing ensquared energy."""
    wcs = WCS()
    data = np.ones(shape=(6, 5)) * 2
    image1 = Image(data=data, wcs=wcs)
    image1.mask_region((2, 2), (1.5, 1.5), inside=False, unit_center=None,
                       unit_radius=None)

    assert image1.ee() == 9 * 2
    assert image1.ee(frac=True) == 1.0
    ee = image1.ee(center=(2, 2), unit_center=None, radius=1, unit_radius=None)
    assert ee == 4 * 2

    r, eer = image1.eer_curve(center=(2, 2), unit_center=None,
                              unit_radius=None, cont=0)
    assert r[1] == 1.0
    assert eer[1] == 1.0

    size = image1.ee_size(center=(2, 2), unit_center=None, unit_size=None,
                          cont=0)
    assert_almost_equal(size[0], 1.775)


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

    image2 = image1.rebin(factor=(2, 2))
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
    assert start[0] == 0.5
    assert start[1] == 0.5


def test_fftconvolve():
    """Image class: testing FFT convolution method."""
    wcs = WCS(cdelt=(0.2, 0.3), crval=(8.5, 12), shape=(40, 30), deg=True)
    data = np.zeros((40, 30))
    data[19, 14] = 1
    ima = Image(wcs=wcs, data=data)
    ima2 = ima.fftconvolve_gauss(center=None, flux=1., fwhm=(20000., 10000.),
                                 peak=False, rot=60., factor=1,
                                 unit_center=u.deg, unit_fwhm=u.arcsec)

    g = ima2.gauss_fit(verbose=False)
    assert_almost_equal(g.fwhm[0], 20000, 2)
    assert_almost_equal(g.fwhm[1], 10000, 2)
    assert_almost_equal(g.center[0], 8.5)
    assert_almost_equal(g.center[1], 12)
    ima2 = ima.fftconvolve_moffat(center=None, flux=1., a=10000, q=1, n=2,
                                  peak=False, rot=60., factor=1,
                                  unit_center=u.deg, unit_a=u.arcsec)
    m = ima2.moffat_fit(verbose=False)
    assert_almost_equal(m.center[0], 8.5)
    assert_almost_equal(m.center[1], 12)
    # ima3 = ima.correlate2d(np.ones((40, 30)))


def test_convolve():
    """Image class: testing discrete convolution method."""

    shape = (12, 25)
    wcs = WCS(cdelt=(1.0, 1.0), crval=(0.0, 0.0), shape=shape)
    data = np.zeros(shape)
    data[7, 5] = 1.0
    mask = np.zeros(shape, dtype=bool)
    mask[5, 3] = True
    ima = Image(wcs=wcs, data=data, mask=mask, copy=False)

    # Create a symmetric convolution kernel with an even number of elements
    # along one dimension and and odd number along the other dimension.
    # Make the kernel symmetric around (shape-1)//2. This requires that
    # the final column be all zeros.
    kern = np.array([[0.1, 0.25, 0.1, 0.0],
                     [0.25, 0.50, 0.25, 0.0],
                     [0.1, 0.25, 0.1, 0.0]])

    # The image should consist of a copy of the convolution kernel, centered
    # such that pixels (kern.shape-1)//2 is at pixel 7,5 of data.
    expected_data = np.ma.array(data=np.zeros(shape), mask=mask)
    expected_data.data[6:9, 4:8] = kern

    res = ima.convolve(kern)
    assert_masked_allclose(res.data, expected_data)

    res = ima.convolve(Image(data=kern))
    assert_masked_allclose(res.data, expected_data)

    res = ima.fftconvolve(kern)
    assert_masked_allclose(res.data, expected_data, atol=1e-15)


def test_dtype():
    """Image class: testing dtype."""
    wcs = WCS(cdelt=(0.2, 0.3), crval=(8.5, 12), shape=(40, 30), deg=True)
    data = np.zeros((40, 30))
    data[19, 14] = 1
    ima = Image(wcs=wcs, data=data, dtype=int)
    ima2 = ima.fftconvolve_gauss(center=None, flux=1., fwhm=(20000., 10000.),
                                 peak=False, rot=60., factor=1,
                                 unit_center=u.deg, unit_fwhm=u.arcsec)

    g = ima2.gauss_fit(verbose=False)
    assert_almost_equal(g.fwhm[0], 20000, 2)
    assert_almost_equal(g.fwhm[1], 10000, 2)
    assert_almost_equal(g.center[0], 8.5)
    assert_almost_equal(g.center[1], 12)
    assert_equal(ima2.dtype, np.float64)

    ima3 = ima2.resample(newdim=(32, 24), newstart=None,
                         newstep=ima2.get_step(unit=u.arcsec) * 0.8)
    assert_equal(ima3.dtype, np.float64)


def test_segment():
    wcs = WCS(cdelt=(0.2, 0.2), crval=(8.5, 12), shape=(100, 100))
    ima = gauss_image(wcs=wcs, fwhm=(2, 2), cont=2.0,
                      unit_center=u.pix, unit_fwhm=u.pix, flux=10, peak=True)
    subimages = ima.segment(minpts=20, background=2.1, median=(5, 5))
    assert len(subimages) == 1
    assert subimages[0].shape == (45, 45)


def test_peak_detection_and_fwhm():
    shape = (101, 101)
    fwhm = (5, 5)
    wcs = WCS(cdelt=(1., 1.), crval=(8.5, 12), shape=shape)
    ima = gauss_image(wcs=wcs, fwhm=fwhm, cont=2.0,
                      unit_center=u.pix, unit_fwhm=u.pix, flux=10, peak=True)

    peaks = ima.peak_detection(5, 2)
    assert peaks.shape == (1, 2)
    assert_allclose(peaks[0], (np.array(shape) - 1) / 2.0)
    assert_allclose(ima.fwhm(unit_radius=None), fwhm, rtol=0.1)


def test_get_item():
    """Image class: testing __getitem__"""
    # Set the shape and contents of the image's data array.
    shape = (4, 5)
    data = np.arange(shape[0] * shape[1]).reshape(shape[0], shape[1])

    # Create a test image with the above data array.
    im = generate_image(data=data, shape=shape)
    im.primary_header['KEY'] = 'primary value'
    im.data_header['KEY'] = 'data value'

    # Select the whole image.
    for r in [im[:, :], im[:]]:
        assert_array_equal(r.shape, im.shape)
        assert_allclose(r.data, im.data)
        assert r.primary_header['KEY'] == im.primary_header['KEY']
        assert r.data_header['KEY'] == im.data_header['KEY']
        assert isinstance(r, Image)
        assert r.wcs.isEqual(im.wcs)
        assert r.wave is None

    # Select a subimage that only has one pixel along the y axis.
    for r in [im[2, :], im[2]]:
        assert_array_equal(r.shape, (1, im.shape[1]))
        assert_allclose(r.data.ravel(), im.data[2, :].ravel())
        assert r.primary_header['KEY'] == im.primary_header['KEY']
        assert r.data_header['KEY'] == im.data_header['KEY']
        assert isinstance(r, Image)
        assert r.wcs.isEqual(im.wcs[2, :])
        assert r.wave is None

    # Select a subimage that only has one pixel along the x axis.
    r = im[:, 2]
    assert_array_equal(r.shape, (im.shape[0], 1))
    assert_allclose(r.data.ravel(), im.data[:, 2].ravel())
    assert r.primary_header['KEY'] == im.primary_header['KEY']
    assert r.data_header['KEY'] == im.data_header['KEY']
    assert isinstance(r, Image)
    assert r.wcs.isEqual(im.wcs[:, 2])
    assert r.wave is None

    # Select a sub-image using a non-scalar slice along each axis.
    r = im[1:2, 2:4]
    assert_array_equal(r.shape, (1, 2))
    assert_allclose(r.data.ravel(), im.data[1:2, 2:4].ravel())
    assert r.primary_header['KEY'] == im.primary_header['KEY']
    assert r.data_header['KEY'] == im.data_header['KEY']
    assert isinstance(r, Image)
    assert r.wcs.isEqual(im.wcs[1:2, 2:4])
    assert r.wave is None

    # Select a single pixel.
    r = im[2, 3]
    assert np.isscalar(r)
    assert_allclose(r, im.data[2, 3])


def test_align_with_image(hdfs_muse_image, hdfs_hst_image):
    muse = hdfs_muse_image
    hst = hdfs_hst_image
    hst_orig = hdfs_hst_image.copy()

    im = hst.align_with_image(muse)

    assert im.wcs.isEqual(muse.wcs)
    assert im.shape == muse.shape

    dy, dx = im.estimate_coordinate_offset(muse)
    assert_almost_equal((dy, dx), (0.15, -0.11), 2)

    sy, sx = im.crop()
    corners = muse.wcs.sky2pix(hst.wcs.pix2sky([[0, 0], hst.shape]),
                               nearest=True).T

    assert (sy.start, sy.stop) == tuple(corners[0])
    # FIXME: check why a -1 is necessary below
    assert (sx.start, sx.stop - 1) == tuple(corners[1])

    assert_array_equal(hst_orig.data, hst.data)


def test_prepare_data():
    image = generate_image(data=2.0)
    image[1, 1] = np.ma.masked
    image[3:5, 2:4] = np.ma.masked

    data = image._prepare_data()
    assert not np.ma.is_masked(data)
    assert np.allclose(data, 2.0)

    data = image._prepare_data(interp='linear')
    assert not np.ma.is_masked(data)
    assert np.allclose(data, 2.0)

    # FIXME: wheck why this doesn't work
    # data = image._prepare_data(interp='spline')
    # assert not np.ma.is_masked(data)
    # assert np.allclose(data, 2.0)
