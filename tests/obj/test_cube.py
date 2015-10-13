"""Test on Cube objects."""
import nose.tools
from nose.plugins.attrib import attr

import astropy.units as u
import numpy as np

from mpdaf.obj import Spectrum, Image, Cube, iter_spe, iter_ima, WCS, WaveCoord
from numpy.testing import assert_almost_equal, assert_array_equal
from operator import add, sub, mul, div
from ..utils import generate_cube, generate_image, generate_spectrum


@attr(speed='fast')
def test_copy():
    """Cube class: testing copy method."""
    cube1 = generate_cube()
    cube2 = cube1.copy()
    s = cube1.data.sum()
    cube1[0, 0, 0] = 1000
    nose.tools.assert_true(cube1.wcs.isEqual(cube2.wcs))
    nose.tools.assert_true(cube1.wave.isEqual(cube2.wave))
    nose.tools.assert_equal(s, cube2.data.sum())


@attr(speed='fast')
def test_arithmetricOperator_Cube():
    """Cube class: tests arithmetic functions"""
    cube1 = generate_cube(uwave=u.nm)
    image1 = generate_image(wcs=cube1.wcs, unit=2 * u.ct)
    spectrum1 = generate_spectrum(scale=2.3, cdelt=30.0, crval=5)
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
                            op(cube1.data.data*cube1.unit, im1).value)

    cube2 = cube1 / 25.3
    cube3 = cube1.clone()
    cube3[:] = cube2
    assert_almost_equal(cube3.data, cube1.data / 25.3)


@attr(speed='fast')
def test_get_Cube():
    """Cube class: tests getters"""
    cube1 = generate_cube()
    assert_array_equal(cube1[2, :, :].shape, (6, 5))
    nose.tools.assert_equal(cube1[:, 2, 3].shape, 10)
    assert_array_equal(cube1[1:7, 0:2, 0:3].shape, (6, 2, 3))
    assert_array_equal(cube1.get_lambda(1.2, 15.6).shape, (6, 6, 5))
    a = cube1[2:4, 0:2, 1:4]
    assert_array_equal(a.get_start(), (3.5, 0, 1))
    assert_array_equal(a.get_end(), (6.5, 1, 3))


@attr(speed='fast')
def test_iterator():
    """Cube class: tests iterators"""
    cube1 = generate_cube()
    for ima, k in iter_ima(cube1, True):
        ima[:, :] = k * np.ones(shape=(6, 5))
    c = np.arange(cube1.shape[0])[:, np.newaxis, np.newaxis]
    assert_array_equal(*np.broadcast_arrays(cube1.data.data, c))

    for (spe, (p, q)) in iter_spe(cube1, True):
        spe[:] = spe + p + q
    z, y, x = [np.arange(sh) for sh in cube1.shape]
    assert_array_equal(cube1.data.data, np.add.reduce(np.meshgrid(y, z, x)))


@attr(speed='fast')
def test_clone():
    """Cube class: tests clone method."""
    cube1 = generate_cube()
    cube2 = cube1.clone()
    assert_array_equal(cube2.data, np.zeros(cube1.shape))


@attr(speed='fast')
def test_resize():
    """Cube class: tests resize method."""
    cube1 = generate_cube()
    cube1.data.mask[0, :, :] = True
    cube1.resize()
    nose.tools.assert_equal(cube1.shape[0], 9)


# @attr(speed='fast')
def test_multiprocess():
    """Cube class: tests multiprocess"""
    cube1 = generate_cube()

    f = Image.sum
    list_spe = cube1.loop_ima_multiprocessing(f, cpu=2, verbose=True, axis=0)
    nose.tools.assert_equal(list_spe[8][1], cube1[8, :, :].sum(axis=0)[1])

    f = Image.ee
    ee = cube1.loop_ima_multiprocessing(f, cpu=2, verbose=True)
    nose.tools.assert_equal(ee[1], cube1[1, :, :].ee())

    f = Image.rotate
    cub2 = cube1.loop_ima_multiprocessing(f, cpu=2, verbose=True, theta=20)
    nose.tools.assert_equal(cub2[4, 3, 2], cube1[4, :, :].rotate(20)[3, 2])

    f = Spectrum.mean
    out = cube1.loop_spe_multiprocessing(f, cpu=2, verbose=True)
    nose.tools.assert_equal(out[2, 3], cube1[:, 2, 3].mean())

    f = Spectrum.resample
    out = cube1.loop_spe_multiprocessing(f, cpu=2, verbose=True, step=1)
    nose.tools.assert_equal(out[8, 3, 2], cube1[:, 3, 2].resample(step=1)[8])


@attr(speed='fast')
def test_mask():
    """Cube class: testing mask functionalities"""
    cube1 = generate_cube()
    cube1.mask((2, 2), (1, 1), lmin=2, lmax=5, inside=False, unit_center=None,
               unit_radius=None, unit_wave=None)
    nose.tools.assert_almost_equal(cube1.sum(), 2.3 * 9 * 3)
    cube1.unmask()
    wcs = WCS(deg=True)
    wave = WaveCoord(cunit=u.angstrom)
    cube1 = Cube(data=cube1.data, wave=wave, wcs=wcs, copy=False)
    cube1.mask(wcs.pix2sky([2, 2]), (3600, 3600), lmin=2, lmax=5, inside=False)
    nose.tools.assert_almost_equal(cube1.sum(), 2.3 * 9 * 3)
    cube1.unmask()
    cube1.mask(wcs.pix2sky([2, 2]), 4000, lmin=2, lmax=5, inside=False)
    nose.tools.assert_almost_equal(cube1.sum(), 2.3 * 5 * 3)
    cube1.unmask()
    cube1.mask_ellipse(wcs.pix2sky([2, 2]), (10000, 3000), 20, lmin=2, lmax=5,
                       inside=False)
    nose.tools.assert_almost_equal(cube1.sum(), 2.3 * 7 * 3)
    ksel = np.where(cube1.data.mask)
    cube1.unmask()
    cube1.mask_selection(ksel)
    nose.tools.assert_almost_equal(cube1.sum(), 2.3 * 7 * 3)


@attr(speed='fast')
def test_truncate():
    """Cube class: testing truncation"""
    cube1 = generate_cube(scale=2, wave=WaveCoord(crval=1))
    coord = [[2, 0, 1], [5, 1, 3]]
    cube2 = cube1.truncate(coord, unit_wcs=cube1.wcs.get_cunit1(),
                           unit_wave=cube1.wave.get_cunit())
    assert_array_equal(cube2.shape, (4, 2, 3))
    assert_array_equal(cube2.get_start(), (2, 0, 1))
    assert_array_equal(cube2.get_end(), (5, 1, 3))


@attr(speed='fast')
def test_sum():
    """Cube class: testing sum, mean and median methods"""
    cube1 = generate_cube(scale=1, wave=WaveCoord(crval=1))
    for i in range(10):
        cube1.data[i, :, :] = i * np.ones((6, 5))
    nose.tools.assert_equal(cube1.sum(), 6 * 5 * 45)
    sum2 = cube1.sum(axis=0)
    nose.tools.assert_equal(sum2.shape[0], 6)
    nose.tools.assert_equal(sum2.shape[1], 5)

    weights = np.ones(shape=(10, 6, 5))
    nose.tools.assert_equal(cube1.sum(weights=weights), 6 * 5 * 45)

    weights = np.ones(shape=(10, 6, 5)) * 2
    nose.tools.assert_equal(cube1.sum(weights=weights), 6 * 5 * 45)

    m = cube1.mean(axis=(1, 2))
    for i in range(10):
        nose.tools.assert_equal(m[i], i)

    m = cube1.median(axis=0)
    nose.tools.assert_equal(m[3, 3], np.median(np.arange(10)))


@attr(speed='fast')
def test_rebin():
    """Cube class: testing rebin methods"""
    cube1 = generate_cube(scale=1, wave=WaveCoord(crval=1))
    cube2 = cube1.rebin_mean(factor=2)
    nose.tools.assert_equal(cube2[0, 0, 0], 1)
    assert_array_equal(cube2.get_start(), (1.5, 0.5, 0.5))
    cube2 = cube1.rebin_mean(factor=2, flux=True, margin='origin')
    nose.tools.assert_equal(cube2[-1, -1, -1], 0.5)
    assert_array_equal(cube2.get_start(), (1.5, 0.5, 0.5))


@attr(speed='fast')
def test_get_image():
    """Cube class: testing get_image method"""
    wave = WaveCoord(crpix=1, cdelt=0.3, crval=200, cunit=u.nm)
    wcs = WCS(crval=(0, 0))
    data = np.ones(shape=(2000, 6, 5)) * 2
    cube1 = Cube(data=data, wave=wave, wcs=wcs)
    cube1[:, 2, 2].add_gaussian(5000, 1200, 20, unit=u.angstrom)
    ima = cube1.get_image(wave=(4800, 5200), is_sum=False, subtract_off=True)
    nose.tools.assert_equal(ima[0, 0], 0)
    nose.tools.assert_almost_equal(ima[2, 2],
                                   cube1[934:1067, 2, 2].mean() - 2, 3)
    ima = cube1.get_image(wave=(4800, 5200), is_sum=False, subtract_off=False)
    nose.tools.assert_equal(ima[0, 0], 2)
    nose.tools.assert_almost_equal(ima[2, 2], cube1[934:1067, 2, 2].mean(), 3)
    ima = cube1.get_image(wave=(4800, 5200), is_sum=True, subtract_off=True)
    nose.tools.assert_equal(ima[0, 0], 0)
    nose.tools.assert_almost_equal(ima[2, 2], cube1[934:1067, 2, 2].sum() -
                                   cube1[934:1067, 0, 0].sum(), 3)
    ima = cube1.get_image(wave=(4800, 5200), is_sum=True, subtract_off=False)
    nose.tools.assert_equal(ima[0, 0], cube1[934:1067, 0, 0].sum())
    nose.tools.assert_almost_equal(ima[2, 2], cube1[934:1067, 2, 2].sum())


@attr(speed='fast')
def test_subcube():
    """Cube class: testing sub-cube extraction methods"""
    cube1 = generate_cube(scale=1, wave=WaveCoord(crval=1))
    cube2 = cube1.subcube(center=(2, 2.8), size=2, lbda=(5, 8),
                          unit_center=None, unit_size=None)
    assert_array_equal(cube2.get_start(), (5, 1, 2))
    assert_array_equal(cube2.shape, (4, 2, 2))
    cube2 = cube1.subcube_circle_aperture(center=(2, 2.8), radius=1,
                                          unit_center=None, unit_radius=None)
    nose.tools.assert_equal(cube2.data.mask[0, 0, 0], True)
    assert_array_equal(cube2.get_start(), (1, 1, 2))
    assert_array_equal(cube2.shape, (10, 2, 2))


@attr(speed='fast')
def test_aperture():
    """Cube class: testing spectrum extraction"""
    cube1 = generate_cube(scale=1, wave=WaveCoord(crval=1))
    spe = cube1.aperture(center=(2, 2.8), radius=1,
                         unit_center=None, unit_radius=None)
    nose.tools.assert_equal(spe.shape, 10)
    nose.tools.assert_equal(spe.get_start(), 1)
