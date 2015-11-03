"""Test on Image objects."""
import nose.tools
from nose.plugins.attrib import attr

import astropy.units as u
import os
import numpy as np
from astropy.io import fits
from mpdaf.obj import DataArray, WaveCoord, WCS
from numpy.testing import assert_array_equal
from os.path import join

from ..utils import generate_cube

DATADIR = join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
TESTIMG = join(DATADIR, 'data', 'obj', 'a370II.fits')
TESTSPE = join(DATADIR, 'data', 'obj', 'Spectrum_lines.fits')


@attr(speed='fast')
def test_fits_img():
    hdu = fits.open(TESTIMG)
    data = DataArray(filename=TESTIMG)
    nose.tools.assert_equal(data.shape, hdu[0].data.shape)


@attr(speed='fast')
def test_fits_spectrum():
    hdu = fits.open(TESTSPE)
    data = DataArray(filename=TESTSPE)
    nose.tools.assert_equal(data.shape, hdu[1].data.shape)


@attr(speed='fast')
def test_from_np():
    data = np.arange(10)
    d = DataArray(data=data)
    nose.tools.assert_tuple_equal(d.shape, data.shape)


@attr(speed='fast')
def test_copy():
    wcs = WCS(deg=True)
    wave = WaveCoord(cunit=u.angstrom)
    cube1 = DataArray(data=np.arange(5 * 4 * 3).reshape(5, 4, 3), wave=wave,
                      wcs=wcs)
    cube2 = cube1.copy()
    s = cube1.data.sum()
    cube1.data[0, 0, 0] = 1000
    nose.tools.assert_true(cube1.wcs.isEqual(cube2.wcs))
    nose.tools.assert_true(cube1.wave.isEqual(cube2.wave))
    nose.tools.assert_equal(s, cube2.data.sum())


@attr(speed='fast')
def test_clone():
    cube1 = generate_cube()
    cube2 = cube1.clone()
    nose.tools.assert_true(cube1.wcs.isEqual(cube2.wcs))
    nose.tools.assert_true(cube1.wave.isEqual(cube2.wave))
    nose.tools.assert_true(cube2.data is None)
    nose.tools.assert_true(cube2.var is None)


@attr(speed='fast')
def test_clone_with_data():
    cube1 = generate_cube()
    cube2 = cube1.clone(data_init=np.zeros, var_init=np.ones)
    assert_array_equal(cube2.data, np.zeros(cube1.shape))
    assert_array_equal(cube2.var, np.ones(cube1.shape))
