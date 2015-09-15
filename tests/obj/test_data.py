"""Test on Image objects."""
import nose.tools
from nose.plugins.attrib import attr

import os
import numpy as np
from astropy.io import fits
from mpdaf.obj.data import DataArray
from os.path import join

DATADIR = join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
TESTIMG = join(DATADIR, 'data', 'obj', 'a370II.fits')
TESTSPE = join(DATADIR, 'data', 'obj', 'Spectrum_lines.fits')


class TestDataArray(object):

    # @attr(speed='fast')
    # def test_fits_img(self):
    #     hdu = fits.open(TESTIMG)
    #     data = DataArray(filename=TESTIMG)
    #     nose.tools.assert_equal(data.shape, hdu[0].data.shape)

    # @attr(speed='fast')
    # def test_fits_spectrum(self):
    #     hdu = fits.open(TESTSPE)
    #     data = DataArray(filename=TESTSPE)
    #     nose.tools.assert_equal(data.shape, hdu[1].data.shape)

    @attr(speed='fast')
    def test_from_np(self):
        data = np.arange(10)
        d = DataArray(data=data)
        nose.tools.assert_tuple_equal(d.shape, data.shape)
