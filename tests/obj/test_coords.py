"""Test on WCS and WaveCoord objects."""

import nose.tools
from nose.plugins.attrib import attr

import numpy as np
from astropy import wcs as pywcs
from mpdaf.obj import WCS, WaveCoord, deg2sexa, sexa2deg
from numpy.testing import assert_allclose, assert_array_equal


class TestWCS(object):

    def setUp(self):
        self.wcs = WCS(crval=(0, 0))
        self.wcs.naxis1 = 6
        self.wcs.naxis2 = 5

    @attr(speed='fast')
    def test_copy(self):
        """WCS class: tests copy"""
        wcs2 = self.wcs.copy()
        nose.tools.assert_true(self.wcs.isEqual(wcs2))

    @attr(speed='fast')
    def test_coordTransform(self):
        """WCS class: tests coordinates transformations"""
        pixcrd = [[0, 0], [2, 3], [3, 2]]
        pixsky = self.wcs.pix2sky(pixcrd)
        pixcrd2 = self.wcs.sky2pix(pixsky)
        assert_array_equal(pixcrd, pixcrd2)

    @attr(speed='fast')
    def test_coordTransform2(self):
        """WCS class: tests transformations with a more complete header."""

        w = pywcs.WCS(naxis=2)
        w.wcs.crpix = [167.401033093, 163.017401336]
        w.wcs.cd = np.array([[-5.5555555555555003e-05, 0],
                             [0, 5.5555555555555003e-05]])
        w.wcs.crval = [338.23092027, -60.56375796]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        wcs = WCS()
        wcs.wcs = w

        pix = np.array([[108.41, 81.34]])
        pix2 = pix[:, [1, 0]]
        pixint = pix2.astype(int)
        ref = np.array([[338.2375, -60.5682]])
        ref2 = ref[:, [1, 0]]
        sky = wcs.pix2sky(pix2)

        assert_allclose(wcs.wcs.wcs_pix2world(pix, 0), ref, rtol=1e-4)
        assert_allclose(sky, ref2, rtol=1e-4)
        assert_allclose(wcs.sky2pix(wcs.pix2sky(pix2)), pix2)
        print wcs.sky2pix(sky, nearest=True), pixint
        assert_allclose(wcs.sky2pix(sky, nearest=True), pixint)

    @attr(speed='fast')
    def test_get(self):
        """WCS class: tests getters"""
        assert_array_equal(self.wcs.get_step(), [1.0, 1.0])
        assert_array_equal(self.wcs.get_start(), [0.0, 0.0])
        assert_array_equal(self.wcs.get_end(), [4.0, 5.0])

        wcs2 = WCS(crval=(0, 0), shape=(5, 6))
        assert_array_equal(wcs2.get_step(), [1.0, 1.0])
        assert_array_equal(wcs2.get_start(), [-2.0, -2.5])
        assert_array_equal(wcs2.get_end(), [2.0, 2.5])


class TestWaveCoord(object):

    def setUp(self):
        self.wave = WaveCoord(crval=0)
        self.wave.shape = 10

    @attr(speed='fast')
    def test_copy(self):
        """WaveCoord class: tests copy"""
        wave2 = self.wave.copy()
        nose.tools.assert_true(self.wave.isEqual(wave2))

    @attr(speed='fast')
    def test_coord_transform(self):
        """WaveCoord class: tests coordinates transformations"""
        pixel = self.wave.pixel(self.wave.coord(5), nearest=True)
        nose.tools.assert_equal(pixel, 5)

        wave = np.arange(10)
        pixel = self.wave.pixel(self.wave.coord(wave), nearest=True)
        np.testing.assert_array_equal(pixel, wave)

        pix = np.arange(self.wave.shape, dtype=np.float)
        np.testing.assert_allclose(self.wave.pixel(self.wave.coord()), pix)

    @attr(speed='fast')
    def test_get(self):
        """WaveCoord class: tests getters"""
        nose.tools.assert_equal(self.wave.get_step(), 1.0)
        nose.tools.assert_equal(self.wave.get_start(), 0.0)
        nose.tools.assert_equal(self.wave.get_end(), 9.0)


class TestCoord(object):

    @attr(speed='fast')
    def test_deg_sexa(self):
        """tests degree/sexagesimal transformations"""
        ra = '23:51:41.268'
        dec = '-26:04:43.032'
        deg = sexa2deg([dec, ra])
        nose.tools.assert_almost_equal(deg[0], -26.07862, 3)
        nose.tools.assert_almost_equal(deg[1], 357.92195, 3)
        sexa = deg2sexa([-26.07862, 357.92195])
        nose.tools.assert_equal(sexa[0], dec)
        nose.tools.assert_equal(sexa[1], ra)
