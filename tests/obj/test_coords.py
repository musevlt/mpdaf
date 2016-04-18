"""Test on WCS and WaveCoord objects."""

from __future__ import absolute_import, division

import nose.tools
from nose.plugins.attrib import attr

import astropy.units as u
import numpy as np
from astropy import wcs as pywcs
from astropy.io import fits
from mpdaf.obj import WCS, WaveCoord, deg2sexa, sexa2deg
from numpy.testing import assert_allclose, assert_array_equal


class TestWCS(object):

    @attr(speed='fast')
    def test_from_hdr(self):
        """WCS class: testing constructor """
        h = fits.getheader('data/obj/a370II.fits')
        wcs = WCS(h)
        h2 = wcs.to_header()
        wcs2 = WCS(h2)
        wcs2.naxis1 = wcs.naxis1
        wcs2.naxis2 = wcs.naxis2
        nose.tools.assert_true(wcs.isEqual(wcs2))

    @attr(speed='fast')
    def test_copy(self):
        """WCS class: tests copy"""
        wcs = WCS(crval=(0, 0), shape=(5, 6))
        wcs2 = wcs.copy()
        nose.tools.assert_true(wcs.isEqual(wcs2))

    @attr(speed='fast')
    def test_coordTransform(self):
        """WCS class: testing coordinates transformations"""
        wcs = WCS(crval=(0, 0), shape=(5, 6))
        pixcrd = [[0, 0], [2, 3], [3, 2]]
        pixsky = wcs.pix2sky(pixcrd)
        pixcrd2 = wcs.sky2pix(pixsky)
        assert_array_equal(pixcrd, pixcrd2)

    @attr(speed='fast')
    def test_coordTransform2(self):
        """WCS class: testing transformations with a more complete header."""

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
        assert_allclose(wcs.sky2pix(sky, nearest=True), pixint)

    @attr(speed='fast')
    def test_get(self):
        """WCS class: testing getters"""
        wcs = WCS(crval=(0, 0), shape=(5, 6), crpix=(1, 1))
        assert_array_equal(wcs.get_step(), [1.0, 1.0])
        assert_array_equal(wcs.get_start(), [0.0, 0.0])
        assert_array_equal(wcs.get_end(), [4.0, 5.0])

        wcs2 = WCS(crval=(0, 0), shape=(5, 6))
        assert_array_equal(wcs2.get_step(), [1.0, 1.0])
        assert_array_equal(wcs2.get_start(), [-2.0, -2.5])
        assert_array_equal(wcs2.get_end(), [2.0, 2.5])

        wcs2.set_step([0.5, 2.5])
        assert_array_equal(wcs2.get_step(), [0.5, 2.5])

        wcs2.set_crval2(-2, unit=2 * u.pix)
        assert_array_equal(wcs2.get_crval2(), -4.0)


class TestWaveCoord(object):

    @attr(speed='fast')
    def test_from_hdr(self):
        """WaveCoord class: testing constructor """
        h = fits.getheader('data/obj/Spectrum_Novariance.fits')
        wave = WaveCoord(h)
        h2 = wave.to_header()
        wave2 = WaveCoord(h2)
        wave2.shape = wave.shape
        nose.tools.assert_true(wave.isEqual(wave2))

    @attr(speed='fast')
    def test_copy(self):
        """WaveCoord class: testing copy"""
        wave = WaveCoord(crval=0, cunit=u.nm, shape=10)
        wave2 = wave.copy()
        nose.tools.assert_true(wave.isEqual(wave2))

    @attr(speed='fast')
    def test_coord_transform(self):
        """WaveCoord class: testing coordinates transformations"""
        wave = WaveCoord(crval=0, cunit=u.nm, shape=10)
        pixel = wave.pixel(wave.coord(5, unit=u.nm), nearest=True, unit=u.nm)
        nose.tools.assert_equal(pixel, 5)

        wave2 = np.arange(10)
        pixel = wave.pixel(wave.coord(wave2, unit=u.nm), nearest=True,
                           unit=u.nm)
        np.testing.assert_array_equal(pixel, wave2)

        pix = np.arange(wave.shape, dtype=np.float)
        np.testing.assert_allclose(wave.pixel(wave.coord(unit=u.nm),
                                              unit=u.nm), pix)

    @attr(speed='fast')
    def test_get(self):
        """WaveCoord class: testing getters"""
        wave = WaveCoord(crval=0, cunit=u.nm, shape=10)
        nose.tools.assert_equal(wave.get_step(unit=u.nm), 1.0)
        nose.tools.assert_equal(wave.get_start(unit=u.nm), 0.0)
        nose.tools.assert_equal(wave.get_end(unit=u.nm), 9.0)

    @attr(speed='fast')
    def test_rebin(self):
        """WCS class: testing rebin method"""
        wave = WaveCoord(crval=0, cunit=u.nm, shape=10)
        wave.rebin(factor=2)
        nose.tools.assert_equal(wave.get_step(unit=u.nm), 2.0)
        nose.tools.assert_equal(wave.get_start(unit=u.nm), 0.5)
        nose.tools.assert_equal(wave.coord(2, unit=u.nm), 4.5)
        nose.tools.assert_equal(wave.shape, 5)

    @attr(speed='fast')
    def test_resample(self):
        """WCS class: testing resampling method"""
        wave = WaveCoord(crval=0, cunit=u.nm, shape=10)
        wave2 = wave.resample(step=2.5, start=20, unit=u.angstrom)
        nose.tools.assert_equal(wave2.get_step(unit=u.nm), 0.25)
        nose.tools.assert_equal(wave2.get_start(unit=u.nm), 2.0)
        nose.tools.assert_equal(wave2.shape, 32)

    @attr(speed='fast')
    def test_muse_header(self):
        """WCS class: testing MUSE header specifities."""
        d = dict(crval=4750., crpix=1., ctype='AWAV', cdelt=1.25,
                 cunit=u.angstrom, shape=3681)
        wave = WaveCoord(**d)
        start = d['crval']
        end = d['crval'] + d['cdelt'] * (d['shape'] - 1)
        nose.tools.assert_equal(wave.get_step(), d['cdelt'])
        nose.tools.assert_equal(wave.get_crval(), start)
        nose.tools.assert_equal(wave.get_start(), start)
        nose.tools.assert_equal(wave.get_end(), end)
        assert_array_equal(wave.get_range(), [start, end])
        nose.tools.assert_equal(wave.get_crpix(), d['crpix'])
        nose.tools.assert_equal(wave.get_ctype(), d['ctype'])

        def to_nm(val):
            return (val * u.angstrom).to(u.nm).value
        nose.tools.assert_equal(wave.get_step(u.nm), to_nm(d['cdelt']))
        nose.tools.assert_equal(wave.get_crval(u.nm), to_nm(start))
        nose.tools.assert_equal(wave.get_start(u.nm), to_nm(start))
        nose.tools.assert_equal(wave.get_end(u.nm), to_nm(end))
        assert_array_equal(wave.get_range(u.nm), [to_nm(start), to_nm(end)])


class TestCoord(object):

    @attr(speed='fast')
    def test_deg_sexa(self):
        """testing degree/sexagesimal transformations"""
        ra = '23:51:41.268'
        dec = '-26:04:43.032'
        deg = sexa2deg([dec, ra])
        nose.tools.assert_almost_equal(deg[0], -26.07862, 3)
        nose.tools.assert_almost_equal(deg[1], 357.92195, 3)
        sexa = deg2sexa([-26.07862, 357.92195])
        nose.tools.assert_equal(sexa[0], dec)
        nose.tools.assert_equal(sexa[1], ra)
