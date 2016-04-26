"""Test on Image objects."""

# import nose.tools
from __future__ import absolute_import
from nose.plugins.attrib import attr

import astropy.units as u
import io
import numpy as np
import unittest
from astropy.io import fits
from mpdaf.drs import PixTable, pixtable
from numpy.testing import assert_array_equal
from six.moves import range

MUSE_ORIGIN_SHIFT_XSLICE = 24
MUSE_ORIGIN_SHIFT_YPIX = 11
MUSE_ORIGIN_SHIFT_IFU = 6


class TestBasicPixTable(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.nrows = 100
        self.xpos = np.linspace(1, 10, self.nrows)
        self.ypos = np.linspace(2, 6, self.nrows)
        self.lbda = np.linspace(5000, 8000, self.nrows)
        self.data = np.linspace(0, 100, self.nrows)
        self.dq = np.linspace(0, 1, self.nrows)
        self.stat = np.linspace(0, 1, self.nrows)

        # generate origin column
        self.aifu = np.random.randint(1, 25, self.nrows)
        self.aslice = np.random.randint(1, 49, self.nrows)
        self.ax = np.random.randint(1, 8192, self.nrows)
        self.ay = np.random.randint(1, 8192, self.nrows)
        self.aoffset = np.random.randint(1, 8192, self.nrows)
        self.origin = (((self.ax - self.aoffset) << MUSE_ORIGIN_SHIFT_XSLICE) |
                       (self.ay << MUSE_ORIGIN_SHIFT_YPIX) |
                       (self.aifu << MUSE_ORIGIN_SHIFT_IFU) | self.aslice)

        prihdu = fits.PrimaryHDU()
        prihdu.header['author'] = ('MPDAF', 'origin of the file')
        prihdu.header['RA'] = 0.0
        prihdu.header['DEC'] = 0.0
        prihdu.header['HIERARCH ESO DRS MUSE PIXTABLE WCS'] = \
            'projected (intermediate)'

        self.pix = PixTable(
            None, xpos=self.xpos, ypos=self.ypos, lbda=self.lbda,
            data=self.data, dq=self.dq, stat=self.stat, origin=self.origin,
            primary_header=prihdu.header)

        self.file = io.BytesIO()
        shape = (self.nrows, 1)
        hdu = fits.HDUList([
            prihdu,
            fits.ImageHDU(name='xpos', data=self.xpos.reshape(shape)),
            fits.ImageHDU(name='ypos', data=self.ypos.reshape(shape)),
            fits.ImageHDU(name='lambda', data=self.lbda.reshape(shape)),
            fits.ImageHDU(name='data', data=self.data.reshape(shape)),
            fits.ImageHDU(name='dq', data=self.dq.reshape(shape)),
            fits.ImageHDU(name='stat', data=self.stat.reshape(shape)),
            fits.ImageHDU(name='origin', data=self.origin.reshape(shape)),
        ])
        hdu[1].header['BUNIT'] = self.pix.wcs.to_string('fits')
        hdu[2].header['BUNIT'] = self.pix.wcs.to_string('fits')
        hdu[3].header['BUNIT'] = self.pix.wave.to_string('fits')
        hdu[4].header['BUNIT'] = self.pix.unit_data.to_string('fits')
        hdu[6].header['BUNIT'] = (self.pix.unit_data**2).to_string('fits')
        hdu.writeto(self.file)
        self.file.seek(0)
        self.pix2 = PixTable(self.file)
        self.file.seek(0)

    @attr(speed='fast')
    def test_empty_pixtable(self):
        pix = PixTable(None)
        self.assertEqual(pix.nrows, 0)
        self.assertEqual(pix.extract(), None)
        self.assertEqual(pix.get_data(), None)

    @attr(speed='fast')
    def test_getters(self):
        self.assertEqual(self.nrows, self.pix.nrows)
        for name in ('xpos', 'ypos', 'data', 'dq', 'stat', 'origin'):
            assert_array_equal(getattr(self.pix, 'get_' + name)(),
                               getattr(self, name))
            assert_array_equal(getattr(self.pix2, 'get_' + name)(),
                               getattr(self, name))

    @attr(speed='fast')
    def test_get_lambda(self):
        assert_array_equal(self.lbda, self.pix.get_lambda())
        assert_array_equal(self.lbda*u.angstrom.to(u.nm),
                           self.pix.get_lambda(unit=u.nm))
        ksel = np.where(self.lbda > 6000)
        assert_array_equal(self.lbda[ksel], self.pix.get_lambda(ksel))
        ksel = (self.lbda > 6000)
        assert_array_equal(self.lbda[ksel], self.pix.get_lambda(ksel))

        assert_array_equal(self.lbda, self.pix2.get_lambda())
        ksel = np.where(self.lbda > 6000)
        assert_array_equal(self.lbda[ksel], self.pix2.get_lambda(ksel))
        ksel = (self.lbda > 6000)
        assert_array_equal(self.lbda[ksel], self.pix2.get_lambda(ksel))

    @attr(speed='fast')
    def test_get_xypos(self):
        assert_array_equal(self.xpos, self.pix2.get_xpos())
        assert_array_equal(self.xpos, self.pix.get_xpos(unit=u.pix))
        assert_array_equal(self.ypos, self.pix2.get_ypos())
        assert_array_equal(self.ypos, self.pix.get_ypos(unit=u.pix))

    @attr(speed='fast')
    def test_get_data(self):
        assert_array_equal(self.data, self.pix2.get_data())
        assert_array_equal(self.data, self.pix.get_data(unit=u.count))
        assert_array_equal(self.stat, self.pix2.get_stat())
        assert_array_equal(self.stat, self.pix.get_stat(unit=u.count**2))

    @attr(speed='fast')
    def test_set_column(self):
        for pixt in (self.pix, self.pix2):
            pix = pixt.copy()
            with self.assertRaises(AssertionError):
                pix.set_xpos(list(range(5)))

        for pix in (self.pix, self.pix2):
            new_xpos = np.linspace(2, 3, self.nrows)
            pix.set_xpos(new_xpos)
            assert_array_equal(new_xpos, pix.get_xpos())

            new_ypos = np.linspace(2, 3, self.nrows)
            pix.set_ypos(new_ypos)
            assert_array_equal(new_ypos, pix.get_ypos())

            new_lambda = np.linspace(4000, 5000, self.nrows)
            pix.set_lambda(new_lambda)
            assert_array_equal(new_lambda, pix.get_lambda())

    @attr(speed='fast')
    def test_set_data(self):
        for pix in (self.pix, self.pix2):
            new_data = pix.get_data()
            new_stat = pix.get_stat()
            ksel = np.where(new_data > 50)
            new_data[ksel] = 0
            new_stat[ksel] = 0
            pix.set_data(0, ksel=ksel)
            assert_array_equal(new_data, pix.get_data())
            pix.set_stat(0, ksel=ksel)
            assert_array_equal(new_stat, pix.get_stat())

            new_data = pix.get_data()
            new_stat = pix.get_stat()
            ksel = (new_data > 60)
            new_data[ksel] = 1
            new_stat[ksel] = 0
            pix.set_data(1, ksel=ksel, unit=u.count)
            assert_array_equal(new_data, pix.get_data())
            pix.set_stat(1, ksel=ksel, unit=u.count**2)
            assert_array_equal(new_stat, pix.get_stat())

    @attr(speed='fast')
    def test_origin_conversion(self):
        origin = self.pix.get_origin()
        ifu = self.pix.origin2ifu(origin)
        sli = self.pix.origin2slice(origin)
        assert_array_equal(self.aifu, ifu)
        assert_array_equal(self.aslice, sli)
        assert_array_equal(self.ay, self.pix.origin2ypix(origin) + 1)
        # TODO: This one needs a pixtable with a header which contains the
        # offset values HIERARCH ESO DRS MUSE PIXTABLE EXP0 IFU* SLICE* XOFFSET
        # assert_array_equal(self.ax,
        #                    self.origin2xpix(origin, ifu=ifu, sli=sli))

    @attr(speed='fast')
    def test_extract(self):
        for numexpr in (True, False):
            if not numexpr:
                _numexpr = pixtable.numexpr
                pixtable.numexpr = False
            pix = self.pix.extract(lbda=(5000, 6000))
            ksel = (self.lbda >= 5000) & (self.lbda < 6000)
            assert_array_equal(self.data[ksel], pix.get_data())

            pix = self.pix.extract(ifu=1)
            ksel = (self.aifu == 1)
            assert_array_equal(self.data[ksel], pix.get_data())

            pix = self.pix.extract(sl=(1, 2, 3))
            ksel = (self.aslice == 1) | (self.aslice == 2) | (self.aslice == 3)
            assert_array_equal(self.data[ksel], pix.get_data())

            if not numexpr:
                pixtable.numexpr = _numexpr
