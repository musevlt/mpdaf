"""Test on Image objects."""

# import nose.tools
from nose.plugins.attrib import attr

import io
import numpy as np
import unittest
from astropy.io import fits
from mpdaf.drs import PixTable
from numpy.testing import assert_array_equal


class TestBasicPixTable(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.nrows = 10
        self.xpos = np.linspace(1, 10, self.nrows)
        self.ypos = np.linspace(2, 6, self.nrows)
        self.lbda = np.linspace(5000, 8000, self.nrows)
        self.data = np.linspace(0, 100, self.nrows)
        self.dq = np.linspace(0, 1, self.nrows)
        self.stat = np.linspace(0, 1, self.nrows)
        self.origin = np.linspace(10, 100, self.nrows)
        prihdu = fits.PrimaryHDU()
        prihdu.header['author'] = ('MPDAF', 'origin of the file')

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
        hdu[1].header['BUNIT'] = self.pix.wcs
        hdu[2].header['BUNIT'] = self.pix.wcs
        hdu[3].header['BUNIT'] = self.pix.wave
        hdu[4].header['BUNIT'] = self.pix.unit_data
        hdu[6].header['BUNIT'] = self.pix.unit_stat
        hdu.writeto(self.file)
        self.file.seek(0)
        self.pix2 = PixTable(self.file)
        self.file.seek(0)

    @attr(speed='fast')
    def test_empty_pixtable(self):
        pix = PixTable(None)
        self.assertEqual(pix.nrows, 0)

    @attr(speed='fast')
    def test_getters(self):
        """Image class: tests getters"""
        self.assertEqual(self.nrows, self.pix.nrows)
        for name in ('xpos', 'ypos', 'data', 'dq', 'stat', 'origin'):
            assert_array_equal(getattr(self.pix, 'get_' + name)(),
                               getattr(self, name))
            assert_array_equal(getattr(self.pix2, 'get_' + name)(),
                               getattr(self, name))

    @attr(speed='fast')
    def test_get_column(self):
        assert_array_equal(self.lbda, self.pix.get_lambda())
        ksel = np.where(self.lbda > 6000)
        assert_array_equal(self.lbda[ksel], self.pix.get_lambda(ksel))
        ksel = (self.lbda > 6000)
        assert_array_equal(self.lbda[ksel], self.pix.get_lambda(ksel))

        assert_array_equal(self.lbda, self.pix2.get_lambda())
        ksel = np.where(self.lbda > 6000)
        assert_array_equal(self.lbda[ksel], self.pix2.get_lambda(ksel))

        # Indexing with a boolean array is not (yet) supported
        # ksel = (self.lbda > 6000)
        # assert_array_equal(self.lbda[ksel], self.pix2.get_lambda(ksel))

    @attr(speed='fast')
    def test_set_column(self):
        for pixtable in (self.pix, self.pix2):
            pix = pixtable.copy()
            with self.assertRaises(AssertionError):
                pix.set_xpos(range(5))

            new_xpos = np.linspace(2, 3, self.nrows)
            pix.set_xpos(new_xpos)
            assert_array_equal(new_xpos, pix.get_xpos())

            new_lambda = np.linspace(4000, 5000, self.nrows)
            pix.set_lambda(new_lambda)
            assert_array_equal(new_lambda, pix.get_lambda())

            new_data = pix.get_data()
            ksel = np.where(new_data > 50)
            new_data[ksel] = 0
            pix.set_data(0, ksel=ksel)
            assert_array_equal(new_data, pix.get_data())

            new_data = pix.get_data()
            ksel = (new_data > 60)
            new_data[ksel] = 1
            pix.set_data(1, ksel=ksel)
            assert_array_equal(new_data, pix.get_data())

    @attr(speed='fast')
    def test_extract(self):
        pix = self.pix.extract(lbda=(5000, 6000))
        ksel = (self.lbda >= 5000 and self.lbda < 6000)
        assert_array_equal(self.data[ksel], pix.get_data())
