"""Test on Image objects."""

# import nose.tools
from nose.plugins.attrib import attr

import io
import numpy as np
import unittest
from astropy.io import fits
from mpdaf.drs import PixTable
from numpy.testing import assert_array_equal

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
        hdu[1].header['BUNIT'] = "{}".format(self.pix.wcs)
        hdu[2].header['BUNIT'] = "{}".format(self.pix.wcs)
        hdu[3].header['BUNIT'] = "{}".format(self.pix.wave)
        hdu[4].header['BUNIT'] = "{}".format(self.pix.unit_data)
        hdu[6].header['BUNIT'] = "{}".format(self.pix.unit_data**2)
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
        pix = self.pix.extract(lbda=(5000, 6000))
        ksel = (self.lbda >= 5000) & (self.lbda < 6000)
        assert_array_equal(self.data[ksel], pix.get_data())

        pix = self.pix.extract(ifu=1)
        ksel = (self.aifu == 1)
        assert_array_equal(self.data[ksel], pix.get_data())

        pix = self.pix.extract(sl=(1, 2, 3))
        ksel = (self.aslice == 1) | (self.aslice == 2) | (self.aslice == 3)
        assert_array_equal(self.data[ksel], pix.get_data())
