"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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
import io
import numpy as np
import pytest
import tempfile
import unittest

from astropy.io import fits
from contextlib import contextmanager
from mpdaf.drs import PixTable, pixtable, PixTableMask, PixTableAutoCalib
from numpy.testing import assert_array_equal, assert_allclose
from os.path import exists, join, basename

from ...tests.utils import DATADIR

MUSE_ORIGIN_SHIFT_XSLICE = 24
MUSE_ORIGIN_SHIFT_YPIX = 11
MUSE_ORIGIN_SHIFT_IFU = 6

EXTERN_DATADIR = join(DATADIR, 'extern')
SERVER_DATADIR = '/home/gitlab-runner/mpdaf-test-data'

if exists(EXTERN_DATADIR):
    SUPP_FILES_PATH = EXTERN_DATADIR
elif exists(SERVER_DATADIR):
    SUPP_FILES_PATH = SERVER_DATADIR
else:
    SUPP_FILES_PATH = None

# Fake dat
NROWS = 100


@contextmanager
def toggle_numexpr(active):
    if not active:
        _numexpr = pixtable.numexpr
        pixtable.numexpr = False
    yield
    if not active:
        pixtable.numexpr = _numexpr


class TestBasicPixTable(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.xpos = np.linspace(1, 10, NROWS)
        self.ypos = np.linspace(2, 6, NROWS)
        self.lbda = np.linspace(5000, 8000, NROWS)
        self.data = np.linspace(0, 100, NROWS)
        self.dq = np.random.randint(0, 2, NROWS)
        self.stat = np.linspace(0, 1, NROWS)

        # generate origin column
        np.random.seed(42)
        self.aifu = np.random.randint(1, 25, NROWS)
        self.aslice = np.random.randint(1, 49, NROWS)
        self.ax = np.random.randint(1, 4112, NROWS)
        self.ay = np.random.randint(1, 4112, NROWS)
        self.aoffset = self.ax // 90 * 90

        # Make sure we have at least one ifu=1 value to avoid random failures
        self.aifu[0] = 1

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
        shape = (NROWS, 1)
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

    def test_empty_pixtable(self):
        pix = PixTable(None)
        self.assertEqual(pix.nrows, 0)
        self.assertEqual(pix.extract(), None)
        self.assertEqual(pix.get_data(), None)

    def test_header(self):
        assert self.pix.fluxcal is False
        assert self.pix.skysub is False
        assert self.pix.projection == 'projected'

        pix2 = self.pix.copy()
        pix2.primary_header['HIERARCH ESO DRS MUSE PIXTABLE FLUXCAL'] = True
        assert pix2.fluxcal is True

    def test_getters(self):
        self.assertEqual(NROWS, self.pix.nrows)
        for name in ('xpos', 'ypos', 'data', 'dq', 'stat', 'origin'):
            assert_array_equal(getattr(self.pix, 'get_' + name)(),
                               getattr(self, name))
            assert_array_equal(getattr(self.pix2, 'get_' + name)(),
                               getattr(self, name))

    def test_get_lambda(self):
        assert_array_equal(self.lbda, self.pix.get_lambda())
        assert_array_equal(self.lbda * u.angstrom.to(u.nm),
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

    def test_get_xypos(self):
        assert_array_equal(self.xpos, self.pix2.get_xpos())
        assert_array_equal(self.xpos, self.pix.get_xpos(unit=u.pix))
        assert_array_equal(self.ypos, self.pix2.get_ypos())
        assert_array_equal(self.ypos, self.pix.get_ypos(unit=u.pix))

    def test_get_data(self):
        assert_array_equal(self.data, self.pix2.get_data())
        assert_array_equal(self.data, self.pix.get_data(unit=u.count))
        assert_array_equal(self.stat, self.pix2.get_stat())
        assert_array_equal(self.stat, self.pix.get_stat(unit=u.count**2))

    def test_set_column(self):
        for pixt in (self.pix, self.pix2):
            pix = pixt.copy()
            with self.assertRaises(AssertionError):
                pix.set_xpos(list(range(5)))

        for pix in (self.pix, self.pix2):
            new_xpos = np.linspace(2, 3, pix.nrows)
            pix.set_xpos(new_xpos)
            assert_array_equal(new_xpos, pix.get_xpos())

            new_ypos = np.linspace(2, 3, pix.nrows)
            pix.set_ypos(new_ypos)
            assert_array_equal(new_ypos, pix.get_ypos())

            new_lambda = np.linspace(4000, 5000, pix.nrows)
            pix.set_lambda(new_lambda)
            assert_array_equal(new_lambda, pix.get_lambda())

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

    def test_extract(self):
        for numexpr in (True, False):
            with toggle_numexpr(numexpr):
                pix = self.pix.extract(lbda=(5000, 6000))
                ksel = (self.lbda >= 5000) & (self.lbda < 6000)
                assert_array_equal(self.data[ksel], pix.get_data())

                pix = self.pix.extract(ifu=1)
                ksel = (self.aifu == 1)
                assert_array_equal(self.data[ksel], pix.get_data())

                pix = self.pix.extract(sl=(1, 2, 3))
                ksel = ((self.aslice == 1) | (self.aslice == 2) |
                        (self.aslice == 3))
                assert_array_equal(self.data[ksel], pix.get_data())

    def test_write(self):
        tmpdir = tempfile.mkdtemp(suffix='.mpdaf-test-pixtable')
        out = join(tmpdir, 'PIX.fits')
        self.pix.write(out)
        pix1 = PixTable(out)

        out = join(tmpdir, 'PIX2.fits')
        self.pix.write(out, save_as_ima=False)
        pix2 = PixTable(out)

        for p in (pix1, pix2):
            self.assertEqual(self.pix.nrows, p.nrows)
            for name in ('xpos', 'ypos', 'data', 'dq', 'stat', 'origin'):
                assert_allclose(getattr(p, 'get_' + name)(),
                                getattr(self, name))


@pytest.mark.skipif(not SUPP_FILES_PATH, reason='Missing test data')
def test_autocalib(tmpdir):
    # testpix-small.fits is a small pixtable with values from 2 IFUs, slices
    # 1 to 23, and lambda between 6500A and 7500A
    pixfile = join(SUPP_FILES_PATH, 'testpix-small.fits')
    maskfile = join(SUPP_FILES_PATH, 'Mask-HDF.fits')
    pix = PixTable(pixfile)
    assert pix.nrows == 2810172
    assert repr(pix) == \
        '<PixTable(2810172 rows, 2 ifus, projected, flux-calibrated)>'

    mask = pix.mask_column(maskfile=maskfile)
    assert np.count_nonzero(mask.maskcol) == 97526

    outmask = str(tmpdir.join('MASK.fits'))
    mask.write(outmask)
    savedmask = PixTableMask(filename=outmask)
    assert savedmask.maskfile == basename(maskfile)
    assert savedmask.pixtable == basename(pixfile)
    assert_array_equal(savedmask.maskcol, mask.maskcol)
    savedmask = None

    sky = pix.sky_ref(pixmask=mask)
    assert sky.shape == (1001,)
    assert_array_equal(sky.get_range(), [6500, 7500])

    with pytest.raises(AttributeError):
        pix.subtract_slice_median(sky, mask)

    with pytest.raises(AttributeError):
        pix.divide_slice_median(sky, mask)

    import mpdaf.drs.pixtable
    mpdaf.drs.pixtable.SKY_SEGMENTS = [6500, 7000, 7500]
    cor = pix.selfcalibrate(mask)
    sel = cor.npts > 0
    assert set(cor.ifu[sel]) == {1, 2}
    assert set(cor.quad[sel]) == {1, 2}

    outcor = str(tmpdir.join('cor.fits'))
    cor.write(outcor)
    savedcor = PixTableAutoCalib(filename=outcor)
    assert savedcor.method == 'drs.pixtable.selfcalibrate'
    assert savedcor.maskfile == basename(maskfile)
    assert savedcor.pixtable == basename(pixfile)


@pytest.mark.skipif(not SUPP_FILES_PATH, reason='Missing test data')
@pytest.mark.parametrize('numexpr', (True, False))
def test_select(numexpr):
    pixfile = join(SUPP_FILES_PATH, 'testpix-small.fits')
    pix = PixTable(pixfile)
    with toggle_numexpr(numexpr):
        pix2 = pix.extract(xpix=[(1000, 2000)], ypix=[(1000, 2000)])
        origin = pix2.get_origin()
        xpix = pix2.origin2xpix(origin)
        ypix = pix2.origin2ypix(origin)
        assert xpix.min() >= 1000
        assert xpix.max() <= 2000
        assert ypix.min() >= 1000
        assert ypix.max() <= 2000


@pytest.mark.skipif(not SUPP_FILES_PATH, reason='Missing test data')
@pytest.mark.parametrize('numexpr', (True, False))
@pytest.mark.parametrize('shape', ('C', 'S'))
def test_select_sky(numexpr, shape):
    pixfile = join(SUPP_FILES_PATH, 'testpix-small.fits')
    pix = PixTable(pixfile)
    with toggle_numexpr(numexpr):
        x, y = pix.get_pos_sky()
        cx = (x.min() + x.max()) / 2
        cy = (y.min() + y.max()) / 2
        pix2 = pix.extract(sky=(cy, cx, 12, shape))
        assert pix2.nrows < pix.nrows


@pytest.mark.skipif(not SUPP_FILES_PATH, reason='Missing test data')
def test_reconstruct():
    pixfile = join(SUPP_FILES_PATH, 'testpix-small.fits')
    pix = PixTable(pixfile).extract(ifu=1)

    im = pix.reconstruct_det_image()
    assert im.shape == (1101, 1920)

    im = pix.reconstruct_det_waveimage()
    assert im.shape == (1101, 1920)
