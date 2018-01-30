# -*- coding: utf-8 -*-

from __future__ import absolute_import, division

import numpy as np
from mpdaf.obj import Image, Cube
from mpdaf.sdetect import Source
from mpdaf.MUSE import create_psf_cube
from numpy.testing import assert_almost_equal
from ...tests.utils import get_data_file


def test_create_psf_cube():
    src = Source.from_file(get_data_file('sdetect', 'origin-00026.fits'))
    cube = Cube(get_data_file('sdetect', 'subcub_mosaic.fits'))
    src.add_FSF(cube)

    wcs = src.images['MUSE_WHITE'].wcs
    shape = src.images['MUSE_WHITE'].shape
    a, b, beta, field = src.get_FSF()
    psf = b * cube.wave.coord() + a

    # Gaussian
    gauss = create_psf_cube(psf.shape + shape, psf, wcs=wcs)
    im = Image(data=gauss[0], wcs=wcs)
    res = im.gauss_fit()
    assert_almost_equal(wcs.sky2pix(res.center)[0], [12., 12.])
    assert np.allclose(res.fwhm, psf[0])
    assert np.allclose(res.flux, 1.0)

    # Moffat
    moff = create_psf_cube(psf.shape + shape, psf, wcs=wcs, beta=beta)
    im = Image(data=moff[0], wcs=wcs)
    res = im.moffat_fit()
    assert_almost_equal(wcs.sky2pix(res.center)[0], [12., 12.])
    assert np.allclose(res.fwhm, psf[0])
    assert np.allclose(res.flux, 1.0, atol=1e-2)
