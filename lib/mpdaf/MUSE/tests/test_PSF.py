# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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

import numpy as np
import pytest
from mpdaf.obj import Image, Cube
from mpdaf.sdetect import Source
from mpdaf.MUSE import create_psf_cube, get_FSF_from_cube_keywords
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


def test_get_FSF_from_cube_keywords():
    cube = Cube(get_data_file('sdetect', 'minicube.fits'))
    with pytest.raises(IOError):
        # This cube has no FSF info
        PSF, fwhm_pix, fwhm_arcsec = get_FSF_from_cube_keywords(cube, 13)

    cube = Cube(get_data_file('sdetect', 'subcub_mosaic.fits'))
    PSF, fwhm_pix, fwhm_arcsec = get_FSF_from_cube_keywords(cube, 13)

    assert len(PSF) == 9
    assert len(fwhm_pix) == 9
    np.testing.assert_allclose(fwhm_pix[0] * 0.2, fwhm_arcsec[0])
