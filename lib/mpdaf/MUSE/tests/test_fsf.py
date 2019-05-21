"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2019 Simon Conseil <simon.conseil@univ-lyon1.fr>

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
from astropy.io import fits
from mpdaf.obj import Cube
from mpdaf.MUSE import get_FSF_from_cube_keywords, FSFModel
from mpdaf.MUSE.fsf import find_model_cls, OldMoffatModel, MoffatModel2
from mpdaf.tools import MpdafWarning
from mpdaf.tests.utils import get_data_file
from numpy.testing import assert_allclose


def test_fsf_model_errors():
    # This cube has no FSF info
    with pytest.raises(ValueError):
        FSFModel.read(get_data_file('sdetect', 'minicube.fits'))

    with pytest.raises(ValueError):
        find_model_cls(fits.Header({'FSFMODE': 5}))

    with pytest.raises(ValueError):
        OldMoffatModel.from_header(fits.Header(), 0)

    for hdr in [fits.Header(),
                fits.Header({'FSFLB1': 5000, 'FSFLB2': 9000}),
                fits.Header({'FSFLB1': 9000, 'FSFLB2': 5000})]:
        with pytest.raises(ValueError):
            MoffatModel2.from_header(hdr, 0)


def test_fsf_model(tmpdir):
    cubename = get_data_file('sdetect', 'subcub_mosaic.fits')
    cube = Cube(cubename)

    # Read FSF model with the old method
    with pytest.warns(MpdafWarning):
        PSF, fwhm_pix, fwhm_arcsec = get_FSF_from_cube_keywords(cube, 13)

    # Read FSF model from file
    fsf = FSFModel.read(cubename)
    assert len(fsf) == 9
    assert fsf[0].model == 'MOFFAT1'
    assert_allclose(fsf[0].get_fwhm(cube.wave.coord()), fwhm_arcsec[0])

    # Read FSF model from cube
    fsf = FSFModel.read(cube)
    assert len(fsf) == 9
    assert fsf[0].model == 'MOFFAT1'
    assert_allclose(fsf[0].get_fwhm(cube.wave.coord()), fwhm_arcsec[0])

    # Read FSF model from header and for a specific field
    hdr = cube.primary_header.copy()
    hdr.update(cube.data_header)
    fsf = FSFModel.read(hdr, field=2)
    assert fsf.model == 'MOFFAT1'
    assert_allclose(fsf.get_fwhm(cube.wave.coord()), fwhm_arcsec[1])

    # test to_header
    assert [str(x).strip() for x in fsf.to_header().cards] == [
        "FSFMODE = 'MOFFAT1 '           / Old model with a fixed beta",
        'FSF00BET=                  2.8',
        'FSF00FWA=                0.825',
        'FSF00FWB=            -3.01E-05'
    ]

    hdr = fits.Header({'FOO': 1})
    outhdr = fsf.to_header(hdr=hdr, field_idx=2)
    assert [str(x).strip() for x in outhdr.cards] == [
        'FOO     =                    1',
        "FSFMODE = 'MOFFAT1 '           / Old model with a fixed beta",
        'FSF02BET=                  2.8',
        'FSF02FWA=                0.825',
        'FSF02FWB=            -3.01E-05'
    ]

    # Convert to model2
    fsf2 = fsf.to_model2()
    assert fsf2.get_beta(7000) == fsf.beta

    assert [str(x).strip() for x in fsf2.to_header().cards] == [
        'FSFMODE =                    2 / Circular MOFFAT beta=poly(lbda) fwhm=poly(lbda)',
        'FSFLB1  =                 5000 / FSF Blue Ref Wave (A)',
        'FSFLB2  =                 9000 / FSF Red Ref Wave (A)',
        'FSF00FNC=                    2 / FSF00 FWHM Poly Ncoef',
        'FSF00F00=              -0.1204 / FSF00 FWHM Poly C00',
        'FSF00F01=               0.6143 / FSF00 FWHM Poly C01',
        'FSF00BNC=                    1 / FSF00 BETA Poly Ncoef',
        'FSF00B00=                  2.8 / FSF00 BETA Poly C00'
    ]

    testfile = str(tmpdir.join('test.fits'))
    outcube = cube.copy()
    fsf2.to_header(hdr=outcube.primary_header)
    outcube.write(testfile)
    fsf3 = FSFModel.read(testfile, field=0)
    assert fsf3.model == 2
    assert fsf3.get_beta(7000) == fsf.get_beta(7000)
    assert fsf3.get_fwhm(7000) == fsf.get_fwhm(7000)
    assert fsf3.get_fwhm(7000, unit='pix') == fsf.get_fwhm(7000, unit='pix')


def test_fsf_arrays():
    cubename = get_data_file('sdetect', 'subcub_mosaic.fits')
    cube = Cube(cubename)
    fsf = FSFModel.read(cube, field=2)
    fsf2 = fsf.to_model2()

    with pytest.raises(ValueError):
        fsf2.get_2darray([7000], (20, 20))

    with pytest.raises(ValueError):
        fsf2.get_image([7000], cube.wcs)

    ima = fsf2.get_image(7000, cube.wcs, center=(10, 10))
    assert np.unravel_index(ima.data.argmax(), ima.shape) == (10, 10)

    tcube = cube[:5, :, :]
    c = fsf2.get_cube(tcube.wave, cube.wcs, center=(10, 10))
    assert c.shape == (5, 30, 30)
    assert np.unravel_index(c[0].data.argmax(), c.shape[1:]) == (10, 10)
