"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2019 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2019 Roland Bacon <roland.bacon@univ-lyon1.fr>

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
from mpdaf.MUSE.fsf import find_model_cls, MoffatModel2
from mpdaf.MUSE.fsf import combine_fsf
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
        MoffatModel2.from_header(fits.Header(), 0)

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
    assert fsf[0].model == 2
    assert_allclose(fsf[0].get_fwhm(cube.wave.coord()), fwhm_arcsec[0])

    # Read FSF model from cube
    fsf = FSFModel.read(cube)
    assert len(fsf) == 9
    assert fsf[0].model == 2
    assert_allclose(fsf[0].get_fwhm(cube.wave.coord()), fwhm_arcsec[0])

    # Read FSF model from header and for a specific field
    hdr = cube.primary_header.copy()
    hdr.update(cube.data_header)
    fsf = FSFModel.read(hdr, field=2)
    assert fsf.model == 2
    assert_allclose(fsf.get_fwhm(cube.wave.coord()), fwhm_arcsec[1])

    # test to_header
    assert [str(x).strip() for x in fsf.to_header().cards] == [
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
    fsf.to_header(hdr=outcube.primary_header)
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

    with pytest.raises(ValueError):
        fsf.get_2darray([7000], (20, 20))

    with pytest.raises(ValueError):
        fsf.get_image([7000], cube.wcs)

    ima = fsf.get_image(7000, cube.wcs, center=(10, 10))
    assert np.unravel_index(ima.data.argmax(), ima.shape) == (10, 10)

    tcube = cube[:5, :, :]
    c = fsf.get_cube(tcube.wave, cube.wcs, center=(10, 10))
    assert c.shape == (5, 30, 30)
    assert np.unravel_index(c[0].data.argmax(), c.shape[1:]) == (10, 10)


def test_fsf_convolve():
    lbrange = [4750.0, 9350.0]
    beta_pol = [0.425572268419153, -0.963126218379342, -0.0014311681713689742,
                -0.0064324103352929405, 0.09098701358534873, 2.0277399948419843]
    fwhm_pol = [0.6321570666462952, -0.06284858095522032, 0.04282359923274102,
                0.045673032671778586, -0.1864068502712748, 0.3693082688212182]
    fsf = MoffatModel2(fwhm_pol, beta_pol, lbrange, 0.2)

    fsf2 = fsf.convolve(cfwhm=0.1)
    assert_allclose(fsf2.get_fwhm(7000), 0.3919, rtol=1e-3)
    assert_allclose(fsf2.get_beta(7000), 2.1509, rtol=1e-3)


def test_combine_fsf():
    lbrange = [4750.0, 9350.0]
    beta_pol = [0.425572268419153, -0.963126218379342, -0.0014311681713689742,
                -0.0064324103352929405, 0.09098701358534873, 2.0277399948419843]
    fwhm_pol = [0.6321570666462952, -0.06284858095522032, 0.04282359923274102,
                0.045673032671778586, -0.1864068502712748, 0.3693082688212182]
    fsf1 = MoffatModel2(fwhm_pol, beta_pol, lbrange, 0.2)

    fwhm_pol = [0.6539648695212446, -0.09803896219961082, 0.0768935513209841,
                0.13029884613164275, -0.30890727537189494, 0.4420737174631386]
    beta_pol = [1.3422018214910905, -1.0824007679002177, 0.0654899276450118,
                0.5566091154793532, -0.4488955513549307, 1.7496593644278122]
    fsf2 = MoffatModel2(fwhm_pol, beta_pol, lbrange, 0.2)

    fsf, cube = combine_fsf([fsf1, fsf1])
    assert_allclose(fsf.get_fwhm(7000), fsf1.get_fwhm(7000), rtol=1.e-6)
    assert_allclose(fsf.get_beta(7000), fsf1.get_beta(7000), rtol=1.e-6)

    fsf, cube = combine_fsf([fsf1, fsf2])
    assert_allclose(fsf.get_fwhm(7000), 0.397959, rtol=1.e-2)
    assert_allclose(fsf.get_beta(7000), 1.843269, rtol=1.e-2)
