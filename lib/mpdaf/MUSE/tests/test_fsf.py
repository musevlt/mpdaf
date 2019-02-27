import pytest
from astropy.io import fits
from mpdaf.obj import Cube
from mpdaf.MUSE import get_FSF_from_cube_keywords, FSFModel
from mpdaf.MUSE.fsf import find_model_cls
from mpdaf.tests.utils import get_data_file
from numpy.testing import assert_allclose


def test_get_FSF_from_cube_keywords(tmpdir):
    # This cube has no FSF info
    with pytest.raises(ValueError):
        FSFModel.read(get_data_file('sdetect', 'minicube.fits'))

    with pytest.raises(ValueError):
        find_model_cls(fits.Header({'FSFMODE': 5}))

    cubename = get_data_file('sdetect', 'subcub_mosaic.fits')
    cube = Cube(cubename)

    # Read FSF model with the old method
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
    assert fsf3.get_beta(7000) == 2.8
    assert fsf3.get_fwhm(7000) == fsf.get_fwhm(7000)
