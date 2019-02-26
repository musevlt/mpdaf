# import numpy as np
import pytest
from mpdaf.obj import Cube
from mpdaf.MUSE import get_FSF_from_cube_keywords, FSFModel
from mpdaf.tests.utils import get_data_file
from numpy.testing import assert_allclose


def test_get_FSF_from_cube_keywords():
    cube = Cube(get_data_file('sdetect', 'minicube.fits'))
    with pytest.raises(IOError):
        # This cube has no FSF info
        PSF, fwhm_pix, fwhm_arcsec = get_FSF_from_cube_keywords(cube, 13)

    cube = Cube(get_data_file('sdetect', 'subcub_mosaic.fits'))
    PSF, fwhm_pix, fwhm_arcsec = get_FSF_from_cube_keywords(cube, 13)

    assert len(PSF) == 9
    assert len(fwhm_pix) == 9
    assert_allclose(fwhm_pix[0] * 0.2, fwhm_arcsec[0])

    model = FSFModel(get_data_file('sdetect', 'subcub_mosaic.fits'))
    assert model.model == 'MOFFAT1'
    assert_allclose(model.get_fwhm(cube.wave.coord()), fwhm_arcsec[0])
