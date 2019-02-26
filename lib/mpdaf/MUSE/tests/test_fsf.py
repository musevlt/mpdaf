# import numpy as np
import pytest
from mpdaf.obj import Cube
from mpdaf.MUSE import get_FSF_from_cube_keywords, FSFModel
from mpdaf.tests.utils import get_data_file
from numpy.testing import assert_allclose


def test_get_FSF_from_cube_keywords():
    # This cube has no FSF info
    with pytest.raises(ValueError):
        FSFModel.read(get_data_file('sdetect', 'minicube.fits'))

    cubename = get_data_file('sdetect', 'subcub_mosaic.fits')
    cube = Cube(cubename)

    # Read FSF model with the old method
    PSF, fwhm_pix, fwhm_arcsec = get_FSF_from_cube_keywords(cube, 13)

    # Read FSF model from file
    model = FSFModel.read(cubename)
    assert model.model == 'MOFFAT1'
    assert_allclose(model.get_fwhm(cube.wave.coord()), fwhm_arcsec[0])

    # Read FSF model from cube
    model = FSFModel.read(cube)
    assert model.model == 'MOFFAT1'
    assert_allclose(model.get_fwhm(cube.wave.coord()), fwhm_arcsec[0])
