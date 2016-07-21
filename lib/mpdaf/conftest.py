# -*- coding: utf-8 -*-

import astropy
import numpy as np
import pytest
import six

from astropy.table import Table
from mpdaf.obj import Image, Cube, Spectrum
from mpdaf.sdetect import Source

from .tests.utils import (get_data_file, generate_cube, generate_image,
                          generate_spectrum)


def pytest_report_header(config):
    return "project deps: Numpy {}, Astropy {}".format(
        np.__version__, astropy.__version__)


@pytest.fixture
def minicube():
    return Cube(get_data_file('sdetect', 'minicube.fits'))


@pytest.fixture
def a478hst():
    return Image(get_data_file('sdetect', 'a478hst-cutout.fits'))


@pytest.fixture
def a370II():
    """Return a test image from a real observation """

    # The CD matrix of the above image includes a small shear term which means
    # that the image can't be displayed accurately with rectangular pixels. All
    # of the functions in MPDAF assume rectangular pixels, so replace the CD
    # matrix with a similar one that doesn't have a shear component.
    ima = Image(get_data_file('obj', 'a370II.fits'))
    ima.wcs.set_cd(np.array([[2.30899476e-5, -5.22301199e-5],
                             [-5.22871997e-5, -2.30647413e-5]]))
    return ima


@pytest.fixture
def spec_var():
    return Spectrum(get_data_file('obj', 'Spectrum_Variance.fits'), ext=[0, 1])


@pytest.fixture
def spec_novar():
    return Spectrum(get_data_file('obj', 'Spectrum_Novariance.fits'))


@pytest.fixture
def cube():
    return generate_cube()


@pytest.fixture
def image():
    return generate_image()


@pytest.fixture
def spectrum():
    return generate_spectrum()


@pytest.fixture
def source1():
    col_lines = ['LBDA_OBS', 'LBDA_OBS_ERR',
                 'FWHM_OBS', 'FWHM_OBS_ERR',
                 'LBDA_REST', 'LBDA_REST_ERR',
                 'FWHM_REST', 'FWHM_REST_ERR',
                 'FLUX', 'FLUX_ERR', 'LINE']
    line1 = [5550, 10, 2.3, 0.2, 5600.0, 11.0, 2.5, 0.4, 28.0, 3.1,
             six.b('[OIII]')]
    line2 = [5550, 10, 2.3, 0.2, 5600.0, 11.0, 2.5, 0.4, 28.0879, 3.1,
             six.b('[OIII]2')]
    lines = Table(names=col_lines, rows=[line1, line2])
    return Source.from_data(ID=1, ra=-65.1349958, dec=140.3057987,
                            origin=('test', 'v0', 'cube.fits', 'v0'),
                            lines=lines)


@pytest.fixture
def source2():
    return Source.from_file(get_data_file('sdetect', 'sing-0032.fits'))
