# -*- coding: utf-8 -*-

import astropy
import numpy as np
import pytest

from mpdaf.obj import Image, Cube
from mpdaf.sdetect import Source

from .utils import (get_data_file, generate_cube, generate_image,
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
def cube():
    return generate_cube()


@pytest.fixture
def image():
    return generate_image()


@pytest.fixture
def spectrum():
    return generate_spectrum()
