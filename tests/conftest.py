# -*- coding: utf-8 -*-

import astropy
import numpy as np
import os
import pytest

from mpdaf.obj import Image
from os.path import join

from .utils import generate_cube, generate_image, generate_spectrum

DATADIR = join(os.path.abspath(os.path.dirname(__file__)), '..', 'data')


def pytest_report_header(config):
    return "project deps: Numpy {}, Astropy {}".format(
        np.__version__, astropy.__version__)


def get_data_file(*paths):
    return join(DATADIR, *paths)


@pytest.fixture
def a370II():
    """Return a test image from a real observation """

    ima = Image(get_data_file('obj', 'a370II.fits'))

    # The CD matrix of the above image includes a small shear term
    # which means that the image can't be displayed accurately with
    # rectangular pixels. All of the functions in MPDAF assume
    # rectangular pixels, so replace the CD matrix with a similar one
    # that doesn't have a shear component.
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
