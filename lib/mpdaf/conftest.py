# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2017 Simon Conseil <simon.conseil@univ-lyon1.fr>

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

import astropy
import matplotlib
import numpy as np
import pytest
import scipy

from astropy.table import Table
from mpdaf.obj import Image, Cube, Spectrum
from mpdaf.sdetect import Source

from .tests.utils import (get_data_file, generate_cube, generate_image,
                          generate_spectrum)


def pytest_report_header(config):
    return "Deps: Numpy {}, Scipy {}, Matplotlib {}, Astropy {}".format(
        np.__version__, scipy.__version__, matplotlib.__version__,
        astropy.__version__)


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
def hdfs_muse_image():
    return Image(get_data_file('obj', 'IMAGE-HDFS-1.34.fits'))


@pytest.fixture
def hdfs_hst_image():
    return Image(get_data_file('obj', 'HST-HDFS.fits'))


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
    line1 = [5550, 10, 2.3, 0.2, 5600.0, 11.0, 2.5, 0.4, 28.0, 3.1, '[OIII]']
    line2 = [5550, 10, 2.3, 0.2, 5600.0, 11.0, 2.5, 0.4, 28.0879, 3.1,
             '[OIII]2']
    lines = Table(names=col_lines, rows=[line1, line2])
    s = Source.from_data(ID=1, ra=-65.1349958, dec=140.3057987,
                         origin=('test', 'v0', 'cube.fits', 'v0'), lines=lines)
    s.add_mag('TEST', 2380, 46)
    s.add_mag('TEST2', 23.5, 0.1)
    s.add_mag('TEST2', 24.5, 0.01)

    s.add_z('z_test', 0.07, errz=0.007)
    s.add_z('z_test2', 1.0, errz=-9999)
    s.add_z('z_test3', 2.0, errz=(1.5, 2.2))
    s.add_z('z_test3', 2.0, errz=(1.8, 2.5))
    return s


@pytest.fixture
def source2():
    return Source.from_file(get_data_file('sdetect', 'sing-0032.fits'))
