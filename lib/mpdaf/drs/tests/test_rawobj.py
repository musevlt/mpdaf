"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
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

from __future__ import absolute_import

import numpy
import pytest
from os.path import join, exists
from mpdaf.drs import RawFile

from ...tests.utils import DATADIR

EXTERN_DATADIR = join(DATADIR, 'extern')
SERVER_DATADIR = '/home/gitlab-runner/mpdaf-test-data'

if exists(EXTERN_DATADIR):
    SUPP_FILES_PATH = EXTERN_DATADIR
elif exists(SERVER_DATADIR):
    SUPP_FILES_PATH = SERVER_DATADIR
else:
    SUPP_FILES_PATH = None


@pytest.fixture
def rawobj():
    return RawFile(join(SUPP_FILES_PATH, 'raw.fits'))


@pytest.mark.skipif(not SUPP_FILES_PATH, reason="Missing test data (raw.fits)")
def test_raw_init(rawobj):
    """Raw objects: tests initialization"""
    chan1 = rawobj.get_channel("CHAN01")
    shape = numpy.shape(chan1.data)
    assert shape == (rawobj.ny, rawobj.nx)


@pytest.mark.skipif(not SUPP_FILES_PATH, reason="Missing test data (raw.fits)")
def test_raw_mask(rawobj):
    """Raw objects: tests strimmed and overscan functionalities"""
    overscan = rawobj[1].data[24, 12]
    pixel = rawobj[1].data[240, 120]
    out = rawobj[1].trimmed() * 10
    assert out.data[24, 12] == overscan
    assert out.data[240, 120] == 10 * pixel

    out = rawobj[1].overscan() * 2
    assert out.data[24, 12] == 2 * overscan
    assert out.data[240, 120] == pixel
