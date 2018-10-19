"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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
from os.path import join, exists
from mpdaf.drs import RawFile
from numpy.testing import assert_array_equal

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
def test_raw(rawobj):
    """Raw objects: tests initialization"""
    assert rawobj.get_keywords('ORIGIN') == 'CRAL-INM'
    assert set(rawobj.get_channels_extname_list()) == {'CHAN02', 'CHAN01'}
    assert len(rawobj) == 2
    assert rawobj.get_channel('CHAN01') is rawobj[1]
    assert rawobj[2] is rawobj.get_channel('CHAN02')

    im = rawobj.reconstruct_white_image()
    assert im.shape == (288, 300)
    assert_array_equal(np.where(im.data.data.sum(axis=1))[0],
                       np.arange(264, 288))


@pytest.mark.skipif(not SUPP_FILES_PATH, reason="Missing test data (raw.fits)")
def test_channel(rawobj):
    chan1 = rawobj.get_channel("CHAN01")
    assert chan1.data.shape == (rawobj.ny, rawobj.nx)
    assert chan1.data.shape == chan1.mask.shape
    assert np.count_nonzero(chan1.mask) > 0

    assert_array_equal(chan1.trimmed().mask, chan1.mask)
    assert_array_equal(chan1.overscan().mask, ~chan1.mask)

    im = chan1.get_image(bias=True)
    assert im.shape == (chan1.ny, chan1.nx)

    im = chan1.get_image(det_out=1, bias=True)
    assert im.shape == (2056 + 64, 2048 + 64)

    with pytest.raises(ValueError):
        im = chan1.get_image(det_out=5)

    assert [chan1.get_bias_level(x) for x in range(1, 5)] == \
        [1498.0, 1499.0, 1499.0, 1500.0]

    im = chan1.get_trimmed_image(det_out=1, bias=False)
    assert im.shape == (2056, 2048)

    im = chan1.get_trimmed_image(bias=True)
    assert im.shape == (chan1.ny - 64*2, chan1.nx - 64*2)

    im = chan1.get_image_mask_overscan(det_out=1)
    assert im.shape == (2056 + 64, 2048 + 64)


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
