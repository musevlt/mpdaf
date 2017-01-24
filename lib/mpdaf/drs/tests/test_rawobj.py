"""Test on RawFile objects to be used with py.test."""

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
