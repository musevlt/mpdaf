"""Test on RawFile objects to be used with py.test."""

from __future__ import absolute_import

import numpy
import os
import pytest
from mpdaf.drs import RawFile

from ..utils import get_data_file

DATA_PATH = get_data_file('drs', 'raw.fits')
MISSING = not os.path.exists(DATA_PATH)


@pytest.fixture
def rawobj():
    return RawFile(DATA_PATH)


@pytest.mark.skipif(MISSING, reason="Missing test data (data/drs/raw.fits)")
def test_raw_init(rawobj):
    """Raw objects: tests initialization"""
    chan1 = rawobj.get_channel("CHAN01")
    shape = numpy.shape(chan1.data)
    assert shape == (rawobj.ny, rawobj.nx)


@pytest.mark.skipif(MISSING, reason="Missing test data (data/drs/raw.fits)")
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
