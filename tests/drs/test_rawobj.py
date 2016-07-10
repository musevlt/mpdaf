"""Test on RawFile objects to be used with py.test."""

from __future__ import absolute_import

import os
import numpy
import unittest

import pytest
from mpdaf.drs import RawFile

DATA_PATH = "data/drs/raw.fits"
DATA_MISSING = not os.path.exists(DATA_PATH)


class TestRawObj(object):

    def setUp(self):
        try:
            self.raw = RawFile(DATA_PATH)
            self.raw.progress = False
        except IOError:
            pass

    @unittest.skipIf(DATA_MISSING, "Missing test data (data/drs/raw.fits)")
    @pytest.mark.slow
    def test_init(self):
        """Raw objects: tests initialization"""
        chan1 = self.raw.get_channel("CHAN01")
        shape = numpy.shape(chan1.data)
        assert shape == (self.raw.ny, self.raw.nx)

    @unittest.skipIf(DATA_MISSING, "Missing test data (data/drs/raw.fits)")
    @pytest.mark.slow
    def test_mask(self):
        """Raw objects: tests strimmed and overscan functionalities"""
        overscan = self.raw[1].data[24, 12]
        pixel = self.raw[1].data[240, 120]
        out = self.raw[1].trimmed() * 10
        assert out.data[24, 12] == overscan
        assert out.data[240, 120] == 10 * pixel

        out = self.raw[1].overscan() * 2
        assert out.data[24, 12] == 2 * overscan
        assert out.data[240, 120] == pixel
