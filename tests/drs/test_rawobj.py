"""Test on RawFile objects to be used with py.test."""

import nose.tools
import os
import numpy
import unittest

from nose.plugins.attrib import attr
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
    @attr(speed='slow')
    def test_init(self):
        """Raw objects: tests initialization"""
        chan1 = self.raw.get_channel("CHAN01")
        shape = numpy.shape(chan1.data)
        nose.tools.assert_equal(shape, (self.raw.ny, self.raw.nx))

    @unittest.skipIf(DATA_MISSING, "Missing test data (data/drs/raw.fits)")
    @attr(speed='slow')
    def test_operator(self):
        """Raw objects: tests arithmetic functions"""
        chan1 = self.raw.get_channel("CHAN01")
        value1 = chan1.data[32, 28]

        out = self.raw - self.raw
        chan2 = out.get_channel("CHAN01")
        value2 = chan2.data[32, 28]
        nose.tools.assert_equal(value2, 0)

        out = self.raw + self.raw
        chan2 = out.get_channel("CHAN01")
        value2 = chan2.data[32, 28]
        nose.tools.assert_equal(value2, 2 * value1)

        out = self.raw * self.raw
        chan2 = out.get_channel("CHAN01")
        value2 = chan2.data[32, 28]
        nose.tools.assert_equal(value2, value1 * value1)

        out = self.raw.sqrt()
        chan2 = out.get_channel("CHAN01")
        value2 = chan2.data[32, 28]
        nose.tools.assert_equal(value2, numpy.sqrt(value1))

        del out
        del chan2

    @unittest.skipIf(DATA_MISSING, "Missing test data (data/drs/raw.fits)")
    @attr(speed='slow')
    def test_copy(self):
        """Raw objects: tests copy"""
        raw2 = self.raw.copy()
        out = self.raw - raw2
        out2 = out.copy()
        del out
        chan2 = out2.get_channel("CHAN02")
        value2 = chan2.data[24, 12]
        nose.tools.assert_equal(value2, 0)
        del out2

    @unittest.skipIf(DATA_MISSING, "Missing test data (data/drs/raw.fits)")
    @attr(speed='slow')
    def test_mask(self):
        """Raw objects: tests strimmed and overscan functionalities"""
        overscan = self.raw[1].data[24, 12]
        pixel = self.raw[1].data[240, 120]
        out = self.raw[1].trimmed() * 10
        nose.tools.assert_equal(out.data[24, 12], overscan)
        nose.tools.assert_equal(out.data[240, 120], 10 * pixel)

        out = self.raw[1].overscan() * 2
        nose.tools.assert_equal(out.data[24, 12], 2 * overscan)
        nose.tools.assert_equal(out.data[240, 120], pixel)
