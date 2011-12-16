"""Test on RawFile objects to be used with py.test."""

import os
import sys
import numpy
import unittest

from drs import RawFile

class TestRawObj(unittest.TestCase):
    
    def setUp(self):
        self.raw = RawFile("data/drs/raw.fits")
        self.raw.progress = False
        
    def tearDown(self):
        del self.raw

    def test_init(self):
        """tests RawFile initialization"""
        chan1 = self.raw.get_channel("CHAN01")
        shape = numpy.shape(chan1.data)
        self.assertEqual(shape, (self.raw.ny, self.raw.nx))

    def test_operator(self):
        """tests arithmetic functions on RawFile objects"""
        chan1 = self.raw.get_channel("CHAN01")
        value1 = chan1.data[32,28]

        out = self.raw - self.raw
        chan2 = out.get_channel("CHAN01")
        value2 = chan2.data[32,28]
        self.assertEqual(value2, 0)

        out = self.raw + self.raw
        chan2 = out.get_channel("CHAN01")
        value2 = chan2.data[32,28]
        self.assertEqual(value2, 2*value1)

        out = self.raw * self.raw
        chan2 = out.get_channel("CHAN01")
        value2 = chan2.data[32,28]
        self.assertEqual(value2, value1 * value1)

        out = self.raw.sqrt()
        chan2 = out.get_channel("CHAN01")
        value2 = chan2.data[32,28]
        self.assertEqual(value2, numpy.sqrt(value1))
        
        del out
        del chan2

    def test_copy(self):
        """tests copy of RawFile objects"""
        raw2 = self.raw.copy()
        out = self.raw - raw2
        out2 = out.copy()
        del out
        chan2 = out2.get_channel("CHAN02")
        value2 = chan2.data[24,12]
        self.assertEqual(value2, 0)
        del out2

    def test_mask(self):
        """tests strimmed and overscan functionalities"""
        overscan = self.raw[1].data[24,12]
        pixel = self.raw[1].data[240,120]
        out = self.raw[1].trimmed() * 10
        self.assertEqual(out.data[24,12],overscan)
        self.assertEqual(out.data[240,120],10*pixel)

        out = self.raw[1].overscan() * 2
        self.assertEqual(out.data[24,12],2*overscan)
        self.assertEqual(out.data[240,120],pixel)

if __name__=='__main__':
    if not os.path.exists("data/drs/raw.fits"):
        print 'IOError: file data/drs/raw.fits not found.'
        print 'Test files are not stored on the git repository to limit its memory size.'
        print 'Please download it from http://urania1.univ-lyon1.fr/mpdaf/login'
        print ''
    else:
        unittest.main()