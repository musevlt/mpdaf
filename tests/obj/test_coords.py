"""Test on WCS and WaveCoord objects."""

import os
import sys
import numpy
import unittest

from mpdaf.obj import WCS
from mpdaf.obj import WaveCoord

class TestWCS(unittest.TestCase):

    def setUp(self):
        self.wcs = WCS()
        self.wcs.wcs.naxis1 = 6
        self.wcs.wcs.naxis2 = 5

    def tearDown(self):
        del self.wcs

    def test_copy(self):
        """tests WCS copy"""
        wcs2 = self.wcs.copy()
        self.assertTrue(self.wcs.isEqual(wcs2))
        del wcs2

    def test_coordTransform(self):
        """tests WCS.sky2pix and WCS.pix2sky methods"""
        pixcrd = [[0,0],[2,3],[3,2]]
        pixsky = self.wcs.pix2sky(pixcrd)
        pixcrd2 =self.wcs.sky2pix(pixsky)
        self.assertEqual(pixcrd[0][0],pixcrd2[0][0])
        self.assertEqual(pixcrd[0][1],pixcrd2[0][1])
        self.assertEqual(pixcrd[1][0],pixcrd2[1][0])
        self.assertEqual(pixcrd[1][1],pixcrd2[1][1])
        self.assertEqual(pixcrd[2][0],pixcrd2[2][0])
        self.assertEqual(pixcrd[2][1],pixcrd2[2][1])
        
    def test_get(self):
        self.assertEqual(self.wcs.get_step()[0],1.0)
        self.assertEqual(self.wcs.get_step()[1],1.0)
        self.assertEqual(self.wcs.get_start()[0],0.0)
        self.assertEqual(self.wcs.get_start()[1],0.0)
        self.assertEqual(self.wcs.get_end()[0],4.0)
        self.assertEqual(self.wcs.get_end()[1],5.0)
        wcs2 = WCS(shape=(5,6))
        self.assertEqual(wcs2.get_step()[0],1.0)
        self.assertEqual(wcs2.get_step()[1],1.0)
        self.assertEqual(wcs2.get_start()[0],-2.0)
        self.assertEqual(wcs2.get_start()[1],-2.5)
        self.assertEqual(wcs2.get_end()[0],2.0)
        self.assertEqual(wcs2.get_end()[1],2.5)
        del wcs2

class TestWaveCoord(unittest.TestCase):

    def setUp(self):
        self.wave = WaveCoord()
        self.wave.shape = 10

    def tearDown(self):
        del self.wave

    def test_copy(self):
        """tests WaveCoord copy"""
        wave2 = self.wave.copy()
        self.assertTrue(self.wave.isEqual(wave2))
        del wave2

    def test_coordTransform(self):
        """tests WaveCoord.coord and WaveCoord.pixel methods"""
        value = self.wave.coord(5)
        pixel = self.wave.pixel(value, nearest=True)
        self.assertEqual(pixel,5)
        
    def test_get(self):
        self.assertEqual(self.wave.get_step(),1.0)
        self.assertEqual(self.wave.get_start(),0.0)
        self.assertEqual(self.wave.get_end(),9.0)

if __name__=='__main__':
    unittest.main()