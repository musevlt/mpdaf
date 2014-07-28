"""Test on WCS and WaveCoord objects."""
import nose.tools
from nose.plugins.attrib import attr

import os
import sys
import numpy

from mpdaf.obj import WCS
from mpdaf.obj import WaveCoord
from mpdaf.obj import deg2sexa,sexa2deg

class TestWCS():

    def setUp(self):
        self.wcs = WCS(crval=(0,0))
        self.wcs.naxis1 = 6
        self.wcs.naxis2 = 5

    def tearDown(self):
        del self.wcs

    @attr(speed='fast')
    def test_copy(self):
        """WCS class: tests copy"""
        wcs2 = self.wcs.copy()
        nose.tools.assert_true(self.wcs.isEqual(wcs2))
        del wcs2

    @attr(speed='fast')
    def test_coordTransform(self):
        """WCS class: tests coordinates transformations"""
        pixcrd = [[0,0],[2,3],[3,2]]
        pixsky = self.wcs.pix2sky(pixcrd)
        pixcrd2 =self.wcs.sky2pix(pixsky)
        nose.tools.assert_equal(pixcrd[0][0],pixcrd2[0][0])
        nose.tools.assert_equal(pixcrd[0][1],pixcrd2[0][1])
        nose.tools.assert_equal(pixcrd[1][0],pixcrd2[1][0])
        nose.tools.assert_equal(pixcrd[1][1],pixcrd2[1][1])
        nose.tools.assert_equal(pixcrd[2][0],pixcrd2[2][0])
        nose.tools.assert_equal(pixcrd[2][1],pixcrd2[2][1])
        
    @attr(speed='fast')
    def test_get(self):
        """WCS class: tests getters"""
        nose.tools.assert_equal(self.wcs.get_step()[0],1.0)
        nose.tools.assert_equal(self.wcs.get_step()[1],1.0)
        nose.tools.assert_equal(self.wcs.get_start()[0],0.0)
        nose.tools.assert_equal(self.wcs.get_start()[1],0.0)
        nose.tools.assert_equal(self.wcs.get_end()[0],4.0)
        nose.tools.assert_equal(self.wcs.get_end()[1],5.0)
        wcs2 = WCS(crval=(0,0),shape=(5,6))
        nose.tools.assert_equal(wcs2.get_step()[0],1.0)
        nose.tools.assert_equal(wcs2.get_step()[1],1.0)
        nose.tools.assert_equal(wcs2.get_start()[0],-2.0)
        nose.tools.assert_equal(wcs2.get_start()[1],-2.5)
        nose.tools.assert_equal(wcs2.get_end()[0],2.0)
        nose.tools.assert_equal(wcs2.get_end()[1],2.5)
        del wcs2

class TestWaveCoord():

    def setUp(self):
        self.wave = WaveCoord(crval=0)
        self.wave.shape = 10

    def tearDown(self):
        del self.wave

    @attr(speed='fast')
    def test_copy(self):
        """WaveCoord class: tests copy"""
        wave2 = self.wave.copy()
        nose.tools.assert_true(self.wave.isEqual(wave2))
        del wave2

    @attr(speed='fast')
    def test_coordTransform(self):
        """WaveCoord class: tests coordinates transformations"""
        value = self.wave.coord(5)
        pixel = self.wave.pixel(value, nearest=True)
        nose.tools.assert_equal(pixel,5)
      
    @attr(speed='fast') 
    def test_get(self):
        """WaveCoord class: tests getters"""
        nose.tools.assert_equal(self.wave.get_step(),1.0)
        nose.tools.assert_equal(self.wave.get_start(),0.0)
        nose.tools.assert_equal(self.wave.get_end(),9.0)
        
class TestCoord():
    
    @attr(speed='fast')
    def test_deg_sexa(self):
        """tests degree/sexagesimal transformations"""
        ra = '23:51:41.268'
        dec = '-26:04:43.032'
        deg = sexa2deg([dec,ra])
        nose.tools.assert_almost_equal(deg[0],-26.07862,3)
        nose.tools.assert_almost_equal(deg[1],357.92195,3)
        sexa = deg2sexa([-26.07862, 357.92195])
        nose.tools.assert_equal(sexa[0],dec)
        nose.tools.assert_equal(sexa[1],ra)
        
