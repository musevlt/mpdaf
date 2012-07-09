"""Test on Image objects."""

import os
import sys
import numpy as np
import unittest

from mpdaf.obj import Spectrum
from mpdaf.obj import Image
from mpdaf.obj import Cube
from mpdaf.obj import WCS
from mpdaf.obj import WaveCoord
from mpdaf.obj import gauss_image

class TestImage(unittest.TestCase):

    def setUp(self):
        wcs = WCS()
        wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit = 'Angstrom')
        self.cube1 = Cube(shape=(10,6,5),data=np.ones(shape=(10,6,5)),wave=wave,wcs=wcs,fscale= 2.3)
        data = np.ones(shape=(6,5))*2
        self.image1 = Image(shape=(6,5),data=data,wcs=wcs)
        self.spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave,fscale= 2.3)

    def tearDown(self):
        del self.cube1
        del self.image1
        del self.spectrum1
    
    def test_arithmetricOperator_Image(self):
        """tests arithmetic functions on Image object"""
        # +
        image3 = self.image1 + self.image1
        self.assertAlmostEqual(image3.data[3,3]*image3.fscale,4*self.image1.fscale)
        self.image1 += 4.2
        self.assertAlmostEqual(self.image1.data[3,3]*self.image1.fscale,(2+4.2)*self.image1.fscale)
        # -
        image3 = self.image1 - self.image1
        self.assertAlmostEqual(image3.data[3,3],0)
        self.image1 -= 4.2
        self.assertAlmostEqual(self.image1.data[3,3]*self.image1.fscale,2)
        # *
        image3 = self.image1 * self.image1
        self.assertAlmostEqual(image3.data[3,3],4)
        self.image1 *= 4.2
        self.assertAlmostEqual(self.image1.data[3,3]*self.image1.fscale,2*4.2)
        # /
        image3 = self.image1 / self.image1
        self.assertAlmostEqual(image3.data[3,3],1)
        self.image1 /= 4.2
        self.assertAlmostEqual(self.image1.data[3,3]*self.image1.fscale,2)
        # with cube
        cube2 = self.image1 + self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.image1 - self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale - self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.image1 * self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale * self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.image1 / self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale / (self.cube1.data[k,j,i]*self.cube1.fscale))
        # spectrum * image
        cube2 = self.image1 * self.spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale * (self.image1.data[j,i]*self.image1.fscale))

    def test_get_Image(self):
        """tests Image[]"""
        ima = self.image1[0:2,1:4]
        self.assertEqual(ima.shape[0],2)
        self.assertEqual(ima.shape[1],3)
        self.assertEqual(ima.get_start()[0],0)
        self.assertEqual(ima.get_start()[1],1)
        self.assertEqual(ima.get_end()[0],1)
        self.assertEqual(ima.get_end()[1],3)
        del ima
        
    def test_resize_Image(self):
        """test Image.resize()"""
        mask = np.ones((6,5),dtype=bool)
        data = self.image1.data.data
        data[2:4,1:4] = 8
        mask[2:4,1:4] = 0
        self.image1.data = np.ma.MaskedArray(data, mask=mask)
        ima2 = self.image1.resize()
        self.assertEqual(ima2.shape[0],2)
        self.assertEqual(ima2.shape[1],3)
        self.assertEqual(ima2.sum(),2*3*8)
        self.assertEqual(ima2.get_start()[0],2)
        self.assertEqual(ima2.get_start()[1],1)
        self.assertEqual(ima2.get_end()[0],3)
        self.assertEqual(ima2.get_end()[1],3)
        del ima2
        
    def test_truncate_Image(self):
        """test Image.truncate()"""
        ima = self.image1.truncate(0,1,1,3)
        self.assertEqual(ima.shape[0],2)
        self.assertEqual(ima.shape[1],3)
        self.assertEqual(ima.get_start()[0],0)
        self.assertEqual(ima.get_start()[1],1)
        self.assertEqual(ima.get_end()[0],1)
        self.assertEqual(ima.get_end()[1],3)
        del ima
        
    def test_sum_Image(self):
        """test Image.sum()"""
        sum1 = self.image1.sum()
        self.assertEqual(sum1,6*5*2)
        sum2 = self.image1.sum(axis=0)
        self.assertEqual(sum2.shape[0],1)
        self.assertEqual(sum2.shape[1],5)
        self.assertEqual(sum2.get_start()[0],0)
        self.assertEqual(sum2.get_start()[1],0)
        self.assertEqual(sum2.get_end()[0],0)
        self.assertEqual(sum2.get_end()[1],4)
        
    def test_gauss_Image(self):
        """test gauss_image and gauss_fit"""
        wcs = WCS (cdelt=(0.2,0.3), crval=(8.5,12),shape=(40,30))
        ima = gauss_image(wcs=wcs,width=(1,2),factor=1, rot = 60)
        gauss = ima.gauss_fit(pos_min=(4, 7), pos_max=(13,17), cont=0)
        ima2 = gauss_image(wcs=wcs,width=(1,2),factor=2, rot = 60)
        gauss2 = ima.gauss_fit(pos_min=(5, 6), pos_max=(12,16), cont=0)
        self.assertAlmostEqual(gauss.center[0], 8.5)
        self.assertAlmostEqual(gauss.center[1], 12)
        self.assertAlmostEqual(gauss.flux, 1)
        self.assertAlmostEqual(gauss2.center[0], 8.5)
        self.assertAlmostEqual(gauss2.center[1], 12)
        self.assertAlmostEqual(gauss2.flux, 1)
        

if __name__=='__main__':
    unittest.main()