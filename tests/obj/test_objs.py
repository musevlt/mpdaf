"""Test on Spectrum, Image and Cube objects."""

import os
import sys
import numpy as np
import unittest

from mpdaf.obj import Spectrum
from mpdaf.obj import Image
from mpdaf.obj import Cube
from mpdaf.obj import WCS
from mpdaf.obj import WaveCoord

class TestObj(unittest.TestCase):

    def setUp(self):
        wcs = WCS(dim=(5,6))
        wave = WaveCoord(dim = 10)
        self.cube1 = Cube(shape=(10,6,5),data=np.ones(shape=(10,6,5)),wave=wave,wcs=wcs,fscale= 2.3)
        data = np.ones(shape=(6,5))*2
        self.image1 = Image(shape=(6,5),data=data,wcs=wcs)
        self.spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave)

    def tearDown(self):
        del self.cube1
        del self.image1
        del self.spectrum1

    def test_selectionOperator_Spectrum(self):
        """tests spectrum > or < number"""
        spectrum2 = self.spectrum1 > 6
        self.assertEqual(spectrum2.data.sum(),24)
        spectrum2 = self.spectrum1 >= 6
        self.assertEqual(spectrum2.data.sum(),30)
        spectrum2 = self.spectrum1 < 6
        self.assertEqual(spectrum2.data.sum(),15.5)
        spectrum2 = self.spectrum1 <= 6
        self.assertEqual(spectrum2.data.sum(),21.5)
        del spectrum2

    def test_arithmetricOperator_Spectrum(self):
        """tests arithmetic functions on Spectrum object"""
        spectrum2 = self.spectrum1 > 6 #[-,-,-,-,-,-,-,7,8,9]
        # +
        spectrum3 = self.spectrum1 + spectrum2
        spectrum3.removeMask()
        self.assertEqual(spectrum3.data[3]*spectrum3.fscale,3*self.spectrum1.fscale)
        self.assertEqual(spectrum3.data[8]*spectrum3.fscale,16*self.spectrum1.fscale)
        spectrum3 = self.spectrum1 + 4.2
        self.assertEqual(spectrum3.data[3]*spectrum3.fscale,(3+4.2)*self.spectrum1.fscale)
        # -
        spectrum3 = self.spectrum1 - spectrum2
        spectrum3.removeMask()
        self.assertEqual(spectrum3.data[3]*spectrum3.fscale,3*self.spectrum1.fscale)
        self.assertEqual(spectrum3.data[8]*spectrum3.fscale,0*self.spectrum1.fscale)
        spectrum3 = self.spectrum1 - 4.2
        self.assertEqual(spectrum3.data[8]*spectrum3.fscale,(8-4.2)*self.spectrum1.fscale)
        # *
        spectrum3 = self.spectrum1 * spectrum2
        spectrum3.removeMask()
        self.assertEqual(spectrum3.data[3]*spectrum3.fscale,3*self.spectrum1.fscale)
        self.assertEqual(spectrum3.data[8]*spectrum3.fscale,64*self.spectrum1.fscale)
        spectrum3 = self.spectrum1 * 4.2
        self.assertEqual(spectrum3.data[9]*spectrum3.fscale,9*4.2*self.spectrum1.fscale)
        # /
        spectrum3 = self.spectrum1 / spectrum2
        spectrum3.removeMask()
        #divide functions that have a validity domain returns the masked constant whenever the input is masked or falls outside the validity domain.
        #self.assertEqual(spectrum3.data[3],3)
        self.assertEqual(spectrum3.data[8]*spectrum3.fscale,1*self.spectrum1.fscale)
        spectrum3 = self.spectrum1 / 4.2
        self.assertEqual(spectrum3.data[5]*spectrum3.fscale,5/4.2*self.spectrum1.fscale)
        # with cube
        cube2 = self.spectrum1 + self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.spectrum1 - self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale - self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.spectrum1 * self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale * self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.spectrum1 / self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale / (self.cube1.data[k,j,i]*self.cube1.fscale))
        # spectrum * image
        cube2 = self.spectrum1 * self.image1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale * (self.image1.data[j,i]*self.image1.fscale))

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

    def test_arithmetricOperator_Cube(self):
        """tests arithmetic functions on Cube object"""
        cube2 = self.image1 + self.cube1
        # +
        cube3 = self.cube1 + cube2
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube3.data[k,j,i]*cube3.fscale,self.cube1.data[k,j,i]*self.cube1.fscale + (cube2.data[k,j,i]*cube2.fscale))
        # -
        cube3 = self.cube1 - cube2
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube3.data[k,j,i]*cube3.fscale,self.cube1.data[k,j,i]*self.cube1.fscale - (cube2.data[k,j,i]*cube2.fscale))
        # *
        cube3 = self.cube1 * cube2
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube3.data[k,j,i]*cube3.fscale,self.cube1.data[k,j,i]*self.cube1.fscale * (cube2.data[k,j,i]*cube2.fscale))
        # /
        cube3 = self.cube1 / cube2
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube3.data[k,j,i]*cube3.fscale,self.cube1.data[k,j,i]*self.cube1.fscale / (cube2.data[k,j,i]*cube2.fscale))
        # with spectrum
        cube2 = self.cube1 + self.spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 - self.spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,-self.spectrum1.data[k]*self.spectrum1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 * self.spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale * self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 / self.spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,(self.cube1.data[k,j,i]*self.cube1.fscale)/(self.spectrum1.data[k]*self.spectrum1.fscale))
        # with image
        cube2 = self.cube1 + self.image1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 - self.image1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,-self.image1.data[j,i]*self.image1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 * self.image1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale * self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 / self.image1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    self.assertAlmostEqual(cube2.data[k,j,i]*cube2.fscale,(self.cube1.data[k,j,i]*self.cube1.fscale) / (self.image1.data[j,i]*self.image1.fscale))

    def test_get_Cube(self):
        """tests Cube[]"""
        a = self.cube1[2,:,:]
        self.assertEqual(a.shape[0],6)
        self.assertEqual(a.shape[1],5)
        a = self.cube1[:,2,3]
        self.assertEqual(a.shape,10)
        a = self.cube1[1:7,0:2,0:3]
        self.assertEqual(a.shape[0],6)
        self.assertEqual(a.shape[1],2)
        self.assertEqual(a.shape[2],3)
        a = self.cube1.get_lambda(1.2,5.6)
        self.assertEqual(a.shape[0],5)
        self.assertEqual(a.shape[1],6)
        self.assertEqual(a.shape[2],5)

    def test_get_Image(self):
        """tests Image[]"""
        a = self.image1[0:2,0:3]
        self.assertEqual(a.shape[0],2)
        self.assertEqual(a.shape[1],3)

    def test_get_Spectrum(self):
        """tests Spectrum[]"""
        a = self.spectrum1[1:7]
        self.assertEqual(a.shape,6)
        a = self.spectrum1.get_lambda(1.2,5.6)
        self.assertEqual(a.shape,5)

if __name__=='__main__':
    unittest.main()