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
from mpdaf.obj import gauss_image

class TestObj(unittest.TestCase):

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

    def test_selectionOperator_Spectrum(self):
        """tests spectrum > or < number"""
        spectrum2 = self.spectrum1 > 13.8
        self.assertEqual(spectrum2.data.sum()*spectrum2.fscale,24*self.spectrum1.fscale)
        spectrum2 = self.spectrum1 >= 6*self.spectrum1.fscale
        self.assertEqual(spectrum2.data.sum()*spectrum2.fscale,30*self.spectrum1.fscale)
        spectrum2 = self.spectrum1 < 6*self.spectrum1.fscale
        self.assertEqual(spectrum2.data.sum()*spectrum2.fscale,15.5*self.spectrum1.fscale)
        spectrum2 = self.spectrum1 <= 6*self.spectrum1.fscale
        self.assertEqual(spectrum2.data.sum()*spectrum2.fscale,21.5*self.spectrum1.fscale)
        del spectrum2

    def test_arithmetricOperator_Spectrum(self):
        """tests arithmetic functions on Spectrum object"""
        spectrum2 = self.spectrum1 > 13.8 #[-,-,-,-,-,-,-,7,8,9]
        # +
        spectrum3 = self.spectrum1 + spectrum2
        self.assertEqual(spectrum3.data.data[3]*spectrum3.fscale,3*self.spectrum1.fscale)
        self.assertEqual(spectrum3.data.data[8]*spectrum3.fscale,16*self.spectrum1.fscale)
        spectrum3 = 4.2 + self.spectrum1
        self.assertEqual(spectrum3.data.data[3]*spectrum3.fscale,3*self.spectrum1.fscale+4.2)
        # -
        spectrum3 = self.spectrum1 - spectrum2
        self.assertEqual(spectrum3.data.data[3]*spectrum3.fscale,3*self.spectrum1.fscale)
        self.assertEqual(spectrum3.data.data[8]*spectrum3.fscale,0*self.spectrum1.fscale)
        spectrum3 = self.spectrum1 - 4.2
        self.assertEqual(spectrum3.data.data[8]*spectrum3.fscale,8*self.spectrum1.fscale - 4.2)
        # *
        spectrum3 = self.spectrum1 * spectrum2
        #self.assertEqual(spectrum3.data.data[3]*spectrum3.fscale,3*self.spectrum1.fscale)
        self.assertEqual(spectrum3.data.data[8]*spectrum3.fscale,64*self.spectrum1.fscale*self.spectrum1.fscale)
        spectrum3 = 4.2 * self.spectrum1
        self.assertEqual(spectrum3.data.data[9]*spectrum3.fscale,9*4.2*self.spectrum1.fscale)
        # /
        spectrum3 = self.spectrum1 / spectrum2
        #divide functions that have a validity domain returns the masked constant whenever the input is masked or falls outside the validity domain.
        self.assertEqual(spectrum3.data.data[8]*spectrum3.fscale,1)
        spectrum3 = 1.0 / (4.2 /self.spectrum1 )
        self.assertEqual(spectrum3.data.data[5]*spectrum3.fscale,5/4.2*self.spectrum1.fscale)
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

    def test_get_Spectrum(self):
        """tests Spectrum[]"""
        a = self.spectrum1[1:7]
        self.assertEqual(a.shape,6)
        a = self.spectrum1.get_lambda(1.2,15.6)
        self.assertEqual(a.shape,6)
        
    def test_rebin(self):
        """tests rebin functions"""
        spectrum2 = self.spectrum1.rebin(0.3) 
        #flux1 = self.spectrum1.data.sum()*self.spectrum1.wave.cdelt
        #flux2 = spectrum2.data.sum()*spectrum2.wave.cdelt
        flux1 = self.spectrum1.sum()*self.spectrum1.wave.cdelt
        flux2 = spectrum2.sum()*spectrum2.wave.cdelt
        self.assertAlmostEqual(flux1,flux2)
        
    def test_spectrum_methods(self):
        """tests spectrum methods"""
        sum1 = self.spectrum1.sum()
        self.assertAlmostEqual(sum1,self.spectrum1.data.sum()*self.spectrum1.fscale)
        spectrum2 = self.spectrum1[1:-2]
        sum1 =  self.spectrum1.sum(lmin=self.spectrum1.wave[1],lmax=self.spectrum1.wave[-2])
        sum2 = spectrum2.sum()
        self.assertAlmostEqual(sum1,sum2)
        mean1 =  self.spectrum1.mean(lmin=self.spectrum1.wave[1],lmax=self.spectrum1.wave[-2])
        mean2 = spectrum2.mean()
        self.assertAlmostEqual(mean1,mean2)
        
    def test_gauss_fit(self):
        wave = WaveCoord(crpix=1, cdelt=3.0, crval=4000, cunit = 'Angstrom')
        data = np.zeros(6000)
        spe = Spectrum(shape=6000, data=data,wave=wave,fscale= 2.3)
        spem = spe.add_gaussian(5000, 1200, 20)
        gauss = spem.gauss_fit(lmin=(4500,4800),lmax=(5200,6000), lpeak = 5000)
        self.assertAlmostEqual(gauss.lpeak,5000,2)
        self.assertAlmostEqual(gauss.flux,1200,2)
        self.assertAlmostEqual(gauss.fwhm,20,2)
    
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
        a = self.cube1.get_lambda(1.2,15.6)
        self.assertEqual(a.shape[0],6)
        self.assertEqual(a.shape[1],6)
        self.assertEqual(a.shape[2],5)
        a = self.cube1[2:4,0:2,1:4]
        self.assertEqual(a.get_start()[0],3.5)
        self.assertEqual(a.get_start()[1],0)
        self.assertEqual(a.get_start()[2],1)
        self.assertEqual(a.get_end()[0],6.5)
        self.assertEqual(a.get_end()[1],1)
        self.assertEqual(a.get_end()[2],3)

    def test_get_Image(self):
        """tests Image[]"""
        ima = self.image1[0:2,1:4]
        self.assertEqual(ima.shape[0],2)
        self.assertEqual(ima.shape[1],3)
        self.assertEqual(ima.get_start()[0],0)
        self.assertEqual(ima.get_start()[1],1)
        self.assertEqual(ima.get_end()[0],1)
        self.assertEqual(ima.get_end()[1],3)
        
    def test_resize_Image(self):
        """test Image.resize()"""
        mask = np.ones((6,5),dtype=bool)
        data = self.image1.data.data
        data[2:4,1:4] = 8
        mask[2:4,1:4] = 0
        self.image1.data = np.ma.MaskedArray(data, mask=mask)
        self.image1.resize()
        self.assertEqual(self.image1.shape[0],2)
        self.assertEqual(self.image1.shape[1],3)
        self.assertEqual(self.image1.sum(),2*3*8)
        self.assertEqual(self.image1.get_start()[0],2)
        self.assertEqual(self.image1.get_start()[1],1)
        self.assertEqual(self.image1.get_end()[0],3)
        self.assertEqual(self.image1.get_end()[1],3)
        
    def test_truncate_Image(self):
        """test Image.truncate()"""
        ima = self.image1.truncate(0,1,1,3)
        self.assertEqual(ima.shape[0],2)
        self.assertEqual(ima.shape[1],3)
        self.assertEqual(ima.get_start()[0],0)
        self.assertEqual(ima.get_start()[1],1)
        self.assertEqual(ima.get_end()[0],1)
        self.assertEqual(ima.get_end()[1],3)
        
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