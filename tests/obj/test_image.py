"""Test on Image objects."""
import nose.tools
from nose.plugins.attrib import attr

import os
import sys
import numpy as np

from mpdaf.obj import Spectrum
from mpdaf.obj import Image
from mpdaf.obj import Cube
from mpdaf.obj import WCS
from mpdaf.obj import WaveCoord
from mpdaf.obj import gauss_image

class TestImage():

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
    
    @attr(speed='fast')
    def test_arithmetricOperator_Image(self):
        """Image class: tests arithmetic functions"""
        # +
        image3 = self.image1 + self.image1
        nose.tools.assert_almost_equal(image3.data[3,3]*image3.fscale,4*self.image1.fscale)
        self.image1 += 4.2
        nose.tools.assert_almost_equal(self.image1.data[3,3]*self.image1.fscale,(2+4.2)*self.image1.fscale)
        # -
        image3 = self.image1 - self.image1
        nose.tools.assert_almost_equal(image3.data[3,3],0)
        self.image1 -= 4.2
        nose.tools.assert_almost_equal(self.image1.data[3,3]*self.image1.fscale,2)
        # *
        image3 = self.image1 * self.image1
        nose.tools.assert_almost_equal(image3.data[3,3],4)
        self.image1 *= 4.2
        nose.tools.assert_almost_equal(self.image1.data[3,3]*self.image1.fscale,2*4.2)
        # /
        image3 = self.image1 / self.image1
        nose.tools.assert_almost_equal(image3.data[3,3],1)
        self.image1 /= 4.2
        nose.tools.assert_almost_equal(self.image1.data[3,3]*self.image1.fscale,2)
        # with cube
        cube2 = self.image1 + self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.image1 - self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale - self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.image1 * self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale * self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.image1 / self.cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale / (self.cube1.data[k,j,i]*self.cube1.fscale))
        # spectrum * image
        cube2 = self.image1 * self.spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale * (self.image1.data[j,i]*self.image1.fscale))
        #
        image2 = (self.image1 *-2).abs()+(self.image1+4).sqrt()-2 
        nose.tools.assert_almost_equal(image2.data[3,3]*image2.fscale,np.abs(self.image1.data[3,3]*self.image1.fscale *-2)+np.sqrt(self.image1.data[3,3]*self.image1.fscale+4)-2 )

    @attr(speed='fast')
    def test_get_Image(self):
        """Image class: tests getters"""
        ima = self.image1[0:2,1:4]
        nose.tools.assert_equal(ima.shape[0],2)
        nose.tools.assert_equal(ima.shape[1],3)
        nose.tools.assert_equal(ima.get_start()[0],0)
        nose.tools.assert_equal(ima.get_start()[1],1)
        nose.tools.assert_equal(ima.get_end()[0],1)
        nose.tools.assert_equal(ima.get_end()[1],3)
        nose.tools.assert_equal(ima.get_step()[0],1)
        nose.tools.assert_equal(ima.get_step()[1],1)
        del ima
      
    @attr(speed='fast')  
    def test_resize_Image(self):
        """Image class: tests resize method"""
        mask = np.ones((6,5),dtype=bool)
        data = self.image1.data.data
        data[2:4,1:4] = 8
        mask[2:4,1:4] = 0
        self.image1.data = np.ma.MaskedArray(data, mask=mask)
        self.image1.resize()
        nose.tools.assert_equal(self.image1.shape[0],2)
        nose.tools.assert_equal(self.image1.shape[1],3)
        nose.tools.assert_equal(self.image1.sum(),2*3*8)
        nose.tools.assert_equal(self.image1.get_start()[0],2)
        nose.tools.assert_equal(self.image1.get_start()[1],1)
        nose.tools.assert_equal(self.image1.get_end()[0],3)
        nose.tools.assert_equal(self.image1.get_end()[1],3)
        nose.tools.assert_equal(self.image1.get_range()[0][0],self.image1.get_start()[0])
        nose.tools.assert_equal(self.image1.get_range()[0][1],self.image1.get_start()[1])
        nose.tools.assert_equal(self.image1.get_range()[1][0],self.image1.get_end()[0])
        nose.tools.assert_equal(self.image1.get_range()[1][1],self.image1.get_end()[1])
        nose.tools.assert_equal(self.image1.get_rot(),0)
     
    @attr(speed='fast')  
    def test_truncate_Image(self):
        """Image class: tests truncation"""
        self.image1 = self.image1.truncate(0,1,1,3)
        nose.tools.assert_equal(self.image1.shape[0],2)
        nose.tools.assert_equal(self.image1.shape[1],3)
        nose.tools.assert_equal(self.image1.get_start()[0],0)
        nose.tools.assert_equal(self.image1.get_start()[1],1)
        nose.tools.assert_equal(self.image1.get_end()[0],1)
        nose.tools.assert_equal(self.image1.get_end()[1],3)
    
    @attr(speed='fast')   
    def test_sum_Image(self):
        """Image class: tests sum"""
        sum1 = self.image1.sum()
        nose.tools.assert_equal(sum1,6*5*2)
        sum2 = self.image1.sum(axis=0)
        nose.tools.assert_equal(sum2.shape,5)
        nose.tools.assert_equal(sum2.get_start(),0)
        nose.tools.assert_equal(sum2.get_end(),4)
#        nose.tools.assert_equal(sum2.shape[0],1)
#        nose.tools.assert_equal(sum2.shape[1],5)
#        nose.tools.assert_equal(sum2.get_start()[0],0)
#        nose.tools.assert_equal(sum2.get_start()[1],0)
#        nose.tools.assert_equal(sum2.get_end()[0],0)
#        nose.tools.assert_equal(sum2.get_end()[1],4)
    
    @attr(speed='fast')   
    def test_gauss_Image(self):
        """Image class: tests Gaussian fit"""
        wcs = WCS (cdelt=(0.2,0.3), crval=(8.5,12),shape=(40,30))
        ima = gauss_image(wcs=wcs,fwhm=(1,2),factor=1, rot = 60)
        #ima2 = gauss_image(wcs=wcs,width=(1,2),factor=2, rot = 60)
        gauss = ima.gauss_fit(pos_min=(4, 7), pos_max=(13,17), cont=0)
        nose.tools.assert_almost_equal(gauss.center[0], 8.5)
        nose.tools.assert_almost_equal(gauss.center[1], 12)
        nose.tools.assert_almost_equal(gauss.flux, 1)
        gauss2 = ima.gauss_fit(pos_min=(5, 6), pos_max=(12,16), cont=0)
        nose.tools.assert_almost_equal(gauss2.center[0], 8.5)
        nose.tools.assert_almost_equal(gauss2.center[1], 12)
        nose.tools.assert_almost_equal(gauss2.flux, 1)
        ima3 = gauss_image(wcs=wcs,fwhm=(1,2))
        #sigma = ima3.fwhm()/(2.*np.sqrt(2.*np.log(2.0)))
        #nose.tools.assert_almost_equal(sigma[0], 1)
        #nose.tools.assert_almost_equal(sigma[1], 2)
        
    @attr(speed='fast')   
    def test_mask_Image(self):
        """Image class: tests mask functionalities"""
        self.image1.mask((2,2),(1,1),pix=True,inside=False)
        nose.tools.assert_equal(self.image1.sum(),2*9)
        self.image1.unmask()
        self.image1.mask((2,2),(3600,3600),inside=False)
        nose.tools.assert_equal(self.image1.sum(),2*9)
        
    @attr(speed='fast')   
    def test_background_Image(self):
        """Image class: tests background value"""
        nose.tools.assert_equal(self.image1.background()[0],2)
        nose.tools.assert_equal(self.image1.background()[1],0)
        
    @attr(speed='fast')   
    def test_peak_Image(self):
        """Image class: tests peak research"""
        self.image1.data[2,2] = 8
        p = self.image1.peak()
        nose.tools.assert_equal(p['p'],2)
        nose.tools.assert_equal(p['q'],2)
        
    @attr(speed='fast')
    def test_clone(self):
        """Image class: tests clone method."""
        ima2 = self.image1.clone()
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(ima2.data[j,i]*ima2.fscale,0)
                
    @attr(speed='fast')
    def test_rotate(self):
        """Image class: tests clone method."""
        ima = Image("data/obj/a370II.fits")
        ima2 = ima.rotate(30)
        
        _theta = -30* np.pi / 180.
        _mrot = np.zeros(shape=(2,2),dtype=np.double)
        _mrot[0] = (np.cos(_theta),np.sin(_theta))
        _mrot[1] = (-np.sin(_theta),np.cos(_theta))
        
        center= (np.array([ima.shape[0],ima.shape[1]])+1)/2. -1
        pixel= np.array([910,1176])
        r = np.dot(pixel - center, _mrot)
        r[0] = r[0] + center[0]
        r[1] = r[1] + center[1]
        nose.tools.assert_almost_equal(ima.wcs.pix2sky(pixel)[0][0],ima2.wcs.pix2sky(r)[0][0])
        nose.tools.assert_almost_equal(ima.wcs.pix2sky(pixel)[0][1],ima2.wcs.pix2sky(r)[0][1])
        
    @attr(speed='fast')
    def test_inside(self):
        """Image class: tests inside method."""
        ima = Image("data/obj/a370II.fits")
        nose.tools.assert_equal(ima.inside((39.951088,-1.4977398)),False)
        
        
        
        