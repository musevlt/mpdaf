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
from mpdaf.obj import moffat_image

class TestImage():
    
    @attr(speed='fast')
    def test_arithmetricOperator_Image(self):
        """Image class: tests arithmetic functions"""
        wcs = WCS()
        wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit = 'Angstrom')
        data = np.ones(shape=(6,5))*2
        image1 = Image(shape=(6,5),data=data,wcs=wcs)
        cube1 = Cube(shape=(10,6,5),data=np.ones(shape=(10,6,5)),wave=wave,wcs=wcs,fscale= 2.3)
        # +
        image3 = image1 + image1
        nose.tools.assert_almost_equal(image3.data[3,3]*image3.fscale,4*image1.fscale)
        image1 += 4.2
        nose.tools.assert_almost_equal(image1.data[3,3]*image1.fscale,(2+4.2)*image1.fscale)
        # -
        image3 = image1 - image1
        nose.tools.assert_almost_equal(image3.data[3,3],0)
        image1 -= 4.2
        nose.tools.assert_almost_equal(image1.data[3,3]*image1.fscale,2)
        # *
        image3 = image1 * image1
        nose.tools.assert_almost_equal(image3.data[3,3],4)
        image1 *= 4.2
        nose.tools.assert_almost_equal(image1.data[3,3]*image1.fscale,2*4.2)
        # /
        image3 = image1 / image1
        nose.tools.assert_almost_equal(image3.data[3,3],1)
        image1 /= 4.2
        nose.tools.assert_almost_equal(image1.data[3,3]*image1.fscale,2)
        # with cube
        cube2 = image1 + cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,image1.data[j,i]*image1.fscale + cube1.data[k,j,i]*cube1.fscale)
        cube2 = image1 - cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,image1.data[j,i]*image1.fscale - cube1.data[k,j,i]*cube1.fscale)
        cube2 = image1 * cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,image1.data[j,i]*image1.fscale * cube1.data[k,j,i]*cube1.fscale)
        cube2 = image1 / cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,image1.data[j,i]*image1.fscale / (cube1.data[k,j,i]*cube1.fscale))
        # spectrum * image
        spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave,fscale= 2.3)
        cube2 = image1 * spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,spectrum1.data[k]*spectrum1.fscale * (image1.data[j,i]*image1.fscale))
        #
        image2 = (image1 *-2).abs()+(image1+4).sqrt()-2 
        nose.tools.assert_almost_equal(image2.data[3,3]*image2.fscale,np.abs(image1.data[3,3]*image1.fscale *-2)+np.sqrt(image1.data[3,3]*image1.fscale+4)-2 )

    @attr(speed='fast')
    def test_get_Image(self):
        """Image class: tests getters"""
        wcs = WCS()
        data = np.ones(shape=(6,5))*2
        image1 = Image(shape=(6,5),data=data,wcs=wcs)
        ima = image1[0:2,1:4]
        nose.tools.assert_equal(ima.shape[0],2)
        nose.tools.assert_equal(ima.shape[1],3)
        nose.tools.assert_equal(ima.get_start()[0],0)
        nose.tools.assert_equal(ima.get_start()[1],1)
        nose.tools.assert_equal(ima.get_end()[0],1)
        nose.tools.assert_equal(ima.get_end()[1],3)
        nose.tools.assert_equal(ima.get_step()[0],1)
        nose.tools.assert_equal(ima.get_step()[1],1)
      
    @attr(speed='fast')  
    def test_resize_Image(self):
        """Image class: tests resize method"""
        wcs = WCS()
        data = np.ones(shape=(6,5))*2
        image1 = Image(shape=(6,5),data=data,wcs=wcs)
        mask = np.ones((6,5),dtype=bool)
        data = image1.data.data
        data[2:4,1:4] = 8
        mask[2:4,1:4] = 0
        image1.data = np.ma.MaskedArray(data, mask=mask)
        image1.resize()
        nose.tools.assert_equal(image1.shape[0],2)
        nose.tools.assert_equal(image1.shape[1],3)
        nose.tools.assert_equal(image1.sum(),2*3*8)
        nose.tools.assert_equal(image1.get_start()[0],2)
        nose.tools.assert_equal(image1.get_start()[1],1)
        nose.tools.assert_equal(image1.get_end()[0],3)
        nose.tools.assert_equal(image1.get_end()[1],3)
        nose.tools.assert_equal(image1.get_range()[0][0],image1.get_start()[0])
        nose.tools.assert_equal(image1.get_range()[0][1],image1.get_start()[1])
        nose.tools.assert_equal(image1.get_range()[1][0],image1.get_end()[0])
        nose.tools.assert_equal(image1.get_range()[1][1],image1.get_end()[1])
        nose.tools.assert_equal(image1.get_rot(),0)
     
    @attr(speed='fast')  
    def test_truncate_Image(self):
        """Image class: tests truncation"""
        wcs = WCS()
        data = np.ones(shape=(6,5))*2
        image1 = Image(shape=(6,5),data=data,wcs=wcs)
        image1 = image1.truncate(0,1,1,3)
        nose.tools.assert_equal(image1.shape[0],2)
        nose.tools.assert_equal(image1.shape[1],3)
        nose.tools.assert_equal(image1.get_start()[0],0)
        nose.tools.assert_equal(image1.get_start()[1],1)
        nose.tools.assert_equal(image1.get_end()[0],1)
        nose.tools.assert_equal(image1.get_end()[1],3)
    
    @attr(speed='fast')   
    def test_sum_Image(self):
        """Image class: tests sum"""
        wcs = WCS()
        data = np.ones(shape=(6,5))*2
        image1 = Image(shape=(6,5),data=data,wcs=wcs)
        sum1 = image1.sum()
        nose.tools.assert_equal(sum1,6*5*2)
        sum2 = image1.sum(axis=0)
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
    def test_moffat_Image(self):
        """Image class: tests Moffat fit"""
        wcs = WCS (cdelt=(0.2,0.3), crval=(8.5,12),shape=(40,30))
        ima = moffat_image(wcs=WCS(),I=12.3, a=1.8, q=1, n=1.6, rot = 0.)
        moffat = ima.moffat_fit(pos_min=(0, 0), pos_max=(100,100), cont=0,plot=True)
        nose.tools.assert_almost_equal(moffat.center[0], 50.)
        nose.tools.assert_almost_equal(moffat.center[1], 50.)
        nose.tools.assert_almost_equal(moffat.I, 12.3)
        nose.tools.assert_almost_equal(moffat.a, 1.8)
        nose.tools.assert_almost_equal(moffat.q, 1)
        nose.tools.assert_almost_equal(moffat.n, 1.6)
        
    @attr(speed='fast')   
    def test_mask_Image(self):
        """Image class: tests mask functionalities"""
        wcs = WCS()
        data = np.ones(shape=(6,5))*2
        image1 = Image(shape=(6,5),data=data,wcs=wcs)
        image1.mask((2,2),(1,1),pix=True,inside=False)
        nose.tools.assert_equal(image1.sum(),2*9)
        image1.unmask()
        image1.mask((2,2),(3600,3600),inside=False)
        nose.tools.assert_equal(image1.sum(),2*9)
        
    @attr(speed='fast')   
    def test_background_Image(self):
        """Image class: tests background value"""
        wcs = WCS()
        data = np.ones(shape=(6,5))*2
        image1 = Image(shape=(6,5),data=data,wcs=wcs)
        (background,std) = image1.background()
        nose.tools.assert_equal(background,2)
        nose.tools.assert_equal(std,0)
        ima = Image("data/obj/a370II.fits")
        (background,std) = ima[1647:1732,618:690].background()
        #compare with IRAF results
        nose.tools.assert_true((background-std<1989) & (background+std>1989))
        
    @attr(speed='fast')   
    def test_peak_Image(self):
        """Image class: tests peak research"""
        wcs = WCS()
        data = np.ones(shape=(6,5))*2
        image1 = Image(shape=(6,5),data=data,wcs=wcs)
        image1.data[2,3] = 8
        p = image1.peak()
        nose.tools.assert_equal(p['p'],2)
        nose.tools.assert_equal(p['q'],3)
        ima = Image("data/obj/a370II.fits")
        p = ima.peak(center=(790,875),radius=20,pix=True,plot=False)
        nose.tools.assert_almost_equal(p['p'],793.3,0.1)
        nose.tools.assert_almost_equal(p['q'],875.8,0.1)
        
    @attr(speed='fast')
    def test_clone(self):
        """Image class: tests clone method."""
        wcs = WCS()
        data = np.ones(shape=(6,5))*2
        image1 = Image(shape=(6,5),data=data,wcs=wcs)
        ima2 = image1.clone()
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(ima2.data[j,i]*ima2.fscale,0)
        ima = Image("data/obj/a370II.fits")
        ima2 = ima.clone()+1000
        nose.tools.assert_equal(ima2.sum(axis=0).data[1000],ima.shape[0]*1000)
        nose.tools.assert_equal(ima2.sum(),ima.shape[0]*ima.shape[1]*1000)
                
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
        
        
        
        