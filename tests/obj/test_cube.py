"""Test on Cube objects."""
import nose.tools
from nose.plugins.attrib import attr

import os
import sys
import numpy as np

from mpdaf.obj import Spectrum
from mpdaf.obj import Image
from mpdaf.obj import Cube, iter_spe, iter_ima
from mpdaf.obj import WCS
from mpdaf.obj import WaveCoord
from mpdaf.obj import gauss_image

class TestCube():

    def setUp(self):
        wcs = WCS(crval=(0,0), crpix = 1.0, shape=(6,5))
        wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit = 'Angstrom', shape=10)
        self.cube1 = Cube(shape=(10,6,5),data=np.ones(shape=(10,6,5)),wave=wave,wcs=wcs,fscale= 2.3)
        data = np.ones(shape=(6,5))*2
        self.image1 = Image(shape=(6,5),data=data,wcs=wcs)
        self.spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave,fscale= 2.3)

    def tearDown(self):
        del self.cube1
        del self.image1
        del self.spectrum1

    @attr(speed='fast')
    def test_arithmetricOperator_Cube(self):
        """Cube class: tests arithmetic functions"""
        cube2 = self.image1 + self.cube1
        # +
        cube3 = self.cube1 + cube2
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube3.data[k,j,i]*cube3.fscale,self.cube1.data[k,j,i]*self.cube1.fscale + (cube2.data[k,j,i]*cube2.fscale))
        # -
        cube3 = self.cube1 - cube2
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube3.data[k,j,i]*cube3.fscale,self.cube1.data[k,j,i]*self.cube1.fscale - (cube2.data[k,j,i]*cube2.fscale))
        # *
        cube3 = self.cube1 * cube2
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube3.data[k,j,i]*cube3.fscale,self.cube1.data[k,j,i]*self.cube1.fscale * (cube2.data[k,j,i]*cube2.fscale))
        # /
        cube3 = self.cube1 / cube2
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube3.data[k,j,i]*cube3.fscale,self.cube1.data[k,j,i]*self.cube1.fscale / (cube2.data[k,j,i]*cube2.fscale))
        # with spectrum
        cube2 = self.cube1 + self.spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 - self.spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,-self.spectrum1.data[k]*self.spectrum1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 * self.spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,self.spectrum1.data[k]*self.spectrum1.fscale * self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 / self.spectrum1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,(self.cube1.data[k,j,i]*self.cube1.fscale)/(self.spectrum1.data[k]*self.spectrum1.fscale))
        # with image
        cube2 = self.cube1 + self.image1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 - self.image1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,-self.image1.data[j,i]*self.image1.fscale + self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 * self.image1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,self.image1.data[j,i]*self.image1.fscale * self.cube1.data[k,j,i]*self.cube1.fscale)
        cube2 = self.cube1 / self.image1
        cube3 = self.cube1.clone()
        cube3[:] = cube2
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube3.data[k,j,i]*cube3.fscale,(self.cube1.data[k,j,i]*self.cube1.fscale) / (self.image1.data[j,i]*self.image1.fscale))

    @attr(speed='fast')
    def test_get_Cube(self):
        """Cube class: tests getters"""
        a = self.cube1[2,:,:]
        nose.tools.assert_equal(a.shape[0],6)
        nose.tools.assert_equal(a.shape[1],5)
        a = self.cube1[:,2,3]
        nose.tools.assert_equal(a.shape,10)
        a = self.cube1[1:7,0:2,0:3]
        nose.tools.assert_equal(a.shape[0],6)
        nose.tools.assert_equal(a.shape[1],2)
        nose.tools.assert_equal(a.shape[2],3)
        a = self.cube1.get_lambda(1.2,15.6)
        nose.tools.assert_equal(a.shape[0],6)
        nose.tools.assert_equal(a.shape[1],6)
        nose.tools.assert_equal(a.shape[2],5)
        a = self.cube1[2:4,0:2,1:4]
        nose.tools.assert_equal(a.get_start()[0],3.5)
        nose.tools.assert_equal(a.get_start()[1],0)
        nose.tools.assert_equal(a.get_start()[2],1)
        nose.tools.assert_equal(a.get_end()[0],6.5)
        nose.tools.assert_equal(a.get_end()[1],1)
        nose.tools.assert_equal(a.get_end()[2],3)
        
    @attr(speed='fast')
    def test_iterator(self):
        """Cube class: tests iterators"""
        for (ima,k) in iter_ima(self.cube1,True):
            ima[:,:] = k*np.ones(shape=(6,5))
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(self.cube1.data[k,j,i]*self.cube1.fscale,k)
        for (spe,(p,q)) in iter_spe(self.cube1,True):
            spe[:]= spe + p + q
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(self.cube1.data[k,j,i]*self.cube1.fscale,k+i+j)
                    
    @attr(speed='fast')
    def test_clone(self):
        """Cube class: tests clone method."""
        cube2 = self.cube1.clone()
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,0)
                    
    @attr(speed='fast')
    def test_resize(self):
        """Cube class: tests resize method."""
        self.cube1.data.mask[0,:,:] = True
        self.cube1.resize()
        nose.tools.assert_equal(self.cube1.shape[0],9)
        
