"""Test on Spectrum objects."""

import nose.tools
from nose.plugins.attrib import attr

import os
import sys
import numpy as np
import pyfits

from mpdaf.obj import Spectrum
from mpdaf.obj import Image
from mpdaf.obj import Cube
from mpdaf.obj import WCS
from mpdaf.obj import WaveCoord

class TestSpectrum():

    @attr(speed='fast')
    def test_selectionOperator_Spectrum(self):
        """Spectrum class: tests operators > and < """
        wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit = 'Angstrom')
        spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave,fscale= 2.3)
        spectrum2 = spectrum1 > 13.8
        nose.tools.assert_equal(spectrum2.data.sum()*spectrum2.fscale,24*spectrum1.fscale)
        spectrum2 = spectrum1 >= 6*spectrum1.fscale
        nose.tools.assert_equal(spectrum2.data.sum()*spectrum2.fscale,30*spectrum1.fscale)
        spectrum2 = spectrum1 < 6*spectrum1.fscale
        nose.tools.assert_equal(spectrum2.data.sum()*spectrum2.fscale,15.5*spectrum1.fscale)
        spectrum2 = spectrum1 <= 6*spectrum1.fscale
        nose.tools.assert_equal(spectrum2.data.sum()*spectrum2.fscale,21.5*spectrum1.fscale)
        del spectrum1, spectrum2

    @attr(speed='fast')
    def test_arithmetricOperator_Spectrum(self):
        """Spectrum class: tests arithmetic functions"""
        wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit = 'Angstrom')
        spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave,fscale= 2.3)
        spectrum2 = spectrum1 > 13.8 #[-,-,-,-,-,-,-,7,8,9]
        # +
        spectrum3 = spectrum1 + spectrum2
        nose.tools.assert_equal(spectrum3.data.data[3]*spectrum3.fscale,3*spectrum1.fscale)
        nose.tools.assert_equal(spectrum3.data.data[8]*spectrum3.fscale,16*spectrum1.fscale)
        spectrum3 = 4.2 + spectrum1
        nose.tools.assert_equal(spectrum3.data.data[3]*spectrum3.fscale,3*spectrum1.fscale+4.2)
        # -
        spectrum3 = spectrum1 - spectrum2
        nose.tools.assert_equal(spectrum3.data.data[3]*spectrum3.fscale,3*spectrum1.fscale)
        nose.tools.assert_equal(spectrum3.data.data[8]*spectrum3.fscale,0*spectrum1.fscale)
        spectrum3 = spectrum1 - 4.2
        nose.tools.assert_equal(spectrum3.data.data[8]*spectrum3.fscale,8*spectrum1.fscale - 4.2)
        # *
        spectrum3 = spectrum1 * spectrum2
        nose.tools.assert_equal(spectrum3.data.data[8]*spectrum3.fscale,64*spectrum1.fscale*spectrum1.fscale)
        spectrum3 = 4.2 * spectrum1
        nose.tools.assert_equal(spectrum3.data.data[9]*spectrum3.fscale,9*4.2*spectrum1.fscale)
        # /
        spectrum3 = spectrum1 / spectrum2
        #divide functions that have a validity domain returns the masked constant whenever the input is masked or falls outside the validity domain.
        nose.tools.assert_equal(spectrum3.data.data[8]*spectrum3.fscale,1)
        spectrum3 = 1.0 / (4.2 /spectrum1 )
        nose.tools.assert_equal(spectrum3.data.data[5]*spectrum3.fscale,5/4.2*spectrum1.fscale)
        del spectrum2, spectrum3
        # with cube
        wcs = WCS()
        cube1 = Cube(shape=(10,6,5),data=np.ones(shape=(10,6,5)),wave=wave,wcs=wcs,fscale= 2.3)
        cube2 = spectrum1 + cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,spectrum1.data[k]*spectrum1.fscale + cube1.data[k,j,i]*cube1.fscale)
        cube2 = spectrum1 - cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,spectrum1.data[k]*spectrum1.fscale - cube1.data[k,j,i]*cube1.fscale)
        cube2 = spectrum1 * cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,spectrum1.data[k]*spectrum1.fscale * cube1.data[k,j,i]*cube1.fscale)
        cube2 = spectrum1 / cube1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,spectrum1.data[k]*spectrum1.fscale / (cube1.data[k,j,i]*cube1.fscale))
        del cube1
        # spectrum * image
        data = np.ones(shape=(6,5))*2
        image1 = Image(shape=(6,5),data=data,wcs=wcs)
        cube2 = spectrum1 * image1
        for k in range(10):
            for j in range(6):
                for i in range(5):
                    nose.tools.assert_almost_equal(cube2.data[k,j,i]*cube2.fscale,spectrum1.data[k]*spectrum1.fscale * (image1.data[j,i]*image1.fscale))
        del image1, cube2, spectrum1

    @attr(speed='fast')
    def test_get_Spectrum(self):
        """Spectrum class: tests getters"""
        wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit = 'Angstrom')
        spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave,fscale= 2.3)
        a = spectrum1[1:7]
        nose.tools.assert_equal(a.shape,6)
        a = spectrum1.get_lambda(1.2,15.6)
        nose.tools.assert_equal(a.shape,6)
        del spectrum1
        spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
        spvarcut=spvar.get_lambda(5560,5590)
        nose.tools.assert_equal(spvarcut.shape,48)
        nose.tools.assert_almost_equal(spvarcut.get_start(),5560.25,2)
        nose.tools.assert_almost_equal(spvarcut.get_end(),5589.89,2)
        nose.tools.assert_almost_equal(spvarcut.get_step(),0.63,2)
        del spvar,spvarcut
        
        
    @attr(speed='fast') 
    def test_spectrum_methods(self):
        """Spectrum class: tests sum/mean/abs/sqrt methods"""
        wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit = 'Angstrom')
        spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave,fscale= 2.3)
        sum1 = spectrum1.sum()
        nose.tools.assert_almost_equal(sum1,spectrum1.data.sum()*spectrum1.fscale)
        spectrum2 = spectrum1[1:-2]
        sum1 =  spectrum1.sum(lmin=spectrum1.wave[1],lmax=spectrum1.wave[-3])
        sum2 = spectrum2.sum()
        nose.tools.assert_almost_equal(sum1,sum2)
        mean1 =  spectrum1.mean(lmin=spectrum1.wave[1],lmax=spectrum1.wave[-3])
        mean2 = spectrum2.mean()
        nose.tools.assert_almost_equal(mean1,mean2)
        del spectrum1, spectrum2
        spnovar=Spectrum('data/obj/Spectrum_Novariance.fits')
        spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
        spvar2=spvar.copy()
        spvar2.abs()
        nose.tools.assert_equal(spvar2[23],np.abs(spvar[23]))
        spvar2.sqrt()
        nose.tools.assert_equal(spvar2[8],np.sqrt(np.abs(spvar[8])))
        nose.tools.assert_almost_equal(spvar.mean(),11.526547845374727)
        nose.tools.assert_almost_equal(spnovar.mean(),11.101086376675089)
        spvarsum=spvar2+4*spvar2-56/spvar2
        nose.tools.assert_almost_equal(spvarsum.mean(),-165.50331027784796)
        nose.tools.assert_almost_equal(spvarsum[10],-71.589502348454999)
        nose.tools.assert_almost_equal(spvar.get_step(),0.630448220641262)
        nose.tools.assert_almost_equal(spvar.get_start(),4602.6040286827802)
        nose.tools.assert_almost_equal(spvar.get_end(),7184.289492208748)
        nose.tools.assert_almost_equal(spvar.get_range()[0],4602.60402868)
        nose.tools.assert_almost_equal(spvar.get_range()[1],7184.28949221)
        del spnovar,spvar,spvar2
    
    @attr(speed='fast')   
    def test_gauss_fit(self):
        """Spectrum class: tests Gaussian fit"""
        wave = WaveCoord(crpix=1, cdelt=3.0, crval=4000, cunit = 'Angstrom')
        data = np.zeros(6000)
        spem = Spectrum(shape=6000, data=data,wave=wave,fscale= 2.3)
        spem.add_gaussian(5000, 1200, 20)
        gauss = spem.gauss_fit(lmin=(4500,4800),lmax=(5200,6000), lpeak = 5000)
        nose.tools.assert_almost_equal(gauss.lpeak,5000,2)
        nose.tools.assert_almost_equal(gauss.flux,1200,2)
        nose.tools.assert_almost_equal(gauss.fwhm,20,2)
        del spem
    
    @attr(speed='fast')
    def test_resize(self):
        """Spectrum class: tests resize method"""
        f= pyfits.open("data/obj/g9-124Tsigspec.fits")
        sig = f[0].data
        f.close()
        spe = Spectrum("data/obj/g9-124Tspec.fits",var=sig*sig)
        spe.mask(lmax=5000)
        spe.mask(lmin=6500)
        spe.resize()
        nose.tools.assert_equal(int((6500-5000)/spe.get_step()),spe.shape)
        del spe
    
    @attr(speed='fast')    
    def test_rebin(self):
        """Spectrum class: tests rebin function"""
        wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit = 'Angstrom')
        spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave,fscale= 2.3)
        flux1 = spectrum1.sum()*spectrum1.wave.cdelt
        spectrum1.rebin(0.3) 
        flux2 = spectrum1.sum()*spectrum1.wave.cdelt
        nose.tools.assert_almost_equal(flux1,flux2)
        del spectrum1
        
    @attr(speed='slow')
    def test_rebin_slow(self):
        """Spectrum class: heavy test of rebin function"""
        f= pyfits.open("data/obj/g9-124Tsigspec.fits")
        sig = f[0].data
        f.close()
        spe = Spectrum("data/obj/g9-124Tspec.fits",var=sig*sig)
        flux1 = spe.sum(weight=False)*spe.wave.cdelt
        spe.rebin(0.3) 
        flux2 = spe.sum(weight=False)*spe.wave.cdelt
        nose.tools.assert_almost_equal(flux1,flux2,1)
        del spe
        
        spnovar = Spectrum('data/obj/Spectrum_Novariance.fits')
        flux1 = spnovar.sum()*spnovar.wave.cdelt
        spnovar.rebin(4)
        flux2 = spnovar.sum()*spnovar.wave.cdelt
        nose.tools.assert_almost_equal(flux1,flux2,0)        
        spvar = Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
        flux1 = spvar.sum(weight=False)*spvar.wave.cdelt
        spvar.rebin(4)
        flux2 = spvar.sum(weight=False)*spvar.wave.cdelt
        nose.tools.assert_almost_equal(flux1,flux2,0)     
        del spnovar,spvar
        
    
    @attr(speed='fast')    
    def test_rebin_factor(self):
        """Spectrum class: tests rebin_factor function"""
        wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit = 'Angstrom')
        spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave,fscale= 2.3)
        flux1 = spectrum1.sum()*spectrum1.wave.cdelt
        spectrum1.rebin_factor(3) 
        flux2 = spectrum1.sum()*spectrum1.wave.cdelt
        nose.tools.assert_almost_equal(flux1,flux2)
        del spectrum1
        
        f= pyfits.open("data/obj/g9-124Tsigspec.fits")
        sig = f[0].data
        f.close()
        spe = Spectrum("data/obj/g9-124Tspec.fits",var=sig*sig)
        flux1 = spe.sum()*spe.wave.cdelt
        spe.rebin_factor(3) 
        flux2 = spe.sum()*spe.wave.cdelt
        nose.tools.assert_almost_equal(flux1,flux2)
        del spe
        
        spnovar = Spectrum('data/obj/Spectrum_Novariance.fits')
        flux1 = spnovar.sum()*spnovar.wave.cdelt
        spnovar.rebin_factor(4)
        flux2 = spnovar.sum()*spnovar.wave.cdelt
        nose.tools.assert_almost_equal(flux1,flux2)        
        spvar = Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
        flux1 = spvar.sum(weight=False)*spvar.wave.cdelt
        #flux1 = spvar.sum()*spvar.wave.cdelt
        spvar.rebin_factor(4)
        flux2 = spvar.sum(weight=False)*spvar.wave.cdelt
        #flux2 = spvar.sum()*spvar.wave.cdelt
        nose.tools.assert_almost_equal(flux1,flux2)     
        del spnovar,spvar
    
    @attr(speed='fast')    
    def test_truncate(self):
        """Spectrum class: tests truncate function"""
        f= pyfits.open("data/obj/g9-124Tsigspec.fits")
        sig = f[0].data
        f.close()
        spe = Spectrum("data/obj/g9-124Tspec.fits",var=sig*sig)
        spe.truncate(4950,5050)
        nose.tools.assert_equal(spe.shape,160)
        del spe
       
    @attr(speed='fast')    
    def test_exception(self):
        """Spectrum class: tests exceptions"""
        spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
        spvarcut=spvar.get_lambda(5560,5590)
        nose.tools.assert_raises(TypeError, spvar+spvarcut, "Operation forbidden for spectra with different sizes")
        del spvar,spvarcut
        
    @attr(speed='fast')    
    def test_interpolation(self):
        """Spectrum class: tests interpolations"""
        spnovar=Spectrum('data/obj/Spectrum_Novariance.fits')
        spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
        spvar.mask(5575,5585)
        spvar.mask(6296,6312)
        spvar.mask(6351,6375)
        spnovar.mask(5575,5585)
        spnovar.mask(6296,6312)
        spnovar.mask(6351,6375)
        nose.tools.assert_almost_equal(spvar.mean(),11.505614537901696)
        spm1=spvar.copy()
        spm1.interp_mask()
        spm2=spvar.copy()
        spm2.interp_mask(spline=True)
        spvarcut1=spvar.get_lambda(5550,5590)
        spvarcut2=spnovar.get_lambda(5550,5590)
        spvarcut3=spm1.get_lambda(5550,5590)
        spvarcut4=spm2.get_lambda(5550,5590)
        nose.tools.assert_almost_equal(spvar.mean(5550,5590),spvarcut1.mean())
        nose.tools.assert_almost_equal(spnovar.mean(5550,5590),spvarcut2.mean())
        nose.tools.assert_almost_equal(spm1.mean(5550,5590),spvarcut3.mean())
        nose.tools.assert_almost_equal(spm2.mean(5550,5590),spvarcut4.mean())
        del spvar,spnovar,spm1,spm2,spvarcut1,spvarcut2,spvarcut3,spvarcut4
     
    @attr(speed='fast')   
    def test_poly_fit(self):
        """Spectrum class: tests polynomial fit"""   
        spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
        polyfit1=spvar.poly_fit(35)
        spfit1=spvar.copy()
        spfit1.poly_val(polyfit1)
        spfit2=spvar.copy()
        spfit2.poly_spec(10)
        spfit3=spvar.copy()
        spfit3.poly_spec(10,weight=False)
        nose.tools.assert_almost_equal(spfit1.mean(),11.5,1)
        nose.tools.assert_almost_equal(spfit2.mean(),11.5,1)
        nose.tools.assert_almost_equal(spfit3.mean(),11.5,1)
        del spvar,spfit1,spfit2,spfit3
      
    @attr(speed='fast')   
    def test_filter(self): 
        """Spectrum class: tests filters"""   
        spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
        nose.tools.assert_almost_equal(spvar.abmag_band(5000.0,1000.0),-22.837,2)
        nose.tools.assert_almost_equal(spvar.abmag_filter([4000,5000,6000],[0.1,1.0,0.3]),-23.077,2)
        nose.tools.assert_almost_equal(spvar.abmag_filter_name('U'),99)
        nose.tools.assert_almost_equal(spvar.abmag_filter_name('B'),-22.278,2)
        del spvar

#    @attr(speed='fast')   
#    def test_fft_convolve(self): 
#        """Spectrum class: tests convolution"""   
#        spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
        

if __name__=='__main__':
    nosetests -v
    #nosetests -v -a speed='fast'
