"""Test on Spectrum objects."""

import nose.tools
from nose.plugins.attrib import attr

import numpy as np

from astropy import units as u
from astropy.io import fits as pyfits
from mpdaf.obj import Spectrum, Image, Cube, WCS, WaveCoord
from numpy.testing import assert_array_almost_equal


@attr(speed='fast')
def test_copy():
    """Spectrum class: testing copy method."""
    spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
    spe = spvar.copy()
    nose.tools.assert_true(spvar.wave.isEqual(spe.wave))
    nose.tools.assert_equal(spvar.data.sum(),spe.data.sum())
    nose.tools.assert_equal(spvar.var.sum(), spe.var.sum())

@attr(speed='fast')
def test_selectionOperator_Spectrum():
    """Spectrum class: testing operators > and < """
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave)
    spectrum2 = spectrum1 > 6
    nose.tools.assert_almost_equal(spectrum2.sum(),24)
    spectrum2 = spectrum1 >= 6
    nose.tools.assert_almost_equal(spectrum2.sum(),30)
    spectrum2 = spectrum1 < 6
    nose.tools.assert_almost_equal(spectrum2.sum(),15.5)
    spectrum2 = spectrum1 <= 6
    spectrum1[:] = spectrum2
    nose.tools.assert_almost_equal(spectrum1.sum(),21.5)
    del spectrum1, spectrum2

@attr(speed='fast')
def test_arithmetricOperator_Spectrum():
    """Spectrum class: testing arithmetic functions"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave)
    spectrum2 = spectrum1 > 6 #[-,-,-,-,-,-,-,7,8,9]
    # +
    spectrum3 = spectrum1 + spectrum2
    nose.tools.assert_equal(spectrum3.data.data[3],3)
    nose.tools.assert_equal(spectrum3.data.data[8],16)
    spectrum3 = 4.2 + spectrum1
    nose.tools.assert_equal(spectrum3.data.data[3],3+4.2)
    # -
    spectrum3 = spectrum1 - spectrum2
    nose.tools.assert_equal(spectrum3.data.data[3],3)
    nose.tools.assert_equal(spectrum3.data.data[8],0)
    spectrum3 = spectrum1 - 4.2
    nose.tools.assert_equal(spectrum3.data.data[8],8 - 4.2)
    # *
    spectrum3 = spectrum1 * spectrum2
    nose.tools.assert_equal(spectrum3.data.data[8],64)
    spectrum3 = 4.2 * spectrum1
    nose.tools.assert_equal(spectrum3.data.data[9],9*4.2)
    # /
    spectrum3 = spectrum1 / spectrum2
    #divide functions that have a validity domain returns the masked constant whenever the input is masked or falls outside the validity domain.
    nose.tools.assert_equal(spectrum3.data.data[8],1)
    spectrum3 = 1.0 / (4.2 /spectrum1 )
    nose.tools.assert_equal(spectrum3.data.data[5],5/4.2)
    del spectrum2, spectrum3
    # with cube
    wcs = WCS()
    cube1 = Cube(shape=(10,6,5),data=np.ones(shape=(10,6,5)),wave=wave,wcs=wcs)
    cube2 = spectrum1 + cube1
    assert_array_almost_equal(
        cube2.data, spectrum1.data[:, np.newaxis, np.newaxis] + cube1.data)

    cube2 = spectrum1 - cube1
    assert_array_almost_equal(
        cube2.data, spectrum1.data[:, np.newaxis, np.newaxis] - cube1.data)

    cube2 = spectrum1 * cube1
    assert_array_almost_equal(
        cube2.data, spectrum1.data[:, np.newaxis, np.newaxis] * cube1.data)

    cube2 = spectrum1 / cube1
    assert_array_almost_equal(
        cube2.data, spectrum1.data[:, np.newaxis, np.newaxis] / cube1.data)

    del cube1
    # spectrum * image
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    cube2 = spectrum1 * image1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2.data[k,j,i],spectrum1.data[k] * (image1.data[j,i]))
    del image1, cube2, spectrum1

@attr(speed='fast')
def test_get_Spectrum():
    """Spectrum class: testing getters"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9])*2.3,wave=wave)
    a = spectrum1[1:7]
    nose.tools.assert_equal(a.shape,6)
    a = spectrum1.get_lambda(1.2, 15.6, unit=u.nm)
    nose.tools.assert_equal(a.shape,6)
    del spectrum1
    spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
    unit = spvar.wave.get_cunit()
    spvarcut=spvar.get_lambda(5560,5590,unit=unit)
    nose.tools.assert_equal(spvarcut.shape,48)
    nose.tools.assert_almost_equal(spvarcut.get_start(unit=unit),5560.25,2)
    nose.tools.assert_almost_equal(spvarcut.get_end(unit=unit),5589.89,2)
    nose.tools.assert_almost_equal(spvarcut.get_step(unit=unit),0.63,2)
    del spvar,spvarcut


@attr(speed='fast')
def test_spectrum_methods():
    """Spectrum class: testing sum/mean/abs/sqrt methods"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave)
    sum1 = spectrum1.sum()
    nose.tools.assert_almost_equal(sum1,spectrum1.data.sum())
    spectrum2 = spectrum1[1:-2]
    sum1 =  spectrum1.sum(lmin=spectrum1.wave[1],lmax=spectrum1.wave[-3], unit=u.nm)
    sum2 = spectrum2.sum()
    nose.tools.assert_almost_equal(sum1,sum2)
    mean1 =  spectrum1.mean(lmin=spectrum1.wave[1],lmax=spectrum1.wave[-3], unit=u.nm)
    mean2 = spectrum2.mean()
    nose.tools.assert_almost_equal(mean1,mean2)
    del spectrum1, spectrum2
    spnovar=Spectrum('data/obj/Spectrum_Novariance.fits')
    spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
    spvar2=spvar.abs()
    nose.tools.assert_equal(spvar2[23],np.abs(spvar[23]))
    spvar2=spvar.abs().sqrt()
    nose.tools.assert_equal(spvar2[8],np.sqrt(np.abs(spvar[8])))
    nose.tools.assert_almost_equal(spvar.mean(),11.526547845374727)
    nose.tools.assert_almost_equal(spnovar.mean(),11.101086376675089)
    spvarsum=spvar2+4*spvar2-56/spvar2

    nose.tools.assert_almost_equal(spvarsum[10], spvar2[10]+4*spvar2[10]-56/spvar2[10])
    nose.tools.assert_almost_equal(spvar.get_step(),0.630448220641262)
    nose.tools.assert_almost_equal(spvar.get_start(),4602.6040286827802)
    nose.tools.assert_almost_equal(spvar.get_end(),7184.289492208748)
    nose.tools.assert_almost_equal(spvar.get_range()[0],4602.60402868)
    nose.tools.assert_almost_equal(spvar.get_range()[1],7184.28949221)
    del spnovar,spvar,spvar2

@attr(speed='fast')
def test_gauss_fit():
    """Spectrum class: testing Gaussian fit"""
    wave = WaveCoord(crpix=1, cdelt=0.3, crval=400, cunit=u.nm)
    data = np.zeros(600)
    spem = Spectrum(shape=600, data=data*2.3,wave=wave)
    spem.add_gaussian(5000, 1200, 20, unit=u.angstrom)
    gauss = spem.gauss_fit(lmin=(4500,4800),lmax=(5200,6000), lpeak = 5000, unit=u.angstrom)
    nose.tools.assert_almost_equal(gauss.lpeak,5000,2)
    nose.tools.assert_almost_equal(gauss.flux,1200,2)
    nose.tools.assert_almost_equal(gauss.fwhm,20,2)
    nose.tools.assert_almost_equal(spem.fwhm(gauss.lpeak), 20, 0)
    gauss = spem.line_gauss_fit(lmin=(4500,4800),lmax=(5200,6000), lpeak = 5000, unit=u.angstrom)
    nose.tools.assert_almost_equal(gauss.flux,1200,2)
    nose.tools.assert_almost_equal(gauss.fwhm,20,2)
    nose.tools.assert_almost_equal(spem.fwhm(gauss.lpeak), 20, 0)
    del spem

@attr(speed='fast')
def test_resize():
    """Spectrum class: testing resize method"""
    f= pyfits.open("data/obj/g9-124Tsigspec.fits")
    sig = f[0].data
    f.close()
    spe = Spectrum("data/obj/g9-124Tspec.fits",var=sig*sig)
    unit = spe.wave.get_cunit()
    spe.mask(lmax=5000, unit=unit)
    spe.mask(lmin=6500, unit=unit)
    spe.resize()
    nose.tools.assert_equal(int((6500-5000)/spe.get_step(unit=unit)),spe.shape)
    del spe

@attr(speed='fast')
def test_resample():
    """Spectrum class: testing resampling function"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9])*2.3,wave=wave)
    flux1 = spectrum1.sum()*spectrum1.wave.get_step()
    spectrum2 = spectrum1.resample(0.3)
    flux2 = spectrum2.sum()*spectrum2.wave.get_step()
    nose.tools.assert_almost_equal(flux1,flux2)
    del spectrum1,spectrum2

@attr(speed='slow')
def test_resampling_slow():
    """Spectrum class: heavy test of resampling function"""
    f= pyfits.open("data/obj/g9-124Tsigspec.fits")
    sig = f[0].data
    f.close()
    spe = Spectrum("data/obj/g9-124Tspec.fits",var=sig*sig)
    unit = spe.wave.get_cunit()
    flux1 = spe.sum(weight=False)*spe.wave.get_step(unit=unit)
    spe2 = spe.resample(0.3, unit=unit)
    flux2 = spe2.sum(weight=False)*spe2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1,flux2,1)
    del spe,spe2

    spnovar = Spectrum('data/obj/Spectrum_Novariance.fits')
    unit = spnovar.wave.get_cunit()
    flux1 = spnovar.sum()*spnovar.wave.get_step(unit=unit)
    spnovar2 = spnovar.resample(4, unit=unit)
    flux2 = spnovar2.sum()*spnovar2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1,flux2,0)
    spvar = Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
    unit = spvar.wave.get_cunit()
    flux1 = spvar.sum(weight=False)*spvar.wave.get_step(unit=unit)
    spvar2 = spvar.resample(4, unit=unit)
    flux2 = spvar2.sum(weight=False)*spvar2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1,flux2,0)
    del spnovar,spvar,spnovar2,spvar2


@attr(speed='fast')
def test_rebin_mean():
    """Spectrum class: testing rebin_mean function"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5)
    spectrum1 = Spectrum(shape=10,
                         data=np.array([0.5,1,2,3,4,5,6,7,8,9])*2.3,
                         wave=wave)
    unit = spectrum1.wave.get_cunit()
    flux1 = spectrum1.sum()*spectrum1.wave.get_step(unit=unit)
    spectrum2 = spectrum1.rebin_mean(3)
    flux2 = spectrum2.sum()*spectrum2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1,flux2)
    del spectrum1,spectrum2

    f= pyfits.open("data/obj/g9-124Tsigspec.fits")
    sig = f[0].data
    f.close()
    spe = Spectrum("data/obj/g9-124Tspec.fits",var=sig*sig)
    unit = spe.wave.get_cunit()
    flux1 = spe.sum()*spe.wave.get_step(unit=unit)
    spe2 = spe.rebin_mean(3)
    flux2 = spe2.sum()*spe2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1,flux2)
    del spe, spe2

    spnovar = Spectrum('data/obj/Spectrum_Novariance.fits')
    unit = spnovar.wave.get_cunit()
    flux1 = spnovar.sum()*spnovar.wave.get_step(unit=unit)
    spnovar2 = spnovar.rebin_mean(4)
    flux2 = spnovar2.sum()*spnovar2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1,flux2)
    spvar = Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
    unit = spvar.wave.get_cunit()
    flux1 = spvar.sum(weight=False)*spvar.wave.get_step(unit=unit)
    #flux1 = spvar.sum()*spvar.wave.cdelt
    spvar2 = spvar.rebin_mean(4)
    flux2 = spvar2.sum(weight=False)*spvar2.wave.get_step(unit=unit)
    #flux2 = spvar.sum()*spvar.wave.cdelt
    nose.tools.assert_almost_equal(flux1,flux2)
    del spnovar,spvar,spnovar2,spvar2

@attr(speed='fast')
def test_rebin_median():
    """Spectrum class: testing rebin_median function"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(shape=10,
                         data=np.array([0.5,1,2,3,4,5,6,7,8,9]),
                         wave=wave)
    spectrum2 = spectrum1.rebin_median(3, margin='right')
    nose.tools.assert_equal(spectrum2[1],4)

@attr(speed='fast')
def test_truncate():
    """Spectrum class: testing truncate function"""
    f= pyfits.open("data/obj/g9-124Tsigspec.fits")
    sig = f[0].data
    f.close()
    spe = Spectrum("data/obj/g9-124Tspec.fits",var=sig*sig)
    unit = spe.wave.get_cunit()
    spe.truncate(4950,5050, unit=unit)
    nose.tools.assert_equal(spe.shape,160)
    del spe

@attr(speed='fast')
def test_interpolation():
    """Spectrum class: testing interpolations"""
    spnovar=Spectrum('data/obj/Spectrum_Novariance.fits')
    uspnovar = spnovar.wave.get_cunit()
    spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
    uspvar = spvar.wave.get_cunit()
    spvar.mask(5575,5585, unit=uspvar)
    spvar.mask(6296,6312, unit=uspvar)
    spvar.mask(6351,6375, unit=uspvar)
    spnovar.mask(5575,5585, unit=uspnovar)
    spnovar.mask(6296,6312, unit=uspnovar)
    spnovar.mask(6351,6375, unit=uspnovar)
    spm1=spvar.copy()
    spm1.interp_mask()
    spm2=spvar.copy()
    spm2.interp_mask(spline=True)
    spvarcut1=spvar.get_lambda(5550,5590, unit=uspvar)
    spvarcut2=spnovar.get_lambda(5550,5590, unit=uspnovar)
    spvarcut3=spm1.get_lambda(5550,5590, unit=uspvar)
    spvarcut4=spm2.get_lambda(5550,5590, unit=uspvar)
    nose.tools.assert_almost_equal(spvar.mean(5550,5590, unit=uspvar),spvarcut1.mean())
    nose.tools.assert_almost_equal(spnovar.mean(5550,5590, unit=uspnovar),spvarcut2.mean())
    nose.tools.assert_almost_equal(spm1.mean(5550,5590, unit=uspvar),spvarcut3.mean())
    nose.tools.assert_almost_equal(spm2.mean(5550,5590, unit=uspvar),spvarcut4.mean())
    del spvar,spnovar,spm1,spm2,spvarcut1,spvarcut2,spvarcut3,spvarcut4

@attr(speed='fast')
def test_poly_fit():
    """Spectrum class: testing polynomial fit"""
    spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
    polyfit1=spvar.poly_fit(12)
    spfit1=spvar.copy()
    spfit1.poly_val(polyfit1)
    spfit2=spvar.poly_spec(10)
    spfit3=spvar.poly_spec(10,weight=False)
    nose.tools.assert_almost_equal(spfit1.mean(),11.1,1)
    nose.tools.assert_almost_equal(spfit2.mean(),11.1,1)
    nose.tools.assert_almost_equal(spfit3.mean(),11.1,1)
    del spvar,spfit1,spfit2,spfit3

@attr(speed='fast')
def test_filter():
    """Spectrum class: testing filters"""
    spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
    spvar.wave.wcs.wcs.cunit[0] = u.angstrom
    nose.tools.assert_almost_equal(spvar.abmag_band(5000.0,1000.0),-22.837,2)
    nose.tools.assert_almost_equal(spvar.abmag_filter([4000,5000,6000],[0.1,1.0,0.3]),-23.077,2)
    nose.tools.assert_almost_equal(spvar.abmag_filter_name('U'),99)
    nose.tools.assert_almost_equal(spvar.abmag_filter_name('B'),-22.278,2)
    del spvar

@attr(speed='fast')
def test_clone():
    """Spectrum class: testing clone method."""
    spvar=Spectrum('data/obj/Spectrum_Variance.fits',ext=[0,1])
    spe = spvar.clone()
    nose.tools.assert_almost_equal(spe.mean(),0)

@attr(speed='fast')
def test_mag():
    """Spectrum class: testing magnitude computations."""
    Vega=Spectrum('data/obj/Vega.fits')
    Vega.unit=u.Unit('2E-17 erg / (A cm2 s)')
    Vega.wave.wcs.wcs.cunit[0] = u.angstrom
    nose.tools.assert_almost_equal(Vega.abmag_filter_name('V'),0,1)
    mag = Vega.abmag_filter_name('Ic')
    nose.tools.assert_true(mag>0.4 and mag<0.5)
    mag = Vega.abmag_band(22500,2500)
    nose.tools.assert_true(mag>1.9 and mag<2.0)

@attr(speed='fast')
def test_integrate():
    """Spectrum class: testing integration"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(shape=10,
                         data=np.array([0.5,1,2,3,4,5,6,7,8,9]),
                         wave=wave,
                         unit=u.Unit('ct/Angstrom'))
    flux = spectrum1.integrate().value
    nose.tools.assert_almost_equal(flux, spectrum1.get_step(unit=u.angstrom)*spectrum1.sum())
    flux = spectrum1.integrate(unit=u.nm).value
    nose.tools.assert_almost_equal(flux, spectrum1.get_step(unit=u.angstrom)*spectrum1.sum())
    flux = spectrum1.integrate(lmin=3.5, lmax=6.5, unit=u.nm).value
    nose.tools.assert_almost_equal(flux, 75)
    #flux = spectrum1.integrate(lmin=2, lmax=4.2, unit=u.nm)
