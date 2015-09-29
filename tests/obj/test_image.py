"""Test on Image objects."""
import nose.tools
from nose.plugins.attrib import attr

import astropy.units as u
import numpy as np
from mpdaf.obj import (Spectrum, Image, Cube, WCS, WaveCoord, gauss_image,
                       moffat_image)
from ..utils import assert_image_equal


@attr(speed='fast')
def test_copy():
    """Image class: testing copy method."""
    wcs = WCS(crval=(0,0), shape=(6,5))
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    image2 = image1.copy()
    s = image1.data.sum()
    image1[0,0] = 10000
    nose.tools.assert_true(image1.wcs.isEqual(image2.wcs))
    nose.tools.assert_equal(s,image2.data.sum())

@attr(speed='fast')
def test_arithmetricOperator():
    """Image class: testing arithmetic functions"""
    wcs = WCS(crval=(0,0), shape=(6,5))
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5)
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs, unit=u.ct)
    image2 = Image(shape=(6,5),data=data*0.5,wcs=wcs, unit=u.Unit('2 ct'))
    cube1 = Cube(shape=(10,6,5),data=np.ones(shape=(10,6,5))*0.5,wave=wave,wcs=wcs, unit=u.Unit('2 ct'))
    # +
    image3 = image1 + image2
    nose.tools.assert_almost_equal(image3[3,3],4)
    image1 += 4.2
    nose.tools.assert_almost_equal(image1[3,3],2+4.2)
    # -
    image3 = image1 - image2
    nose.tools.assert_almost_equal(image3[3,3],4.2)
    image1 -= 4.2
    nose.tools.assert_almost_equal(image1[3,3],2)
    # *
    image3 = image1 * image1
    nose.tools.assert_almost_equal(image3[3,3],4)
    image1 *= 4.2
    nose.tools.assert_almost_equal(image1[3,3],2*4.2)
    # /
    image3 = image1 / image1
    nose.tools.assert_almost_equal(image3[3,3],1)
    image1 /= 4.2
    nose.tools.assert_almost_equal(image1[3,3],2)
    # with cube
    cube2 = image2 + cube1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],image2[j,i] + cube1[k,j,i])
    cube2 = image2 - cube1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],image2[j,i] - cube1[k,j,i])
    cube2 = image2 * cube1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],image2[j,i] * cube1[k,j,i])
    cube2 = image2 / cube1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],image2[j,i] / cube1[k,j,i])
    # spectrum * image
    spectrum1 = Spectrum(shape=10, data=np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave)
    cube2 = image1 * spectrum1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],spectrum1[k] * image1[j,i])
    #
    image2 = (image1 *-2).abs()+(image1+4).sqrt()-2
    image3 = image1.clone()
    image3[:]=image2
    nose.tools.assert_almost_equal(image3[3,3],np.abs(image1[3,3] *-2)+np.sqrt(image1[3,3]+4)-2 )

@attr(speed='fast')
def test_get():
    """Image class: testing getters"""
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    ima = image1[0:2,1:4]
    assert_image_equal(ima, shape=(2, 3), start=(0, 1), end=(1, 3), step=(1, 1))

@attr(speed='fast')
def test_resize():
    """Image class: testing resize method"""
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    mask = np.ones((6,5),dtype=bool)
    data = image1.data.data
    data[2:4,1:4] = 8
    mask[2:4,1:4] = 0
    image1.data = np.ma.MaskedArray(data, mask=mask)
    image1.resize()
    assert_image_equal(image1, shape=(2, 3), start=(2, 1), end=(3, 3))
    nose.tools.assert_equal(image1.sum(),2*3*8)
    nose.tools.assert_equal(image1.get_range()[0][0],image1.get_start()[0])
    nose.tools.assert_equal(image1.get_range()[0][1],image1.get_start()[1])
    nose.tools.assert_equal(image1.get_range()[1][0],image1.get_end()[0])
    nose.tools.assert_equal(image1.get_range()[1][1],image1.get_end()[1])
    nose.tools.assert_equal(image1.get_rot(),0)

@attr(speed='fast')
def test_truncate():
    """Image class: testing truncation"""
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5), data=data, wcs=wcs)
    image1 = image1.truncate(0,1,1,3, unit=wcs.get_cunit1())
    assert_image_equal(image1, shape=(2, 3), start=(0, 1), end=(1, 3))

@attr(speed='fast')
def test_sum():
    """Image class: testing sum"""
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    sum1 = image1.sum()
    nose.tools.assert_equal(sum1,6*5*2)
    sum2 = image1.sum(axis=0)
    nose.tools.assert_equal(sum2.shape,5)
    nose.tools.assert_equal(sum2.get_start(),0)
    nose.tools.assert_equal(sum2.get_end(),4)

@attr(speed='fast')
def test_gauss():
    """Image class: testing Gaussian fit"""
    wcs = WCS (cdelt=(0.2,0.3), crval=(8.5,12),shape=(40,30))
    ima = gauss_image(wcs=wcs,fwhm=(2,1),factor=1, rot = 60, cont = 2.0, unit_center=u.pix, unit_fwhm=u.pix)
    #ima2 = gauss_image(wcs=wcs,width=(1,2),factor=2, rot = 60)
    gauss = ima.gauss_fit(cont=2.0, fit_back=False, verbose=False, unit_center=None, unit_fwhm=None)
    nose.tools.assert_almost_equal(gauss.center[0], 19.5)
    nose.tools.assert_almost_equal(gauss.center[1], 14.5)
    nose.tools.assert_almost_equal(gauss.flux, 1)
    ima += 10.3
    gauss2 = ima.gauss_fit(cont=2.0+10.3,fit_back=True, verbose=False, unit_center=None, unit_fwhm=None)
    nose.tools.assert_almost_equal(gauss2.center[0], 19.5)
    nose.tools.assert_almost_equal(gauss2.center[1], 14.5)
    nose.tools.assert_almost_equal(gauss2.flux, 1)
    nose.tools.assert_almost_equal(gauss2.cont, 12.3)

@attr(speed='fast')
def test_moffat():
    """Image class: testing Moffat fit"""
    ima = moffat_image(wcs=WCS(crval=(0,0)),flux=12.3, fwhm=(1.8,1.8), n=1.6, rot = 0., cont=8.24, unit_center=u.pix, unit_fwhm=u.pix)
    moffat = ima.moffat_fit(fit_back=True, verbose=False, unit_center=None, unit_fwhm=None)
    nose.tools.assert_almost_equal(moffat.center[0], 50.)
    nose.tools.assert_almost_equal(moffat.center[1], 50.)
    nose.tools.assert_almost_equal(moffat.flux, 12.3)
    nose.tools.assert_almost_equal(moffat.fwhm[0], 1.8)
    nose.tools.assert_almost_equal(moffat.n, 1.6)
    nose.tools.assert_almost_equal(moffat.cont, 8.24)

@attr(speed='fast')
def test_mask():
    """Image class: testing mask functionalities"""
    wcs = WCS()
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    image1.mask((2,2),(1,1),inside=False,unit_center=None, unit_radius=None)
    nose.tools.assert_equal(image1.sum(),2*9)
    image1.unmask()
    wcs = WCS(deg=True)
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    image1.mask(wcs.pix2sky([2,2]),(3600,3600),inside=False)
    nose.tools.assert_equal(image1.sum(),2*9)
    image1.unmask()
    image1.mask(wcs.pix2sky([2,2]),4000,inside=False)
    nose.tools.assert_equal(image1.sum(),2*5)
    image1.unmask()
    image1.mask_ellipse(wcs.pix2sky([2,2]),(10000,3000),20,inside=False)
    nose.tools.assert_equal(image1.sum(),2*7)
    ksel = np.where(image1.data.mask)
    image1.unmask()
    image1.mask_selection(ksel)
    nose.tools.assert_equal(image1.sum(),2*7)

@attr(speed='fast')
def test_background():
    """Image class: testing background value"""
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
def test_peak():
    """Image class: testing peak research"""
    wcs = WCS()
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    image1.data[2,3] = 8
    p = image1.peak()
    nose.tools.assert_equal(p['p'],2)
    nose.tools.assert_equal(p['q'],3)
    ima = Image("data/obj/a370II.fits")
    p = ima.peak(center=(790,875),radius=20,plot=False, unit_center=None,
         unit_radius=None)
    nose.tools.assert_almost_equal(p['p'],793.1,1)
    nose.tools.assert_almost_equal(p['q'],875.9,1)

@attr(speed='fast')
def test_clone():
    """Image class: testing clone method."""
    wcs = WCS()
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    ima2 = image1.clone()
    for j in range(6):
        for i in range(5):
            nose.tools.assert_almost_equal(ima2[j,i],0)
    ima = Image("data/obj/a370II.fits")
    ima2 = ima.clone()+1000
    nose.tools.assert_equal(ima2.sum(axis=0).data[1000],ima.shape[0]*1000)
    nose.tools.assert_equal(ima2.sum(),ima.shape[0]*ima.shape[1]*1000)

@attr(speed='fast')
def test_rotate():
    """Image class: testing rotation."""
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
def test_inside():
    """Image class: testing inside method."""
    ima = Image("data/obj/a370II.fits")
    nose.tools.assert_equal(ima.inside((39.951088,-1.4977398),unit=ima.wcs.get_cunit1()),False)

@attr(speed='fast')
def test_subimage():
    """Image class: testing sub-image extraction."""
    ima = Image("data/obj/a370II.fits")
    subima = ima.subimage(center=(790,875), size=40, unit_center=None, unit_size=None)
    nose.tools.assert_equal(subima.peak()['data'], 3035.0)

@attr(speed='fast')
def test_ee():
    """Image class: testing ensquared energy."""
    wcs = WCS()
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    image1.mask((2,2),(1,1),inside=False,unit_center=None, unit_radius=None)
    nose.tools.assert_equal(image1.ee(), 9*2)
    ee = image1.ee(center=(2,2), unit_center=None, radius=1, unit_radius=None)
    nose.tools.assert_equal(ee,4*2)
    r,eer = image1.eer_curve(center=(2,2),unit_center=None, unit_radius=None,cont=0)
    nose.tools.assert_equal(r[1], 1.0)
    nose.tools.assert_equal(eer[1], 1.0)
    size = image1.ee_size(center=(2,2),unit_center=None, unit_size=None,cont=0)
    nose.tools.assert_almost_equal(size[0], 1.775)

@attr(speed='fast')
def test_rebin_mean():
    """Image class: testing rebin methods."""
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data,wcs=wcs)
    image1.mask((2,2),(1,1),inside=False,unit_center=None, unit_radius=None)
    image2 = image1.rebin_mean(2)
    nose.tools.assert_equal(image2[0,0], 0.5)
    nose.tools.assert_equal(image2[1,1], 2)
    start = image2.get_start()
    nose.tools.assert_equal(start[0],0.5)
    nose.tools.assert_equal(start[1],0.5)
    image2 = image1.rebin_median(2)
    nose.tools.assert_equal(image2[0,0], 2)
    nose.tools.assert_equal(image2[1,1], 2)

# TODO test_resample: pb rotation

@attr(speed='fast')
def test_add():
    """Image class: testing add method."""
    ima = Image("data/obj/a370II.fits")
    subima = ima.subimage(center=(790,875), size=40, unit_center=None, unit_size=None)
    ima.add(subima*4)
    nose.tools.assert_equal(ima[800,885], subima[30,30]*5)

@attr(speed='fast')
def test_fftconvolve():
    """Image class: testing convolution methods."""
    wcs = WCS (cdelt=(0.2,0.3), crval=(8.5,12),shape=(40,30),deg=True)
    data = np.zeros((40,30))
    data[19,14]=1
    ima = Image(wcs=wcs, data=data)
    ima2 = ima.fftconvolve_gauss(center=None, flux=1., fwhm=(20000., 10000.),
                      peak=False, rot=60., factor=1, unit_center=u.deg,
                      unit_fwhm=u.arcsec)
    g = ima2.gauss_fit(verbose=False)
    nose.tools.assert_almost_equal(g.fwhm[0], 20000, 2)
    nose.tools.assert_almost_equal(g.fwhm[1], 10000, 2)
    nose.tools.assert_almost_equal(g.center[0], 8.5)
    nose.tools.assert_almost_equal(g.center[1], 12)
    ima2 = ima.fftconvolve_moffat(center=None, flux=1., a= 10000, q=1, n=2, peak=False, rot=60., factor=1, unit_center=u.deg, unit_a=u.arcsec)
    m = ima2.moffat_fit(verbose=False)
    nose.tools.assert_almost_equal(m.center[0], 8.5)
    nose.tools.assert_almost_equal(m.center[1], 12)
    ima3 = ima.correlate2d(np.ones((40,30)))
