"""Test on Cube objects."""
import nose.tools
from nose.plugins.attrib import attr

import astropy.units as u
import numpy as np

from mpdaf.obj import Spectrum
from mpdaf.obj import Image
from mpdaf.obj import Cube, iter_spe, iter_ima
from mpdaf.obj import WCS
from mpdaf.obj import WaveCoord

@attr(speed='fast')
def test_copy():
    """Cube class: testing copy method."""
    wcs = WCS(crval=(0,0), crpix = 1.0, shape=(6,5))
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=10)
    cube1 = Cube(shape=(10,6,5),data=2.3*np.ones(shape=(10,6,5)),wave=wave,wcs=wcs)
    cube2 = cube1.copy()
    s = cube1.data.sum()
    cube1[0,0,0] = 1000
    nose.tools.assert_true(cube1.wcs.isEqual(cube2.wcs))
    nose.tools.assert_true(cube1.wave.isEqual(cube2.wave))
    nose.tools.assert_equal(s,cube2.data.sum())

@attr(speed='fast')
def test_arithmetricOperator_Cube():
    """Cube class: tests arithmetic functions"""
    wcs = WCS(crval=(0,0), crpix = 1.0, shape=(6,5))
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=10, cunit=u.nm)
    cube1 = Cube(shape=(10,6,5),data=2.3*np.ones(shape=(10,6,5)),wave=wave,wcs=wcs, unit=u.ct)
    data = np.ones(shape=(6,5))*2
    image1 = Image(shape=(6,5),data=data*0.5,wcs=wcs,unit=2*u.ct)
    wave2 = WaveCoord(crpix=2.0, cdelt=30.0, crval=5, shape=10, cunit=u.angstrom)
    spectrum1 = Spectrum(shape=10, data=2.3*np.array([0.5,1,2,3,4,5,6,7,8,9]),wave=wave2)
    cube2 = image1 + cube1
    # +
    cube3 = cube1 + cube2
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube3[k,j,i],cube1[k,j,i] + cube2[k,j,i])
    # -
    cube3 = cube1 - cube2
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube3[k,j,i],cube1[k,j,i] - cube2[k,j,i])
    # *
    cube3 = cube1 * cube2
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube3[k,j,i],cube1[k,j,i] * cube2[k,j,i])
    # /
    cube3 = cube1 / cube2
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube3[k,j,i],cube1[k,j,i] / cube2[k,j,i])
    # with spectrum
    cube2 = cube1 + spectrum1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],spectrum1[k] + cube1[k,j,i])
    cube2 = cube1 - spectrum1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],-spectrum1[k] + cube1[k,j,i])
    cube2 = cube1 * spectrum1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],spectrum1[k] * cube1[k,j,i])
    cube2 = cube1 / spectrum1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],cube1[k,j,i]/spectrum1[k])
    # with image
    cube2 = cube1 + image1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],image1[j,i]*2 + cube1[k,j,i])
    cube2 = cube1 - image1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],-image1[j,i]*2 + cube1[k,j,i])
    cube2 = cube1 * image1
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],image1[j,i] * cube1[k,j,i])
    cube2 = cube1 / 25.3
    cube3 = cube1.clone()
    cube3[:] = cube2
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube3[k,j,i],cube1[k,j,i] / 25.3)

@attr(speed='fast')
def test_get_Cube():
    """Cube class: tests getters"""
    wcs = WCS(crval=(0,0), crpix = 1.0, shape=(6,5))
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=10)
    cube1 = Cube(shape=(10,6,5),data=2.3*np.ones(shape=(10,6,5)),wave=wave,wcs=wcs)
    a = cube1[2,:,:]
    nose.tools.assert_equal(a.shape[0],6)
    nose.tools.assert_equal(a.shape[1],5)
    a = cube1[:,2,3]
    nose.tools.assert_equal(a.shape,10)
    a = cube1[1:7,0:2,0:3]
    nose.tools.assert_equal(a.shape[0],6)
    nose.tools.assert_equal(a.shape[1],2)
    nose.tools.assert_equal(a.shape[2],3)
    a = cube1.get_lambda(1.2,15.6)
    nose.tools.assert_equal(a.shape[0],6)
    nose.tools.assert_equal(a.shape[1],6)
    nose.tools.assert_equal(a.shape[2],5)
    a = cube1[2:4,0:2,1:4]
    nose.tools.assert_equal(a.get_start()[0],3.5)
    nose.tools.assert_equal(a.get_start()[1],0)
    nose.tools.assert_equal(a.get_start()[2],1)
    nose.tools.assert_equal(a.get_end()[0],6.5)
    nose.tools.assert_equal(a.get_end()[1],1)
    nose.tools.assert_equal(a.get_end()[2],3)

@attr(speed='fast')
def test_iterator():
    """Cube class: tests iterators"""
    wcs = WCS(crval=(0,0), crpix = 1.0, shape=(6,5))
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=10)
    cube1 = Cube(shape=(10,6,5),data=2.3*np.ones(shape=(10,6,5)),wave=wave,wcs=wcs)
    for (ima,k) in iter_ima(cube1,True):
        ima[:,:] = k*np.ones(shape=(6,5))
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube1[k,j,i],k)
    for (spe,(p,q)) in iter_spe(cube1,True):
        spe[:]= spe + p + q
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube1[k,j,i],k+i+j)

@attr(speed='fast')
def test_clone():
    """Cube class: tests clone method."""
    wcs = WCS(crval=(0,0), crpix = 1.0, shape=(6,5))
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=10)
    cube1 = Cube(shape=(10,6,5),data=2.3*np.ones(shape=(10,6,5)),wave=wave,wcs=wcs)
    cube2 = cube1.clone()
    for k in range(10):
        for j in range(6):
            for i in range(5):
                nose.tools.assert_almost_equal(cube2[k,j,i],0)

@attr(speed='fast')
def test_resize():
    """Cube class: tests resize method."""
    wcs = WCS(crval=(0,0), crpix = 1.0, shape=(6,5))
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=10)
    cube1 = Cube(shape=(10,6,5),data=2.3*np.ones(shape=(10,6,5)),wave=wave,wcs=wcs)
    cube1.data.mask[0,:,:] = True
    cube1.resize()
    nose.tools.assert_equal(cube1.shape[0],9)

@attr(speed='fast')
def test_multiprocess():
    """Cube class: tests multiprocess"""
    wcs = WCS(crval=(0,0), crpix = 1.0, shape=(6,5))
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=10)
    cube1 = Cube(shape=(10,6,5),data=2.3*np.ones(shape=(10,6,5)),wave=wave,wcs=wcs)
    f = Image.sum
    list_spe = cube1.loop_ima_multiprocessing(f, cpu=2, verbose=False, axis=0)
    nose.tools.assert_equal(list_spe[8][1], cube1[8,:,:].sum(axis=0)[1])
    f = Image.ee
    ee = cube1.loop_ima_multiprocessing(f, cpu=2, verbose=False)
    nose.tools.assert_equal(ee[1], cube1[1,:,:].ee())
    f = Image.rotate
    cub2 = cube1.loop_ima_multiprocessing(f, cpu=2, verbose=False, theta=20)
    nose.tools.assert_equal(cub2[4,3,2], cube1[4,:,:].rotate(20)[3,2])
    f = Spectrum.mean
    out = cube1.loop_spe_multiprocessing(f, cpu=2, verbose=False)
    nose.tools.assert_equal(out[2,3], cube1[:,2,3].mean())
    f = Spectrum.resample
    out = cube1.loop_spe_multiprocessing(f, cpu=2, verbose=False, step=1)
    nose.tools.assert_equal(out[8,3,2], cube1[:,3,2].resample(step=1)[8])

@attr(speed='fast')
def test_mask():
    """Cube class: testing mask functionalities"""
    wcs = WCS()
    wave = WaveCoord()
    data = np.ones(shape=(10,6,5))*2
    cube1 = Cube(shape=(10,6,5),data=data,wave=wave,wcs=wcs)
    cube1.mask((2,2),(1,1),lmin=2, lmax=5,inside=False,unit_center=None, unit_radius=None, unit_wave=None)
    nose.tools.assert_equal(cube1.sum(),2*9*3)
    cube1.unmask()
    wcs = WCS(deg=True)
    wave = WaveCoord(cunit=u.angstrom)
    cube1 = Cube(shape=(10,6,5),data=data,wave=wave,wcs=wcs)
    cube1.mask(wcs.pix2sky([2,2]),(3600,3600),lmin=2, lmax=5,inside=False)
    nose.tools.assert_equal(cube1.sum(),2*9*3)
    cube1.unmask()
    cube1.mask(wcs.pix2sky([2,2]),4000,lmin=2, lmax=5,inside=False)
    nose.tools.assert_equal(cube1.sum(),2*5*3)
    cube1.unmask()
    cube1.mask_ellipse(wcs.pix2sky([2,2]),(10000,3000),20,lmin=2, lmax=5,inside=False)
    nose.tools.assert_equal(cube1.sum(),2*7*3)
    ksel = np.where(cube1.data.mask)
    cube1.unmask()
    cube1.mask_selection(ksel)
    nose.tools.assert_equal(cube1.sum(),2*7*3)

@attr(speed='fast')
def test_truncate():
    """Cube class: testing truncation"""
    wave = WaveCoord(crval=1)
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(10,6,5))*2
    cube1 = Cube(shape=(10,6,5),data=data,wave=wave,wcs=wcs)
    coord = [[2,0,1], [5,1,3]]
    cube2 = cube1.truncate(coord, unit_wcs=wcs.get_cunit1(), unit_wave=wave.get_cunit())
    nose.tools.assert_equal(cube2.shape[0],4)
    nose.tools.assert_equal(cube2.shape[1],2)
    nose.tools.assert_equal(cube2.shape[2],3)
    nose.tools.assert_equal(cube2.get_start()[0],2)
    nose.tools.assert_equal(cube2.get_start()[1],0)
    nose.tools.assert_equal(cube2.get_start()[2],1)
    nose.tools.assert_equal(cube2.get_end()[0],5)
    nose.tools.assert_equal(cube2.get_end()[1],1)
    nose.tools.assert_equal(cube2.get_end()[2],3)

@attr(speed='fast')
def test_sum():
    """Cube class: testing sum, mean and median methods"""
    wave = WaveCoord(crval=1)
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(10,6,5))
    for i in range(10):
        data[i,:,:] = i*np.ones((6,5))
    cube1 = Cube(shape=(10,6,5),data=data,wave=wave,wcs=wcs)
    sum1 = cube1.sum()
    nose.tools.assert_equal(sum1,6*5*45)
    sum2 = cube1.sum(axis=0)
    nose.tools.assert_equal(sum2.shape[0],6)
    nose.tools.assert_equal(sum2.shape[1],5)

    weights = np.ones(shape=(10,6,5))
    sum1 = cube1.sum(weights=weights)
    nose.tools.assert_equal(sum1,6*5*45)

    weights = np.ones(shape=(10,6,5))*2
    sum1 = cube1.sum(weights=weights)
    nose.tools.assert_equal(sum1,6*5*45)

    m = cube1.mean(axis=(1,2))
    for i in range(10):
        nose.tools.assert_equal(m[i],i)

    m = cube1.median(axis=0)
    nose.tools.assert_equal(m[3,3], np.median(np.arange(10)))

@attr(speed='fast')
def test_rebin():
    """Cube class: testing rebin methods"""
    wave = WaveCoord(crval=1)
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(10,6,5))
    cube1 = Cube(shape=(10,6,5),data=data,wave=wave,wcs=wcs)
    cube2 = cube1.rebin_mean(factor=2)
    nose.tools.assert_equal(cube2[0,0,0],1)
    start = cube2.get_start()
    nose.tools.assert_equal(start[0],1.5)
    nose.tools.assert_equal(start[1],0.5)
    nose.tools.assert_equal(start[2],0.5)
    cube2 = cube1.rebin_mean(factor=2, flux=True, margin='origin')
    nose.tools.assert_equal(cube2[-1,-1,-1],0.5)
    start = cube2.get_start()
    nose.tools.assert_equal(start[0],1.5)
    nose.tools.assert_equal(start[1],0.5)
    nose.tools.assert_equal(start[2],0.5)

@attr(speed='fast')
def test_get_image():
    """Cube class: testing get_image method"""
    wave = WaveCoord(crpix=1, cdelt=0.3, crval=200, cunit=u.nm)
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(2000,6,5))*2
    cube1 = Cube(shape=(2000,6,5),data=data,wave=wave,wcs=wcs)
    cube1[:,2,2].add_gaussian(5000, 1200, 20, unit=u.angstrom)
    ima = cube1.get_image(wave=(4800,5200), is_sum=False, subtract_off=True)
    nose.tools.assert_equal(ima[0,0],0)
    nose.tools.assert_almost_equal(ima[2,2], cube1[934:1067,2,2].mean()-2,3)
    ima = cube1.get_image(wave=(4800,5200), is_sum=False, subtract_off=False)
    nose.tools.assert_equal(ima[0,0],2)
    nose.tools.assert_almost_equal(ima[2,2], cube1[934:1067,2,2].mean(),3)
    ima = cube1.get_image(wave=(4800,5200), is_sum=True, subtract_off=True)
    nose.tools.assert_equal(ima[0,0],0)
    nose.tools.assert_almost_equal(ima[2,2], cube1[934:1067,2,2].sum()-cube1[934:1067,0,0].sum(),3)
    ima = cube1.get_image(wave=(4800,5200), is_sum=True, subtract_off=False)
    nose.tools.assert_equal(ima[0,0],cube1[934:1067,0,0].sum())
    nose.tools.assert_almost_equal(ima[2,2], cube1[934:1067,2,2].sum())

@attr(speed='fast')
def test_subcube():
    """Cube class: testing sub-cube extraction methods"""
    wave = WaveCoord(crval=1)
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(10,6,5))
    cube1 = Cube(shape=(10,6,5),data=data,wave=wave,wcs=wcs)
    cube2 = cube1.subcube(center=(2,2.8), size=2, lbda=(5,8),
                          unit_center=None, unit_size=None)
    start = cube2.get_start()
    nose.tools.assert_equal(start[0],5)
    nose.tools.assert_equal(start[1],1)
    nose.tools.assert_equal(start[2],2)
    shape = cube2.shape
    nose.tools.assert_equal(shape[0],4)
    nose.tools.assert_equal(shape[1],2)
    nose.tools.assert_equal(shape[2],2)
    cube2=cube1.subcube_circle_aperture(center=(2,2.8), radius=1,
                                        unit_center=None, unit_radius=None)
    start = cube2.get_start()
    nose.tools.assert_equal(cube2.data.mask[0,0,0],True)
    nose.tools.assert_equal(start[0],1)
    nose.tools.assert_equal(start[1],1)
    nose.tools.assert_equal(start[2],2)
    shape = cube2.shape
    nose.tools.assert_equal(shape[0],10)
    nose.tools.assert_equal(shape[1],2)
    nose.tools.assert_equal(shape[2],2)

@attr(speed='fast')
def test_aperture():
    """Cube class: testing spectrum extraction"""
    wave = WaveCoord(crval=1)
    wcs = WCS(crval=(0,0))
    data = np.ones(shape=(10,6,5))
    cube1 = Cube(shape=(10,6,5),data=data,wave=wave,wcs=wcs)
    spe = cube1.aperture(center=(2,2.8), radius=1,
                        unit_center=None, unit_radius=None)
    nose.tools.assert_equal(spe.shape, 10)
    nose.tools.assert_equal(spe.get_start(), 1)
