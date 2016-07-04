"""Test on Spectrum objects."""

from __future__ import absolute_import, division

import nose.tools
from nose.plugins.attrib import attr

import numpy as np

from astropy import units as u
from astropy.io import fits
from mpdaf.obj import Spectrum, Image, Cube, WCS, WaveCoord
from numpy.testing import assert_array_almost_equal, assert_array_equal
from tempfile import NamedTemporaryFile

from ..utils import generate_spectrum


@attr(speed='fast')
def test_copy():
    """Spectrum class: testing copy method."""
    spvar = Spectrum('data/obj/Spectrum_Variance.fits', ext=[0, 1])
    spe = spvar.copy()
    nose.tools.assert_true(spvar.wave.isEqual(spe.wave))
    nose.tools.assert_equal(spvar.data.sum(), spe.data.sum())
    nose.tools.assert_equal(spvar.var.sum(), spe.var.sum())


@attr(speed='fast')
def test_selection():
    """Spectrum class: testing operators > and < """
    spectrum1 = generate_spectrum(uwave=u.nm)
    spectrum2 = spectrum1 > 6
    nose.tools.assert_almost_equal(spectrum2.sum(), 24)
    spectrum2 = spectrum1 >= 6
    nose.tools.assert_almost_equal(spectrum2.sum(), 30)
    spectrum2 = spectrum1 < 6
    nose.tools.assert_almost_equal(spectrum2.sum(), 15.5)
    spectrum2 = spectrum1 <= 6
    spectrum1[:] = spectrum2
    nose.tools.assert_almost_equal(spectrum1.sum(), 21.5)


@attr(speed='fast')
def test_arithmetric():
    """Spectrum class: testing arithmetic functions"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                         wave=wave)
    spectrum2 = spectrum1 > 6  # [-,-,-,-,-,-,-,7,8,9]
    # +
    spectrum3 = spectrum1 + spectrum2
    nose.tools.assert_equal(spectrum3.data.data[3], 3)
    nose.tools.assert_equal(spectrum3.data.data[8], 16)
    spectrum3 = 4.2 + spectrum1
    nose.tools.assert_equal(spectrum3.data.data[3], 3 + 4.2)
    # -
    spectrum3 = spectrum1 - spectrum2
    nose.tools.assert_equal(spectrum3.data.data[3], 3)
    nose.tools.assert_equal(spectrum3.data.data[8], 0)
    spectrum3 = spectrum1 - 4.2
    nose.tools.assert_equal(spectrum3.data.data[8], 8 - 4.2)
    # *
    spectrum3 = spectrum1 * spectrum2
    nose.tools.assert_equal(spectrum3.data.data[8], 64)
    spectrum3 = 4.2 * spectrum1
    nose.tools.assert_equal(spectrum3.data.data[9], 9 * 4.2)
    # /
    spectrum3 = spectrum1 / spectrum2
    # divide functions that have a validity domain returns the masked constant
    # whenever the input is masked or falls outside the validity domain.
    nose.tools.assert_equal(spectrum3.data.data[8], 1)
    spectrum3 = 1.0 / (4.2 / spectrum1)
    nose.tools.assert_equal(spectrum3.data.data[5], 5 / 4.2)

    # with cube
    wcs = WCS()
    cube1 = Cube(data=np.ones(shape=(10, 6, 5)), wave=wave, wcs=wcs)
    cube2 = spectrum1 + cube1
    sp1data = spectrum1.data[:, np.newaxis, np.newaxis]
    assert_array_almost_equal(cube2.data, sp1data + cube1.data)

    cube2 = spectrum1 - cube1
    assert_array_almost_equal(cube2.data, sp1data - cube1.data)

    cube2 = spectrum1 * cube1
    assert_array_almost_equal(cube2.data, sp1data * cube1.data)

    cube2 = spectrum1 / cube1
    assert_array_almost_equal(cube2.data, sp1data / cube1.data)

    # spectrum * image
    data = np.ones(shape=(6, 5)) * 2
    image1 = Image(data=data, wcs=wcs)
    cube2 = spectrum1 * image1
    assert_array_almost_equal(cube2.data,
                              sp1data * image1.data[np.newaxis, :, :])


@attr(speed='fast')
def test_get_Spectrum():
    """Spectrum class: testing getters"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2.3,
                         wave=wave)
    a = spectrum1[1:7]
    nose.tools.assert_equal(a.shape[0], 6)
    a = spectrum1.subspec(1.2, 15.6, unit=u.nm)
    nose.tools.assert_equal(a.shape[0], 6)

    spvar = Spectrum('data/obj/Spectrum_Variance.fits', ext=[0, 1])
    unit = spvar.wave.unit
    spvarcut = spvar.subspec(5560, 5590, unit=unit)
    nose.tools.assert_equal(spvarcut.shape[0], 48)
    nose.tools.assert_almost_equal(spvarcut.get_start(unit=unit), 5560.25, 2)
    nose.tools.assert_almost_equal(spvarcut.get_end(unit=unit), 5589.89, 2)
    nose.tools.assert_almost_equal(spvarcut.get_step(unit=unit), 0.63, 2)


@attr(speed='fast')
def test_spectrum_methods():
    """Spectrum class: testing sum/mean/abs/sqrt methods"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm, shape=10)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                         wave=wave)
    sum1 = spectrum1.sum()
    nose.tools.assert_almost_equal(sum1, spectrum1.data.sum())
    spectrum2 = spectrum1[1:-2]
    sum1 = spectrum1.sum(lmin=spectrum1.wave.coord(1), lmax=spectrum1.wave.coord(10 - 3),
                         unit=u.nm)
    sum2 = spectrum2.sum()
    nose.tools.assert_almost_equal(sum1, sum2)
    mean1 = spectrum1.mean(lmin=spectrum1.wave.coord(1), lmax=spectrum1.wave.coord(10 - 3),
                           unit=u.nm)
    mean2 = spectrum2.mean()
    nose.tools.assert_almost_equal(mean1, mean2)

    spnovar = Spectrum('data/obj/Spectrum_Novariance.fits')
    spvar = Spectrum('data/obj/Spectrum_Variance.fits', ext=[0, 1])
    spvar2 = spvar.abs()
    nose.tools.assert_equal(spvar2[23], np.abs(spvar[23]))
    spvar2 = spvar.abs().sqrt()
    nose.tools.assert_equal(spvar2[8], np.sqrt(np.abs(spvar[8])))
    nose.tools.assert_almost_equal(spvar.mean(), 11.526547845374727)
    nose.tools.assert_almost_equal(spnovar.mean(), 11.101086376675089)
    spvarsum = spvar2 + 4 * spvar2 - 56 / spvar2

    nose.tools.assert_almost_equal(spvarsum[10], spvar2[10] + 4 * spvar2[10] - 56 / spvar2[10])
    nose.tools.assert_almost_equal(spvar.get_step(), 0.630448220641262)
    nose.tools.assert_almost_equal(spvar.get_start(), 4602.6040286827802)
    nose.tools.assert_almost_equal(spvar.get_end(), 7184.289492208748)
    nose.tools.assert_almost_equal(spvar.get_range()[0], 4602.60402868)
    nose.tools.assert_almost_equal(spvar.get_range()[1], 7184.28949221)


@attr(speed='fast')
def test_gauss_fit():
    """Spectrum class: testing Gaussian fit"""
    wave = WaveCoord(crpix=1, cdelt=0.3, crval=400, cunit=u.nm, shape=10)
    data = np.zeros(600)
    spem = Spectrum(data=data * 2.3, wave=wave)
    spem.add_gaussian(5000, 1200, 20, unit=u.angstrom)
    gauss = spem.gauss_fit(lmin=(4500, 4800), lmax=(5200, 6000), lpeak=5000,
                           unit=u.angstrom)
    nose.tools.assert_almost_equal(gauss.lpeak, 5000, 2)
    nose.tools.assert_almost_equal(gauss.flux, 1200, 2)
    nose.tools.assert_almost_equal(gauss.fwhm, 20, 2)
    nose.tools.assert_almost_equal(spem.fwhm(gauss.lpeak), 20, 0)
    gauss = spem.line_gauss_fit(lmin=(4500, 4800), lmax=(5200, 6000),
                                lpeak=5000, unit=u.angstrom)
    nose.tools.assert_almost_equal(gauss.flux, 1200, 2)
    nose.tools.assert_almost_equal(gauss.fwhm, 20, 2)
    nose.tools.assert_almost_equal(spem.fwhm(gauss.lpeak), 20, 0)


@attr(speed='fast')
def test_crop():
    """Spectrum class: testing resize method"""
    sig = fits.getdata("data/obj/g9-124Tsigspec.fits")
    spe = Spectrum("data/obj/g9-124Tspec.fits", var=sig * sig)
    unit = spe.wave.unit
    spe.mask_region(lmax=5000, unit=unit)
    spe.mask_region(lmin=6500, unit=unit)
    spe.crop()
    nose.tools.assert_equal(int((6500 - 5000) / spe.get_step(unit=unit)),
                            spe.shape[0])


@attr(speed='fast')
def test_resample():
    """Spectrum class: testing resampling function"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm, shape=10)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2.3,
                         wave=wave)
    flux1 = spectrum1.sum() * spectrum1.wave.get_step()
    spectrum2 = spectrum1.resample(0.3)
    flux2 = spectrum2.sum() * spectrum2.wave.get_step()
    nose.tools.assert_almost_equal(flux1, flux2)


@attr(speed='veryslow')
def test_resampling_slow():
    """Spectrum class: heavy test of resampling function"""
    sig = fits.getdata("data/obj/g9-124Tsigspec.fits")
    spe = Spectrum("data/obj/g9-124Tspec.fits", var=sig * sig)
    unit = spe.wave.unit
    flux1 = spe.sum(weight=False) * spe.wave.get_step(unit=unit)
    spe2 = spe.resample(0.3, unit=unit)
    flux2 = spe2.sum(weight=False) * spe2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1, flux2, 1)

    spnovar = Spectrum('data/obj/Spectrum_Novariance.fits')
    unit = spnovar.wave.unit
    flux1 = spnovar.sum() * spnovar.wave.get_step(unit=unit)
    spnovar2 = spnovar.resample(4, unit=unit)
    flux2 = spnovar2.sum() * spnovar2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1, flux2, 0)

    spvar = Spectrum('data/obj/Spectrum_Variance.fits', ext=[0, 1])
    unit = spvar.wave.unit
    flux1 = spvar.sum(weight=False) * spvar.wave.get_step(unit=unit)
    spvar2 = spvar.resample(4, unit=unit)
    flux2 = spvar2.sum(weight=False) * spvar2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1, flux2, 0)


@attr(speed='fast')
def test_rebin():
    """Spectrum class: testing rebin function"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=10)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2.3,
                         wave=wave)
    unit = spectrum1.wave.unit
    factor = 3
    s = slice(0, factor * (spectrum1.shape[0] // factor))  # The rebinned slice
    flux1 = spectrum1[s].sum() * spectrum1[s].wave.get_step(unit=unit)
    spectrum2 = spectrum1.rebin(factor, margin='left')
    flux2 = spectrum2.sum() * spectrum2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1, flux2)

    sig = fits.getdata("data/obj/g9-124Tsigspec.fits")
    spe = Spectrum("data/obj/g9-124Tspec.fits", var=sig * sig)
    unit = spe.wave.unit
    factor = 3
    s = slice(0, factor * (spe.shape[0] // factor))
    flux1 = spe[s].sum() * spe[s].wave.get_step(unit=unit)
    spe2 = spe.rebin(factor, margin='left')
    flux2 = spe2.sum() * spe2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1, flux2)

    spnovar = Spectrum('data/obj/Spectrum_Novariance.fits')
    unit = spnovar.wave.unit
    factor = 4
    s = slice(0, factor * (spnovar.shape[0] // factor))
    flux1 = spnovar[s].sum() * spnovar[s].wave.get_step(unit=unit)
    spnovar2 = spnovar.rebin(factor, margin='left')
    flux2 = spnovar2.sum() * spnovar2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1, flux2)

    spvar = Spectrum('data/obj/Spectrum_Variance.fits', ext=[0, 1])
    unit = spvar.wave.unit
    factor = 4
    s = slice(0, factor * (spvar.shape[0] // factor))
    flux1 = spvar[s].sum(weight=False) * spvar[s].wave.get_step(unit=unit)
    spvar2 = spvar.rebin(factor, margin='left')
    flux2 = spvar2.sum(weight=False) * spvar2.wave.get_step(unit=unit)
    nose.tools.assert_almost_equal(flux1, flux2)


@attr(speed='fast')
def test_truncate():
    """Spectrum class: testing truncate function"""
    sig = fits.getdata("data/obj/g9-124Tsigspec.fits")
    spe = Spectrum("data/obj/g9-124Tspec.fits", var=sig * sig)
    unit = spe.wave.unit
    spe.truncate(4950, 5050, unit=unit)
    nose.tools.assert_equal(spe.shape[0], 160)


@attr(speed='fast')
def test_interpolation():
    """Spectrum class: testing interpolations"""
    spnovar = Spectrum('data/obj/Spectrum_Novariance.fits')
    uspnovar = spnovar.wave.unit
    spvar = Spectrum('data/obj/Spectrum_Variance.fits', ext=[0, 1])
    uspvar = spvar.wave.unit
    spvar.mask_region(5575, 5585, unit=uspvar)
    spvar.mask_region(6296, 6312, unit=uspvar)
    spvar.mask_region(6351, 6375, unit=uspvar)
    spnovar.mask_region(5575, 5585, unit=uspnovar)
    spnovar.mask_region(6296, 6312, unit=uspnovar)
    spnovar.mask_region(6351, 6375, unit=uspnovar)
    spm1 = spvar.copy()
    spm1.interp_mask()
    spm2 = spvar.copy()
    spm2.interp_mask(spline=True)
    spvarcut1 = spvar.subspec(5550, 5590, unit=uspvar)
    spvarcut2 = spnovar.subspec(5550, 5590, unit=uspnovar)
    spvarcut3 = spm1.subspec(5550, 5590, unit=uspvar)
    spvarcut4 = spm2.subspec(5550, 5590, unit=uspvar)
    nose.tools.assert_almost_equal(spvar.mean(5550, 5590, unit=uspvar),
                                   spvarcut1.mean())
    nose.tools.assert_almost_equal(spnovar.mean(5550, 5590, unit=uspnovar),
                                   spvarcut2.mean())
    nose.tools.assert_almost_equal(spm1.mean(5550, 5590, unit=uspvar),
                                   spvarcut3.mean())
    nose.tools.assert_almost_equal(spm2.mean(5550, 5590, unit=uspvar),
                                   spvarcut4.mean())


@attr(speed='fast')
def test_poly_fit():
    """Spectrum class: testing polynomial fit"""
    spvar = Spectrum('data/obj/Spectrum_Variance.fits', ext=[0, 1])
    polyfit1 = spvar.poly_fit(12)
    spfit1 = spvar.copy()
    spfit1.poly_val(polyfit1)
    spfit2 = spvar.poly_spec(10)
    spfit3 = spvar.poly_spec(10, weight=False)
    nose.tools.assert_almost_equal(spfit1.mean(), 11.1, 1)
    nose.tools.assert_almost_equal(spfit2.mean(), 11.1, 1)
    nose.tools.assert_almost_equal(spfit3.mean(), 11.1, 1)


@attr(speed='fast')
def test_filter():
    """Spectrum class: testing filters"""
    sp = Spectrum('data/obj/Spectrum_Variance.fits', ext=[0, 1])
    sp.unit = u.Unit('erg/cm2/s/Angstrom')
    sp.wave.unit = u.angstrom
    nose.tools.assert_almost_equal(sp.abmag_band(5000.0, 1000.0), -22.837, 2)
    nose.tools.assert_almost_equal(
        sp.abmag_filter([4000, 5000, 6000], [0.1, 1.0, 0.3]), -23.077, 2)
    nose.tools.assert_almost_equal(sp.abmag_filter_name('U'), 99)
    nose.tools.assert_almost_equal(sp.abmag_filter_name('B'), -22.278, 2)


@attr(speed='fast')
def test_mag():
    """Spectrum class: testing magnitude computations."""
    Vega = Spectrum('data/obj/Vega.fits')
    Vega.unit = u.Unit('2E-17 erg / (Angstrom cm2 s)')
    Vega.wave.wcs.wcs.cunit[0] = u.angstrom
    nose.tools.assert_almost_equal(Vega.abmag_filter_name('V'), 0, 1)
    mag = Vega.abmag_filter_name('Ic')
    nose.tools.assert_true(mag > 0.4 and mag < 0.5)
    mag = Vega.abmag_band(22500, 2500)
    nose.tools.assert_true(mag > 1.9 and mag < 2.0)


@attr(speed='fast')
def test_integrate():
    """Spectrum class: testing integration"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                         wave=wave, unit=u.Unit('ct/Angstrom'))

    # Integrate the whole spectrum, by not specifying starting or ending
    # wavelengths. This should be the sum of the pixel values multiplied
    # by cdelt in angstroms (because the flux units are per angstrom).
    result = spectrum1.integrate()
    expected = spectrum1.get_step(unit=u.angstrom) * spectrum1.sum()
    nose.tools.assert_almost_equal(result.value, expected)
    nose.tools.assert_equal(result.unit, u.ct)

    # The result should not change if we change the wavelength units of
    # the wavelength limits to nanometers.
    result = spectrum1.integrate(unit=u.nm)
    expected = spectrum1.get_step(unit=u.angstrom) * spectrum1.sum()
    nose.tools.assert_almost_equal(result.value, expected)
    nose.tools.assert_equal(result.unit, u.ct)

    # Integrate over a wavelength range 3.5 to 6.5 nm. The WCS
    # conversion equation from wavelength to pixel index is,
    #
    #  index = crpix-1 + (lambda-crval)/cdelt
    #  index = 1 + (lambda - 0.5) / 3.0
    #
    # So wavelengths 3.5 and 6.5nm, correspond to pixel indexes
    # of 2.0 and 3.0. These are the centers of pixels 2 and 3.
    # Thus the integration should be the value of pixel 2 times
    # half of cdelt, plus the value of pixel 3 times half of cdelt.
    # This comes to 2*3.0/2 + 3*3.0/2 = 7.5 ct/Angstrom*nm, which
    # should be rescaled to 75 ct, since nm/Angstrom is 10.0.
    result = spectrum1.integrate(lmin=3.5, lmax=6.5, unit=u.nm)
    nose.tools.assert_almost_equal(result.value, 75)
    nose.tools.assert_equal(result.unit, u.ct)

    # Do the same test, but specify the wavelength limits in angstroms.
    # The result should be the same as before.
    result = spectrum1.integrate(lmin=35.0, lmax=65.0, unit=u.angstrom)
    nose.tools.assert_almost_equal(result.value, 75)
    nose.tools.assert_equal(result.unit, u.ct)

    # Do the same experiment yet again, but this time after changing
    # the flux units of the spectrum to simple counts, without any per
    # wavelength units. Since there are no wavelength units in the
    # flux units, the result should not be rescaled from the native
    # value of 7.5, and because we specified a wavelength range in
    # angstroms, the resulting units should be counts * nm.
    spectrum1.unit = u.ct
    result = spectrum1.integrate(lmin=3.5, lmax=6.5, unit=u.nm)
    nose.tools.assert_almost_equal(result.value, 7.5)
    nose.tools.assert_equal(result.unit, u.ct * u.nm)


@attr(speed='fast')
def test_write():
    """Spectrum class: testing write."""
    sp = Spectrum(data=np.arange(10), wave=WaveCoord(cunit=u.nm))
    fobj = NamedTemporaryFile()
    sp.write(fobj.name)

    hdu = fits.open(fobj)
    # print repr(hdu[0].header)
    # print '========='
    # print repr(hdu[1].header)
    assert_array_equal(hdu[1].data.shape, sp.shape)

    hdr = hdu[1].header
    nose.tools.assert_equal(hdr['EXTNAME'], 'DATA')
    nose.tools.assert_equal(hdr['NAXIS'], 1)
    nose.tools.assert_equal(u.Unit(hdr['CUNIT1']), u.nm)
    nose.tools.assert_equal(hdr['NAXIS1'], sp.shape[0])

    # Same with Angstrom
    sp = Spectrum(data=np.arange(10), wave=WaveCoord(cunit=u.angstrom))
    fobj = NamedTemporaryFile()
    sp.write(fobj.name)

    hdu = fits.open(fobj)
    hdr = hdu[1].header
    nose.tools.assert_equal(u.Unit(hdr['CUNIT1']), u.angstrom)
