"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>
Copyright (c)      2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import pytest
import numpy as np

from astropy import units as u
from astropy.io import ascii, fits
from mpdaf.log import setup_logging
from mpdaf.obj import Spectrum, Image, Cube, WCS, WaveCoord, airtovac, vactoair
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal, assert_allclose)

from ...tests.utils import (get_data_file, generate_spectrum)


def test_copy(spec_var):
    """Spectrum class: testing copy method."""
    spe = spec_var.copy()
    assert spec_var.wave.isEqual(spe.wave)
    assert spec_var.data.sum() == spe.data.sum()
    assert spec_var.var.sum() == spe.var.sum()


def test_selection(spectrum):
    """Spectrum class: testing operators > and < """
    spectrum2 = spectrum > 6
    assert_almost_equal(spectrum2.sum()[0], 24)
    spectrum2 = spectrum >= 6
    assert_almost_equal(spectrum2.sum()[0], 30)
    spectrum2 = spectrum < 6
    assert_almost_equal(spectrum2.sum()[0], 15.5)
    spectrum2 = spectrum <= 6
    spectrum[:] = spectrum2
    assert_almost_equal(spectrum.sum()[0], 21.5)


def test_arithmetric():
    """Spectrum class: testing arithmetic functions"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                         wave=wave)
    spectrum2 = spectrum1 > 6  # [-,-,-,-,-,-,-,7,8,9]
    # +
    spectrum3 = spectrum1 + spectrum2
    assert spectrum3.data.data[3] == 3
    assert spectrum3.data.data[8] == 16
    spectrum3 = 4.2 + spectrum1
    assert spectrum3.data.data[3] == 3 + 4.2
    # -
    spectrum3 = spectrum1 - spectrum2
    assert spectrum3.data.data[3] == 3
    assert spectrum3.data.data[8] == 0
    spectrum3 = spectrum1 - 4.2
    assert spectrum3.data.data[8] == 8 - 4.2
    # *
    spectrum3 = spectrum1 * spectrum2
    assert spectrum3.data.data[8] == 64
    spectrum3 = 4.2 * spectrum1
    assert spectrum3.data.data[9] == 9 * 4.2
    # /
    spectrum3 = spectrum1 / spectrum2
    # divide functions that have a validity domain returns the masked constant
    # whenever the input is masked or falls outside the validity domain.
    assert spectrum3.data.data[8] == 1
    spectrum3 = 1.0 / (4.2 / spectrum1)
    assert spectrum3.data.data[5] == 5 / 4.2

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


def test_get_Spectrum(spec_var):
    """Spectrum class: testing getters"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2.3,
                         wave=wave)
    a = spectrum1[1:7]
    assert a.shape[0] == 6
    a = spectrum1.subspec(1.2, 15.6, unit=u.nm)
    assert a.shape[0] == 6

    unit = spec_var.wave.unit
    spvarcut = spec_var.subspec(5560, 5590, unit=unit)
    assert spvarcut.shape[0] == 48
    assert_almost_equal(spvarcut.get_start(unit=unit), 5560.25, 2)
    assert_almost_equal(spvarcut.get_end(unit=unit), 5589.89, 2)
    assert_almost_equal(spvarcut.get_step(unit=unit), 0.63, 2)


def test_spectrum_methods(spec_var, spec_novar):
    """Spectrum class: testing sum/mean/abs/sqrt methods"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm, shape=10)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                         wave=wave)
    sum1 = spectrum1.sum()
    assert_almost_equal(sum1[0], spectrum1.data.sum())
    spectrum2 = spectrum1[1:-2]
    sum1 = spectrum1.sum(lmin=spectrum1.wave.coord(1),
                         lmax=spectrum1.wave.coord(10 - 3),
                         unit=u.nm)
    sum2 = spectrum2.sum()
    assert_almost_equal(sum1, sum2)
    mean1 = spectrum1.mean(lmin=spectrum1.wave.coord(1),
                           lmax=spectrum1.wave.coord(10 - 3),
                           unit=u.nm)
    mean2 = spectrum2.mean()
    assert_almost_equal(mean1, mean2)

    spvar2 = spec_var.abs()
    assert spvar2[23] == np.abs(spec_var[23])
    spvar2 = spec_var.abs().sqrt()
    assert spvar2[8] == np.sqrt(np.abs(spec_var[8]))
    assert_almost_equal(spec_var.mean()[0], 11.526, 2)
    assert_almost_equal(spec_novar.mean()[0], 11.101, 2)
    spvarsum = spvar2 + 4 * spvar2 - 56 / spvar2

    assert_almost_equal(spvarsum[10],
                        spvar2[10] + 4 * spvar2[10] - 56 / spvar2[10], 2)
    assert_almost_equal(spec_var.get_step(), 0.630, 2)
    assert_almost_equal(spec_var.get_start(), 4602.604, 2)
    assert_almost_equal(spec_var.get_end(), 7184.289, 2)
    assert_almost_equal(spec_var.get_range()[0], 4602.604, 2)
    assert_almost_equal(spec_var.get_range()[1], 7184.289, 2)


@pytest.mark.parametrize('cont', (5, None))
def test_gauss_fit(capsys, cont):
    """Spectrum class: testing Gaussian fit"""
    contval = cont or 0
    wave = WaveCoord(crpix=1, cdelt=0.3, crval=400, cunit=u.nm)
    spem = Spectrum(data=np.zeros(600) + contval, wave=wave)
    spem.add_gaussian(5000, 1200, 20, unit=u.angstrom)

    setup_logging()
    gauss = spem.gauss_fit(lmin=(4500, 4800), lmax=(5200, 6000), lpeak=5000,
                           cont=cont, unit=u.angstrom)
    gauss.print_param()
    out, err = capsys.readouterr()
    assert '[INFO] Gaussian center = 5000 ' in err

    assert_almost_equal(gauss.lpeak, 5000, 2)
    assert_almost_equal(gauss.flux, 1200, 2)
    assert_almost_equal(gauss.fwhm, 20, 2)
    assert_allclose(spem.fwhm(gauss.lpeak, cont=contval), 20, atol=0.2)

    gauss = spem.line_gauss_fit(lmin=(4500, 4800), lmax=(5200, 6000),
                                lpeak=5000, cont=cont, unit=u.angstrom)
    assert_almost_equal(gauss.flux, 1200, 2)
    assert_almost_equal(gauss.fwhm, 20, 2)
    assert_allclose(spem.fwhm(gauss.lpeak, cont=contval), 20, atol=0.2)


def test_crop(spec_var):
    """Spectrum class: testing resize method"""
    spec_var.mask_region(lmax=5000)
    spec_var.mask_region(lmin=6500)
    spec_var.crop()
    assert int((6500 - 5000) / spec_var.get_step(unit=spec_var.wave.unit)) == \
        spec_var.shape[0]


def test_resample():
    """Spectrum class: Test resampling"""

    # Choose the dimensions of the spectrum, choosing a large number that is
    # *not* a convenient power of 2.
    oldshape = 4000

    # Choose the wavelength pixel size and the default wavelength units.
    oldstep = 1.0
    oldunit = u.angstrom

    # Create the wavelength axis coordinates.
    wave = WaveCoord(crpix=2.0, cdelt=oldstep, crval=0.5, cunit=oldunit,
                     shape=oldshape)

    # Specify the desired increase in pixel size, and the resulting pixel size.
    factor = 6.5
    newstep = ((factor * oldstep) * oldunit).to(u.nm).value

    # Specify the wavelength at which the peak of the resampled spectrum should
    # be expected.
    expected_peak_wave = 3000.0

    # Create the array in which the test spectrum will be composed.
    data = np.zeros(oldshape)

    # Get the wavelength coordinates of each pixel in the spectrum.
    w = wave.coord()

    # Add the following list gaussians to the spectrum, where each
    # gaussian is specified as: (amplitude, sigma_in_pixels,
    # center_wavelength). Given that narrow gaussians are reduced in
    # amplitude by resampling more than wide gaussians, we arrange
    # that the peak gaussian before and after correctly resampling are
    # different.
    gaussians = [(0.5, 12.0, 800.0),
                 (0.7, 5.0, 1200.0),
                 (0.4, 700.0, 1600.0),
                 (1.5, 2.6, 1980.0),             # Peak before resampling
                 (1.2, 2.6, 2000.0),
                 (1.3, 15.0, expected_peak_wave),  # Peak if resampled correctly
                 (1.0, 2.0, 3200.0)]
    for amp, sigma, center in gaussians:
        sigma *= oldstep
        data += amp * np.exp(-0.5 * ((center - w) / sigma)**2)

    # Fill the variance array with a simple window function.
    var = np.hamming(oldshape)

    # Add gaussian random noise to the spectrum, but leave 3 output
    # pixel widths zero at each end of the spectrum so that the PSF of
    # the output grid doesn't spread flux from the edges off the edge
    # of the output grid. It takes about 3 pixel widths for the gaussian
    # PSF to drop to about 0.01 of its peak.
    margin = np.ceil(3 * factor).astype(int)
    data[margin:-margin] += np.random.normal(scale=0.1, size=data.shape - 2 * margin)

    # Install the spectral data in a Spectrum container.
    oldsp = Spectrum(data=data, var=var, wave=wave)

    # Mask a few pixels.
    masked_slice = slice(900, 910)
    oldsp.mask[masked_slice] = True

    # Create a down-sampled version of the input spectrum.
    newsp = oldsp.resample(newstep, unit=u.nm)

    # Check that the integral flux in the resampled spectrum matches that of
    # the original spectrum.
    expected_flux = oldsp.sum(weight=False)[0] * oldsp.wave.get_step(unit=oldunit)
    actual_flux = newsp.sum(weight=False)[0] * newsp.wave.get_step(unit=oldunit)
    assert_allclose(actual_flux, expected_flux, 1e-2)

    # Do the same test, but with fluxes weighted by the inverse of the variances.
    expected_flux = oldsp.sum(weight=True)[0] * oldsp.wave.get_step(unit=oldunit)
    actual_flux = newsp.sum(weight=True)[0] * newsp.wave.get_step(unit=oldunit)
    assert_allclose(actual_flux, expected_flux, 1e-2)

    # Check that the peak of the resampled spectrum is at the wavelength
    # where the strongest gaussian was centered in the input spectrum.
    assert_allclose(np.argmax(newsp.data),
                    newsp.wave.pixel(expected_peak_wave, nearest=True))

    # Now upsample the downsampled spectrum to the original pixel size.
    # This won't recover the same spectrum, since higher spatial frequencies
    # are lost when downsampling, but the total flux should be about the
    # same, and the peak should be at the same wavelength as the peak in
    # original spectrum within one pixel width of the downsampled spectrum.
    newsp2 = newsp.resample(oldstep, unit=oldunit)

    # Check that the doubly resampled spectrum has the same integrated flux
    # as the original.
    expected_flux = oldsp.sum(weight=False)[0] * oldsp.wave.get_step(unit=oldunit)
    actual_flux = newsp2.sum(weight=False)[0] * newsp2.wave.get_step(unit=oldunit)
    assert_allclose(actual_flux, expected_flux, 1e-2)

    # Check that the peak of the up-sampled spectrum is at the wavelength
    # of the peak of the down-sampled spectrum to within the pixel resolution
    # of the downsampled spectrum.
    assert_allclose(newsp.wave.pixel(newsp2.wave.coord(np.argmax(newsp2.data)),
                                     nearest=True),
                    newsp.wave.pixel(expected_peak_wave, nearest=True))

    # Check that pixels that were masked in the input spectrum are still
    # masked in the final spectrum.
    np.testing.assert_equal(newsp2.mask[masked_slice], oldsp.mask[masked_slice])


def test_rebin(spec_var, spec_novar):
    """Spectrum class: testing rebin function"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, shape=10)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * 2.3,
                         wave=wave)
    unit = spectrum1.wave.unit
    factor = 3
    s = slice(0, factor * (spectrum1.shape[0] // factor))  # The rebinned slice
    flux1 = spectrum1[s].sum()[0] * spectrum1[s].wave.get_step(unit=unit)
    spectrum2 = spectrum1.rebin(factor, margin='left')
    flux2 = spectrum2.sum()[0] * spectrum2.wave.get_step(unit=unit)
    assert_almost_equal(flux1, flux2, 2)

    unit = spec_novar.wave.unit
    factor = 4
    s = slice(0, factor * (spec_novar.shape[0] // factor))
    flux1 = spec_novar[s].sum()[0] * spec_novar[s].wave.get_step(unit=unit)
    spnovar2 = spec_novar.rebin(factor, margin='left')
    flux2 = spnovar2.sum()[0] * spnovar2.wave.get_step(unit=unit)
    assert_almost_equal(flux1, flux2, 2)

    unit = spec_var.wave.unit
    factor = 4
    s = slice(0, factor * (spec_var.shape[0] // factor))
    flux1 = spec_var[s].sum(weight=False)[0] * \
        spec_var[s].wave.get_step(unit=unit)
    spvar2 = spec_var.rebin(factor, margin='left')
    flux2 = spvar2.sum(weight=False)[0] * spvar2.wave.get_step(unit=unit)
    assert_almost_equal(flux1, flux2, 2)


def test_truncate(spec_var):
    """Spectrum class: testing truncate function"""
    spec_var.truncate(4950, 5050)
    assert spec_var.shape[0] == 160


def test_interpolation(spec_var, spec_novar):
    """Spectrum class: testing interpolations"""
    uspnovar = spec_novar.wave.unit
    uspvar = spec_var.wave.unit
    spec_var.mask_region(5575, 5585, unit=uspvar)
    spec_var.mask_region(6296, 6312, unit=uspvar)
    spec_var.mask_region(6351, 6375, unit=uspvar)
    spec_novar.mask_region(5575, 5585, unit=uspnovar)
    spec_novar.mask_region(6296, 6312, unit=uspnovar)
    spec_novar.mask_region(6351, 6375, unit=uspnovar)
    spm1 = spec_var.copy()
    spm1.interp_mask()
    spm2 = spec_var.copy()
    spm2.interp_mask(spline=True)
    spvarcut1 = spec_var.subspec(5550, 5590, unit=uspvar)
    spvarcut2 = spec_novar.subspec(5550, 5590, unit=uspnovar)
    spvarcut3 = spm1.subspec(5550, 5590, unit=uspvar)
    spvarcut4 = spm2.subspec(5550, 5590, unit=uspvar)
    assert_almost_equal(spec_var.mean(5550, 5590, unit=uspvar),
                        spvarcut1.mean())
    assert_almost_equal(spec_novar.mean(5550, 5590, unit=uspnovar)[0],
                        spvarcut2.mean()[0])
    assert_almost_equal(spm1.mean(5550, 5590, unit=uspvar), spvarcut3.mean())
    assert_almost_equal(spm2.mean(5550, 5590, unit=uspvar), spvarcut4.mean())


def test_poly_fit(spec_var):
    """Spectrum class: testing polynomial fit"""
    polyfit1 = spec_var.poly_fit(12)
    spfit1 = spec_var.copy()
    spfit1.poly_val(polyfit1)
    spfit2 = spec_var.poly_spec(10)
    spfit3 = spec_var.poly_spec(10, weight=False)
    assert_almost_equal(spfit1.mean()[0], 11.1, 1)
    assert_almost_equal(spfit2.mean()[0], 11.1, 1)
    assert_almost_equal(spfit3.mean()[0], 11.1, 1)


def test_filter(spec_var):
    """Spectrum class: testing filters"""
    spec_var.unit = u.Unit('erg/cm2/s/Angstrom')
    spec_var.wave.unit = u.angstrom
    assert_almost_equal(spec_var.abmag_band(5000.0, 1000.0)[0], -22.837, 2)
    assert_almost_equal(spec_var.abmag_filter([4000, 5000, 6000],
                                              [0.1, 1.0, 0.3])[0],
                        -23.077, 2)
    assert_almost_equal(spec_var.abmag_filter_name('U')[0], 99)
    assert_almost_equal(spec_var.abmag_filter_name('B')[0], -22.278, 2)


def test_mag():
    """Spectrum class: testing magnitude computations."""
    Vega = Spectrum(get_data_file('obj', 'Vega.fits'))
    Vega.unit = u.Unit('2E-17 erg / (Angstrom cm2 s)')
    Vega.wave.wcs.wcs.cunit[0] = u.angstrom
    assert_almost_equal(Vega.abmag_filter_name('V')[0], 0, 1)
    mag = Vega.abmag_filter_name('Ic')[0]
    assert mag > 0.4 and mag < 0.5
    mag = Vega.abmag_band(22500, 2500)[0]
    assert mag > 1.9 and mag < 2.0


def test_integrate():
    """Spectrum class: testing integration"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(data=np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                         wave=wave, unit=u.Unit('ct/Angstrom'))

    # Integrate the whole spectrum, by not specifying starting or ending
    # wavelengths. This should be the sum of the pixel values multiplied
    # by cdelt in angstroms (because the flux units are per angstrom).
    result = spectrum1.integrate()[0]
    expected = spectrum1.get_step(unit=u.angstrom) * spectrum1.sum()[0]
    assert_almost_equal(result.value, expected)
    assert result.unit == u.ct

    # The result should not change if we change the wavelength units of
    # the wavelength limits to nanometers.
    result = spectrum1.integrate(unit=u.nm)[0]
    expected = spectrum1.get_step(unit=u.angstrom) * spectrum1.sum()[0]
    assert_almost_equal(result.value, expected)
    assert result.unit == u.ct

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
    result = spectrum1.integrate(lmin=3.5, lmax=6.5, unit=u.nm)[0]
    assert_almost_equal(result.value, 75)
    assert result.unit == u.ct

    # Do the same test, but specify the wavelength limits in angstroms.
    # The result should be the same as before.
    result = spectrum1.integrate(lmin=35.0, lmax=65.0, unit=u.angstrom)[0]
    assert_almost_equal(result.value, 75)
    assert result.unit == u.ct

    # Do the same experiment yet again, but this time after changing
    # the flux units of the spectrum to simple counts, without any per
    # wavelength units. Since there are no wavelength units in the
    # flux units, the result should not be rescaled from the native
    # value of 7.5, and because we specified a wavelength range in
    # angstroms, the resulting units should be counts * nm.
    spectrum1.unit = u.ct
    result = spectrum1.integrate(lmin=3.5, lmax=6.5, unit=u.nm)[0]
    assert_almost_equal(result.value, 7.5)
    assert result.unit == u.ct * u.nm


def test_write(tmpdir):
    """Spectrum class: testing write."""
    testfile = str(tmpdir.join('spec.fits'))
    sp = Spectrum(data=np.arange(10), wave=WaveCoord(cunit=u.nm))
    sp.write(testfile)

    with fits.open(testfile) as hdu:
        assert_array_equal(hdu[1].data.shape, sp.shape)

    hdr = hdu[1].header
    assert hdr['EXTNAME'] == 'DATA'
    assert hdr['NAXIS'] == 1
    assert u.Unit(hdr['CUNIT1']) == u.nm
    assert hdr['NAXIS1'] == sp.shape[0]

    # Same with Angstrom
    sp = Spectrum(data=np.arange(10), wave=WaveCoord(cunit=u.angstrom))
    sp.write(testfile)

    with fits.open(testfile) as hdu:
        assert u.Unit(hdu[1].header['CUNIT1']) == u.angstrom


def test_resample2():
    """Spectrum class: testing resampling function
    with a spectrum of integers and resampling to a smaller pixel size"""
    wave = WaveCoord(crpix=2.0, cdelt=3.0, crval=0.5, cunit=u.nm)
    spectrum1 = Spectrum(data=np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
                         wave=wave)
    flux1 = spectrum1.sum()[0] * spectrum1.wave.get_step()
    spectrum2 = spectrum1.resample(0.3)
    flux2 = spectrum2.sum()[0] * spectrum2.wave.get_step()
    assert_almost_equal(flux1, flux2, 2)


def test_get_item():
    """Spectrum class: testing __getitem__"""
    # Set the shape and contents of the spectrum's data array.
    shape = (5,)
    data = np.arange(shape[0])

    # Create a test spectrum with the above data array.
    s = generate_spectrum(data=data, shape=shape, wave=WaveCoord(crval=1, cunit=u.angstrom))
    s.primary_header['KEY'] = 'primary value'
    s.data_header['KEY'] = 'data value'

    # Select the whole spectrum.
    r = s[:]
    assert_array_equal(r.shape, s.shape)
    assert_allclose(r.data, s.data)
    assert r.primary_header['KEY'] == s.primary_header['KEY']
    assert r.data_header['KEY'] == s.data_header['KEY']
    assert isinstance(r, Spectrum)
    assert r.wcs is None
    assert r.wave.isEqual(s.wave)

    # Select a sub-spectrum.
    r = s[1:3]
    assert_array_equal(r.shape, (2))
    assert_allclose(r.data, s.data[1:3])
    assert r.primary_header['KEY'] == s.primary_header['KEY']
    assert r.data_header['KEY'] == s.data_header['KEY']
    assert isinstance(r, Spectrum)
    assert r.wave.isEqual(s.wave[1:3])
    assert r.wcs is None

    # Select a single pixel of the spectrum.
    r = s[2]
    assert np.isscalar(r)
    assert_allclose(r, s.data[2])


def test_airtovac():
    # From http://classic.sdss.org/dr7/products/spectra/vacwavelength.html
    tbl = ascii.read("""\
line air vacuum
H-beta 4861.363 4862.721
[OIII] 4958.911 4960.295
[OIII] 5006.843 5008.239
[NII] 6548.05  6549.86
H-alpha 6562.801 6564.614
[NII] 6583.45  6585.27
[SII] 6716.44  6718.29
[SII] 6730.82  6732.68
""")

    for row in tbl:
        assert_allclose(airtovac(row['air']), row['vacuum'], atol=1e-2)

    assert_allclose(vactoair(row['vacuum']), row['air'], atol=1e-2)
