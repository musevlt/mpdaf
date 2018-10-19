"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2012-2017 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2014-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>
Copyright (c) 2016-2018 Roland Bacon <roland.bacon@univ-lyon1.fr>
Copyright (c)      2018 Yannick Roehlly <yannick.roehlly@univ-lyon1.fr>

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

import matplotlib.pyplot as plt
import numpy as np
import types

import astropy.units as u
from astropy.io import fits
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
from os.path import join, abspath, dirname
from scipy import interpolate, signal
from scipy.optimize import leastsq

from . import ABmag_filters, wavelet1D
from .arithmetic import ArithmeticMixin
from .data import DataArray
from .fitting import Gauss1D
from .objs import flux2mag
from ..tools import deprecated

__all__ = ('Spectrum', 'vactoair', 'airtovac')


def vactoair(vacwl):
    """Calculate the approximate wavelength in air for vacuum wavelengths.

    Parameters
    ----------
    vacwl : ndarray
       Vacuum wavelengths.

    This uses an approximate formula from the IDL astronomy library
    https://idlastro.gsfc.nasa.gov/ftp/pro/astro/vactoair.pro

    """
    wave2 = vacwl * vacwl
    n = 1.0 + 2.735182e-4 + 131.4182 / wave2 + 2.76249e8 / (wave2 * wave2)

    # Do not extrapolate to very short wavelengths.
    if not isinstance(vacwl, np.ndarray):
        if vacwl < 2000:
            n = 1.0
    else:
        ignore = np.where(vacwl < 2000)
        n[ignore] = 1.0

    return vacwl / n


def airtovac(airwl):
    """Convert air wavelengths to vacuum wavelengths.

    Parameters
    ----------
    vacwl : ndarray
       Vacuum wavelengths.

    This uses the IAU standard as implemented in the IDL astronomy library
    https://idlastro.gsfc.nasa.gov/ftp/pro/astro/airtovac.pro

    """
    sigma2 = (1e4 / airwl)**2.        # Convert to wavenumber squared
    n = 1.0 + (6.4328e-5 + 2.94981e-2 / (146. - sigma2) +
               2.5540e-4 / (41. - sigma2))

    if not isinstance(airwl, np.ndarray):
        if airwl < 2000:
            n = 1.0
    else:
        ignore = np.where(airwl < 2000)
        n[ignore] = 1.0

    return airwl * n


class Spectrum(ArithmeticMixin, DataArray):

    """Spectrum objects contain 1D arrays of numbers, optionally
    accompanied by corresponding variances. These numbers represent
    sample fluxes along a regularly spaced grid of wavelengths.

    The spectral pixel values and their variances, if any, are
    available as arrays[q that can be accessed via properties of the
    Spectrum object called .data and .var, respectively. These arrays
    are usually masked arrays, which share a boolean masking array
    that can be accessed via a property called .mask. In principle,
    these arrays can also be normal numpy arrays without masks, in
    which case the .mask property holds the value,
    numpy.ma.nomask. However non-masked arrays are only supported by a
    subset of mpdaf functions at this time, so masked arrays should be
    used where possible.

    When a new Spectrum object is created, the data, variance and mask
    arrays can either be specified as arguments, or the name of a FITS
    file can be provided to load them from.

    Parameters
    ----------
    filename : string
        An optional FITS file name from which to load the spectrum.
        None by default. This argument is ignored if the data
        argument is not None.
    ext : int or (int,int) or string or (string,string)
        The optional number/name of the data extension
        or the numbers/names of the data and variance extensions.
    wave : `mpdaf.obj.WaveCoord`
        The wavelength coordinates of the spectrum.
    unit : str or `astropy.units.Unit`
        The physical units of the data values. Defaults to
        `astropy.units.dimensionless_unscaled`.
    data : float array
        An optional 1 dimensional array containing the values of each
        pixel of the spectrum, stored in ascending order of wavelength
        (None by default). Where given, this array should be 1
        dimensional.
    var : float array
        An optional 1 dimensional array containing the estimated
        variances of each pixel of the spectrum, stored in ascending
        order of wavelength (None by default).

    Attributes
    ----------
    filename : string
        The name of the originating FITS file, if any. Otherwise None.
    unit : `astropy.units.Unit`
        The physical units of the data values.
    primary_header : `astropy.io.fits.Header`
        The FITS primary header instance, if a FITS file was provided.
    data_header : `astropy.io.fits.Header`
        The FITS header of the DATA extension.
    wave : `mpdaf.obj.WaveCoord`
        The wavelength coordinates of the spectrum.

    """

    # Tell the DataArray base-class that Spectrum objects require 1 dimensional
    # data arrays and wavelength coordinates.
    _ndim_required = 1
    _has_wave = True

    def subspec(self, lmin, lmax=None, unit=u.angstrom):
        """Return the flux at a given wavelength, or the sub-spectrum
        of a specified wavelength range.

        A single flux value is returned if the lmax argument is None
        (the default), or if the wavelengths assigned to the lmin and
        lmax arguments are both within the same pixel. The value that
        is returned is the value of the pixel whose wavelength is
        closest to the wavelength specified by the lmin argument.

        Note that is a wavelength range is asked for, a view on the original
        spectrum is returned and both will be modified at the same time.  If
        you need to modify only the sub-spectrum, you'll need to copy() it
        before.

        Parameters
        ----------
        lmin : float
            The minimum wavelength of a wavelength range, or the wavelength
            of a single pixel if lmax is None.
        lmax : float or None
            The maximum wavelength of the wavelength range.
        unit : `astropy.units.Unit`
            The wavelength units of the lmin and lmax arguments. The
            default is angstroms. If unit is None, then lmin and lmax
            are interpreted as array indexes within the spectrum.

        Returns
        -------
        out : float or `~mpdaf.obj.Spectrum`

        """
        if self.wave is None:
            raise ValueError('Operation forbidden without world coordinates '
                             'along the spectral direction')

        if lmax is None:
            lmax = lmin

        # Are lmin and lmax array indexes?
        if unit is None:
            pix_min = max(0, int(lmin + 0.5))
            pix_max = min(self.shape[0], int(lmax + 0.5))
        # Convert wavelengths to the nearest spectrum array indexes.
        else:
            pix_min = max(0, self.wave.pixel(lmin, nearest=True, unit=unit))
            pix_max = min(self.shape[0],
                          self.wave.pixel(lmax, nearest=True, unit=unit) + 1)

        # If the start and end of the wavelength range select the same pixel,
        # return just the value of that pixel.
        if (pix_min + 1) == pix_max:
            return self[pix_min]
        # Otherwise return a sub-spectrum.
        else:
            return self[pix_min:pix_max]

    def get_step(self, unit=None):
        """Return the wavelength step size.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The units of the returned step-size.

        Returns
        -------
        out : float
            The width of a spectrum pixel.
        """
        if self.wave is not None:
            return self.wave.get_step(unit)

    def get_start(self, unit=None):
        """Return the wavelength value of the first pixel of the spectrum.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The units of the returned wavelength.

        Returns
        -------
        out : float
            The wavelength of the first pixel of the spectrum.
        """
        if self.wave is not None:
            return self.wave.get_start(unit)

    def get_end(self, unit=None):
        """Return the wavelength of the last pixel of the spectrum.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The units of the returned wavelength.

        Returns
        -------
        out : float
            The wavelength of the final pixel of the spectrum.
        """
        if self.wave is not None:
            return self.wave.get_end(unit)

    def get_range(self, unit=None):
        """Return the wavelength range (Lambda_min, Lambda_max) of the spectrum.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            The units of the returned wavelengths.

        Returns
        -------
        out : float array
            The minimum and maximum wavelengths.
        """
        if self.wave is not None:
            return self.wave.get_range(unit)

    def mask_region(self, lmin=None, lmax=None, inside=True, unit=u.angstrom):
        """Mask spectrum pixels inside or outside a wavelength range, [lmin,lmax].

        Parameters
        ----------
        lmin : float
            The minimum wavelength of the range, or None to choose the
            wavelength of the first pixel in the spectrum.
        lmax : float
            The maximum wavelength of the range, or None to choose the
            wavelength of the last pixel in the spectrum.
        unit : `astropy.units.Unit`
            The wavelength units of lmin and lmax. If None, lmin and
            lmax are assumed to be pixel indexes.
        inside : bool
            If True, pixels inside the range [lmin,lmax] are masked.
            If False, pixels outside the range [lmin,lmax] are masked.

        """
        if self.wave is None:
            raise ValueError('Operation forbidden without world coordinates '
                             'along the spectral direction')
        else:
            if lmin is None:
                pix_min = 0
            else:
                if unit is None:
                    pix_min = max(0, int(lmin + 0.5))
                else:
                    pix_min = max(0, self.wave.pixel(lmin, nearest=True,
                                                     unit=unit))
            if lmax is None:
                pix_max = self.shape[0]
            else:
                if unit is None:
                    pix_max = min(self.shape[0], int(lmax + 0.5))
                else:
                    pix_max = min(self.shape[0],
                                  self.wave.pixel(lmax, nearest=True,
                                                  unit=unit) + 1)

            if inside:
                self.data[pix_min:pix_max] = np.ma.masked
            else:
                self.data[:pix_min] = np.ma.masked
                self.data[pix_max + 1:] = np.ma.masked

    def _wavelengths_to_slice(self, lmin, lmax, unit):
        """Return the slice that selects a specified wavelength range.

        Parameters
        ----------
        lmin : float
            The minimum wavelength of a wavelength range, or the wavelength
            of a single pixel if lmax is None.
        lmax : float or None
            The maximum wavelength of the wavelength range.
        unit : `astropy.units.Unit`
            The wavelength units of the lmin and lmax arguments. The
            default is angstroms. If unit is None, then lmin and lmax
            are interpreted as array indexes within the spectrum.

        Returns
        -------
        out : slice
            The slice needed to select pixels within the specified wavelength
            range.
        """

        if unit is not None and self.wave is None:
            raise ValueError('Operation forbidden without world coordinates '
                             'along the spectral direction')

        # Get the pixel index that corresponds to the minimum wavelength.
        if lmin is None:
            i1 = 0
        else:
            if unit is None:
                if lmin > self.shape[0]:
                    raise ValueError('Minimum and maximum wavelengths '
                                     'are outside the spectrum range')
                i1 = max(0, int(lmin + 0.5))
            else:
                i1 = self.wave.pixel(lmin, nearest=False, unit=unit)
                if i1 > self.shape[0]:
                    raise ValueError('Minimum and maximum wavelengths '
                                     'are outside the spectrum range')
                i1 = self.wave.pixel(lmin, nearest=True, unit=unit)

        # Get the pixel index that corresponds to the maximum wavelength.
        if lmax is None:
            i2 = self.shape[0]
        else:
            if unit is None:
                if lmax < 0:
                    raise ValueError('Minimum and maximum wavelengths '
                                     'are outside the spectrum range')
                i2 = min(self.shape[0], int(lmax + 0.5))
            else:
                i2 = self.wave.pixel(lmax, nearest=False, unit=unit)
                if i2 < 0:
                    raise ValueError('Minimum and maximum wavelengths '
                                     'are outside the spectrum range')
                i2 = self.wave.pixel(lmax, nearest=True, unit=unit) + 1

        return slice(i1, i2)

    def _interp(self, wavelengths, spline=False):
        """return the interpolated values corresponding to the wavelength
        array.

        Parameters
        ----------
        wavelengths : array of float
            wavelength values
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates
        spline : bool
            False: linear interpolation (use `scipy.interpolate.interp1d`),
            True: spline interpolation (use `scipy.interpolate.splrep`
            and `scipy.interpolate.splev`).

        """
        lbda = self.wave.coord()
        data = np.pad(self.data.compressed(), 1, 'edge')
        w = np.concatenate(([self.get_start() - 0.5 * self.get_step()],
                            np.compress(~self._mask, lbda),
                            [self.get_end() + 0.5 * self.get_step()]))

        if spline:
            if self._var is not None:
                _weight = 1. / np.sqrt(np.abs(self.var.filled(np.inf)))
                if self.mask is np.ma.nomask:
                    weight = np.empty(self.shape + 2, dtype=float)
                    weight[1:-1] = _weight
                else:
                    ksel = np.where(self.mask == False)
                    weight = np.empty(np.shape(ksel)[1] + 2)
                    weight[1:-1] = _weight[ksel]
                weight[0] = weight[1]
                weight[-1] = weight[-2]
            else:
                weight = None
            tck = interpolate.splrep(w, data, w=weight)
            return interpolate.splev(wavelengths, tck, der=0)
        else:
            f = interpolate.interp1d(w, data)
            return f(wavelengths)

    def _interp_data(self, spline=False):
        """Return data array with interpolated values for masked pixels.

        Parameters
        ----------
        spline : bool
            False: linear interpolation (use `scipy.interpolate.interp1d`),
            True: spline interpolation (use `scipy.interpolate.splrep`
            and `scipy.interpolate.splev`).

        """
        if np.count_nonzero(self._mask) in (0, self.shape[0]):
            return self._data

        lbda = self.wave.coord()
        wnew = lbda[self._mask]
        data = self._data.copy()
        data[self._mask] = self._interp(wnew, spline)
        return data

    def interp_mask(self, spline=False):
        """Interpolate masked pixels.

        Parameters
        ----------
        spline : bool
            False: linear interpolation (use `scipy.interpolate.interp1d`),
            True: spline interpolation (use `scipy.interpolate.splrep`
            and `scipy.interpolate.splev`).

        """
        self.data = np.ma.masked_invalid(self._interp_data(spline))

    def rebin(self, factor, margin='center', inplace=False):
        """Combine neighboring pixels to reduce the size of a spectrum by an
        integer factor.

        Each output pixel is the mean of n pixels, where n is the
        specified reduction factor.

        Parameters
        ----------
        factor : int
            The integer reduction factor by which the spectrum should
            be shrunk.
        margin : string in 'center'|'right'|'left'|'origin'
            When the dimension of the input spectrum is not an integer
            multiple of the reduction factor, the spectrum is
            truncated to remove just enough pixels that its length is
            a multiple of the reduction factor. This sub-spectrum is
            then rebinned in place of the original spectrum. The
            margin parameter determines which pixels of the input
            spectrum are truncated, and which remain.

            The options are:
              'origin' or 'center':
                 The start of the output spectrum is coincident
                 with the start of the input spectrum.
              'center':
                 The center of the output spectrum is aligned
                 with the center of the input spectrum, within
                 one pixel.
              'right':
                 The end of the output spectrum is coincident
                 with the end of the input spectrum.
        inplace : bool
            If False, return a rebinned copy of the spectrum (the default).
            If True, rebin the original spectrum in-place, and return that.

        Returns
        -------
        out : Spectrum

        """
        # Delegate the rebinning to the generic DataArray function.
        return self._rebin(factor, margin, inplace)

    def _decimation_filter(self, newstep, atten, unit=None):
        """This is a private function Spectrum.resample(), used to apply
        a decimation filter prior to resampling.

        Parameters
        ----------
        step : float
            The new pixel size along the wavelength axis of the spectrum.
        atten : float
            The minimum attenuation (dB), of the antialiasing
            decimation filter at the Nyquist folding frequency of the
            new pixel size. Larger attenuations suppress aliasing
            better at the expense of worsened resolution. A good value
            to choose is 40dB, which produces a response that is very
            similar to a blackman filter applied within the Fourier
            plane, but with less ringing in the image plane.
        unit : `astropy.units.Unit`
            The wavelength units of the step argument. A value of None
            is equivalent to specifying self.wave.unit.

        """
        # Convert the attenuation from dB to a linear scale factor.
        gcut = 10.0**(-atten / 20.0)

        # Calculate the Nyquist folding frequency of the new pixel size.
        nyquist_folding_freq = 0.5 / newstep

        # Calculate the standard deviation of a Gaussian whose Fourier
        # transform drops from unity at the center to gcut at the Nyquist
        # folding frequency.
        sigma = (0.5 / np.pi / nyquist_folding_freq *
                 np.sqrt(-2.0 * np.log(gcut)))

        # Convert the standard deviation from wavelength units to input pixels.
        sigma /= self.get_step(unit=unit)

        # Choose dimensions for the gaussian filtering kernel. Choose an
        # extent from -4*sigma to +4*sigma. This truncates the gaussian
        # where it drops to about 3e-4 of its peak.  The following
        # calculation ensures that the dimensions of the array are odd, so
        # that the gaussian will be symmetrically sampled either side of a
        # central pixel. This prevents spectral shifts.
        gshape = int(np.ceil(4.0 * sigma)) * 2 + 1

        # fftconvolve requires that the kernel be no larger than the array
        # that it is convolving, so reduce the size of the kernel array if
        # needed. Be careful to choose an odd sized array.
        n = self.shape[0]
        if gshape > n:
            gshape = n if n % 2 != 0 else (n - 1)

        # Sample the gaussian filter symmetrically around the central pixel.
        gx = np.arange(gshape, dtype=float) - gshape // 2
        gy = np.exp(-0.5 * (gx / sigma)**2)

        # Area-normalize the gaussian profile.
        gy /= gy.sum()

        # Filter the spectrum with the gaussian filter.
        self.fftconvolve(gy, inplace=True)

    def resample(self, step, start=None, shape=None, unit=u.angstrom,
                 inplace=False, atten=40.0, cutoff=0.25):
        """Resample a spectrum to have a different wavelength interval.

        Parameters
        ----------
        step : float
            The new pixel size along the wavelength axis of the spectrum.
        start : float
            The wavelength at the center of the first pixel of the resampled
            spectrum.  If None (the default) the center of the first pixel
            has the same wavelength before and after resampling.
        unit : `astropy.units.Unit`
            The wavelength units of the step and start arguments.
            The default is u.angstrom.
        shape : int
            The dimension of the array of the new spectrum (ie. the number
            of spectral pixels). If this is not specified, the shape is
            selected to encompass the wavelength range from the chosen
            start wavelength to the ending wavelength of the input spectrum.
        inplace : bool
            If False, return a resampled copy of the spectrum (the default).
            If True, resample the original spectrum in-place, and return that.
        atten : float
            The minimum attenuation (dB), of the antialiasing
            decimation filter at the Nyquist folding frequency of the
            new pixel size. Larger attenuations suppress aliasing
            better at the expense of worsened resolution. The default
            attenuation is 40.0 dB. To disable antialiasing, specify
            atten=0.0.
        cutoff : float
            Mask each output pixel of which at least this fraction of the
            pixel was interpolated from masked input pixels.

        Returns
        -------
        out : Spectrum

        """
        out = self if inplace else self.copy()

        # Don't allow the spectrum to be started beyond the far end of
        # the spectrum, because this would result in an empty spectrum.
        if start is not None and start > self.get_end(unit):
            raise ValueError('The start value is past the end of the '
                             'spectrum range')

        # Get wavelength world coordinates of the output spectrum.
        newwave = self.wave.resample(step, start, unit)

        # How many pixels should there be in the resampled spectrum?
        # If the user didn't specify this, use newwave.shape, which
        # holds the number of pixels of size 'step' needed to sample
        # from 'start' to the end of the current wavelength range.
        if shape is not None:
            newwave.shape = shape

        # Get the existing wavelength step size in the new units.
        oldstep = self.wave.get_step(unit)

        # If the spectrum is being resampled to a larger pixel size,
        # then a decimation filter should be applied before
        # resampling, to ensure that the new pixel size doesn't
        # undersample rapidly changing features in the spectrum.

        if step > oldstep and atten > 0.0:
            out._decimation_filter(step, atten, unit=unit)

        # Get the data, mask (and variance) arrays, and replace bad pixels with
        # zeros.
        if out._mask is not None:         # Is out.data a masked array?
            data = out.data.filled(0.0)
            if out._var is not None:
                var = out.var.filled(0.0)
            else:
                var = None
            mask = out._mask
        else:                             # Is out.data just a numpy array?
            mask = ~np.isfinite(out._data)
            data = out._data.copy()
            data[mask] = 0.0
            if out.var is not None:
                var = out.var.copy()
                var[mask] = 0.0
            else:
                var = None

        # Get the coordinates of the pixels of the input and output spectra.
        xi = self.wave.coord()
        xo = newwave.coord()

        # Get a resampled versions of the data array, optionally the variance
        # array, and a floating point version of the mask array. Note that the
        # choice of linear interpolation is required to preserve flux.
        data = interpolate.griddata(xi, data, xo, method="linear",
                                    fill_value=np.nan)
        if var is not None:
            var = interpolate.griddata(xi, var, xo, method="linear",
                                       fill_value=np.nan)
        mask = interpolate.griddata(xi, mask.astype(float), xo,
                                    method="linear", fill_value=1.0)

        # Create a new boolean mask in which all pixels that had an integrated
        # contribution of more than 'cutoff' originally masked pixels are
        # masked. Note that setting the cutoff to the "obvious" value of zero
        # results in lots of pixels being masked that are far away from any
        # masked pixels, due to precision errors in the griddata()
        # function.  Limit the minimum value of the cutoff to avoid this.
        mask = np.greater(mask, max(cutoff, 1.0e-6))

        # If masked arrays were not in use in the original spectrum, fill
        # bad pixels with NaNs.
        if out._mask is None:
            data[mask] = np.nan
            if var is not None:
                var[mask] = np.nan
            mask = None

        # Install the resampled arrays.
        out._data = data
        out._var = var
        out._mask = mask

        # Install the new wavelength world coordinates.
        out.wave = newwave

        # When up-sampling, decimation filter the output spectrum. The
        # combination of this and the linear interpolation of the preceding
        # griddata() produces a much better interpolation than a cubic spline
        # filter can. In particular, a spline interpolation does not conserve
        # flux, whereas linear interpolation plus decimation filtering does.
        if step < oldstep and atten > 0.0:
            out._decimation_filter(step, atten, unit=unit)

        return out

    def mean(self, lmin=None, lmax=None, weight=True, unit=u.angstrom):
        """Compute the mean flux over a specified wavelength range.

        Parameters
        ----------
        lmin : float
            The minimum wavelength of the range, or None to choose the
            wavelength of the first pixel in the spectrum.
        lmax : float
            The maximum wavelength of the range, or None to choose the
            wavelength of the last pixel in the spectrum.
        unit : `astropy.units.Unit`
            The wavelength units of lmin and lmax. If None, lmin and
            lmax are assumed to be pixel indexes.
        weight : bool
            If weight is True, compute the weighted mean, inversely
            weighting each pixel by its variance.

        Returns
        -------
        out : (float, float)
            The mean flux and its error.

        """

        # Don't attempt to perform a weighted mean if there are no variances.
        if self._var is None:
            weight = False

        # Get the slice that selects the specified wavelength range.
        try:
            lambda_slice = self._wavelengths_to_slice(lmin, lmax, unit)
        except ValueError:
            return (0.0, np.inf)

        # Obtain the mean flux of the sub-spectrum.
        if weight:
            weights = 1.0 / self.var[lambda_slice].filled(np.inf)
            flux, wsum = np.ma.average(self.data[lambda_slice],
                                       weights=weights, returned=True)
            if self.var is not None:
                err_flux = np.sqrt(
                    np.ma.sum(self.var[lambda_slice] * weights**2) / wsum**2)
            else:
                err_flux = np.inf
        else:
            flux, wsum = np.ma.average(self.data[lambda_slice], returned=True)
            if self.var is not None:
                err_flux = np.sqrt(np.ma.sum(self.var[lambda_slice])) / wsum**2
            else:
                err_flux = np.inf
        return (flux, err_flux)

    def sum(self, lmin=None, lmax=None, weight=True, unit=u.angstrom):
        """Obtain the sum of the fluxes within a specified wavelength range.

        Parameters
        ----------
        lmin : float
            The minimum wavelength of the range, or None to choose the
            wavelength of the first pixel in the spectrum.
        lmax : float
            The maximum wavelength of the range, or None to choose the
            wavelength of the last pixel in the spectrum.
        unit : `astropy.units.Unit`
            The wavelength units of lmin and lmax. If None, lmin and
            lmax are assumed to be pixel indexes.
        weight : bool
            If weight is True, compute the weighted sum, inversely
            weighting each pixel by its variance.

        Returns
        -------
        out : float, float
            The total flux and its error.
        """

        # Get the slice that selects the specified wavelength range.
        try:
            lambda_slice = self._wavelengths_to_slice(lmin, lmax, unit)
        except ValueError:
            return (0.0, np.inf)

        # Perform a weighted sum?
        if weight and self._var is not None:
            weights = 1.0 / self.var[lambda_slice].filled(np.inf)

            # How many unmasked pixels will be averaged?
            nsum = np.ma.count(self.data[lambda_slice])

            fmean, wsum = np.ma.average(self.data[lambda_slice],
                                        weights=weights, returned=True)

            # The weighted average multiplied by the number of unmasked pixels.
            flux = fmean * nsum

            if self.var is not None:
                err_flux = np.sqrt(
                    np.ma.sum(self.var[lambda_slice] * weights**2) /
                    wsum**2 * nsum**2)
            else:
                err_flux = np.inf
        else:
            flux = self.data[lambda_slice].sum()
            if self.var is not None:
                err_flux = np.sqrt(np.ma.sum(self.var[lambda_slice]))
            else:
                err_flux = np.inf
        return (flux, err_flux)

    def integrate(self, lmin=None, lmax=None, unit=u.angstrom):
        """Integrate the flux over a specified wavelength range.

        The units of the integrated flux depend on the flux units of
        the spectrum and the wavelength units, as follows:

        If the flux units of the spectrum, self.unit, are something
        like Q per angstrom, Q per nm, or Q per um, then the
        integrated flux will have the units of Q. For example, if the
        fluxes have units of 1e-20 erg/cm2/Angstrom/s, then the units
        of the integration will be 1e-20 erg/cm2/s.

        Alternatively, if unit is not None, then the unit of the
        returned number will be the product of the units in self.unit
        and unit. For example, if the flux units are counts/s, and
        unit=u.angstrom, then the integrated flux will have units
        counts*Angstrom/s.

        Finally, if unit is None, then the units of the returned
        number will be the product of self.unit and the units of the
        wavelength axis of the spectrum (ie. self.wave.unit).

        The result of the integration is returned as an astropy
        Quantity, which holds the integrated value and its physical
        units.  The units of the returned number can be determined
        from the .unit attribute of the return value. Alternatively
        the returned value can be converted to another unit, using the
        to() method of astropy quantities.

        Parameters
        ----------
        lmin : float
            The minimum wavelength of the range to be integrated,
            or None (the default), to select the minimum wavelength
            of the first pixel of the spectrum. If this is below the
            minimum wavelength of the spectrum, the integration
            behaves as though the flux in the first pixel extended
            down to that wavelength.

            If the unit argument is None, lmin is a pixel index, and
            the wavelength of the center of this pixel is used as the
            lower wavelength of the integration.
        lmax : float
            The maximum wavelength of the range to be integrated,
            or None (the default), to select the maximum wavelength
            of the last pixel of the spectrum. If this is above the
            maximum wavelength of the spectrum, the integration
            behaves as though the flux in the last pixel extended
            up to that wavelength.

            If the unit argument is None, lmax is a pixel index, and
            the wavelength of the center of this pixel is used as the
            upper wavelength of the integration.
        unit : `astropy.units.Unit`
            The wavelength units of lmin and lmax, or None to indicate
            that lmin and lmax are pixel indexes.

        Returns
        -------
        out : `astropy.units.Quantity`, `astropy.units.Quantity`
            The result of the integration and its error, expressed as
            a floating point number with accompanying units. The integrated
            value and its physical units can be extracted using the .value and
            .unit attributes of the returned quantity. The value can also be
            converted to different units, using the .to() method of the
            returned objected.

        """
        # Get the index of the first pixel within the wavelength range,
        # and the minimum wavelength of the integration.
        if lmin is None:
            i1 = 0
            lmin = self.wave.coord(-0.5, unit=unit)
        else:
            if unit is None:
                l1 = lmin
                lmin = self.wave.coord(max(-0.5, l1))
            else:
                l1 = self.wave.pixel(lmin, False, unit)
            i1 = max(0, int(l1))

        # Get the index of the last pixel within the wavelength range, plus
        # 1, and the maximum wavelength of the integration.
        if lmax is None:
            i2 = self.shape[0]
            lmax = self.wave.coord(i2 - 0.5, unit=unit)
        else:
            if unit is None:
                l2 = lmax
                lmax = self.wave.coord(min(self.shape[0] - 0.5, l2))
            else:
                l2 = self.wave.pixel(lmax, False, unit)
            i2 = min(self.shape[0], int(l2) + 1)

        # Get the lower wavelength of each pixel, including one extra
        # pixel at the end of the range.
        d = self.wave.coord(-0.5 + np.arange(i1, i2 + 1), unit=unit)

        # Change the wavelengths of the first and last pixels to
        # truncate or extend those pixels to the starting and ending
        # wavelengths of the spectrum.
        d[0] = lmin
        d[-1] = lmax

        if unit is None:
            unit = self.wave.unit

        # Get the data of the subspectrum covered by the integration.
        data = self.data[i1:i2]

        # If the spectrum has been calibrated, the flux units will be
        # per angstrom, per nm, per um etc. If these wavelength units
        # don't match the units of the wavelength axis of the
        # integration, then although the results will be correct, they
        # will have inconvenient units. In such cases attempt to
        # convert the units of the wavelength axis to match the flux
        # units.
        if unit in self.unit.bases:      # The wavelength units already agree.
            out_unit = self.unit * unit
        else:
            try:
                # Attempt to determine the wavelength units of the flux density
                wunit = (set(self.unit.bases) &
                         set([u.pm, u.angstrom, u.nm, u.um])).pop()

                # Scale the wavelength axis to have the same wavelength units.
                d *= unit.to(wunit)

                # Get the final units of the integration.
                out_unit = self.unit * wunit

            # If the wavelength units of the flux weren't recognized,
            # simply return the units unchanged.
            except:
                out_unit = self.unit * unit

        # Integrate the spectrum by multiplying the value of each pixel
        # by the difference in wavelength from the start of that pixel to
        # the start of the next pixel.
        flux = (data * np.diff(d)).sum() * out_unit
        if self.var is None:
            err_flux = np.inf * out_unit
        else:
            err_flux = (self.var[i1:i2] * np.diff(d**2)).sum() * out_unit
        return (flux, err_flux)

    def poly_fit(self, deg, weight=True, maxiter=0,
                 nsig=(-3.0, 3.0), verbose=False):
        """Perform polynomial fit on normalized spectrum and returns polynomial
        coefficients.

        Parameters
        ----------
        deg : int
            Polynomial degree.
        weight : bool
            If weight is True, the weight is computed as the inverse of
            variance.
        maxiter : int
            Maximum allowed iterations (0)
        nsig : (float,float)
            The low and high rejection factor in std units (-3.0,3.0)

        Returns
        -------
        out : ndarray, shape.
              Polynomial coefficients ordered from low to high.
        """
        if self.shape[0] <= deg + 1:
            raise ValueError('Too few points to perform polynomial fit')

        if self._var is None:
            weight = False

        if weight:
            vec_weight = 1.0 / np.sqrt(np.abs(self.var.filled(np.inf)))
        else:
            vec_weight = None

        if self._mask is np.ma.nomask:
            d = self._data
            w = self.wave.coord()
        else:
            mask = ~self._mask
            d = self._data[mask]
            w = self.wave.coord()[mask]
            if weight:
                vec_weight = vec_weight[mask]

        # normalize w
        w0 = np.min(w)
        dw = np.max(w) - w0
        w = (w - w0) / dw

        p = np.polynomial.polynomial.polyfit(w, d, deg, w=vec_weight)

        if maxiter > 0:
            err = d - np.polynomial.polynomial.polyval(w, p)
            sig = np.std(err)
            n_p = len(d)
            for it in range(maxiter):
                ind = np.where((err >= nsig[0] * sig) &
                               (np.abs(err) <= nsig[1] * sig))
                if len(ind[0]) == n_p:
                    break
                if len(ind[0]) <= deg + 1:
                    raise ValueError('Too few points to perform '
                                     'polynomial fit')
                if vec_weight is not None:
                    vec_weight = vec_weight[ind]
                p = np.polynomial.polynomial.polyfit(w[ind], d[ind],
                                                     deg, w=vec_weight)
                err = d[ind] - np.polynomial.polynomial.polyval(w[ind], p)
                sig = np.std(err)
                n_p = len(ind[0])

                if verbose:
                    self._logger.info('Number of iteration: %d Std: %10.4e '
                                      'Np: %d Frac: %4.2f', it + 1, sig, n_p,
                                      100. * n_p / self.shape[0])

        return p

    def poly_val(self, z):
        """Update in place the spectrum data from polynomial coefficients.

        Uses `numpy.poly1d`.

        Parameters
        ----------
        z : array
            The polynomial coefficients, in increasing powers:

                data = z0 + z1(lbda-min(lbda))/(max(lbda)-min(lbda)) + ...
                + zn ((lbda-min(lbda))/(max(lbda)-min(lbda)))**n

        """
        l = self.wave.coord()
        w0 = np.min(l)
        dw = np.max(l) - w0
        w = (l - w0) / dw
        self._data = np.polynomial.polynomial.polyval(w, z)
        if self._mask is not np.ma.nomask:
            self._mask = ~(np.isfinite(self._data))
        self._var = None

    def poly_spec(self, deg, weight=True, maxiter=0,
                  nsig=(-3.0, 3.0), verbose=False):
        """Return a spectrum containing a polynomial fit.

        Parameters
        ----------
        deg : int
            Polynomial degree.
        weight : bool
            If weight is True, the weight is computed as the inverse of
            variance.
        maxiter : int
            Maximum allowed iterations (0)
        nsig : (float,float)
            The low and high rejection factor in std units (-3.0,3.0)

        Returns
        -------
        out : Spectrum
        """
        z = self.poly_fit(deg, weight, maxiter, nsig, verbose)
        res = self.clone()
        res.poly_val(z)
        return res

    def abmag_band(self, lbda, dlbda):
        """Compute AB magnitude corresponding to the wavelength band.

        Parameters
        ----------
        lbda : float
            Mean wavelength in Angstrom.
        dlbda : float
            Width of the wavelength band in Angstrom.

        Returns
        -------
        out : float, float
              Magnitude value and its error
        """
        vflux, err_flux = self.mean(lmin=lbda - dlbda / 2.0,
                                    lmax=lbda + dlbda / 2.0,
                                    weight=None, unit=u.angstrom)
        if vflux == 0:
            return (99, 0)
        else:
            unit = u.Unit('erg.s-1.cm-2.Angstrom-1')
            vflux2 = (vflux * self.unit).to(unit)
            err_flux2 = (err_flux * self.unit).to(unit)
            return flux2mag(vflux2.value, err_flux2.value, lbda)

    def abmag_filter_name(self, name):
        """Compute AB magnitude using the filter name.

        Parameters
        ----------
        name : string
            'U', 'B', 'V', 'Rc', 'Ic', 'z', 'R-Johnson',
            'F606W', 'F775W', 'F814W', 'F850LP'

        Returns
        -------
        out : float, float
              Magnitude value and its error
        """

        if name == 'U':
            return self.abmag_band(3663., 650.)
        elif name == 'B':
            return self.abmag_band(4361., 890.)
        elif name == 'V':
            return self.abmag_band(5448., 840.)
        elif name == 'Rc':
            return self.abmag_band(6410., 1600.)
        elif name == 'Ic':
            return self.abmag_band(7980., 1500.)
        elif name == 'z':
            return self.abmag_band(8930., 1470.)
        elif name == 'R-Johnson':
            (l0, lmin, lmax, tck) = ABmag_filters.mag_RJohnson()
            return self._filter(l0, lmin, lmax, tck)
        else:
            FILTERS = join(abspath(dirname(__file__)), 'filters',
                           'filter_list.fits')
            filtname = 'ACS_' + name

            with fits.open(FILTERS) as hdul:
                if filtname not in hdul:
                    raise ValueError("filter '{}' not found".format(filtname))
                lbda = hdul[filtname].data['lambda']
                thr = hdul[filtname].data['throughput']

            return self.abmag_filter(lbda, thr)

    def abmag_filter(self, lbda, eff):
        """Compute AB magnitude using array filter.

        Parameters
        ----------
        lbda : float array
            Wavelength values in Angstrom.
        eff : float array
            Efficiency values.

        Returns
        -------
        out : float, float
              Magnitude value and its error
        """
        lbda = np.asarray(lbda)
        eff = np.asarray(eff)
        if np.shape(lbda) != np.shape(eff):
            raise TypeError('lbda and eff inputs have not the same size.')
        l0 = np.average(lbda, weights=eff)
        lmin = lbda[0]
        lmax = lbda[-1]
        k = 3 if lbda.shape[0] > 3 else 1
        tck = interpolate.splrep(lbda, eff, k=k)
        return self._filter(l0, lmin, lmax, tck)

    def _filter(self, l0, lmin, lmax, tck):
        """Compute AB magnitude.

        Parameters
        ----------
        l0 : float
            Mean wavelength in Angstrom.
        lmin : float
            Minimum wavelength in Angstrom.
        lmax : float
            Maximum wavelength in Angstrom.
        tck : 3-tuple
            (t,c,k) contains the spline representation. t = the knot-points,
            c = coefficients and k = the order of the spline.
        """
        try:
            lambda_slice = self._wavelengths_to_slice(lmin, lmax, u.Angstrom)
        except ValueError:
            raise ValueError('Spectrum outside Filter band')

        if lambda_slice.start == (lambda_slice.stop - 1):
            raise ValueError('Filter band smaller than spectrum step')

        lb = self.wave.coord(np.arange(lambda_slice.start, lambda_slice.stop),
                             unit=u.Angstrom)
        w = interpolate.splev(lb, tck, der=0)

        vflux, wsum = np.ma.average(self.data[lambda_slice], weights=w,
                                    returned=True)
        if self.var is not None:
            err_flux = np.sqrt(np.ma.sum(self.var[lambda_slice] * w**2) /
                               wsum**2)
        else:
            err_flux = np.inf

        unit = u.Unit('erg.s-1.cm-2.Angstrom-1')
        vflux2 = (vflux * self.unit).to(unit)
        err_flux2 = (err_flux * self.unit).to(unit)
        return flux2mag(vflux2.value, err_flux2.value, l0)

    def wavelet_filter(self, levels=9, sigmaCutoff=5.0, epsilon=0.05,
                       inplace=False):
        """Perform a wavelet filtering on the spectrum in 1 dimension.

        Code contributed by Markus Rexroth (EPFL, 2016), and used in
        https://arxiv.org/abs/1703.09239 (with funding from ERC Advanced
        Grant LIDA, Swiss National Science Foundation, ERC Starting Grant
        336736-CALENDS).

        Parameters
        ----------
        levels : int
            Highest  scale level.
        sigmaCutoff : float
            Cleaning threshold.
            By default 5 for a 5 sigma cleaning in wavelet space.
        epsilon : float in ]0,1[
            Residual criterion used to perform the cleaning
        inplace : bool
            If False, return a filtered copy of the spectrum (the default).
            If True, filter the original spectrum in-place, and return that.

        Returns
        -------
        out : Spectrum

        """
        res = self if inplace else self.copy()
        deNoisedSignal = wavelet1D.cleanSignal(self._data, np.sqrt(self._var),
                                               levels, sigmaCutoff=sigmaCutoff,
                                               epsilon=epsilon)
        res._data = deNoisedSignal
        res._var = None
        return res

    def truncate(self, lmin=None, lmax=None, unit=u.angstrom):
        """Truncate the wavelength range of a spectrum in-place.

        Parameters
        ----------
        lmin : float
            The minimum wavelength of a wavelength range, or the wavelength
            of a single pixel if lmax is None.
        lmax : float or None
            The maximum wavelength of the wavelength range.
        unit : `astropy.units.Unit`
            The wavelength units of the lmin and lmax arguments. The
            default is angstroms. If unit is None, then lmin and lmax
            are interpreted as array indexes within the spectrum.

        """

        # Get the slice that selects the specified wavelength range.
        lambda_slice = self._wavelengths_to_slice(lmin, lmax, unit)

        if lambda_slice.start == (lambda_slice.stop - 1):
            raise ValueError('Minimum and maximum wavelengths are equal')

        res = self[lambda_slice]
        self._data = res._data
        self._var = res._var
        self._mask = res._mask
        self.wave = res.wave

    def fwhm(self, l0, cont=0, spline=False, unit=u.angstrom):
        """Return the fwhm of a peak.

        Parameters
        ----------
        l0 : float
            Wavelength value corresponding to the peak position.
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        cont : int
            The continuum [default 0].
        spline : bool
            Linear/spline interpolation to interpolate masked values.

        Returns
        -------
        out : float
        """
        if unit is None:
            k0 = int(l0 + 0.5)
            step = 1
        else:
            k0 = self.wave.pixel(l0, nearest=True, unit=unit)
            step = self.wave.get_step(unit=unit)
        d = self._interp_data(spline) - cont
        f2 = d[k0] / 2
        try:
            k2 = np.argwhere(d[k0:-1] < f2)[0][0] + k0
            i2 = np.interp(f2, d[k2:k2 - 2:-1], [k2, k2 - 1])
            k1 = k0 - np.argwhere(d[k0:-1] < f2)[0][0]
            i1 = np.interp(f2, d[k1:k1 + 2], [k1, k1 + 1])
            fwhm = (i2 - i1) * step
            return fwhm
        except:
            try:
                k2 = np.argwhere(d[k0:-1] > f2)[0][0] + k0
                i2 = np.interp(f2, d[k2:k2 - 2:-1], [k2, k2 - 1])
                k1 = k0 - np.argwhere(d[k0:-1] > f2)[0][0]
                i1 = np.interp(f2, d[k1:k1 + 2], [k1, k1 + 1])
                fwhm = (i2 - i1) * step
                return fwhm
            except:
                raise ValueError('Error in fwhm estimation')

    def gauss_fit(self, lmin, lmax, lpeak=None, flux=None, fwhm=None,
                  cont=None, peak=False, spline=False, weight=True,
                  plot=False, plot_factor=10, unit=u.angstrom,
                  fix_lpeak=False):
        """Perform a Gaussian fit.

        Uses `scipy.optimize.leastsq` to minimize the sum of squares.

        Parameters
        ----------
        lmin : float or (float,float)
            Minimum wavelength value or wavelength range
            used to initialize the gaussian left value (in angstrom)
        lmax : float or (float,float)
            Maximum wavelength or wavelength range
            used to initialize the gaussian right value (in angstrom)
        lpeak : float
            Input gaussian center (in angstrom), if None it is estimated
            with the wavelength corresponding to the maximum value
            in [max(lmin), min(lmax)]
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        flux : float
            Integrated gaussian flux or gaussian peak value if peak is True.
        fwhm : float
            Input gaussian fwhm (in angstrom), if None it is estimated.
        peak : bool
            If true, flux contains the gaussian peak value .
        cont : float
            Continuum value, if None it is estimated by the line through points
            (max(lmin),mean(data[lmin])) and (min(lmax),mean(data[lmax])).
        spline : bool
            Linear/spline interpolation to interpolate masked values.
        weight : bool
            If weight is True, the weight is computed as the inverse of
            variance.
        plot : bool
            If True, the Gaussian is plotted.
        plot_factor : double
            oversampling factor for the overplotted fit

        Returns
        -------
        out : `mpdaf.obj.Gauss1D`
        """
        # truncate the spectrum and compute right and left gaussian values
        if np.isscalar(lmin):
            fmin = None
        else:
            lmin = np.array(lmin, dtype=float)
            fmin = self.mean(lmin[0], lmin[1], unit=unit)[0]
            lmin = (lmin[0] + lmin[1]) / 2.

        if np.isscalar(lmax):
            fmax = None
        else:
            lmax = np.array(lmax, dtype=float)
            fmax = self.mean(lmax[0], lmax[1], unit=unit)[0]
            lmax = (lmax[0] + lmax[1]) / 2.

        spec = self.subspec(lmin, lmax, unit=unit)
        data = spec._interp_data(spline)
        if unit is None:
            l = np.arange(self.shape, dtype=float)
        else:
            l = spec.wave.coord(unit=unit)

        lmin = l[0]
        lmax = l[-1]
        if fmin is None:
            fmin = data[0]
        if fmax is None:
            fmax = data[-1]

        # initial gaussian peak position
        if lpeak is None:
            lpeak = l[data.argmax()]

        # continuum value
        if cont is None:
            cont0 = ((fmax - fmin) * lpeak + lmax *
                     fmin - lmin * fmax) / (lmax - lmin)
        else:
            cont0 = cont

        # initial sigma value
        if fwhm is None:
            try:
                fwhm = spec.fwhm(lpeak, cont0, spline, unit=unit)
            except:
                lpeak2 = l[data.argmin()]
                fwhm = spec.fwhm(lpeak2, cont0, spline, unit=unit)
        sigma = fwhm * gaussian_fwhm_to_sigma

        # initial gaussian integrated flux
        if flux is None:
            if unit is None:
                pixel = int(lpeak + 0.5)
            else:
                pixel = spec.wave.pixel(lpeak, nearest=True, unit=unit)
            peak = data[pixel] - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        elif peak is True:
            peak = flux - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        else:
            pass

        # 1d gaussian function: p = (ampl, sigma, center)
        if fix_lpeak:
            if cont is None:
                gaussfit = lambda p, x: \
                    ((fmax - fmin) * x + lmax * fmin - lmin * fmax) \
                    / (lmax - lmin) + np.abs(p[0]) \
                    * (1 / np.sqrt(2 * np.pi * (p[1] ** 2))) \
                    * np.exp(-(x - lpeak) ** 2 / (2 * p[1] ** 2))
            else:
                gaussfit = lambda p, x: \
                    cont + p[0] * (1 / np.sqrt(2 * np.pi * (p[1] ** 2))) \
                    * np.exp(-(x - lpeak) ** 2 / (2 * p[1] ** 2))
        else:
            if cont is None:
                gaussfit = lambda p, x: \
                    ((fmax - fmin) * x + lmax * fmin - lmin * fmax) \
                    / (lmax - lmin) + p[0] \
                    * (1 / np.sqrt(2 * np.pi * (p[1] ** 2))) \
                    * np.exp(-(x - p[2]) ** 2 / (2 * p[1] ** 2))
            else:
                gaussfit = lambda p, x: \
                    cont + p[0] * (1 / np.sqrt(2 * np.pi * (p[1] ** 2))) \
                    * np.exp(-(x - p[2]) ** 2 / (2 * p[1] ** 2))

        if spec.var is not None and weight:
            wght = 1.0 / np.sqrt(np.abs(spec.var))
            np.ma.fix_invalid(wght, copy=False, fill_value=0)
        else:
            wght = np.ones(spec.shape)

        # inital guesses for Gaussian Fit
        v0 = [flux, sigma]
        if not fix_lpeak:
            v0.append(lpeak)

        # Minimize the sum of squares
        e_gauss_fit = lambda p, x, y, w: w * (gaussfit(p, x) - y)
        v, covar, info, mesg, success = leastsq(e_gauss_fit, v0[:],
                                                args=(l, data, wght),
                                                maxfev=100000, full_output=1)

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i])) *
                            np.sqrt(np.abs(chisq / dof))
                            for i in range(len(v))])
        else:
            err = [np.nan] * len(v)

        if plot:
            xxx = np.arange(l[0], l[-1], (l[1] - l[0]) / plot_factor)
            ccc = gaussfit(v, xxx)
            plt.plot(xxx, ccc, 'r--')

        # return a Gauss1D object
        v = list(v)
        err = list(err)
        if not fix_lpeak:
            lpeak = v.pop()
            err_lpeak = err.pop()
        else:
            err_lpeak = 0

        flux, sigma = v[0], np.abs(v[1])
        fwhm = sigma * gaussian_sigma_to_fwhm
        peak = flux / np.sqrt(2 * np.pi * (sigma ** 2))

        err_flux, err_sigma = err
        err_fwhm = err_sigma * gaussian_sigma_to_fwhm
        err_peak = np.abs(1. / np.sqrt(2 * np.pi) *
                          (err_flux * sigma - flux * err_sigma) /
                          sigma / sigma)

        return Gauss1D(lpeak, peak, flux, fwhm, cont0, err_lpeak,
                       err_peak, err_flux, err_fwhm, chisq, dof)

    def add_gaussian(self, lpeak, flux, fwhm, cont=0, peak=False,
                     unit=u.angstrom):
        """Add a gaussian on spectrum in place.

        Parameters
        ----------
        lpeak : float
            Gaussian center.
        flux : float
            Integrated gaussian flux or gaussian peak value if peak is True.
        fwhm : float
            Gaussian fwhm.
        cont : float
            Continuum value.
        peak : bool
            If true, flux contains the gaussian peak value
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        """
        gauss = lambda p, x: cont \
            + p[0] * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) \
            * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))

        sigma = fwhm * gaussian_fwhm_to_sigma

        if peak is True:
            flux = flux * np.sqrt(2 * np.pi * (sigma ** 2))

        lmin = lpeak - 5 * sigma
        lmax = lpeak + 5 * sigma
        if unit is None:
            imin = int(lmin + 0.5)
            imax = int(lmax + 0.5)
        else:
            imin = self.wave.pixel(lmin, nearest=True, unit=unit)
            imax = self.wave.pixel(lmax, nearest=True, unit=unit)
        if imin == imax:
            if imin == 0 or imin == self.shape[0]:
                raise ValueError('Gaussian outside spectrum wavelength range')

        if unit is None:
            wave = np.arange(imin, imax, dtype=float)
        else:
            wave = self.wave.coord(unit=unit)[imin:imax]
        v = [flux, lpeak, sigma]

        self.data[imin:imax] += gauss(v, wave)

    def gauss_dfit(self, lmin, lmax, wratio, lpeak_1=None,
                   flux_1=None, fratio=1., fwhm=None, cont=None,
                   peak=False, spline=False, weight=True,
                   plot=False, plot_factor=10, unit=u.angstrom):
        """Truncate the spectrum and fit it as a sum of two gaussian functions.

        Returns the two gaussian functions as `mpdaf.obj.Gauss1D` objects.

        From Johan Richard and Vera Patricio.

        Parameters
        ----------
        lmin : float or (float,float)
            Minimum wavelength value or wavelength range
            used to initialize the gaussian left value.
        lmax : float or (float,float)
            Maximum wavelength or wavelength range
            used to initialize the gaussian right value.
        wratio : float
            Ratio between the two gaussian centers
        lpeak_1 : float
            Input gaussian center of the first gaussian. if None it is
            estimated with the wavelength corresponding to the maximum value
            in [max(lmin), min(lmax)]
        flux_1 : float
            Integrated gaussian flux or gaussian peak value if peak is True.
        fratio : float
            Ratio between the two integrated gaussian fluxes.
        fwhm : float
            Input gaussian fwhm, if None it is estimated.
        peak : bool
            If true, flux contains the gaussian peak value .
        cont : float
            Continuum value, if None it is estimated by the line through points
            (max(lmin),mean(data[lmin])) and (min(lmax),mean(data[lmax])).
        spline : bool
            Linear/spline interpolation to interpolate masked values.
        weight : bool
            If weight is True, the weight is computed as the inverse of
            variance.
        plot : bool
            If True, the resulted fit is plotted.
        plot_factor : double
            oversampling factor for the overplotted fit
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.

        Returns
        -------
        out : `mpdaf.obj.Gauss1D`, `mpdaf.obj.Gauss1D`
        """
        if np.isscalar(lmin):
            fmin = None
        else:
            lmin = np.array(lmin, dtype=float)
            fmin = self.mean(lmin[0], lmin[1], weight=False, unit=unit)[0]
            lmin = lmin[1]

        if np.isscalar(lmax):
            fmax = None
        else:
            lmax = np.array(lmax, dtype=float)
            fmax = self.mean(lmax[0], lmax[1], weight=False, unit=unit)[0]
            lmax = lmax[0]

        # spec = self.truncate(lmin, lmax)
        spec = self.subspec(lmin, lmax, unit=unit)
        data = spec._interp_data(spline)
        if unit is None:
            l = np.arange(self.shape, dtype=float)
        else:
            l = spec.wave.coord(unit=unit)

        lmin = l[0]
        lmax = l[-1]
        if fmin is None:
            fmin = data[0]
        if fmax is None:
            fmax = data[-1]

        # initial gaussian peak position
        if lpeak_1 is None:
            lpeak_1 = l[data.argmax()]

        # continuum value
        if cont is None:
            cont0 = ((fmax - fmin) * lpeak_1 + lmax *
                     fmin - lmin * fmax) / (lmax - lmin)
        else:
            cont0 = cont

        # initial sigma value
        if fwhm is None:
            try:
                fwhm = spec.fwhm(lpeak_1, cont0, spline, unit=unit)
            except:
                lpeak_1 = l[data.argmin()]
                fwhm = spec.fwhm(lpeak_1, cont0, spline, unit=unit)
        sigma = fwhm * gaussian_fwhm_to_sigma

        # initial gaussian integrated flux
        if flux_1 is None:
            if unit is None:
                pixel = int(lpeak_1 + 0.5)
            else:
                pixel = spec.wave.pixel(lpeak_1, nearest=True, unit=unit)
            peak_1 = data[pixel] - cont0
            flux_1 = peak_1 * np.sqrt(2 * np.pi * (sigma ** 2))
        elif peak is True:
            peak_1 = flux_1 - cont0
            flux_1 = peak_1 * np.sqrt(2 * np.pi * (sigma ** 2))
        else:
            pass

        flux_2 = fratio * flux_1

        # 1d gaussian function
        # p[0]: flux 1, p[1]:center 1, p[2]: fwhm, p[3] = peak 2
        gaussfit = lambda p, x: cont0 + \
            p[0] * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) * \
            np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2)) + \
            p[3] * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) * \
            np.exp(-(x - (p[1] * wratio)) ** 2 / (2 * p[2] ** 2))

        # 1d gaussian fit
        if spec.var is not None and weight:
            wght = 1.0 / np.sqrt(np.abs(spec.var))
            np.ma.fix_invalid(wght, copy=False, fill_value=0)
        else:
            wght = np.ones(spec.shape)

        e_gauss_fit = lambda p, x, y, w: w * (gaussfit(p, x) - y)

        # inital guesses for Gaussian Fit
        v0 = [flux_1, lpeak_1, sigma, flux_2]
        # Minimize the sum of squares
        v, covar, info, mesg, success = leastsq(
            e_gauss_fit, v0[:], args=(l, data, wght), maxfev=100000,
            full_output=1)

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i])) *
                            np.sqrt(np.abs(chisq / dof))
                            for i in range(len(v))])
        else:
            err = None

        # plot
        if plot:
            xxx = np.arange(l[0], l[-1], (l[1] - l[0]) / plot_factor)
            ccc = gaussfit(v, xxx)
            plt.plot(xxx, ccc, 'r--')

        # return a Gauss1D object
        flux_1 = v[0]
        flux_2 = v[3]
        lpeak_1 = v[1]
        lpeak_2 = lpeak_1 * wratio
        sigma = np.abs(v[2])
        fwhm = sigma * gaussian_sigma_to_fwhm
        peak_1 = flux_1 / np.sqrt(2 * np.pi * (sigma ** 2))
        peak_2 = flux_2 / np.sqrt(2 * np.pi * (sigma ** 2))
        if err is not None:
            err_flux_1 = err[0]
            err_flux_2 = err[3]
            err_lpeak_1 = err[1]
            err_lpeak_2 = err[1] * wratio
            err_sigma = err[2]
            err_fwhm = err_sigma * gaussian_sigma_to_fwhm
            err_peak_1 = np.abs(1. / np.sqrt(2 * np.pi) *
                                (err_flux_1 * sigma - flux_1 * err_sigma) /
                                sigma / sigma)
            err_peak_2 = np.abs(1. / np.sqrt(2 * np.pi) *
                                (err_flux_2 * sigma - flux_2 * err_sigma) /
                                sigma / sigma)
        else:
            err_flux_1 = np.NAN
            err_flux_2 = np.NAN
            err_lpeak_1 = np.NAN
            err_lpeak_2 = np.NAN
            err_sigma = np.NAN
            err_fwhm = np.NAN
            err_peak_1 = np.NAN
            err_peak_2 = np.NAN

        return (Gauss1D(lpeak_1, peak_1, flux_1, fwhm, cont0, err_lpeak_1,
                        err_peak_1, err_flux_1, err_fwhm, chisq, dof),
                Gauss1D(lpeak_2, peak_2, flux_2, fwhm, cont0, err_lpeak_2,
                        err_peak_2, err_flux_2, err_fwhm, chisq, dof))

    def gauss_asymfit(self, lmin, lmax, lpeak=None, flux=None, fwhm=None,
                      cont=None, peak=False, spline=False, weight=True,
                      plot=False, plot_factor=10, ax=None, unit=u.angstrom):
        """Truncate the spectrum and fit it with an asymetric gaussian
        function.

        Returns the two gaussian functions (right and left) as
        `mpdaf.obj.Gauss1D` objects.

        From Johan Richard and Vera Patricio, modified by Jeremy Blaizot.

        Parameters
        ----------
        lmin : float or (float,float)
            Minimum wavelength value or wavelength range
            used to initialize the gaussian left value.
        lmax : float or (float,float)
            Maximum wavelength or wavelength range
            used to initialize the gaussian right value.
        lpeak : float
            Input gaussian center. if None it is estimated with the wavelength
            corresponding to the maximum value in ``[max(lmin), min(lmax)]``.
        flux : float
            Integrated gaussian flux or gaussian peak value if peak is True.
        fwhm : float
            Input gaussian fwhm, if None it is estimated.
        peak : bool
            If true, flux contains the gaussian peak value .
        cont : float
            Continuum value, if None it is estimated by the line through points
            (max(lmin),mean(data[lmin])) and (min(lmax),mean(data[lmax])).
        spline : bool
            Linear/spline interpolation to interpolate masked values.
        weight : bool
            If weight is True, the weight is computed as the inverse of
            variance.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates. If None, inputs are in pixels.
        plot : bool
            If True, the resulted fit is plotted.
        plot_factor : double
            oversampling factor for the overplotted fit

        Returns
        -------
        out : `mpdaf.obj.Gauss1D`, `mpdaf.obj.Gauss1D`
            Left and right Gaussian functions.

        """
        if np.isscalar(lmin):
            fmin = None
        else:
            lmin = np.array(lmin, dtype=float)
            fmin = self.mean(lmin[0], lmin[1], weight=False, unit=unit)[0]
            lmin = lmin[1]

        if np.isscalar(lmax):
            fmax = None
        else:
            lmax = np.array(lmax, dtype=float)
            fmax = self.mean(lmax[0], lmax[1], weight=False, unit=unit)[0]
            lmax = lmax[0]

        spec = self.subspec(lmin, lmax, unit=unit)
        data = spec._interp_data(spline)
        if unit is None:
            l = np.arange(self.shape, dtype=float)
        else:
            l = spec.wave.coord(unit=unit)

        lmin = l[0]
        lmax = l[-1]
        if fmin is None:
            fmin = data[0]
        if fmax is None:
            fmax = data[-1]

        # initial gaussian peak position
        if lpeak is None:
            lpeak = l[data.argmax()]

        # continuum value
        if cont is None:
            cont0 = ((fmax - fmin) * lpeak + lmax * fmin -
                     lmin * fmax) / (lmax - lmin)
        else:
            cont0 = cont

        # initial sigma value
        if fwhm is None:
            try:
                fwhm = spec.fwhm(lpeak, cont0, spline, unit=unit)
            except:
                lpeak = l[data.argmin()]
                fwhm = spec.fwhm(lpeak, cont0, spline, unit=unit)
        sigma = fwhm * gaussian_fwhm_to_sigma

        # initial gaussian integrated flux
        if flux is None:
            if unit is None:
                pixel = int(lpeak + 0.5)
            else:
                pixel = spec.wave.pixel(lpeak, nearest=True, unit=unit)
            peak = data[pixel] - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        elif peak is True:
            peak = flux - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        else:
            pass

        # Asymetric gaussian function (p[0]: flux of the right-hand side if it
        # was full... ; p[1]: lambda peak; p[2]:sigma_right; p[3]: sigma_left)
        asymfit = lambda p, x: np.where(
            x > p[1],
            cont0 + p[0] / np.sqrt(2 * np.pi) / p[2] *
            np.exp(-(x - p[1]) ** 2 / (2. * p[2] ** 2)),
            cont0 + p[0] * p[3] / p[2] / np.sqrt(2 * np.pi) / p[3] *
            np.exp(-(x - p[1]) ** 2 / (2. * p[3] ** 2))
        )

        # 1d Gaussian fit
        if spec.var is not None and weight:
            wght = 1.0 / np.sqrt(np.abs(spec.var))
            np.ma.fix_invalid(wght, copy=False, fill_value=0)
        else:
            wght = np.ones(spec.shape)

        e_asym_fit = lambda p, x, y, w: w * (asymfit(p, x) - y)

        # inital guesses for Gaussian Fit
        v0 = [peak, lpeak, sigma, sigma]

        # Minimize the sum of squares
        v, covar, info, mesg, success = leastsq(
            e_asym_fit, v0[:], args=(l, data, wght), maxfev=100000,
            full_output=1)

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i])) *
                            np.sqrt(np.abs(chisq / dof))
                            for i in range(len(v))])
        else:
            err = None

        if plot:
            # Same wavelenght grid as input spectrum
            xxx = np.arange(l[0], l[-1], (l[1] - l[0]) / plot_factor)
            ccc = asymfit(v, xxx)
            if ax is None:
                plt.plot(xxx, ccc, 'm--', label='Asymmetric')
            else:
                ax.plot(xxx, ccc, 'm--', label='Asymmetric')

        # return a Gauss1D object
        sigma_right = np.abs(v[2])
        sigma_left = np.abs(v[3])
        flux_right = 0.5 * v[0]
        flux_left = flux_right * sigma_left / sigma_right
        flux = flux_right + flux_left
        fwhm_right = sigma_right * gaussian_sigma_to_fwhm
        fwhm_left = sigma_left * gaussian_sigma_to_fwhm
        lpeak = v[1]
        peak = flux_right / np.sqrt(2 * np.pi * sigma_right ** 2)
        if err is not None:
            err_flux = err[0]
            err_lpeak = err[1]
            err_sigma_right = err[2]
            err_sigma_left = err[3]
            err_fwhm_right = err_sigma_right * gaussian_sigma_to_fwhm
            err_fwhm_left = err_sigma_left * gaussian_sigma_to_fwhm
            err_peak = np.abs(
                1. / np.sqrt(2 * np.pi) *
                (err_flux * sigma_right - flux * err_sigma_right) /
                sigma_right / sigma_right)
        else:
            err_flux = np.NAN
            err_lpeak = np.NAN
            err_sigma_right = np.NAN
            err_sigma_left = np.NAN
            err_fwhm_right = np.NAN
            err_fwhm_left = np.NAN
            err_peak = np.NAN

        return (Gauss1D(lpeak, peak, flux_left, fwhm_left, cont0, err_lpeak,
                        err_peak, err_flux / 2, err_fwhm_left, chisq, dof),
                Gauss1D(lpeak, peak, flux_right, fwhm_right, cont0, err_lpeak,
                        err_peak, err_flux / 2, err_fwhm_right, chisq, dof))

    def add_asym_gaussian(self, lpeak, flux, fwhm_right, fwhm_left, cont=0,
                          peak=False, unit=u.angstrom):
        """Add an asymetric gaussian on spectrum in place.

        Parameters
        ----------
        lpeak : float
            Gaussian center.
        flux : float
            Integrated gaussian flux or gaussian peak value if peak is True.
        fwhm_right : float
            Gaussian fwhm on the right (red) side
        fwhm_left : float
            Gaussian fwhm on the right (red) side
        cont : float
            Continuum value.
        peak : bool
            If true, flux contains the gaussian peak value.
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        """

        asym_gauss = lambda p, x: np.where(
            x > p[1],
            cont + p[0] / np.sqrt(2 * np.pi) / p[2] *
            np.exp(-(x - p[1]) ** 2 / (2. * p[2] ** 2)),
            cont + p[0] * p[3] / p[2] / np.sqrt(2 * np.pi) / p[3] *
            np.exp(-(x - p[1]) ** 2 / (2. * p[3] ** 2))
        )
        sigma_left = fwhm_left * gaussian_fwhm_to_sigma
        sigma_right = fwhm_right * gaussian_fwhm_to_sigma

#         if peak is True:
#             right_norm = flux * np.sqrt(2. * np.pi * sigma_right ** 2)
#         else:
#             right_norm = 2. * flux / (1. + sigma_left / sigma_right)

        lmin = lpeak - 5 * sigma_left
        lmax = lpeak + 5 * sigma_right
        if unit is None:
            imin = int(lmin + 0.5)
            imax = int(lmax + 0.5)
        else:
            imin = self.wave.pixel(lmin, True, unit=unit)
            imax = self.wave.pixel(lmax, True, unit=unit)
        if imin == imax:
            if imin == 0 or imin == self.shape[0]:
                raise ValueError('Gaussian outside spectrum wavelength range')

        if unit is None:
            wave = np.arange(imin, imax, dtype=float)
        else:
            wave = self.wave.coord(unit=unit)[imin:imax]
        v = [flux, lpeak, sigma_right, sigma_left]
        self.data[imin:imax] = self.data[imin:imax] + asym_gauss(v, wave)

    @deprecated('deprecated in favor of gauss_fit with fix_lpeak=True')
    def line_gauss_fit(self, *args, **kwargs):
        kwargs['fix_lpeak'] = True
        return self.gauss_fit(*args, **kwargs)

    def median_filter(self, kernel_size=1., spline=False, unit=u.angstrom,
                      inplace=False):
        """Perform a median filter on the spectrum.

        Uses `scipy.signal.medfilt`.

        Parameters
        ----------
        kernel_size : float
            Size of the median filter window.
        unit : `astropy.units.Unit`
            unit ot the kernel size
        inplace : bool
            If False, return a filtered copy of the spectrum (the default).
            If True, filter the original spectrum in-place, and return that.

        Returns
        -------
        out : Spectrum
        """
        res = self if inplace else self.copy()

        if unit is not None:
            kernel_size = kernel_size / res.get_step(unit=unit)
        ks = int(kernel_size / 2) * 2 + 1

        data = np.empty(res.shape[0] + 2 * ks)
        data[ks:-ks] = res._interp_data(spline)
        data[:ks] = data[ks:2 * ks][::-1]
        data[-ks:] = data[-2 * ks:-ks][::-1]
        data = signal.medfilt(data, ks)
        res._data = data[ks:-ks]
        res._var = None
        return res

    def convolve(self, other, inplace=False):
        """Convolve a Spectrum with a 1D array or another Spectrum, using
        the discrete convolution equation.

        This function, which uses the discrete convolution equation,
        is usually slower than Image.fftconvolve(). However it can be
        faster when other.data.size is small, and it always uses much
        less memory, so it is sometimes the only practical choice.

        Masked values in self.data and self.var are replaced with zeros before
        the convolution is performed, but are masked again after the
        convolution.

        If self.var exists, the variances are propagated using the equation:

          result.var = self.var (*) other**2

        where (*) indicates convolution. This equation can be derived by
        applying the usual rules of error-propagation to the discrete
        convolution equation.

        The speed of this function scales as O(Nd x No) where
        Nd=self.data.size and No=other.data.size.

        Uses `scipy.signal.convolve`.

        Parameters
        ----------
        other : Spectrum or np.ndarray
            The 1D array with which to convolve the spectrum in self.data.
            This can be an array of the same size as self, or it can be a
            smaller array, such as a small gaussian profile to use to smooth
            the spectrum.

            When other.data contains a symmetric filtering function, such as a
            gaussian profile, the center of the function should be placed at
            the center of pixel:

             ``(other.shape - 1) // 2``

            If ``other`` is an MPDAF Spectrum object, note that only its data
            array is used. Masked values in this array are treated as
            zero. Any variances found in other.var are ignored.
        inplace : bool
            If False (the default), return the results in a new Spectrum.
            If True, record the result in self and return that.

        Returns
        -------
        out : `~mpdaf.obj.Spectrum`

        """
        # Delegate the task to DataArray._convolve()
        return self._convolve(signal.convolve, other=other, inplace=inplace)

    def fftconvolve(self, other, inplace=False):
        """Convolve a Spectrum with a 1D array or another Spectrum, using
        the Fourier convolution theorem.

        This function, which performs the convolution by multiplying the
        Fourier transforms of the two arrays, is usually much faster than
        Spectrum.convolve(), except when other.data.size is small. However it
        uses much more memory, so Spectrum.convolve() is sometimes a better
        choice.

        Masked values in self.data and self.var are replaced with zeros before
        the convolution is performed, but they are masked again after the
        convolution.

        If self.var exists, the variances are propagated using the equation:

          result.var = self.var (*) other**2

        where (*) indicates convolution. This equation can be derived by
        applying the usual rules of error-propagation to the discrete
        convolution equation.

        The speed of this function scales as O(Nd x log(Nd)) where
        Nd=self.data.size.  This function temporarily allocates a pair of
        arrays that have the sum of the shapes of self.shape and other.shape,
        rounded up to a power of two along each axis. This can involve a lot
        of memory being allocated. For this reason, when other.shape is small,
        Spectrum.convolve() may be more efficient than Spectrum.fftconvolve().

        Uses `scipy.signal.fftconvolve`.

        Parameters
        ----------
        other : Spectrum or np.ndarray
            The 1D array with which to convolve the spectrum in self.data.
            This can be an array of the same size as self.data, or it can be a
            smaller array, such as a small gaussian to use to smooth the
            spectrum.

            When ``other`` contains a symmetric filtering function, such as a
            gaussian profile, the center of the function should be placed at
            the center of pixel:

             ``(other.shape - 1) // 2``

            If other is an MPDAF Spectrum object, note that only its data
            array is used. Masked values in this array are treated as
            zero. Any variances found in other.var are ignored.
        inplace : bool
            If False (the default), return the results in a new Spectrum.
            If True, record the result in self and return that.

        Returns
        -------
        out : `~mpdaf.obj.Spectrum`

        """
        # Delegate the task to DataArray._convolve()
        return self._convolve(signal.fftconvolve, other=other, inplace=inplace)

    def correlate(self, other, inplace=False):
        """Cross-correlate the spectrum with a other spectrum or an array.

        Uses `scipy.signal.correlate`. self and other must have the same
        size.

        Parameters
        ----------
        other : 1d-array or Spectrum
            Second spectrum or 1d-array.
        inplace : bool
            If True, replace the input spectrum with the correlation.

        Returns
        -------
        out : Spectrum

        """
        res = self if inplace else self.copy()

        try:
            if isinstance(other, Spectrum):
                if res.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '
                                  'with different sizes')
                else:
                    data = other._data
                    if res.unit != other.unit:
                        data = (data * other.unit).to(res.unit).value
                    res._data = signal.correlate(res._data, data, mode='same')
                    if res._var is not None:
                        res._var = signal.correlate(res._var, data,
                                                    mode='same')
        except IOError as e:
            raise e
        except:
            try:
                res._data = signal.correlate(res._data, other, mode='same')
                if res._var is not None:
                    res._var = signal.correlate(res._var, other, mode='same')
            except:
                raise IOError('Operation forbidden')
        return res

    def fftconvolve_gauss(self, fwhm, nsig=5, unit=u.angstrom, inplace=False):
        """Convolve the spectrum with a Gaussian using fft.

        Parameters
        ----------
        fwhm : float
            Gaussian fwhm.
        nsig : int
            Number of standard deviations.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates
        inplace : bool
            If True, convolve the original spectrum in-place, and return that.

        Returns
        -------
        out : Spectrum

        """
        from scipy import special

        res = self if inplace else self.copy()

        sigma = fwhm * gaussian_fwhm_to_sigma
        if unit is None:
            s = sigma
        else:
            s = sigma / res.get_step(unit=unit)
        n = nsig * int(s + 0.5)
        n = int(n / 2) * 2
        data = np.arange(-n, n + 1)
        kernel = special.erf((1 + 2 * data) / (2 * np.sqrt(2) * s)) \
            + special.erf((1 - 2 * data) / (2 * np.sqrt(2) * s))
        kernel /= kernel.sum()

        res._data = signal.correlate(res._data, kernel, mode='same')
        if res._var is not None:
            res._var = signal.correlate(res._var, kernel, mode='same')

        return res

    def LSF_convolve(self, lsf, size, **kwargs):
        """Convolve spectrum with LSF.

        Parameters
        ----------
        lsf : python function
            `mpdaf.MUSE.LSF` object or function f describing the LSF.

            The first three parameters of the function f must be lbda
            (wavelength value in A), step (in A) and size (odd integer).

            f returns an np.array with shape=2*(size/2)+1 and centered in lbda

            Example: from mpdaf.MUSE import LSF
        size : odd int
            size of LSF in pixels.
        kwargs : kwargs
            it can be used to set function arguments.

        Returns
        -------
        out : `~mpdaf.obj.Spectrum`
        """
        res = self.clone()
        if self._data.sum() == 0:
            return res
        step = self.get_step(u.angstrom)
        lbda = self.wave.coord(u.angstrom)

        if size % 2 == 0:
            raise ValueError('Size must be an odd number')
        else:
            k = size // 2

        if isinstance(lsf, types.FunctionType):
            f = lsf
        else:
            try:
                f = getattr(lsf, 'get_LSF')
            except:
                raise ValueError('lsf parameter is not valid')

        data = np.empty(len(self._data) + 2 * k)
        data[k:-k] = self._data
        data[:k] = self._data[k:0:-1]
        data[-k:] = self._data[-2:-k - 2:-1]

        res._data = np.array([(f(lbda[i], step, size, **kwargs) *
                               data[i:i + size]).sum()
                              for i in range(self.shape[0])])
        res._mask = self._mask

        if self._var is None:
            res._var = None
        else:
            res._var = np.array([(f(lbda[i], step, size, **kwargs) *
                                  data[i:i + size]).sum()
                                 for i in range(self.shape[0])])
        return res

    def plot(self, max=None, title=None, noise=False, snr=False,
             lmin=None, lmax=None, ax=None, stretch='linear', unit=u.angstrom,
             noise_kwargs=None, **kwargs):
        """Plot the spectrum.

        By default, the matplotlib drawstyle option is set to
        'steps-mid'. The reason for this is that in MPDAF integer
        pixel indexes correspond to the centers of their pixels, and
        the floating point pixel indexes of a pixel extend from half a
        pixel below the integer central position to half a pixel above
        it.

        Parameters
        ----------
        max : float
            If max is not None (the default), it should be a floating
            point value. The plotted data will be renormalized such
            that the peak in the plot has this value.
        title : string
            The title to give the figure (None by default).
        noise : bool
            If noise is True, colored extensions above and below
            the plotted points indicate the square-root of the
            variances of each pixel (if any).
        snr : bool
            If snr is True, data/sqrt(var) is plotted.
        lmin : float
            The minimum wavelength to be plotted, or None (the default)
            to start the plot from the minimum wavelength in the spectrum.
        lmax : float
            The maximum wavelength to be plotted, or None (the default)
            to start the plot from the maximum wavelength in the spectrum.
        ax : matplotlib.Axes
            The Axes instance in which the spectrum is drawn, or None
            (the default), to request that an Axes object be created
            internally.
        unit : `astropy.units.Unit`
            The wavelength units of the lmin and lmax arguments, or None
            to indicate that lmin and lmax are floating point pixel
            indexes.
        noise_kwargs : dict
            Properties for the noise plot (if ``noise=True``). Default to
            ``color='0.75', facecolor='0.75', alpha=0.5``.
        kwargs : dict
            kwargs can be used to set properties of the plot such as:
            line label (for auto legends), linewidth, anitialising,
            marker face color, etc.

        """

        # Create an Axes instance for the plot?
        if ax is None:
            ax = plt.gca()

        # If a sub-set of the spectrum's wavelengths have been
        # specified, get a truncated copy of the spectrum that just
        # contains this range.
        res = self.copy()
        if lmin is not None or lmax is not None:
            res.truncate(lmin, lmax, unit)

        # Get the wavelengths to be plotted along the X axis,
        # preferably in the units specified for lmin and lmax,
        # if any. If the specified units can't be used, use
        # the native wavelength units of the spectrum.
        try:
            x = res.wave.coord(unit=unit)
        except u.UnitConversionError:
            unit = res.wave.unit
            x = res.wave.coord(unit=unit)

        # Get the pixel values to be plotted.
        data = res.data

        # Disable the noise and snr options if no variances
        # are available.
        if res.var is None:
            noise = False
            snr = False

        # Compute the SNR?
        if snr:
            data /= np.sqrt(res.var)

        # Renormalize to make the peak value equal to max?
        if max is not None:
            data = data * max / data.max()

        # Set the default plot arguments.
        kwargs.setdefault('drawstyle', 'steps-mid')
        kwargs.setdefault('linewidth', 1)

        # Plot the data with a linear or logarithmic Y axis.
        if stretch == 'linear':
            ax.plot(x, data, **kwargs)
        elif stretch == 'log':
            ax.semilogy(x, data, **kwargs)
        else:
            raise ValueError("Unknow stretch '{}'".format(stretch))

        # Plot extensions above and below the points to represent
        # their uncertainties?
        if noise:
            sigma = np.sqrt(res.var)
            noisekw = dict(color='0.75', facecolor='0.75', alpha=0.5)
            if noise_kwargs is not None:
                noisekw.update(noise_kwargs)
            ax.fill_between(x, data + sigma, data - sigma, **noisekw)

        # Label the plot.
        if title is not None:
            ax.set_title(title)
        if unit is not None:
            ax.set_xlabel(r'$\lambda$ (%s)' % unit)
        if res.unit is not None:
            ax.set_ylabel(res.unit)

        if lmin is None and lmax is None:
            ax.set_xlim(*self.wave.get_range(unit=unit))

        # Arrange for cursor motion events to display corresponding
        # coordinates and values below the plot.
        def _on_move(event):  # pragma: no cover
            """print xc,yc,k,lbda and data in the figure toolbar."""
            if event.inaxes is not None:
                xc, yc = event.xdata, event.ydata
                try:
                    i = self.wave.pixel(xc, nearest=True, unit=unit)
                    x = self.wave.coord(i, unit=self._unit)
                    event.canvas.toolbar.set_message(
                        'xc= %g yc=%g k=%d lbda=%g data=%g' %
                        (xc, yc, i, x, self._data[i]))
                except Exception:
                    pass

        self._unit = unit
        plt.connect('motion_notify_event', _on_move)
        self._plot_id = len(ax.lines) - 1
