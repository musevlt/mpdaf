"""spectrum.py defines Spectrum objects."""

from __future__ import absolute_import, print_function

import logging
import matplotlib.pyplot as plt
import numpy as np
import types
import warnings

import astropy.units as u
from scipy import integrate, interpolate, signal
from scipy.optimize import leastsq
from six.moves import range

from . import ABmag_filters
from .data import DataArray
from .objs import is_float, is_int, flux2mag, UnitMaskedArray, UnitArray
from ..gui.clicks import SpectrumClicks
from ..tools import deprecated

__all__ = ('Gauss1D', 'Spectrum')


class Gauss1D(object):

    """This class stores 1D Gaussian parameters.

    Attributes
    ----------
    cont : float
        Continuum value.
    fwhm : float
        Gaussian fwhm.
    lpeak : float
        Gaussian center.
    peak : float
        Gaussian peak value.
    flux : float
        Gaussian integrated flux.
    err_fwhm : float
        Estimated error on Gaussian fwhm.
    err_lpeak : float
        Estimated error on Gaussian center.
    err_peak : float
        Estimated error on Gaussian peak value.
    err_flux : float
        Estimated error on Gaussian integrated flux.
    chisq : float
        minimization process info (Chi-sqr)
    dof : float
        minimization process info (number of points - number of parameters)

    """

    def __init__(self, lpeak, peak, flux, fwhm, cont, err_lpeak,
                 err_peak, err_flux, err_fwhm, chisq, dof):
        self.cont = cont
        self.fwhm = fwhm
        self.lpeak = lpeak
        self.peak = peak
        self.flux = flux
        self.err_fwhm = err_fwhm
        self.err_lpeak = err_lpeak
        self.err_peak = err_peak
        self.err_flux = err_flux
        self._logger = logging.getLogger(__name__)
        self.chisq = chisq
        self.dof = dof

    def copy(self):
        """Copy Gauss1D object in a new one and returns it."""
        return Gauss1D(self.lpeak, self.peak, self.flux, self.fwhm,
                       self.cont, self.err_lpeak, self.err_peak,
                       self.err_flux, self.err_fwhm, self.chisq, self.dof)

    def print_param(self):
        """Print Gaussian parameters."""

        msg = 'Gaussian center = %g (error:%g)' % (self.lpeak, self.err_lpeak)
        self._logger.info(msg)

        msg = 'Gaussian integrated flux = %g (error:%g)' % \
            (self.flux, self.err_flux)
        self._logger.info(msg)

        msg = 'Gaussian peak value = %g (error:%g)' % \
            (self.peak, self.err_peak)
        self._logger.info(msg)

        msg = 'Gaussian fwhm = %g (error:%g)' % (self.fwhm, self.err_fwhm)
        self._logger.info(msg)

        msg = 'Gaussian continuum = %g' % self.cont
        self._logger.info(msg)


class Spectrum(DataArray):

    """Spectrum class manages spectrum, optionally including a variance and a
    bad pixel mask.

    Parameters
    ----------
    filename : string
        Possible FITS filename.
    ext : integer or (integer,integer) or string or (string,string)
        Number/name of the data extension or numbers/names
        of the data and variance extensions.
    wave : `mpdaf.obj.WaveCoord`
        Wavelength coordinates.
    unit : string
        Data unit type. u.dimensionless_unscaled by default.
    data : float array
        Array containing the pixel values of the spectrum. None by default.
    var : float array
        Array containing the variance. None by default.

    Attributes
    ----------
    filename : string
        Possible FITS filename.
    unit : `astropy.units.Unit`
        Data unit type.
    primary_header : pyfits.Header
        Possible FITS primary header instance.
    data_header : pyfits.Header
        Possible FITS data header instance.
    data : masked array
        Array containing the pixel values of the spectrum.
    shape : tuple
        Size of spectrum.
    var : array
        Array containing the variance.
    wave : `mpdaf.obj.WaveCoord`
        Wavelength coordinates.

    """

    _ndim_required = 1
    _has_wave = True

    def __init__(self, filename=None, ext=None, unit=u.dimensionless_unscaled,
                 data=None, var=None, wave=None, copy=True, dtype=float,
                 **kwargs):
        super(Spectrum, self).__init__(
            filename=filename, ext=ext, wave=wave, unit=unit, data=data,
            var=var, copy=copy, dtype=dtype, **kwargs)
        self._clicks = None

    def __add__(self, other):
        """Operator +.

        spectrum1 + number = spectrum2
        (spectrum2[k] = spectrum1[k] + number)

        spectrum1 + spectrum2 = spectrum3
        (spectrum3[k] = spectrum1[k] + spectrum2[k])

        spectrum + cube1 = cube2
        (cube2[k,p,q] = cube1[k,p,q] + spectrum[k])

        Parameters
        ----------
        other : number or Spectrum or Cube object.
            It is Spectrum : Dimensions and wavelength coordinates must be
            the same.
            It is Cube : The first dimension of cube1 must be equal to the
            spectrum dimension. Wavelength coordinates must be the same.

        Returns
        -------
        out : Spectrum or Cube object.
        """
        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res._data = self._data + other
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            # coordinates
            if self.wave is not None and other.wave is not None \
                    and not self.wave.isEqual(other.wave):
                raise IOError('Operation forbidden for spectra '
                              'with different world coordinates')

            if other.ndim == 1:
                # spectrum1 + spectrum2 = spectrum3
                if self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '
                                  'with different sizes')
                res = self.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data + other.data
                else:
                    res.data = self.data + UnitMaskedArray(
                        other.data, other.unit, self.unit)
                # variance
                if res._var is not None:
                    if self._var is None:
                        if other.unit == self.unit:
                            res._var = other._var
                        else:
                            res._var = UnitArray(other._var, other.unit**2,
                                                self.unit**2)
                    else:
                        if other.unit == self.unit:
                            res._var = self._var + other._var
                        else:
                            res._var = self._var + UnitArray(
                                other._var, other.unit**2, self.unit**2)
                # return
                return res
            elif other.ndim == 3:
                # spectrum + cube1 = cube2
                res = other.__add__(self)
                return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """ Operator -.

        spectrum1 - number = spectrum2
        (spectrum2[k] = spectrum1[k] - number)

        spectrum1 - spectrum2 = spectrum3
        (spectrum3[k] = spectrum1[k] - spectrum2[k])

        spectrum - cube1 = cube2
        (cube2[k,p,q] = spectrum[k] - cube1[k,p,q])

        Parameters
        ----------
        other : number or Spectrum or Cube object.
            It is Spectrum : Dimensions and wavelength coordinates must be the
            same.
            It is Cube : The first dimension of cube1 must be equal to the
            spectrum dimension.  Wavelength coordinates must be the same.

        Returns
        -------
        out : Spectrum or Cube object.
        """
        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res._data = self._data - other
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            # coordinates
            if self.wave is not None and other.wave is not None \
                    and not self.wave.isEqual(other.wave):
                raise IOError('Operation forbidden for spectra '
                              'with different world coordinates')

            if other.ndim == 1:
                # spectrum1 + spectrum2 = spectrum3
                if self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '
                                  'with different sizes')
                res = self.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data - other.data
                else:
                    res.data = self.data - UnitMaskedArray(other.data,
                                                           other.unit,
                                                           self.unit)
                # variance
                if other._var is not None:
                    if self._var is None:
                        if other.unit == self.unit:
                            res._var = other._var
                        else:
                            res._var = UnitArray(other._var,
                                                other.unit**2, self.unit**2)
                    else:
                        if other.unit == self.unit:
                            res._var = self._var + other._var
                        else:
                            res._var = self._var + UnitArray(other._var,
                                                           other.unit**2,
                                                           self.unit**2)
                return res
            else:
                # spectrum - cube1 = cube2
                if self.shape[0] != other.shape[0]:
                    raise IOError('Operation forbidden for objects'
                                  ' with different sizes')

                res = other.copy()
                # data
                if other.unit == self.unit:
                    res.data = self.data[:, np.newaxis, np.newaxis] - other.data
                else:
                    res.data = self.data[:, np.newaxis, np.newaxis] \
                        - UnitMaskedArray(other.data, self.unit, other.unit)
                # variance
                if self._var is not None:
                    if other._var is None:
                        if other.unit == self.unit:
                            res._var = self._var
                        else:
                            res._var = UnitArray(self._var, self.unit**2, other.unit**2)
                    else:
                        if other.unit == self.unit:
                            res._var = self._var + other._var
                        else:
                            res._var = other._var + UnitArray(self._var,
                                                            self.unit**2,
                                                            other.unit**2)
                return res

    def __rsub__(self, other):
        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res._data = other - self._data
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            return other.__sub__(self)

    def __mul__(self, other):
        """ Operator \*.

        spectrum1 \* number = spectrum2
        (spectrum2[k] = spectrum1[k] \* number)

        spectrum1 \* spectrum2 = spectrum3
        (spectrum3[k] = spectrum1[k] \* spectrum2[k])

        spectrum \* cube1 = cube2
        (cube2[k,p,q] = spectrum[k] \* cube1[k,p,q])

        spectrum \* image = cube (cube[k,p,q]=image[p,q] \* spectrum[k]

        Parameters
        ----------
        other : number or Spectrum or Image or Cube object.
            It is Spectrum : Dimensions and wavelength coordinates
            must be the same.
            It is Cube : The first dimension of cube1 must be equal to the
            spectrum dimension.  Wavelength coordinates must be the same.

        Returns
        -------
        out : Spectrum or Cube object.
        """
        if not isinstance(other, DataArray):
            # spectrum1 * number = spectrum2
            # (spectrum2[k]=spectrum1[k]*number)
            try:
                res = self.copy()
                res._data *= other
                if self._var is not None:
                    res._var *= other ** 2
                return res
            except:
                raise IOError('Operation forbidden')
        elif other.ndim == 1:
            # spectrum1 * spectrum2 = spectrum3
            if self.shape != other.shape:
                raise IOError('Operation forbidden for spectra '
                              'with different sizes')
            # coordinates
            if self.wave is not None and other.wave is not None \
                    and not self.wave.isEqual(other.wave):
                raise IOError('Operation forbidden for spectra '
                              'with different world coordinates')

            res = self.copy()
            # data
            res.data = self.data * other.data
            # variance
            if self._var is None and other._var is None:
                res._var = None
            elif self._var is None:
                res._var = other._var * self._data * self._data
            elif other._var is None:
                res._var = self._var * other._data * other._data
            else:
                res._var = (other._var * self._data * self._data +
                           self._var * other._data * other._data)
            # unit
            res.unit = self.unit * other.unit
            # return
            return res
        else:
            res = other.__mul__(self)
            return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """Operator /.

        Note : divide functions that have a validity domain returns
        the masked constant whenever the input is masked or falls
        outside the validity domain.

        spectrum1 / number = spectrum2
        (spectrum2[k] = spectrum1[k] / number)

        spectrum1 / spectrum2 = spectrum3
        (spectrum3[k] = spectrum1[k] / spectrum2[k])

        spectrum / cube1 = cube2
        (cube2[k,p,q] = spectrum[k] / cube1[k,p,q])

        Parameters
        ----------
        other : number or Spectrum or Cube object.
            It is Spectrum : Dimensions and wavelength coordinates
            must be the same.
            It is Cube : The first dimension of cube1 must be equal to
            the spectrum dimension. Wavelength coordinates must be the same.

        Returns
        -------
        out : Spectrum or Cube object.
        """
        if not isinstance(other, DataArray):
            try:
                # spectrum1 / number =
                # spectrum2 (spectrum2[k]=spectrum1[k]/number)
                res = self.copy()
                res._data /= other
                if self._var is not None:
                    res._var /= other ** 2
                    return res
            except:
                raise IOError('Operation forbidden')
        else:
            # coordinates
            if self.wave is not None and other.wave is not None \
                    and not self.wave.isEqual(other.wave):
                raise IOError('Operation forbidden for spectra '
                              'with different world coordinates')
            if other.ndim == 1:
                # spectrum1 / spectrum2 = spectrum3
                if self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '
                                  'with different sizes')

                res = self.copy()
                # data
                res.data = self.data / other.data
                # variance
                if self._var is None and other._var is None:
                    res._var = None
                elif self._var is None:
                    res._var = other._var * self._data * self._data \
                        / (other._data ** 4)
                elif other._var is None:
                    res._var = self._var * other._data * other._data \
                        / (other._data ** 4)
                else:
                    res._var = (other._var * self._data * self._data +
                               self._var * other._data * other._data) \
                        / (other._data ** 4)
                # unit
                res.unit = self.unit / other.unit
                return res
            else:
                # spectrum / cube1 = cube2
                if self.shape[0] != other.shape[0]:
                    raise IOError('Operation forbidden for objects '
                                  'with different sizes')
                # data
                res = other.copy()
                res.data = self.data[:, np.newaxis, np.newaxis] \
                    / other.data
                # variance
                if self._var is None and other._var is None:
                    res._var = None
                elif self._var is None:
                    res._var = other._var \
                        * self._data[:, np.newaxis, np.newaxis] \
                        * self._data[:, np.newaxis, np.newaxis] \
                        / (other._data ** 4)
                elif other._var is None:
                    res._var = self._var[:, np.newaxis, np.newaxis] \
                        * other._data * other._data / (other._data ** 4)
                else:
                    res._var = \
                        (other._var *
                         self._data[:, np.newaxis, np.newaxis] *
                         self._data[:, np.newaxis, np.newaxis] +
                         self._var[:, np.newaxis, np.newaxis] *
                         other._data * other._data) / (other._data ** 4)
                # unit
                res.unit = self.unit / other.unit
                return res

    def __rdiv__(self, other):
        if not isinstance(other, DataArray):
            try:
                res = self.copy()
                res._data = other / self._data
                if self._var is not None:
                    res._var = other ** 2 / self._var
                return res
            except:
                raise IOError('Operation forbidden')
        else:
            return other.__div__(self)

    def get_lambda(self, lmin, lmax=None, unit=u.angstrom):
        """ Return the flux value corresponding to a wavelength,
        or return the sub-spectrum corresponding to a wavelength range.

        Parameters
        ----------
        lmin : float
            minimum wavelength.
        lmax : float
            maximum wavelength.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates. if None, inputs are in pixels.

        Returns
        -------
        out : float or Spectrum
        """
        if lmax is None:
            lmax = lmin
        if self.wave is None:
            raise ValueError('Operation forbidden without world coordinates '
                             'along the spectral direction')
        else:
            if unit is None:
                pix_min = max(0, int(lmin + 0.5))
                pix_max = min(self.shape[0], int(lmax + 0.5))
            else:
                pix_min = max(0, self.wave.pixel(lmin, nearest=True, unit=unit))
                pix_max = min(self.shape[0],
                              self.wave.pixel(lmax, nearest=True, unit=unit) + 1)
            if (pix_min + 1) == pix_max:
                return self[pix_min]
            else:
                return self[pix_min:pix_max]

    def get_step(self, unit=None):
        """Return the wavelength step.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates

        Returns
        -------
        out : float
        """
        if self.wave is not None:
            return self.wave.get_step(unit)
        else:
            return None

    def get_start(self, unit=None):
        """Return the wavelength value of the first pixel.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates

        Returns
        -------
        out : float
        """
        if self.wave is not None:
            return self.wave.get_start(unit)
        else:
            return None

    def get_end(self, unit=None):
        """Return the wavelength value of the last pixel.

        Parameters
        ----------
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates

        Returns
        -------
        out : float
        """
        if self.wave is not None:
            return self.wave.get_end(unit)
        else:
            return None

    def get_range(self, unit=None):
        """Return the wavelength range (Lambda_min, Lambda_max).

        Parameters
        ----------
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates

        Returns
        -------
        out : float array
        """
        if self.wave is not None:
            return self.wave.get_range(unit)
        else:
            return None

    def set_wcs(self, wave):
        """Set the world coordinates.

        Parameters
        ----------
        wave : `mpdaf.obj.WaveCoord`
            Wavelength coordinates.
        """
        if wave.shape is not None and wave.shape != self.shape:
            self._logger.warning('wavelength coordinates and data have '
                                 'not the same dimensions')
        self.wave = wave.copy()
        self.wave.shape = self.shape

    def mask_region(self, lmin=None, lmax=None, inside=True, unit=u.angstrom):
        """Mask the spectrum inside/outside [lmin,lmax].

        Parameters
        ----------
        lmin : float
            minimum wavelength.
        lmax : float
            maximum wavelength.
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        inside : boolean
            If inside is True, pixels inside [lmin,lmax] are masked.
            If inside is False, pixels outside [lmin,lmax] are masked.
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
                    pix_min = max(0, self.wave.pixel(lmin, nearest=True, unit=unit))
            if lmax is None:
                pix_max = self.shape[0]
            else:
                if unit is None:
                    pix_max = min(self.shape[0], int(lmax + 0.5))
                else:
                    pix_max = min(self.shape[0],
                                  self.wave.pixel(lmax, nearest=True, unit=unit) + 1)

            if inside:
                self.data[pix_min:pix_max] = np.ma.masked
            else:
                self.data[:pix_min] = np.ma.masked
                self.data[pix_max + 1:] = np.ma.masked

    def _interp(self, wavelengths, spline=False):
        """return the interpolated values corresponding to the wavelength
        array.

        Parameters
        ----------
        wavelengths : array of float
            wavelength values
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates
        spline : boolean
            False: linear interpolation (`scipy.interpolate.interp1d` used),
            True: spline interpolation (`scipy.interpolate.splrep/splev` used).
        """
        lbda = self.wave.coord()
        if self.mask is np.ma.nomask:
            d = np.empty(self.shape + 2, dtype=float)
            d[1:-1] = self._data
            w = np.empty(self.shape + 2, dtype=float)
            w[1:-1] = lbda
        else:
            ksel = np.where(self.mask == False)
            d = np.empty(np.shape(ksel)[1] + 2, dtype=float)
            d[1:-1] = self._data[ksel]
            w = np.empty(np.shape(ksel)[1] + 2)
            w[1:-1] = lbda[ksel]
        d[0] = d[1]
        d[-1] = d[-2]
        w[0] = self.get_start() - 0.5 * self.get_step()
        w[-1] = self.get_end() + 0.5 * self.get_step()

        if spline:
            if self._var is not None:
                _weight = 1. / np.sqrt(np.abs(self.var.filled(np.inf)))
                if self.mask is np.ma.nomask:
                    weight = np.empty(self.shape + 2, dtype=float)
                    weight[1:-1] = _weight
                else:
                    weight = np.empty(np.shape(ksel)[1] + 2)
                    weight[1:-1] = _weight[ksel]
                weight[0] = weight[1]
                weight[-1] = weight[-2]
            else:
                weight = None
            tck = interpolate.splrep(w, d, w=weight)
            return interpolate.splev(wavelengths, tck, der=0)
        else:
            f = interpolate.interp1d(w, d)
            return f(wavelengths)

    def _interp_data(self, spline=False):
        """Return data array with interpolated values for masked pixels.

        Parameters
        ----------
        spline : boolean
            False: linear interpolation (`scipy.interpolate.interp1d` used),
            True: spline interpolation (`scipy.interpolate.splrep/splev` used).
        """
        if np.ma.count_masked(self.data) == 0:
            return self.data.data
        else:
            lbda = self.wave.coord()
            ksel = np.where(self._mask == True)
            wnew = lbda[ksel]
            data = self._data.copy()
            data[ksel] = self._interp(wnew, spline)
            return data

    def interp_mask(self, spline=False):
        """Interpolate masked pixels.

        Parameters
        ----------
        spline : boolean
            False: linear interpolation (`scipy.interpolate.interp1d` used),
            True: spline interpolation (`scipy.interpolate.splrep/splev` used).
        """
        self.data = np.ma.masked_invalid(self._interp_data(spline))

    def _rebin_mean_(self, factor):
        """Shrink the size of the spectrum by factor. New size is an integer
        multiple of the original size.

        Parameters
        ----------
        factor : integer
            Factor.
        """
        assert not np.sometrue(np.mod(self.shape[0], factor))
        # new size is an integer multiple of the original size
        sh = self.shape[0] / factor

        if self.mask is np.ma.nomask:
            self._data = self._data.reshape(sh, factor).sum(1) / factor
            if self._var is not None:
                self._var = self._var.reshape(sh, factor).sum(1) / (factor * factor)
        else:
            mask_count = (~self.mask).reshape(sh, factor).sum(1)
            self._data = self.data.reshape(sh, factor).sum(1).data / mask_count
            if self._var is not None:
                self._var = self.var.reshape(sh, factor).sum(1).data / (mask_count * mask_count)
            self._mask = (mask_count==0)
        self._ndim = self._data.ndim

        try:
            self.wave.rebin(factor)
        except:
            self.wave = None

    def _rebin_mean(self, factor, margin='center'):
        """Shrink the size of the spectrum by factor.

        Parameters
        ----------
        factor : integer
            factor
        margin : string in 'center'|'right'|'left'
            This parameters is used if new size is not an integer multiple of
            the original size.

            - 'center' : two pixels added, on the left and on the right of
              the spectrum.
            - 'right': one pixel added on the right of the spectrum.
            - 'left': one pixel added on the left of the spectrum.

        """

        # The divisor must be in the range 1 to (shape - 1).

        if factor <= 1 or factor >= self.shape[0]:
            raise ValueError('factor must be in ]1,shape[')
        factor = int(factor)

        # How many pixels need to be removed to make the dimension
        # of the array an integer multiple of the division factor?

        n = self.shape[0] % factor

        # If needed, compute the slice needed to truncate the
        # size of the spectrum to an integer multiple of the
        # division factor.

        if n != 0:

            # Calculate the index of the first pixel of the spectrum
            # that will be re-binned.

            if margin == 'left' or n==1:
                ia = 0
            elif margin == 'right':
                ia = n
            elif margin == 'center':
                ia = n//2
            else:
                raise ValueError('Unsupported margin: %s' % margin)

            # Compute the slice to be used to select the sub-spectrum.

            sl = slice(ia, self.shape[0] - (n - ia))

            # Slice the data and variance arrays.

            self._data = self._data[sl]
            if self._var is not None:
                self._var = self._var[sl]
            self._mask = self._mask[sl]

            # Update the world coordinates to match the truncated
            # array.

            self.wave = self.wave[sl]

        # At this point the spectrum dimension is an integer multiple of
        # the division factor. What is the shape of the output spectrum?

        newshape = self.shape[0] // factor
        data = self.data.reshape(newshape, factor)

        # Compute how many unmasked pixels in the input spectrum will
        # contribute to each mean pixel in the output spectrum.

        unmasked = data.count(1)

        # Reduce the size of the data array by taking the mean of
        # successive groups of 'factor' pixels. Note that in the
        # following, np.ma.mean() takes account of masked pixels.

        self._data = data.mean(1).data

        # The treatment of the variance array is complicated by the
        # possibility of masked pixels in the data array. A sum of N
        # data pixels p[i] of variance v[i] has a variance of
        # sum(v[i] / N^2), where N^2 is the number of unmasked pixels
        # in that particular sum.

        if self._var is not None:
            self._var = (self.var.reshape(newshape, factor).sum(1) / unmasked**2)

        # Mask all pixels in the output array that come from zero
        # unmasked pixels of the input array.

        self._mask = unmasked < 1

        # Update the world-coordinate information.

        try:
            self.wave.rebin(factor)
        except:
            self.wave = None

    def rebin_mean(self, factor, margin='center', inplace=False):
        """Return a spectrum that shrinks the size of the current spectrum by
        factor.

        Parameters
        ----------
        factor : integer
            factor
        margin : string in 'center'|'right'|'left'
            This parameters is used if new size is not an integer multiple of
            the original size.

            - 'center' : two pixels added, on the left
              and on the right of the spectrum.
            - 'right': one pixel added on the right of the spectrum.
            - 'left': one pixel added on the left of the spectrum.
        inplace : bool
            If False, return a rebinned copy of the spectrum (the default).
            If True, rebin the original spectrum in-place, and return that.

        Returns
        -------
        out : Spectrum
        """
        # Should we rebin the spectrum in-place, or rebin a copy of the spectrum?

        res = self if inplace else self.copy()

        # Rebin the result object in-place.

        res._rebin_mean(factor, margin)
        return res

    def _rebin_median_(self, factor):
        """Shrink the size of the spectrum by factor. Median values used. New
        size is an integer multiple of the original size.

        Parameters
        ----------
        factor : integer
            factor
        """
        assert not np.sometrue(np.mod(self.shape[0], factor))
        # new size is an integer multiple of the original size
        shape = self.shape[0] / factor
        self._data = np.ma.median(self.data.reshape(shape, factor), 1)
        if self._mask is not np.ma.nomask:
            self._mask = ((~self._mask).reshape(shape, factor).sum(1) == 0)
        self.var = None
        self._ndim = self._data.ndim
        try:
            self.wave.rebin(factor)
        except:
            self.wave = None

    def rebin_median(self, factor, margin='center'):
        """Shrink the size of the spectrum by factor. Median values are used.

        Parameters
        ----------
        factor : integer
            factor
        margin : string in 'center'|'right'|'left'
            This parameters is used if new size is not
            an integer multiple of the original size.

            - 'center' : data lost on the left
              and on the right of the spectrum.
            - 'right': data lost on the right of the spectrum.
            - 'left': data lost on the left of the spectrum.

        Returns
        -------
        out `~mpdaf.obj.Spectrum`
        """
        if factor <= 1 or factor >= self.shape[0]:
            raise ValueError('factor must be in ]1,shape[')
        # assert not np.sometrue(np.mod( self.shape, factor ))
        if not np.sometrue(np.mod(self.shape[0], factor)):
            # new size is an integer multiple of the original size
            res = self.copy()
        else:
            newshape = self.shape[0] / factor
            n = self.shape[0] - newshape * factor
            if margin == 'center' and n == 1:
                margin = 'right'
            if margin == 'center':
                n_left = n / 2
                n_right = self.shape[0] - n + n_left
                res = self[n_left:n_right]
            elif margin == 'right':
                res = self[0:self.shape[0] - n]
            elif margin == 'left':
                res = self[n:]
            else:
                raise ValueError('margin must be center|right|left')
        res._rebin_median_(factor)
        return res

    def _resample(self, step, start=None, shape=None,
                  spline=False, notnoise=False, unit=u.angstrom):
        """Resample spectrum data to different wavelength step size.

        Uses `scipy.integrate.quad`.

        Parameters
        ----------
        step : float
            New pixel size in spectral direction.
        start : float
            Spectral position of the first new pixel.
            It can be set or kept at the edge of the old first one.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates
        shape : integer
            Size of the new spectrum.
        spline : boolean
            Linear/spline interpolation to interpolate masked values.
        notnoise : boolean
            True if the noise Variance spectrum is not interpolated
            (if it exists).

        """
        lrange = self.get_range(unit)
        if start is not None and start > lrange[1]:
            raise ValueError('Start value outside the spectrum range')
        if start is not None and start < lrange[0]:
            n = int((lrange[0] - start) / step)
            start = lrange[0] + (n + 1) * step

        newwave = self.wave.resample(step, start, unit)
        if shape is None:
            newshape = newwave.shape
        else:
            newshape = min(shape, newwave.shape)
            newwave.shape = newshape

        dmin = np.ma.min(self.data)
        dmax = np.ma.max(self.data)
        if dmin == dmax:
            self._data = np.ones(newshape, dtype=np.float) * dmin
        else:
            data = self._interp_data(spline)
            f = lambda x: data[self.wave.pixel(x, unit=unit, nearest=True)]
            self._data = np.empty(newshape, dtype=np.float)
            pix = np.arange(newshape + 1, dtype=np.float)
            x = (pix - newwave.get_crpix() + 1) * newwave.get_step(unit) \
                + newwave.get_crval(unit) - 0.5 * newwave.get_step(unit)

            lbdamax = self.get_end(unit) + 0.5 * self.get_step(unit)
            if x[-1] > lbdamax:
                x[-1] = lbdamax

            for i in range(newshape):
                self._data[i] = \
                    integrate.quad(f, x[i], x[i + 1], full_output=1)[0] \
                    / newwave.get_step(unit)

        if self._mask is not np.ma.nomask:
            self._mask = ~(np.isfinite(self._data))

        if self._var is not None and not notnoise:
            dmin = np.min(self._var)
            dmax = np.max(self._var)
            if dmin == dmax:
                self._var = np.ones(newshape, dtype=np.float) * dmin
            else:
                f = lambda x: self._var[int(self.wave.pixel(x, unit=unit) + 0.5)]
                var = np.empty(newshape, dtype=np.float)
                for i in range(newshape):
                    var[i] = \
                        integrate.quad(f, x[i], x[i + 1], full_output=1)[0] \
                        / newwave.get_step(unit)
                self._var = var

            if self._mask is not np.ma.nomask:
                self._mask = self._mask | ~(np.isfinite(self._var)) | (self._var<=0)

        else:
            self._var = None

        self.wave = newwave

    def resample(self, step, start=None, shape=None,
                 spline=False, notnoise=False, unit=u.angstrom, inplace=False):
        """Return a spectrum with data resample to different wavelength step
        size.

        Uses `scipy.integrate.quad`.

        Parameters
        ----------
        step : float
            New pixel size in spectral direction.
        start : float
            Spectral position of the first new pixel.
            It can be set or kept at the edge of the old first one.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates
        shape : integer
            Size of the new spectrum.
        spline : boolean
            Linear/spline interpolation to interpolate masked values.
        notnoise : boolean
        inplace : bool
            If False, return a resampled copy of the spectrum (the default).
            If True, resample the original spectrum in-place, and return that.

        Returns
        -------
        out : Spectrum
        """
        # Should we resample the spectrum in-place, or resample a copy?

        res = self if inplace else self.copy()

        # Resample the result object in-place.

        res._resample(step, start, shape, spline, notnoise, unit)
        return res

    def mean(self, lmin=None, lmax=None, weight=True, unit=u.angstrom):
        """Compute the mean flux value over a wavelength range.

        Parameters
        ----------
        lmin : float
            Minimum wavelength.
        lmax : float
            Maximum wavelength.
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        weight : boolean
            If weight is True, compute the weighted average
            with the inverse of variance as weight.

        Returns
        -------
        out : float
        """
        if self._var is None:
            weight = False
        if lmin is None:
            i1 = 0
        else:
            if unit is None:
                i1 = max(0, int(lmin + 0.5))
            else:
                i1 = max(0, self.wave.pixel(lmin, nearest=True, unit=unit))
        if lmax is None:
            i2 = self.shape[0]
        else:
            if unit is None:
                i2 = min(self.shape[0], int(lmax + 0.5))
            else:
                i2 = min(self.shape[0],
                         self.wave.pixel(lmax, nearest=True, unit=unit) + 1)

        subspe = self[i1:i2]
        if weight:
            weights = 1.0 / subspe.var.filled(np.inf)
            flux = np.ma.average(subspe.data, weights=weights)
        else:
            flux = np.ma.average(subspe.data)
        return flux

    def sum(self, lmin=None, lmax=None, weight=True, unit=u.angstrom):
        """Sum the flux value over [lmin,lmax].

        Parameters
        ----------
        lmin : float
            Minimum wavelength.
        lmax : float
            Maximum wavelength.
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        weight : boolean
            If weight is True, compute the weighted average
            with the inverse of variance as weight.

        Returns
        -------
        out : float
        """
        if lmin is None:
            i1 = 0
        else:
            if unit is None:
                i1 = int(lmin + 0.5)
            else:
                i1 = max(0, self.wave.pixel(lmin, True, unit))
        if lmax is None:
            i2 = self.shape[0]
        else:
            if unit is None:
                i2 = int(lmax + 0.5)
            else:
                i2 = min(self.shape[0], self.wave.pixel(lmax, True, unit) + 1)

        subspe = self[i1:i2]
        if weight and self._var is not None:
            weights = 1.0 / subspe.var.filled(np.inf)

            # How many unmasked pixels will be averaged?

            nsum = np.ma.count(subspe.data)

            # The weighted average multiplied by the number of unmasked pixels.

            flux = nsum * np.ma.average(subspe.data, weights=weights)
        else:
            flux = subspe.data.sum()
        return flux

    def integrate(self, lmin=None, lmax=None, unit=u.angstrom):
        """Integrate the flux value over [lmin,lmax].

        Parameters
        ----------
        lmin : float
            Minimum wavelength.
        lmax : float
            Maximum wavelength.
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.

        Returns
        -------
        out : float
        """
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

        d = self.wave.coord(-0.5 + np.arange(i1, i2 + 1), unit=unit)
        d[0] = lmin
        d[-1] = lmax

        if unit is None:
            unit = self.wave.unit

        subspe = self[i1:i2]

        if u.angstrom in self.unit.bases and unit is not u.angstrom:
            try:
                return np.ma.sum(subspe.data *
                              ((np.diff(d) * unit).to(u.angstrom).value)
                              ) * self.unit * u.angstrom
            except:
                return (subspe.data * np.diff(d)).sum() * self.unit * unit
        else:
            return (subspe.data * np.diff(d)).sum() * self.unit * unit

    def poly_fit(self, deg, weight=True, maxiter=0,
                 nsig=(-3.0, 3.0), verbose=False):
        """Perform polynomial fit on normalized spectrum and returns polynomial
        coefficients.

        Parameters
        ----------
        deg : integer
            Polynomial degree.
        weight : boolean
            If weight is True, the weight is computed as the inverse of
            variance.
        maxiter : integer
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
            mask = np.array(1 - self._mask, dtype=bool)
            d = self._data.compress(mask)
            w = self.wave.coord().compress(mask)
            if weight:
                vec_weight = vec_weight.compress(mask)

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
                    msg = 'Number of iteration: '\
                        '%d Std: %10.4e Np: %d Frac: %4.2f' \
                        % (it + 1, sig, n_p, 100. * n_p / self.shape[0])
                    self._logger.info(msg)

        return p

    def poly_val(self, z):
        """Update in place the spectrum data from polynomial coefficients.

        Uses `numpy.poly1d`.

        Parameters
        ----------
        z : array
            The polynomial coefficients, in increasing powers.

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
        deg : integer
            Polynomial degree.
        weight : boolean
            If weight is True, the weight is computed as the inverse of
            variance.
        maxiter : integer
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

    def abmag_band(self, lbda, dlbda, out=1):
        """Compute AB magnitude corresponding to the wavelength band.

        Parameters
        ----------
        lbda : float
            Mean wavelength in Angstrom.
        dlbda : float
            Width of the wavelength band in Angstrom.
        out : 1 or 2
            1: the magnitude is returned,
            2: the magnitude, mean flux and mean wavelength are returned.

        Returns
        -------
        out : magnitude value (out=1)
            or float array containing magnitude,
            mean flux and mean wavelength (out=2).
        """
        i1 = max(0, self.wave.pixel(lbda - dlbda / 2.0, nearest=True,
                                    unit=u.angstrom))
        i2 = min(self.shape[0], self.wave.pixel(lbda + dlbda / 2.0,
                                                nearest=True, unit=u.angstrom))
        if i1 == i2:
            return 99
        else:
            vflux = self.data[i1:i2 + 1].mean()
            vflux2 = (vflux * self.unit).to(u.Unit('erg.s-1.cm-2.Angstrom-1'))
            mag = flux2mag(vflux2.value, lbda)
            if out == 1:
                return mag
            if out == 2:
                return np.array([mag, vflux, lbda])

    def abmag_filter_name(self, name, out=1):
        """Compute AB magnitude using the filter name.

        Parameters
        ----------
        name : string
            'U', 'B', 'V', 'Rc', 'Ic', 'z', 'R-Johnson','F606W'
        out : 1 or 2
            1: the magnitude is returned,
            2: the magnitude, mean flux and mean wavelength are returned.

        Returns
        -------
        out : magnitude value (out=1) or magnitude,
            mean flux and mean wavelength in angstrom (out=2).
        """
        if name == 'U':
            return self.abmag_band(3663, 650, out)
        elif name == 'B':
            return self.abmag_band(4361, 890, out)
        elif name == 'V':
            return self.abmag_band(5448, 840, out)
        elif name == 'Rc':
            return self.abmag_band(6410, 1600., out)
        elif name == 'Ic':
            return self.abmag_band(7980, 1500., out)
        elif name == 'z':
            return self.abmag_band(8930, 1470., out)
        elif name == 'R-Johnson':
            (l0, lmin, lmax, tck) = ABmag_filters.mag_RJohnson()
            return self._filter(l0, lmin, lmax, tck, out)
        elif name == 'F606W':
            (l0, lmin, lmax, tck) = ABmag_filters.mag_F606W()
            return self._filter(l0, lmin, lmax, tck, out)
        else:
            pass

    def abmag_filter(self, lbda, eff, out=1):
        """Compute AB magnitude using array filter.

        Parameters
        ----------
        lbda : float array
            Wavelength values in Angstrom.
        eff : float array
            Efficiency values.
        out : 1 or 2
            1: the magnitude is returned,
            2: the magnitude, mean flux and mean wavelength are returned.

        Returns
        -------
        out : magnitude value (out=1) or magnitude,
            mean flux and mean wavelength (out=2).
        """
        lbda = np.array(lbda)
        eff = np.array(eff)
        if np.shape(lbda) != np.shape(eff):
            raise TypeError('lbda and eff inputs have not the same size.')
        l0 = np.average(lbda, weights=eff)
        lmin = lbda[0]
        lmax = lbda[-1]
        if np.shape(lbda)[0] > 3:
            tck = interpolate.splrep(lbda, eff, k=min(np.shape(lbda)[0], 3))
        else:
            tck = interpolate.splrep(lbda, eff, k=1)
        return self._filter(l0, lmin, lmax, tck, out)

    def _filter(self, l0, lmin, lmax, tck, out=1):
        """compute AB magnitude.

        Parameters
        ----------
        l0 : float
            Mean wavelength in Angstrom.
        lmin : float
            Minimum wavelength in Angstrom.
        lmax : float
            Maximum wavelength in Angstrom.
        tck : 3-tuple
            (t,c,k) contains the spline representation.
            t = the knot-points, c = coefficients and k = the order of the spline.
        out : 1 or 2
            1: the magnitude is returned
            2: the magnitude, mean flux and mean lbda are returned
        """
        imin = self.wave.pixel(lmin, True, u.Angstrom)
        imax = self.wave.pixel(lmax, True, u.Angstrom)
        if imin == imax:
            if imin == 0 or imin == self.shape[0]:
                raise ValueError('Spectrum outside Filter band')
            else:
                raise ValueError('filter band smaller than spectrum step')
        lb = (np.arange(imin, imax) - self.wave.get_crpix() + 1) \
            * self.wave.get_step(u.angstrom) + self.wave.get_crval(u.angstrom)
        w = interpolate.splev(lb, tck, der=0)
        vflux = np.ma.average(self.data[imin:imax], weights=w)
        vflux2 = (vflux * self.unit).to(u.Unit('erg.s-1.cm-2.Angstrom-1')).value
        mag = flux2mag(vflux2, l0)
        if out == 1:
            return mag
        if out == 2:
            return np.array([mag, vflux, l0])

    def truncate(self, lmin=None, lmax=None, unit=u.angstrom):
        """Truncate a spectrum in place.

        Parameters
        ----------
        lmin : float
            Minimum wavelength.
        lmax : float
            Maximum wavelength.
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        """
        if lmin is None:
            i1 = 0
        else:
            if unit is None:
                i1 = max(0, int(lmin + 0.5))
            else:
                i1 = max(0, self.wave.pixel(lmin, nearest=True, unit=unit))
        if lmax is None:
            i2 = self.shape[0]
        else:
            if unit is None:
                i2 = min(self.shape[0], int(lmax + 0.5))
            else:
                i2 = min(self.shape[0],
                         self.wave.pixel(lmax, nearest=True, unit=unit) + 1)

        if i1 == i2:
            raise ValueError('Minimum and maximum wavelengths are equal')

        if i2 == i1 + 1:
            raise ValueError('Minimum and maximum wavelengths '
                             'are outside the spectrum range')

        res = self.__getitem__(slice(i1, i2, 1))
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
        cont : integer
            The continuum [default 0].
        spline : boolean
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
                  plot=False, plot_factor=10, unit=u.angstrom):
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
        peak : boolean
            If true, flux contains the gaussian peak value .
        cont : float
            Continuum value, if None it is estimated by the line through points
            (max(lmin),mean(data[lmin])) and (min(lmax),mean(data[lmax])).
        spline : boolean
            Linear/spline interpolation to interpolate masked values.
        weight : boolean
            If weight is True, the weight is computed as the inverse of
            variance.
        plot : boolean
            If True, the Gaussian is plotted.
        plot_factor : double
            oversampling factor for the overplotted fit

        Returns
        -------
        out : `mpdaf.obj.Gauss1D`
        """
        # truncate the spectrum and compute right and left gaussian values
        if is_int(lmin) or is_float(lmin):
            fmin = None
        else:
            lmin = np.array(lmin, dtype=float)
            fmin = self.mean(lmin[0], lmin[1], unit=unit)
            lmin = (lmin[0] + lmin[1]) / 2.

        if is_int(lmax) or is_float(lmax):
            fmax = None
        else:
            lmax = np.array(lmax, dtype=float)
            fmax = self.mean(lmax[0], lmax[1], unit=unit)
            lmax = (lmax[0] + lmax[1]) / 2.

        # spec = self.truncate(lmin, lmax)
        spec = self.get_lambda(lmin, lmax, unit=unit)
        data = spec._interp_data(spline)
        if unit is None:
            l = np.arange(self.shape, dtype=float)
        else:
            l = spec.wave.coord(unit=unit)
        d = data

        lmin = l[0]
        lmax = l[-1]
        if fmin is None:
            fmin = d[0]
        if fmax is None:
            fmax = d[-1]

        # initial gaussian peak position
        if lpeak is None:
            lpeak = l[d.argmax()]

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
                lpeak2 = l[d.argmin()]
                fwhm = spec.fwhm(lpeak2, cont0, spline, unit=unit)
        sigma = fwhm / (2. * np.sqrt(2. * np.log(2.0)))

        # initial gaussian integrated flux
        if flux is None:
            if unit is None:
                pixel = int(lpeak + 0.5)
            else:
                pixel = spec.wave.pixel(lpeak, nearest=True, unit=unit)
            peak = d[pixel] - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        elif peak is True:
            peak = flux - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        else:
            pass

        # 1d gaussian function
        if cont is None:
            gaussfit = lambda p, x: \
                ((fmax - fmin) * x + lmax * fmin - lmin * fmax) \
                / (lmax - lmin) + p[0] \
                * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) \
                * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))
        else:
            gaussfit = lambda p, x: \
                cont + p[0] * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) \
                * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))
        # 1d Gaussian fit
        if spec.var is not None and weight:
            wght = 1.0 / np.sqrt(np.abs(spec.var))
            np.ma.fix_invalid(wght, copy=False, fill_value=0)
        else:
            wght = np.ones(spec.shape)
        e_gauss_fit = lambda p, x, y, w: w * (gaussfit(p, x) - y)

        # inital guesses for Gaussian Fit
        v0 = [flux, lpeak, sigma]
        # Minimize the sum of squares
        v, covar, info, mesg, success = leastsq(e_gauss_fit, v0[:],
                                                args=(l, d, wght),
                                                maxfev=100000, full_output=1)

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
        flux = v[0]
        lpeak = v[1]
        sigma = np.abs(v[2])
        fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
        peak = flux / np.sqrt(2 * np.pi * (sigma ** 2))
        if err is not None:
            err_flux = err[0]
            err_lpeak = err[1]
            err_sigma = err[2]
            err_fwhm = err_sigma * 2 * np.sqrt(2 * np.log(2))
            err_peak = np.abs(1. / np.sqrt(2 * np.pi) *
                              (err_flux * sigma - flux * err_sigma) /
                              sigma / sigma)
        else:
            err_flux = np.NAN
            err_lpeak = np.NAN
            err_sigma = np.NAN
            err_fwhm = np.NAN
            err_peak = np.NAN

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
        peak : boolean
            If true, flux contains the gaussian peak value
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        """
        gauss = lambda p, x: cont \
            + p[0] * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) \
            * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))

        sigma = fwhm / (2. * np.sqrt(2. * np.log(2.0)))

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

        self.data[imin:imax] = self.data[imin:imax] \
            + gauss(v, wave)

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
        peak : boolean
            If true, flux contains the gaussian peak value .
        cont : float
            Continuum value, if None it is estimated by the line through points
            (max(lmin),mean(data[lmin])) and (min(lmax),mean(data[lmax])).
        spline : boolean
            Linear/spline interpolation to interpolate masked values.
        weight : boolean
            If weight is True, the weight is computed as the inverse of
            variance.
        plot : boolean
            If True, the resulted fit is plotted.
        plot_factor : double
            oversampling factor for the overplotted fit
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.

        Returns
        -------
        out : `mpdaf.obj.Gauss1D`, `mpdaf.obj.Gauss1D`
        """
        if is_int(lmin) or is_float(lmin):
            fmin = None
        else:
            lmin = np.array(lmin, dtype=float)
            fmin = self.mean(lmin[0], lmin[1], weight=False, unit=unit)
            lmin = lmin[1]

        if is_int(lmax) or is_float(lmax):
            fmax = None
        else:
            lmax = np.array(lmax, dtype=float)
            fmax = self.mean(lmax[0], lmax[1], weight=False, unit=unit)
            lmax = lmax[0]

        # spec = self.truncate(lmin, lmax)
        spec = self.get_lambda(lmin, lmax, unit=unit)
        data = spec._interp_data(spline)
        if unit is None:
            l = np.arange(self.shape, dtype=float)
        else:
            l = spec.wave.coord(unit=unit)
        d = data

        lmin = l[0]
        lmax = l[-1]
        if fmin is None:
            fmin = d[0]
        if fmax is None:
            fmax = d[-1]

        # initial gaussian peak position
        if lpeak_1 is None:
            lpeak_1 = l[d.argmax()]

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
                lpeak_1 = l[d.argmin()]
                fwhm = spec.fwhm(lpeak_1, cont0, spline, unit=unit)
        sigma = fwhm / (2. * np.sqrt(2. * np.log(2.0)))

        # initial gaussian integrated flux
        if flux_1 is None:
            if unit is None:
                pixel = int(lpeak_1 + 0.5)
            else:
                pixel = spec.wave.pixel(lpeak_1, nearest=True, unit=unit)
            peak_1 = d[pixel] - cont0
            flux_1 = peak_1 * np.sqrt(2 * np.pi * (sigma ** 2))
        elif peak is True:
            peak_1 = flux_1 - cont0
            flux_1 = peak_1 * np.sqrt(2 * np.pi * (sigma ** 2))
        else:
            pass

        flux_2 = fratio * flux_1

        # 1d gaussian function
        # p[0]: flux 1, p[1]:center 1, p[2]: fwhm, p[3] = peak 2
        gaussfit = lambda p, x: cont0 + p[0] * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2)) \
                                      + p[3] * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) * np.exp(-(x - (p[1] * wratio)) ** 2 / (2 * p[2] ** 2))

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
        v, covar, info, mesg, success = leastsq(e_gauss_fit, v0[:], args=(l, d, wght), maxfev=100000, full_output=1)  # Gauss Fit

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i])) * np.sqrt(np.abs(chisq / dof)) for i in range(len(v))])
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
        fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
        peak_1 = flux_1 / np.sqrt(2 * np.pi * (sigma ** 2))
        peak_2 = flux_2 / np.sqrt(2 * np.pi * (sigma ** 2))
        if err is not None:
            err_flux_1 = err[0]
            err_flux_2 = err[3]
            err_lpeak_1 = err[1]
            err_lpeak_2 = err[1] * wratio
            err_sigma = err[2]
            err_fwhm = err_sigma * 2 * np.sqrt(2 * np.log(2))
            err_peak_1 = np.abs(1. / np.sqrt(2 * np.pi) *
                                (err_flux_1 * sigma - flux_1 * err_sigma) / sigma / sigma)
            err_peak_2 = np.abs(1. / np.sqrt(2 * np.pi) *
                                (err_flux_2 * sigma - flux_2 * err_sigma) / sigma / sigma)
        else:
            err_flux_1 = np.NAN
            err_flux_2 = np.NAN
            err_lpeak_1 = np.NAN
            err_lpeak_2 = np.NAN
            err_sigma = np.NAN
            err_fwhm = np.NAN
            err_peak_1 = np.NAN
            err_peak_2 = np.NAN

        return Gauss1D(lpeak_1, peak_1, flux_1, fwhm, cont0, err_lpeak_1,
                       err_peak_1, err_flux_1, err_fwhm, chisq, dof), \
            Gauss1D(lpeak_2, peak_2, flux_2, fwhm, cont0, err_lpeak_2,
                    err_peak_2, err_flux_2, err_fwhm, chisq, dof)

    def gauss_asymfit(self, lmin, lmax, lpeak=None, flux=None, fwhm=None,
                      cont=None, peak=False, spline=False, weight=True,
                      plot=False, plot_factor=10, unit=u.angstrom):
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
        peak : boolean
            If true, flux contains the gaussian peak value .
        cont : float
            Continuum value, if None it is estimated by the line through points
            (max(lmin),mean(data[lmin])) and (min(lmax),mean(data[lmax])).
        spline : boolean
            Linear/spline interpolation to interpolate masked values.
        weight : boolean
            If weight is True, the weight is computed as the inverse of
            variance.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates. If None, inputs are in pixels.
        plot : boolean
            If True, the resulted fit is plotted.
        plot_factor : double
            oversampling factor for the overplotted fit

        Returns
        -------
        out : `mpdaf.obj.Gauss1D`, `mpdaf.obj.Gauss1D`
            Left and right Gaussian functions.

        """
        if is_int(lmin) or is_float(lmin):
            fmin = None
        else:
            lmin = np.array(lmin, dtype=float)
            fmin = self.mean(lmin[0], lmin[1], weight=False, unit=unit)
            lmin = lmin[1]

        if is_int(lmax) or is_float(lmax):
            fmax = None
        else:
            lmax = np.array(lmax, dtype=float)
            fmax = self.mean(lmax[0], lmax[1], weight=False, unit=unit)
            lmax = lmax[0]

        spec = self.get_lambda(lmin, lmax, unit=unit)
        data = spec._interp_data(spline)
        if unit is None:
            l = np.arange(self.shape, dtype=float)
        else:
            l = spec.wave.coord(unit=unit)
        d = data

        lmin = l[0]
        lmax = l[-1]
        if fmin is None:
            fmin = d[0]
        if fmax is None:
            fmax = d[-1]

        # initial gaussian peak position
        if lpeak is None:
            lpeak = l[d.argmax()]

        # continuum value
        if cont is None:
            cont0 = ((fmax - fmin) * lpeak + lmax * fmin - lmin * fmax) / (lmax - lmin)
        else:
            cont0 = cont

        # initial sigma value
        if fwhm is None:
            try:
                fwhm = spec.fwhm(lpeak, cont0, spline, unit=unit)
            except:
                lpeak = l[d.argmin()]
                fwhm = spec.fwhm(lpeak, cont0, spline, unit=unit)
        sigma = fwhm / (2. * np.sqrt(2. * np.log(2.0)))

        # initial gaussian integrated flux
        if flux is None:
            if unit is None:
                pixel = int(lpeak + 0.5)
            else:
                pixel = spec.wave.pixel(lpeak, nearest=True, unit=unit)
            peak = d[pixel] - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        elif peak is True:
            peak = flux - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        else:
            pass

        # Asymetric gaussian function (p[0]: flux of the right-hand side if it was full... ; p[1]: lambda peak; p[2]:sigma_right; p[3]: sigma_left)
        asymfit = lambda p, x: np.where(x > p[1], cont0 + p[0] / np.sqrt(2 * np.pi) / p[2] * np.exp(-(x - p[1]) ** 2 / (2. * p[2] ** 2)), cont0 + p[0] * p[3] / p[2] / np.sqrt(2 * np.pi) / p[3] * np.exp(-(x - p[1]) ** 2 / (2. * p[3] ** 2)))
        #cont + p[0] / np.sqrt(2*np.pi)/p[2] * np.exp(-(x-p[1])**2/(2.*p[2]**2)) if x > p[1] else cont + p[0] * p[3]/p[2] / np.sqrt(2*np.pi)/p[3] * np.exp(-(x-p[1])**2/(2.*p[3]**2))

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
        v, covar, info, mesg, success = leastsq(e_asym_fit, v0[:], args=(l, d, wght), maxfev=100000, full_output=1)

        # calculate the errors from the estimated covariance matrix
        chisq = sum(info["fvec"] * info["fvec"])
        dof = len(info["fvec"]) - len(v)
        if covar is not None:
            err = np.array([np.sqrt(np.abs(covar[i, i])) * np.sqrt(np.abs(chisq / dof)) for i in range(len(v))])
        else:
            err = None

        # plot
        if plot:
            xxx = np.arange(l[0], l[-1], (l[1] - l[0]) / plot_factor)  # Same wavelenght grid as input spectrum
            ccc = asymfit(v, xxx)
            plt.plot(xxx, ccc, 'm--', label='Asymmetric')

        # return a Gauss1D object
        sigma_right = np.abs(v[2])
        sigma_left = np.abs(v[3])
        flux_right = 0.5 * v[0]
        flux_left = flux_right * sigma_left / sigma_right
        flux = flux_right + flux_left
        fwhm_right = sigma_right * 2 * np.sqrt(2 * np.log(2))
        fwhm_left = sigma_left * 2 * np.sqrt(2 * np.log(2))
        lpeak = v[1]
        peak = flux_right / np.sqrt(2 * np.pi * sigma_right ** 2)
        if err is not None:
            err_flux = err[0]
            err_lpeak = err[1]
            err_sigma_right = err[2]
            err_sigma_left = err[3]
            err_fwhm_right = err_sigma_right * 2 * np.sqrt(2 * np.log(2))
            err_fwhm_left = err_sigma_left * 2 * np.sqrt(2 * np.log(2))
            err_peak = np.abs(1. / np.sqrt(2 * np.pi) * (err_flux * sigma_right - flux * err_sigma_right) / sigma_right / sigma_right)
        else:
            err_flux = np.NAN
            err_lpeak = np.NAN
            err_sigma_right = np.NAN
            err_sigma_left = np.NAN
            err_fwhm_right = np.NAN
            err_fwhm_left = np.NAN
            err_peak = np.NAN

        return Gauss1D(lpeak, peak, flux_left, fwhm_left, cont0, err_lpeak,
                       err_peak, err_flux / 2, err_fwhm_left, chisq, dof), \
            Gauss1D(lpeak, peak, flux_right, fwhm_right, cont0, err_lpeak,
                    err_peak, err_flux / 2, err_fwhm_right, chisq, dof)

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
        peak : boolean
            If true, flux contains the gaussian peak value.
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        """

        asym_gauss = lambda p, x: np.where(x > p[1], cont + p[0] / np.sqrt(2 * np.pi) / p[2] * np.exp(-(x - p[1]) ** 2 / (2. * p[2] ** 2)), cont + p[0] * p[3] / p[2] / np.sqrt(2 * np.pi) / p[3] * np.exp(-(x - p[1]) ** 2 / (2. * p[3] ** 2)))
        sigma_left = fwhm_left / (2. * np.sqrt(2. * np.log(2.0)))
        sigma_right = fwhm_right / (2. * np.sqrt(2. * np.log(2.0)))

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

    def line_gauss_fit(self, lpeak, lmin, lmax, flux=None, fwhm=None,
                       cont=None, peak=False, spline=False, weight=True,
                       plot=False, plot_factor=10, unit=u.angstrom):
        """Perform a Gaussian fit on a line (fixed Gaussian center).

        Uses `scipy.optimize.leastsq` to minimize the sum of squares.

        Parameters
        ----------
        lmin : float or (float,float)
            Minimum wavelength value or wavelength range
            used to initialize the gaussian left value.
        lmax : float or (float,float)
            Maximum wavelength or wavelength range
            used to initialize the gaussian right value.
        lpeak : float
            Input gaussian center, if None it is estimated
            with the wavelength corresponding to the maximum value
            in [max(lmin), min(lmax)]
        flux : float
            Integrated gaussian flux or gaussian peak value if peak is True.
        fwhm : float
            Input gaussian fwhm, if None it is estimated.
        peak : boolean
            If true, flux contains the gaussian peak value .
        cont : float
            Continuum value, if None it is estimated
            by the line through points (max(lmin),mean(data[lmin]))
            and (min(lmax),mean(data[lmax])).
        spline : boolean
            Linear/spline interpolation to interpolate masked values.
        weight : boolean
            If weight is True, the weight is computed as the inverse of
            variance.
        plot : boolean
            If True, the Gaussian is plotted.
        plot_factor : double
            oversampling factor for the overplotted fit

        Returns
        -------
        out : `mpdaf.obj.Gauss1D`
        """
        # truncate the spectrum and compute right and left gaussian values
        if is_int(lmin) or is_float(lmin):
            fmin = None
        else:
            lmin = np.array(lmin, dtype=float)
            fmin = self.mean(lmin[0], lmin[1])
            lmin = (lmin[0] + lmin[1]) / 2.

        if is_int(lmax) or is_float(lmax):
            fmax = None
        else:
            lmax = np.array(lmax, dtype=float)
            fmax = self.mean(lmax[0], lmax[1])
            lmax = (lmax[0] + lmax[1]) / 2.

        # spec = self.truncate(lmin, lmax)
        spec = self.get_lambda(lmin, lmax, unit=unit)
        data = spec._interp_data(spline)
        l = spec.wave.coord(unit=unit)
        d = data

        lmin = l[0]
        lmax = l[-1]
        if fmin is None:
            fmin = d[0]
        if fmax is None:
            fmax = d[-1]

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
                lpeak = l[d.argmin()]
                fwhm = spec.fwhm(lpeak, cont0, spline, unit=unit)
        sigma = fwhm / (2. * np.sqrt(2. * np.log(2.0)))

        # initial gaussian integrated flux
        if flux is None:
            pixel = spec.wave.pixel(lpeak, nearest=True, unit=unit)
            peak = d[pixel] - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        elif peak is True:
            peak = flux - cont0
            flux = peak * np.sqrt(2 * np.pi * (sigma ** 2))
        else:
            pass

        # 1d gaussian function
        if cont is None:
            gaussfit = lambda p, x: \
                ((fmax - fmin) * x + lmax * fmin - lmin * fmax) \
                / (lmax - lmin) + np.abs(p[0]) \
                * (1 / np.sqrt(2 * np.pi * (p[1] ** 2))) \
                * np.exp(-(x - lpeak) ** 2 / (2 * p[1] ** 2))
        else:
            gaussfit = lambda p, x: \
                cont + np.abs(p[0]) * (1 / np.sqrt(2 * np.pi * (p[1] ** 2))) \
                * np.exp(-(x - lpeak) ** 2 / (2 * p[1] ** 2))
        # 1d Gaussian fit
        if spec.var is not None and weight:
            wght = 1.0 / np.sqrt(np.abs(spec.var))
            np.ma.fix_invalid(wght, copy=False, fill_value=0)
        else:
            wght = np.ones(spec.shape)
        e_gauss_fit = lambda p, x, y, w: w * (gaussfit(p, x) - y)

        # inital guesses for Gaussian Fit
        v0 = [flux, sigma]
        # Minimize the sum of squares
        v, covar, info, mesg, success = leastsq(e_gauss_fit, v0[:],
                                                args=(l, d, wght),
                                                maxfev=100000, full_output=1)

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
        flux = np.abs(v[0])
        sigma = np.abs(v[1])
        fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
        peak = flux / np.sqrt(2 * np.pi * (sigma ** 2))
        if err is not None:
            err_flux = np.abs(err[0])
            err_lpeak = 0
            err_sigma = err[1]
            err_fwhm = err_sigma * 2 * np.sqrt(2 * np.log(2))
            err_peak = np.abs(1. / np.sqrt(2 * np.pi) *
                              (err_flux * sigma - flux * err_sigma) /
                              sigma / sigma)
        else:
            err_flux = np.NAN
            err_lpeak = np.NAN
            err_sigma = np.NAN
            err_fwhm = np.NAN
            err_peak = np.NAN

        return Gauss1D(lpeak, peak, flux, fwhm, cont0, err_lpeak,
                       err_peak, err_flux, err_fwhm, chisq, dof)

    def _median_filter(self, kernel_size=1., spline=False, unit=u.angstrom):
        """Perform a median filter on the spectrum.

        Uses `scipy.signal.medfilt`.

        Parameters
        ----------
        kernel_size : float
            Size of the median filter window.
        unit : `astropy.units.Unit`
            unit ot the kernekl size. If None, inputs are in pixels.
        """
        if unit is not None:
            kernel_size = kernel_size / self.get_step(unit=unit)
        ks = int(kernel_size / 2) * 2 + 1

        data = np.empty(self.shape[0] + 2 * ks)
        data[ks:-ks] = self._interp_data(spline)
        data[:ks] = data[ks:2 * ks][::-1]
        data[-ks:] = data[-2 * ks:-ks][::-1]
        data = signal.medfilt(data, ks)
        self._data = data[ks:-ks]

    def median_filter(self, kernel_size=1., spline=False, unit=u.angstrom,
                      inplace=False):
        """Return a spectrum resulted on a median filter on the current
        spectrum.

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
        # Should we filter the spectrum in-place, or filter a copy?

        res = self if inplace else self.copy()

        # Filter the result object in-place.

        res._median_filter(kernel_size, spline, unit)
        return res

    def _convolve(self, other):
        """Convolve the spectrum with a other spectrum or an array.

        Uses `scipy.signal.convolve`. self and other must have the same
        size.

        Parameters
        ----------
        other : 1d-array or Spectrum
            Second spectrum or 1d-array.
        """
        try:
            if isinstance(other, Spectrum):
                if self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '
                                  'with different sizes')
                else:
                    data = other._data
                    if self.unit != other.unit:
                        data = (data * other.unit).to(self.unit).value
                    self._data = signal.convolve(self._data, data, mode='same')
                    if self._var is not None:
                        self._var = signal.convolve(self._var, data, mode='same')
        except IOError as e:
            raise e
        except:
            try:
                self._data = signal.convolve(self._data, other, mode='same')
                if self._var is not None:
                    self._var = signal.convolve(self._var, other, mode='same')
            except:
                raise IOError('Operation forbidden')
                return None

    def convolve(self, other, inplace=False):
        """Return the convolution of the spectrum with a other spectrum or an
        array.

        Uses `scipy.signal.convolve`. self and other must have the same
        size.

        Parameters
        ----------
        other : 1d-array or Spectrum
            Second spectrum or 1d-array.
        inplace : bool
            If False, return a convolved copy of the spectrum (the default).
            If True, convolve the original spectrum in-place, and return that.

        Returns
        -------
        out : Spectrum
        """
        # Should we convolve the spectrum in-place, or convolve a copy?

        res = self if inplace else self.copy()

        # Convolve the result object in-place.

        res._convolve(other)
        return res

    def _fftconvolve(self, other):
        """Convolve the spectrum with a other spectrum or an array using fft.

        Uses `scipy.signal.fftconvolve`. self and other must have the
        same size.

        Parameters
        ----------
        other : 1d-array or Spectrum
            Second spectrum or 1d-array.
        """
        try:
            if isinstance(other, Spectrum):
                if self.shape != other.shape:
                    raise IOError('Operation forbidden '
                                  'for spectra with different sizes')
                else:
                    data = other._data
                    if self.unit != other.unit:
                        data = (data * other.unit).to(self.unit).value
                    self._data = signal.fftconvolve(self._data, data, mode='same')
                    if self._var is not None:
                        self._var = signal.fftconvolve(self._var, data, mode='same')
        except IOError as e:
            raise e
        except:
            try:
                self._data = signal.fftconvolve(self._data, other, mode='same')
                if self._var is not None:
                    self._var = signal.fftconvolve(self._var, other, mode='same')
            except:
                raise IOError('Operation forbidden')

    def fftconvolve(self, other, inplace=False):
        """Return the convolution of the spectrum with a other spectrum or an
        array using fft.

        Uses `scipy.signal.fftconvolve`. self and other must have the
        same size.

        Parameters
        ----------
        other : 1d-array or Spectrum
            Second spectrum or 1d-array.
        inplace : bool
            If False, return a convolved copy of the spectrum (the default).
            If True, convolve the original spectrum in-place, and return that.

        Returns
        -------
        out : Spectrum
        """
        # Should we convolve the spectrum in-place, or convolve a copy?

        res = self if inplace else self.copy()

        # Convolve the result object in-place.

        res._fftconvolve(other)
        return res

    def _correlate(self, other):
        """Cross-correlate the spectrum with a other spectrum or an array.

        Uses `scipy.signal.correlate`. self and other must have the same
        size.

        Parameters
        ----------
        other : 1d-array or Spectrum
            Second spectrum or 1d-array.
        """
        try:
            if isinstance(other, Spectrum):
                if self.shape != other.shape:
                    raise IOError('Operation forbidden for spectra '
                                  'with different sizes')
                else:
                    data = other._data
                    if self.unit != other.unit:
                        data = (data * other.unit).to(self.unit).value
                    self._data = signal.correlate(self._data, data, mode='same')
                    if self._var is not None:
                        self._var = signal.correlate(self._var, data, mode='same')
        except IOError as e:
            raise e
        except:
            try:
                self._data = signal.correlate(self._data, other, mode='same')
                if self._var is not None:
                    self._var = signal.correlate(self._var, other, mode='same')
            except:
                raise IOError('Operation forbidden')

    def correlate(self, other, inplace=False):
        """Return the cross-correlation of the spectrum with a other spectrum
        or an array.

        Uses `scipy.signal.correlate`. self and other must have the same
        size.

        Parameters
        ----------
        other : 1d-array or Spectrum
            Second spectrum or 1d-array.
        inplace : bool
            If False, return the correlation in a new spectrum object (default).
            If True, replace the input spectrum with the correlation.

        Returns
        -------
        out : Spectrum
        """
        # Should we perform the correlation in-place, or to a copy of the spectrum?

        res = self if inplace else self.copy()

        # Perform the correlation in-place.

        res._correlate(other)
        return res

    def _fftconvolve_gauss(self, fwhm, nsig=5, unit=u.angstrom):
        """Convolve the spectrum with a Gaussian using fft.

        Parameters
        ----------
        fwhm : float
            Gaussian fwhm in angstrom
        nsig : integer
            Number of standard deviations.
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        """
        from scipy import special

        sigma = fwhm / (2. * np.sqrt(2. * np.log(2.0)))
        if unit is None:
            s = sigma
        else:
            s = sigma / self.get_step(unit=unit)
        n = nsig * int(s + 0.5)
        n = int(n / 2) * 2
        d = np.arange(-n, n + 1)
        kernel = special.erf((1 + 2 * d) / (2 * np.sqrt(2) * s)) \
            + special.erf((1 - 2 * d) / (2 * np.sqrt(2) * s))
        kernel /= kernel.sum()

        self._data = signal.correlate(self._data, kernel, mode='same')
        if self._var is not None:
            self._var = signal.correlate(self._var, kernel, mode='same')

    def fftconvolve_gauss(self, fwhm, nsig=5, unit=u.angstrom, inplace=False):
        """Return the convolution of the spectrum with a Gaussian using fft.

        Parameters
        ----------
        fwhm : float
            Gaussian fwhm.
        nsig : integer
            Number of standard deviations.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates
        inplace : bool
            If False, return a convolved copy of the spectrum (the default).
            If True, convolve the original spectrum in-place, and return that.

        Returns
        -------
        out : Spectrum
        """
        # Should we convolve the spectrum in-place, or convolve a copy?

        res = self if inplace else self.copy()

        # Convolve the result object in-place.

        res._fftconvolve_gauss(fwhm, nsig, unit)
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
        size : odd integer
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
            k = size / 2

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

        res._data = np.array([(f(lbda[i], step, size, **kwargs)
                                              * data[i:i + size]).sum() for i in range(self.shape[0])])
        res._mask = self._mask

        if self._var is None:
            res._var = None
        else:
            res._var = np.array([(f(lbda[i], step, size, **kwargs)
                                              * data[i:i + size]).sum() for i in range(self.shape[0])])
        return res

    def peak_detection(self, kernel_size=None, unit=u.angstrom):
        """Return a list of peak locations.

        Uses `scipy.signal.medfilt`.

        Parameters
        ----------
        kernel_size : float
            size of the median filter window
        unit : `astropy.units.Unit`
            Type of the wavelength coordinates. If None, inputs are in pixels.
        """
        d = np.abs(self.data - signal.medfilt(self.data, kernel_size))
        cont = self.poly_spec(5)
        ksel = np.where(d > cont.data)
        if unit is None:
            return ksel[0]
        else:
            wave = self.wave.coord(unit=unit)
            return wave[ksel]

    def plot(self, max=None, title=None, noise=False, snr=False,
             lmin=None, lmax=None, ax=None, stretch='linear', unit=u.angstrom,
             noise_kwargs=None, **kwargs):
        """Plot the spectrum. By default, drawstyle is 'steps-mid'.

        Parameters
        ----------
        max : boolean
            If max is True, the plot is normalized to peak at max value.
        title : string
            Figure title (None by default).
        noise : boolean
            If noise is True, the +/- standard deviation is overplotted.
        snr : boolean
            If snr is True, data/sqrt(var) is plotted.
        lmin : float
            Minimum wavelength.
        lmax : float
            Maximum wavelength.
        ax : matplotlib.Axes
            The Axes instance in which the spectrum is drawn.
        unit : `astropy.units.Unit`
            Unit of the wavelength coordinates.
        noise_kwargs : dict
            Properties for the noise plot (if ``noise=True``). Default to
            ``color='0.75', facecolor='0.75', alpha=0.5``.
        kwargs : dict
            kwargs can be used to set plot properties: line label (for auto
            legends), linewidth, anitialising, marker face color, etc.
        """

        if ax is None:
            ax = plt.gca()

        if lmin is not None or lmax is not None:
            res = self.copy()
            res.truncate(lmin, lmax, unit)
        else:
            res = self

        try:
            x = res.wave.coord(unit=unit)
        except u.UnitConversionError:
            unit = res.wave.unit
            x = res.wave.coord(unit=unit)

        data = res.data
        if res.var is None:
            noise = False
            snr = False
        if snr:
            data /= np.sqrt(res.var)
        if max is not None:
            data = data * max / data.max()

        # default plot arguments
        kwargs.setdefault('drawstyle', 'steps-mid')

        if stretch == 'linear':
            ax.plot(x, data, **kwargs)
        elif stretch == 'log':
            ax.semilogy(x, data, **kwargs)
        else:
            raise ValueError("Unknow stretch '{}'".format(stretch))

        if noise:
            sigma = np.sqrt(res.var)
            noisekw = dict(color='0.75', facecolor='0.75', alpha=0.5)
            if noise_kwargs is not None:
                noisekw.update(noise_kwargs)
            ax.fill_between(x, data + sigma, data - sigma, **noisekw)

        if title is not None:
            ax.set_title(title)
        if unit is not None:
            ax.set_xlabel(r'$\lambda$ (%s)' % unit)
        if res.unit is not None:
            ax.set_ylabel(res.unit)

        self._fig = plt.get_current_fig_manager()
        self._unit = unit
        plt.connect('motion_notify_event', self._on_move)
        self._plot_id = len(ax.lines) - 1

    def log_plot(self, max=None, title=None, noise=False, snr=False,
                 lmin=None, lmax=None, ax=None, unit=u.angstrom,
                 **kwargs):
        """Plot the spectrum with y logarithmic scale.

        Shortcut for `mpdaf.obj.Spectrum.plot` with `stretch='log'`.
        By default, drawstyle is 'steps-mid'.

        Parameters
        ----------
        max : boolean
            If max is True, the plot is normalized to peak at max value.
        title : string
            Figure title (None by default).
        noise : boolean
            If noise is True, the +/- standard deviation is overplotted.
        snr : boolean
            If snr is True, data/sqrt(var) is plotted.
        lmin : float
            Minimum wavelength.
        lmax : float
            Maximum wavelength.
        unit : `astropy.units.Unit`
            type of the wavelength coordinates
        ax : matplotlib.Axes
            the Axes instance in which the spectrum is drawn
        kwargs : matplotlib.lines.Line2D
            kwargs can be used to set line properties: line label (for auto
            legends), linewidth, anitialising, marker face color, etc.
        """
        self.plot(max=max, title=title, noise=noise, snr=snr,
                  lmin=lmin, lmax=lmax, ax=ax, stretch='log', unit=unit,
                  **kwargs)

    def _on_move(self, event):
        """print xc,yc,k,lbda and data in the figure toolbar."""
        if event.inaxes is not None:
            xc, yc = event.xdata, event.ydata
            try:
                i = self.wave.pixel(xc, True)
                x = self.wave.coord(i, unit=self._unit)
                val = self.data.data[i]
                s = 'xc= %g yc=%g k=%d lbda=%g data=%g' % (xc, yc, i, x, val)
                self._fig.toolbar.set_message(s)
            except:
                pass

    def ipos(self, filename='None'):
        """Print cursor position in interactive mode.

        xc and yc correspond to the cursor position, k is the nearest pixel,
        lbda contains the wavelength value and data contains spectrum data
        value.

        To read cursor position, click on the left mouse button.

        To remove a cursor position, click on the left mouse button + <d>

        To quit the interactive mode, click on the right mouse button.

        At the end, clicks are saved in self.clicks as dictionary
        {'xc','yc','k','lbda','data'}.

        Parameters
        ----------
        filename : string
            If filename is not None, the cursor values are saved as a fits
            table with columns labeled 'XC'|'YC'|'I'|'X'|'DATA'
        """
        msg = 'To read cursor position, click on the left mouse button'
        self._logger.info(msg)
        msg = 'To remove a cursor position, '\
            'click on the left mouse button + <d>'
        self._logger.info(msg)
        msg = 'To quit the interactive mode, click on the right mouse button.'
        self._logger.info(msg)
        msg = 'After quit, clicks are saved in self.clicks '\
            'as dictionary {xc,yc,k,lbda,data}.'
        self._logger.info(msg)

        if self._clicks is None:
            binding_id = plt.connect('button_press_event', self._on_click)
            self._clicks = SpectrumClicks(binding_id, filename)
            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")
        else:
            self._clicks.filename = filename

    def _on_click(self, event):
        """print xc,yc,k,lbda and data corresponding to the cursor position."""
        if event.key == 'd':
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        xc, yc = event.xdata, event.ydata
                        self._clicks.remove(xc)
                        msg = "new selection:"
                        self._logger.info(msg)
                        for i in range(len(self._clicks.xc)):
                            self._clicks.iprint(i)
                    except:
                        pass
        else:
            if event.button == 1:
                if event.inaxes is not None:
                    try:
                        xc, yc = event.xdata, event.ydata
                        i = self.wave.pixel(xc, True, unit=self._unit)
                        x = self.wave.coord(i, unit=self._unit)
                        val = self.data[i]
                        if len(self._clicks.k) == 0:
                            print('')
                        self._clicks.add(xc, yc, i, x, val)
                        self._clicks.iprint(len(self._clicks.k) - 1)
                    except:
                        pass
            else:
                self._clicks.write_fits()
                # save clicks in a dictionary {'xc','yc','x','data'}
                d = {'xc': self._clicks.xc, 'yc': self._clicks.yc,
                     'k': self._clicks.k, 'lbda': self._clicks.lbda,
                     'data': self._clicks.data}
                self.clicks = d
                # clear
                self._clicks.clear()
                self._clicks = None
                fig = plt.gcf()
                fig.canvas.stop_event_loop_default()

    def idist(self):
        """Get distance and center from 2 cursor positions (interactive mode)

        To quit the interactive mode, click on the right mouse button.
        """
        msg = 'Use 2 mouse clicks to get center and distance.'
        self._logger.info(msg)
        msg = 'To quit the interactive mode, click on the right mouse button.'
        self._logger.info(msg)
        if self._clicks is None:
            binding_id = plt.connect('button_press_event',
                                     self._on_click_dist)
            self._clicks = SpectrumClicks(binding_id)

            warnings.filterwarnings(action="ignore")
            fig = plt.gcf()
            fig.canvas.start_event_loop_default(timeout=-1)
            warnings.filterwarnings(action="default")

    def _on_click_dist(self, event):
        """Print distance and center between 2 cursor positions."""
        if event.button == 1:
            if event.inaxes is not None:
                try:
                    xc, yc = event.xdata, event.ydata
                    i = self.wave.pixel(xc, True, unit=self._unit)
                    x = self.wave.coord(i, unit=self._unit)
                    val = self.data[i]
                    if len(self._clicks.k) == 0:
                        print('')
                    self._clicks.add(xc, yc, i, x, val)
                    self._clicks.iprint(len(self._clicks.k) - 1)
                    if np.sometrue(np.mod(len(self._clicks.k), 2)) == False:
                        dx = np.abs(self._clicks.xc[-1] - self._clicks.xc[-2])
                        xc = (self._clicks.xc[-1] + self._clicks.xc[-2]) / 2
                        msg = 'Center: %f Distance: %f' % (xc, dx)
                        self._logger.info(msg)
                except:
                    pass
        else:
            self._clicks.clear()
            self._clicks = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def igauss_fit(self, nclicks=5):
        """Perform and plots a gaussian fit on spectrum.

        To select minimum, peak and maximum wavelengths, click on the left
        mouse button.

        To quit the interactive mode, click on the right mouse button.

        The parameters of the last gaussian are saved in self.gauss
        (`mpdaf.obj.Gauss1D`)

        Parameters
        ----------
        nclicks : integer (3 or 5)
            3 or 5 clicks.

            Use 3 mouse clicks to get minimim, peak and maximum wavelengths.

            Use 5 mouse clicks: the two first select a range
            of minimum wavelengths, the 3th selects the peak wavelength and
            the two last clicks select a range of maximum wavelengths
            - see `mpdaf.obj.Spectrum.gauss_fit`.
        """
        if nclicks == 3:
            msg = 'Use 3 mouse clicks to get minimim, '\
                'peak and maximum wavelengths.'
            self._logger.info(msg)
            msg = 'To quit the interactive mode, '\
                'click on the right mouse button.'
            self._logger.info(msg)
            msg = 'The parameters of the last '\
                'gaussian are saved in self.gauss.'
            self._logger.info(msg)
            if self._clicks is None:
                binding_id = plt.connect('button_press_event',
                                         self._on_3clicks_gauss_fit)
                self._clicks = SpectrumClicks(binding_id)
                warnings.filterwarnings(action="ignore")
                fig = plt.gcf()
                fig.canvas.start_event_loop_default(timeout=-1)
                warnings.filterwarnings(action="default")
        else:
            msg = 'Use the 2 first mouse clicks to get the wavelength '\
                'range to compute the gaussian left value.'
            self._logger.info(msg)
            msg = 'Use the next click to get the peak wavelength.'
            self._logger.info(msg)
            msg = 'Use the 2 last mouse clicks to get the wavelength range '\
                'to compute the gaussian rigth value.'
            self._logger.info(msg)
            msg = 'To quit the interactive mode, '\
                'click on the right mouse button.'
            self._logger.info(msg)
            msg = 'The parameters of the last gaussian '\
                'are saved in self.gauss.'
            self._logger.info(msg)
            if self._clicks is None:
                binding_id = plt.connect('button_press_event',
                                         self._on_5clicks_gauss_fit)
                self._clicks = SpectrumClicks(binding_id)
                warnings.filterwarnings(action="ignore")
                fig = plt.gcf()
                fig.canvas.start_event_loop_default(timeout=-1)
                warnings.filterwarnings(action="default")

    def _on_3clicks_gauss_fit(self, event):
        """Perform polynomial fit on spectrum (interactive mode)."""
        if event.button == 1:
            if event.inaxes is not None:
                try:
                    xc, yc = event.xdata, event.ydata
                    i = self.wave.pixel(xc, True, unit=self._unit)
                    x = self.wave.coord(i, unit=self._unit)
                    val = self.data[i]
                    if len(self._clicks.k) == 0:
                        print('')
                    self._clicks.add(xc, yc, i, x, val)
                    if np.sometrue(np.mod(len(self._clicks.k), 3)) == False:
                        lmin = self._clicks.lbda[-3]
                        lpeak = self._clicks.lbda[-2]
                        lmax = self._clicks.lbda[-1]
                        self.gauss = self.gauss_fit(lmin, lmax,
                                                    lpeak=lpeak, plot=True,
                                                    unit=self._unit)
                        self.gauss.print_param()
                        self._clicks.id_lines.append(len(plt.gca().lines) - 1)
                except:
                    pass
        else:
            self._clicks.clear()
            self._clicks = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def _on_5clicks_gauss_fit(self, event):
        """Perform polynomial fit on spectrum (interactive mode)."""
        if event.button == 1:
            if event.inaxes is not None:
                try:
                    xc, yc = event.xdata, event.ydata
                    i = self.wave.pixel(xc, True, unit=self._unit)
                    x = self.wave.coord(i, unit=self._unit)
                    val = self.data[i]
                    if len(self._clicks.k) == 0:
                        print('')
                    self._clicks.add(xc, yc, i, x, val)
                    if np.sometrue(np.mod(len(self._clicks.k), 5)) == False:
                        lmin1 = self._clicks.lbda[-5]
                        lmin2 = self._clicks.lbda[-4]
                        lpeak = self._clicks.lbda[-3]
                        lmax1 = self._clicks.lbda[-2]
                        lmax2 = self._clicks.lbda[-1]
                        self.gauss = self.gauss_fit((lmin1, lmin2),
                                                    (lmax1, lmax2),
                                                    lpeak=lpeak, plot=True,
                                                    unit=self._unit)
                        self.gauss.print_param()
                        self._clicks.id_lines.append(len(plt.gca().lines) - 1)
                except:
                    pass
        else:
            self._clicks.clear()
            self._clicks = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def imask(self):
        """Over-plot masked values (interactive mode)."""
        try:
            try:
                del plt.gca().lines[self._plot_mask_id]
            except:
                pass
            lbda = self.wave.coord(unit=self._unit)
            drawstyle = plt.gca().lines[self._plot_id].get_drawstyle()
            plt.plot(lbda, self.data.data, drawstyle=drawstyle,
                     hold=True, alpha=0.3)
            self._plot_mask_id = len(plt.gca().lines) - 1
        except:
            pass

    def igauss_asymfit(self, nclicks=5):
        """Performs and plots a asymetric gaussian fit on spectrum.

        To select minimum, peak and maximum wavelengths, click on the left
        mouse button.

        To quit the interactive mode, click on the right mouse button.

        The parameters of the returned gaussian functions are saved in
        self.gauss2 (`mpdaf.obj.Gauss1D`, `mpdaf.obj.Gauss1D`)

        Parameters
        ----------
        nclicks : integer (3 or 5)
            3 or 5 clicks.

            Use 3 mouse clicks to get minimim, peak and maximum wavelengths.

            Use 5 mouse clicks: the two first select a range
            of minimum wavelengths, the 3th selects the peak wavelength and
            the two last clicks select a range of maximum wavelengths
            - see `mpdaf.obj.Spectrum.gauss_symfit`.
        """
        if nclicks == 3:
            msg = 'Use 3 mouse clicks to get minimim, '\
                'peak and maximum wavelengths.'
            self._logger.info(msg)
            msg = 'To quit the interactive mode, '\
                'click on the right mouse button.'
            self._logger.info(msg)
            msg = 'The parameters of the '\
                'gaussian functions are saved in self.gauss2.'
            self._logger.info(msg)
            if self._clicks is None:
                binding_id = plt.connect('button_press_event',
                                         self._on_3clicks_gauss_asymfit)
                self._clicks = SpectrumClicks(binding_id)
                warnings.filterwarnings(action="ignore")
                fig = plt.gcf()
                fig.canvas.start_event_loop_default(timeout=-1)
                warnings.filterwarnings(action="default")
        else:
            msg = 'Use the 2 first mouse clicks to get the wavelength '\
                'range to compute the gaussian left value.'
            self._logger.info(msg)
            msg = 'Use the next click to get the peak wavelength.'
            self._logger.info(msg)
            msg = 'Use the 2 last mouse clicks to get the wavelength range '\
                'to compute the gaussian rigth value.'
            self._logger.info(msg)
            msg = 'To quit the interactive mode, '\
                'click on the right mouse button.'
            self._logger.info(msg)
            msg = 'The parameters of the resulted gaussian functions'\
                'are saved in self.gauss2.'
            self._logger.info(msg)
            if self._clicks is None:
                binding_id = plt.connect('button_press_event',
                                         self._on_5clicks_gauss_asymfit)
                self._clicks = SpectrumClicks(binding_id)
                warnings.filterwarnings(action="ignore")
                fig = plt.gcf()
                fig.canvas.start_event_loop_default(timeout=-1)
                warnings.filterwarnings(action="default")

    def _on_3clicks_gauss_asymfit(self, event):
        """Performs asymetrical gaussian fit on spectrum (interactive mode)."""
        if event.button == 1:
            if event.inaxes is not None:
                try:
                    xc, yc = event.xdata, event.ydata
                    i = self.wave.pixel(xc, True, unit=self._unit)
                    x = self.wave.coord(i, unit=self._unit)
                    val = self.data[i]
                    if len(self._clicks.k) == 0:
                        print('')
                    self._clicks.add(xc, yc, i, x, val)
                    if np.sometrue(np.mod(len(self._clicks.k), 3)) == False:
                        lmin = self._clicks.lbda[-3]
                        lpeak = self._clicks.lbda[-2]
                        lmax = self._clicks.lbda[-1]
                        self.gauss2 = self.gauss_asymfit(
                            lmin, lmax, lpeak=lpeak, plot=True,
                            unit=self._unit)
                        self._logger.info('left:')
                        self.gauss2[0].print_param()
                        self._logger.info('right:')
                        self.gauss2[1].print_param()
                        self._clicks.id_lines.append(len(plt.gca().lines) - 1)
                except:
                    pass
        else:
            self._clicks.clear()
            self._clicks = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    def _on_5clicks_gauss_asymfit(self, event):
        """Performs asymetrical gaussian fit on spectrum (interactive mode)."""
        if event.button == 1:
            if event.inaxes is not None:
                try:
                    xc, yc = event.xdata, event.ydata
                    i = self.wave.pixel(xc, True, unit=self._unit)
                    x = self.wave.coord(i, unit=self._unit)
                    val = self.data[i]
                    if len(self._clicks.k) == 0:
                        print('')
                    self._clicks.add(xc, yc, i, x, val)
                    if np.sometrue(np.mod(len(self._clicks.k), 5)) == False:
                        lmin1 = self._clicks.lbda[-5]
                        lmin2 = self._clicks.lbda[-4]
                        lpeak = self._clicks.lbda[-3]
                        lmax1 = self._clicks.lbda[-2]
                        lmax2 = self._clicks.lbda[-1]
                        self.gauss2 = self.gauss_asymfit(
                            (lmin1, lmin2), (lmax1, lmax2), lpeak=lpeak,
                            plot=True, unit=self._unit)
                        self._logger.info('left:')
                        self.gauss2[0].print_param()
                        self._logger.info('right:')
                        self.gauss2[1].print_param()
                        self._clicks.id_lines.append(len(plt.gca().lines) - 1)
                except:
                    pass
        else:
            self._clicks.clear()
            self._clicks = None
            fig = plt.gcf()
            fig.canvas.stop_event_loop_default()

    @deprecated('rebin_factor method is deprecated in favor of rebin_mean')
    def rebin_factor(self, factor, margin='center'):
        return self.rebin_mean(factor, margin)

    @deprecated('rebin method is deprecated in favor of resample')
    def rebin(self, step, start=None, shape=None,
              spline=False, notnoise=False, unit=u.angstrom):
        return self.resample(step, start, shape, spline, notnoise, unit)
