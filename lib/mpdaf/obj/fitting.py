# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2017 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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

import astropy.units as u
import logging
import numpy as np

from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.models import Disk2D
from collections import OrderedDict
from scipy.special import j1, jn_zeros

__all__ = ('Gauss1D', 'Gauss2D', 'Moffat2D', 'SmoothDisk2D',
           'SmoothOuterDisk2D', 'EllipticalMoffat2D', 'AiryDisk2D')


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
        self.chisq = chisq
        self.dof = dof

    def copy(self):
        """Copy Gauss1D object in a new one and returns it."""
        return Gauss1D(self.lpeak, self.peak, self.flux, self.fwhm,
                       self.cont, self.err_lpeak, self.err_peak,
                       self.err_flux, self.err_fwhm, self.chisq, self.dof)

    def print_param(self):
        """Print Gaussian parameters."""
        info = logging.getLogger(__name__).info
        info('Gaussian center = %g (error:%g)', self.lpeak, self.err_lpeak)
        info('Gaussian integrated flux = %g (error:%g)',
             self.flux, self.err_flux)
        info('Gaussian peak value = %g (error:%g)', self.peak, self.err_peak)
        info('Gaussian fwhm = %g (error:%g)', self.fwhm, self.err_fwhm)
        info('Gaussian continuum = %g', self.cont)


class Gauss2D(object):

    """This class stores 2D gaussian parameters.

    Attributes
    ----------
    center : (float,float)
        Gaussian center (y,x).
    flux : float
        Gaussian integrated flux.
    fwhm : (float,float)
        Gaussian fwhm (fhwm_y,fwhm_x).
    cont : float
        Continuum value.
    rot : float
        Rotation in degrees.
    peak : float
        Gaussian peak value.
    err_center : (float,float)
        Estimated error on Gaussian center.
    err_flux : float
        Estimated error on Gaussian integrated flux.
    err_fwhm : (float,float)
        Estimated error on Gaussian fwhm.
    err_cont : float
        Estimated error on continuum value.
    err_rot : float
        Estimated error on rotation.
    err_peak : float
        Estimated error on Gaussian peak value.
    ima : `~mpdaf.obj.Image`
        Gaussian image

    """

    def __init__(self, center, flux, fwhm, cont, rot, peak, err_center,
                 err_flux, err_fwhm, err_cont, err_rot, err_peak, ima=None):
        self.center = center
        self.flux = flux
        self.fwhm = fwhm
        self.cont = cont
        self.rot = rot
        self.peak = peak
        self.err_center = err_center
        self.err_flux = err_flux
        self.err_fwhm = err_fwhm
        self.err_cont = err_cont
        self.err_rot = err_rot
        self.err_peak = err_peak
        self.ima = ima

    def copy(self):
        """Copy Gauss2D object in a new one and returns it."""
        return Gauss2D(self.center, self.flux, self.fwhm, self.cont,
                       self.rot, self.peak, self.err_center, self.err_flux,
                       self.err_fwhm, self.err_cont, self.err_rot,
                       self.err_peak)

    def print_param(self):
        """Print Gaussian parameters."""
        info = logging.getLogger(__name__).info
        info('Gaussian center = (%g,%g) (error:(%g,%g))', self.center[0],
             self.center[1], self.err_center[0], self.err_center[1])
        info('Gaussian integrated flux = %g (error:%g)',
             self.flux, self.err_flux)
        info('Gaussian peak value = %g (error:%g)', self.peak, self.err_peak)
        info('Gaussian fwhm = (%g,%g) (error:(%g,%g))',
             self.fwhm[0], self.fwhm[1], self.err_fwhm[0], self.err_fwhm[1])
        info('Rotation in degree: %g (error:%g)', self.rot, self.err_rot)
        info('Gaussian continuum = %g (error:%g)', self.cont, self.err_cont)


class Moffat2D(object):

    """This class stores 2D moffat parameters.

    Attributes
    ----------
    center : (float,float)
        peak center (y,x).
    flux : float
        integrated flux.
    fwhm : (float,float)
        fwhm (fhwm_y,fwhm_x).
    cont : float
        Continuum value.
    n : int
        Atmospheric scattering coefficient.
    rot : float
        Rotation in degrees.
    peak : float
        intensity peak value.
    err_center : (float,float)
        Estimated error on center.
    err_flux : float
        Estimated error on integrated flux.
    err_fwhm : (float,float)
        Estimated error on fwhm.
    err_cont : float
        Estimated error on continuum value.
    err_n : float
        Estimated error on n coefficient.
    err_rot : float
        Estimated error on rotation.
    err_peak : float
        Estimated error on peak value.
    ima : `~mpdaf.obj.Image`
        Moffat image

    """

    def __init__(self, center, flux, fwhm, cont, n, rot, peak, err_center,
                 err_flux, err_fwhm, err_cont, err_n, err_rot, err_peak,
                 ima=None):
        self.center = center
        self.flux = flux
        self.fwhm = fwhm
        self.cont = cont
        self.rot = rot
        self.peak = peak
        self.n = n
        self.err_center = err_center
        self.err_flux = err_flux
        self.err_fwhm = err_fwhm
        self.err_cont = err_cont
        self.err_rot = err_rot
        self.err_peak = err_peak
        self.err_n = err_n
        self.ima = ima

    def copy(self):
        """Return a copy of a Moffat2D object."""
        return Moffat2D(self.center, self.flux, self.fwhm, self.cont,
                        self.n, self.rot, self.peak, self.err_center,
                        self.err_flux, self.err_fwhm, self.err_cont,
                        self.err_n, self.err_rot, self.err_peak)

    def print_param(self):
        """Print Moffat parameters."""
        info = logging.getLogger(__name__).info
        info('center = (%g,%g) (error:(%g,%g))', self.center[0],
             self.center[1], self.err_center[0], self.err_center[1])
        info('integrated flux = %g (error:%g)', self.flux, self.err_flux)
        info('peak value = %g (error:%g)', self.peak, self.err_peak)
        info('fwhm = (%g,%g) (error:(%g,%g))',
             self.fwhm[0], self.fwhm[1], self.err_fwhm[0], self.err_fwhm[1])
        info('n = %g (error:%g)', self.n, self.err_n)
        info('rotation in degree: %g (error:%g)', self.rot, self.err_rot)
        info('continuum = %g (error:%g)', self.cont, self.err_cont)


class SmoothDisk2D(Disk2D):
    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, R_0):
        """Two dimensional Disk model function"""
        rr = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
        return amplitude / (1 + np.exp((rr - R_0)))


class SmoothOuterDisk2D(Disk2D):
    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, R_0):
        """Two dimensional Disk model function"""
        rr = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
        return amplitude * (1 - 1 / (1 + np.exp((rr - R_0))))


RZ = jn_zeros(1, 1)[0] / np.pi


class AiryDisk2D(Fittable2DModel):
    """ Two dimensional Airy disk model.

    Optimized version from astropy.modeling.AiryDisk2D

    """

    amplitude = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    radius = Parameter(default=1)

    @classmethod
    def evaluate(cls, x, y, amplitude, x_0, y_0, radius):
        """Two dimensional Airy model function"""
        # Since r can be zero, we have to take care to treat that case
        # separately so as not to raise a numpy warning
        r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
        z = np.ones(r.shape)
        mask = r > 0
        rt = r[mask] * (np.pi * RZ / radius)
        z[mask] = (2.0 * j1(rt) / rt) ** 2
        z *= amplitude
        return z


class EllipticalMoffat2D(Fittable2DModel):
    """
    Two dimensional Moffat model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the model.
    x_0 : float
        x position of the maximum of the Moffat model.
    y_0 : float
        y position of the maximum of the Moffat model.
    gamma : float
        Core width of the Moffat model.
    alpha : float
        Power index of the Moffat model.

    See Also
    --------
    Gaussian2D, Box2D

    Notes
    -----
    Model formula:

    .. math::

        f(x, y) = A \\left(1 + \\frac{\\left(x - x_{0}\\right)^{2} +
        \\left(y - y_{0}\\right)^{2}}{\\gamma^{2}}\\right)^{- \\alpha}
    """

    # TODO: implement this...
    fit_deriv = None

    amplitude = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    x_gamma = Parameter(default=1)
    y_gamma = Parameter(default=1)
    alpha = Parameter(default=1)
    theta = Parameter(default=0)

    @property
    def x_fwhm(self):
        """Moffat full width at half maximum."""
        return 2.0 * self.x_gamma * np.sqrt(2.0 ** (1.0 / self.alpha) - 1.0)

    @property
    def y_fwhm(self):
        """Moffat full width at half maximum."""
        return 2.0 * self.y_gamma * np.sqrt(2.0 ** (1.0 / self.alpha) - 1.0)

    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, x_gamma, y_gamma, alpha, theta):
        """Two dimensional Moffat model function"""

        cost2 = np.cos(theta) ** 2
        sint2 = np.sin(theta) ** 2
        sin2t = np.sin(2. * theta)
        xstd2 = x_gamma ** 2
        ystd2 = y_gamma ** 2
        xdiff = x - x_0
        ydiff = y - y_0

        a = ((cost2 / xstd2) + (sint2 / ystd2))
        b = ((sin2t / xstd2) - (sin2t / ystd2))
        c = ((sint2 / xstd2) + (cost2 / ystd2))

        rr_gg = (a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2)
        return amplitude * (1 + rr_gg) ** (-alpha)

    @property
    def input_units(self):
        if self.x_0.unit is None:
            return None
        else:
            return {'x': self.x_0.unit,
                    'y': self.y_0.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit['x'] != inputs_unit['y']:
            raise u.UnitsError("Units of 'x' and 'y' inputs should match")
        return OrderedDict([('x_0', inputs_unit['x']),
                            ('y_0', inputs_unit['x']),
                            ('x_gamma', inputs_unit['x']),
                            ('y_gamma', inputs_unit['x']),
                            ('amplitude', outputs_unit['z'])])
