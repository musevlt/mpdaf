# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
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

import logging

__all__ = ('Gauss1D', 'Gauss2D', 'Moffat2D')


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
