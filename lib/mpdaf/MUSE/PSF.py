"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2013-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2014-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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
import numpy as np

from astropy.modeling.models import Moffat2D, Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
from scipy import special


class LSF(object):

    """This class offers Line Spread Function models for MUSE.

    Attributes
    ----------
    typ : str
        LSF type.

    """

    def __init__(self, typ="qsim_v1"):
        """Manages LSF model.

        *qsim_v1* : simple model where the LSF is supposed to be constant over
        the FoV. It is a convolution of a step function with a Gaussian.  The
        resulting function is then sample by the pixel size.  The slit width is
        assumed to be constant (2.09 pixels).  The Gaussian sigma parameter is
        a polynomial approximation of order 3 with wavelength.

        Parameters
        ----------
        typ : str
            type of LSF

        """
        self.typ = typ

    def get_LSF(self, lbda, step, size, **kargs):
        """Return an array containing the LSF.

        Parameters
        ----------
        lbda : float
            Wavelength value in A.
        step : float
            Size of the pixel in A.
        size : odd int
            Number of pixels.
        kargs : dict
            kargs can be used to set LSF parameters.

        Returns
        -------
        out : array
            array containing the LSF

        """
        if self.typ == "qsim_v1":
            T = lambda x: np.exp((-x ** 2) / 2.0) + np.sqrt(2.0 * np.pi) \
                * x * special.erf(x / np.sqrt(2.0)) / 2.0
            c = np.array([-0.09876662, 0.44410609, -0.03166038, 0.46285363])
            sigma = lambda x: c[3] + c[2] * x + c[1] * x ** 2 + c[0] * x ** 3

            x = (lbda - 6975.0) / 4650.0
            h_2 = 2.09 / 2.0
            sig = sigma(x)
            dy_2 = step / 1.25 / 2.0

            k = size // 2
            y = np.arange(-k, k + 1)

            y1 = (y - h_2) / sig
            y2 = (y + h_2) / sig

            lsf = T(y2 + dy_2) - T(y2 - dy_2) - T(y1 + dy_2) + T(y1 - dy_2)
        else:
            raise IOError('Invalid LSF type')

        lsf /= lsf.sum()
        return lsf

    def size(self, lbda, step, epsilon, **kargs):
        """Return the LSF size in pixels.

        Parameters
        ----------
        lbda : float
            Wavelength value in A.
        step : float
            Size of the pixel in A.
        epsilon : float
            This factor is used to determine the size of LSF
                min(LSF) < max(LSF)*epsilon

        Returns
        -------
        out : int
            Size in pixels.

        """
        x0 = lbda
        # x = np.array([x0])
        diff = -1
        k = 0
        while diff < 0:
            k = k + 1
            g = self.get_LSF(x0, step, 2 * k + 1, **kargs)
            diff = epsilon * g[1] - g[0]
        size = k * 2 + 1
        return size


def Moffat(step_arcsec, Nfsf, beta, fwhm):
    """Compute FSF with a Moffat function

    Parameters
    ----------
    step_arcsec : float
        Size of the pixel in arcsec.
    Nfsf : int
        Spatial dimension of the FSF.
    beta : float
        Power index of the Moffat.
    fwhm : float or array of float
        Moffat fwhm in arcsec.

    Returns
    -------
    PSF_Moffat : array (Nz, Nfsf, Nfsf)
        MUSE PSF
    fwhm_pix : array (Nz)
        fwhm of the PSF in pixels
    fwhm_arcsec : array (Nz)
        fwhm of the PSF in arcsec

    """
    # conversion fwhm arcsec -> pixels
    fwhm_pix = fwhm / step_arcsec

    # alpha coefficient in pixel
    alpha = fwhm_pix / (2 * np.sqrt(2**(1 / beta) - 1))
    amplitude = (beta - 1) * (np.pi * alpha**2)
    center = Nfsf // 2
    yy, xx = np.mgrid[:Nfsf, :Nfsf]

    if np.isscalar(alpha):
        moffat = Moffat2D(amplitude, center, center, alpha, beta)
        PSF_Moffat = moffat(xx, yy)
        # Normalization
        # PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat)
    else:
        Nz = alpha.shape[0]
        moffat = Moffat2D(amplitude, [center] * Nz, [center] * Nz,
                          alpha, [beta] * Nz, n_models=Nz)
        PSF_Moffat = moffat(xx, yy, model_set_axis=False)
        # Normalization
        # PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat, axis=(1, 2))\
        #     [:, np.newaxis, np.newaxis]

    return PSF_Moffat, fwhm_pix


def MOFFAT1(lbda, step_arcsec, Nfsf, beta, a, b):
    """Compute PSF with a Moffat function.

    Parameters
    ----------
    lbda : float or array of float
        Wavelength values in Angstrom.
    step_arcsec : float
        Size of the pixel in arcsec
    Nfsf : int
        Spatial dimension of the FSF.
    beta : float
        Power index of the Moffat.
    a : float
        Moffat parameter in arcsec (fwhm=a+b*lbda)
    b : float
        Moffat parameter (fwhm=a+b*lbda)

    Returns
    -------
    PSF_Moffat : array (Nz, Nfsf, Nfsf)
        MUSE PSF
    fwhm_pix : array (Nz)
        fwhm of the PSF in pixels
    fwhm_arcsec : array (Nz)
        fwhm of the PSF in arcsec

    """
    fwhm_arcsec = a + b * lbda
    PSF_Moffat, fwhm_pix = Moffat(step_arcsec, Nfsf, beta, fwhm_arcsec)
    return PSF_Moffat, fwhm_pix, fwhm_arcsec


class FSF(object):
    """This class offers Field Spread Function (FSF) models for MUSE.

    The only supported model currently is "MOFFAT1".

    MOFFAT1: Moffat function with a FWHM which varies linearly with the
    wavelength. Parameters:

        - beta (float) Power index of the Moffat.
        - a (float) constant in arcsec which defined the FWHM (fwhm=a+b*lbda)
        - b (float) constant which defined the FWHM (fwhm=a+b*lbda)

    Attributes
    ----------
    typ : str
        FSF type. Only "MOFFAT1" is supported currently.

    """

    def __init__(self, typ="MOFFAT1"):
        self.typ = typ

    def get_FSF(self, lbda, step, size, **kwargs):
        """Return an array containing the FSF for a given wavelength.

        Parameters
        ----------
        lbda : float or array of float
            Wavelength value in A.
        step : float
            Size of the pixel in arcsec.
        size : int
            Number of pixels.
        kwargs : dict
            Additional arguments are passed to the FSF function (e.g.
            ``MOFFAT1``).

        Returns
        -------
        FSF : array (size, size)
            MUSE FSF
        fwhm_pix : float
            fwhm of the FSF in pixels
        fwhm_arcsec : array
            fwhm of the FSF in arcsec

        """
        lbda = np.asarray(lbda)
        if self.typ == "MOFFAT1":
            return MOFFAT1(lbda, step, size, **kwargs)
        else:
            raise IOError('Invalid FSF type')

    def get_FSF_cube(self, cube, size, **kargs):
        """Return a cube of FSFs corresponding to the MUSE data cube
        given as input: a FSF per MUSE spectral pixels, the step of
        the FSF pixel is equal to the spatial step of the MUSE data cube.

        Parameters
        ----------
        cube : `mpdaf.obj.Cube`
            MUSE data cube
        size : int
            FSF size in pixels.
        kargs : dict
            kargs can be used to set FSF parameters.

        Returns
        -------
        FSF : array (cube.shape[0], size, size)
            Cube containing MUSE FSF (one per wavelength)
        fwhm_pix : array(cube.shape[0])
            fwhm of the FSF in pixels
        fwhm_arcsec : array(cube.shape[0])
            fwhm of the FSF in arcsec

        """
        # size of the pixel in arcsec.
        step = cube.wcs.get_step(unit=u.arcsec)[0]
        # wavelength coordinates of the MUSE spectral pixels.
        lbda = cube.wave.coord()
        if self.typ == "MOFFAT1":
            return MOFFAT1(lbda, step, size, **kargs)
        else:
            raise IOError('Invalid FSF type')


def get_FSF_from_cube_keywords(cube, size):
    """Return a cube of FSFs corresponding to the keywords presents in the
    MUSE data cube primary header ('FSF***')

    The step of the FSF pixel is equal to the spatial step of the MUSE data
    cube.  If the cube corresponds to mosaic of several fields ('NFIELDS'>1),
    a list of FSF cubes is returned.

    Parameters
    ----------
    cube : `mpdaf.obj.Cube`
        MUSE data cube
    size : int
        FSF size in pixels.

    Returns
    -------
    FSF : array (cube.shape[0], size, size) or list of arrays
        Cube containing MUSE FSF (one per wavelength). One cube per field.
    fwhm_pix : array(cube.shape[0]) or list of arrays
        fwhm of the FSF in pixels
    fwhm_arcsec : array(cube.shape[0]) or list of arrays
        fwhm of the FSF in arcsec

    """
    if 'FSFMODE' not in cube.primary_header:
        raise IOError('No FSF keywords in the cube primary header')

    FSF_mode = cube.primary_header['FSFMODE']
    if FSF_mode != 'MOFFAT1':
        raise IOError('This method is coded only for FSFMODE=MOFFAT1')

    nfields = cube.primary_header['NFIELDS']
    FSF_model = FSF(FSF_mode)
    if nfields == 1:  # just one FSF
        nf = 0
        beta = cube.primary_header['FSF%02dBET' % nf]
        a = cube.primary_header['FSF%02dFWA' % nf]
        b = cube.primary_header['FSF%02dFWB' % nf]
        return FSF_model.get_FSF_cube(cube, size, beta=beta, a=a, b=b)
    else:
        l_PSF = []
        l_fwhm_pix = []
        l_fwhm_arcsec = []
        for i in range(1, nfields + 1):
            beta = cube.primary_header['FSF%02dBET' % i]
            a = cube.primary_header['FSF%02dFWA' % i]
            b = cube.primary_header['FSF%02dFWB' % i]
            PSF, fwhm_pix, fwhm_arcsec = \
                FSF_model.get_FSF_cube(cube, size, beta=beta, a=a, b=b)
            l_PSF.append(PSF)
            l_fwhm_pix.append(fwhm_pix)
            l_fwhm_arcsec.append(fwhm_arcsec)
        return l_PSF, l_fwhm_pix, l_fwhm_arcsec


def create_psf_cube(shape, fwhm, beta=None, wcs=None, unit_fwhm=u.arcsec):
    """Create a PSF cube with FWHM varying along each wavelength plane.

    Depending on the value of the 'fwhm' parameter, the PSF can be a Gaussian
    or a Moffat.

    Parameters
    ----------
    fwhm : list
        List of FHHM values for each wavelength plane.
    beta : float or none
        if not none, the PSF is a Moffat function with beta value,
        else it is a Gaussian.

    """
    if len(fwhm) != shape[0]:
        raise ValueError('fwhm length ({}) and input shape ({}) do not match'
                         .format(len(fwhm), shape[0]))

    nl = shape[0]
    y0, x0 = (np.array(shape[1:]) - 1) / 2.0
    yy, xx = np.mgrid[:shape[1], :shape[2]]

    fwhm = np.asarray(fwhm)
    if unit_fwhm is not None:
        fwhm = fwhm / wcs.get_step(unit=unit_fwhm)[0]

    if beta is None:
        # a Gaussian expected.
        stddev = fwhm * gaussian_fwhm_to_sigma
        m = Gaussian2D(amplitude=[1] * nl, theta=[0] * nl,
                       x_mean=[x0] * nl, y_mean=[y0] * nl,
                       x_stddev=stddev, y_stddev=stddev, n_models=nl)
    else:
        alpha = fwhm / (2 * np.sqrt(2 ** (1.0 / beta) - 1.0))
        m = Moffat2D(amplitude=[1] * nl, x_0=[x0] * nl, y_0=[y0] * nl,
                     gamma=alpha, alpha=[beta] * nl, n_models=nl)

    cube = m(xx, yy, model_set_axis=False)
    cube /= cube.sum(axis=(1, 2))[:, None, None]
    return cube
