"""Copyright 2010-2016 CNRS/CRAL

This file is part of MPDAF.

MPDAF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version

MPDAF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MPDAF.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import absolute_import, division

from astropy.convolution import Model2DKernel
from astropy.modeling.functional_models import Moffat2D
import astropy.units as u
import numpy as np
from scipy import special

from six.moves import range


class LSF(object):

    """This class offers Line Spread Function models for MUSE.

    Attributes
    ----------

    typ (string) : LSF type.
    """

    def __init__(self, typ="qsim_v1"):
        """Manages LSF model.

        Parameters
        ----------
        typ : string
              type of LSF
              
        qsim_v1 : simple model where the LSF is supposed to be constant over
        the FoV. It is a convolution of a step function with a Gaussian.
        The resulting function is then sample by the pixel size.
        The slit width is assumed to be constant (2.09 pixels).
        The Gaussian sigma parameter is a polynomial approximation of order 3
        with wavelength.
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
        size : odd integer
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
        out : integer
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
    Nfsf        : int
                  Spatial dimension of the FSF.
    beta        : float
                  Power index of the Moffat.
    fwhm        : float or array of float
                  Moffat fwhm in arcsec.
    

    Returns
    -------
    PSF_Moffat : array (Nz, Nfsf, Nfsf)
                 MUSE PSF
    fwhm_pix   : array (Nz)
                 fwhm of the PSF in pixels
    fwhm_arcsec : array (Nz)
                  fwhm of the PSF in arcsec
    """
    # conversion fwhm arcsec -> pixels
    fwhm_pix = fwhm / step_arcsec

    # alpha coefficient in pixel
    alpha = fwhm_pix / (2 * np.sqrt(2**(1 / beta) - 1))
    amplitude = (beta-1) * (np.pi * alpha**2)
    
    if np.isscalar(alpha):
        amplitude = (beta-1) * (np.pi * alpha**2)
        moffat = Moffat2D(amplitude, 0, 0, alpha, beta)
        moffat_kernel = Model2DKernel(moffat, x_size=Nfsf, y_size=Nfsf)
        PSF_Moffat = moffat_kernel.array
        # Normalization
#         PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat)
    else:
        Nz = alpha.shape[0]
        PSF_Moffat = np.empty((Nz, Nfsf, Nfsf))
        for i in range(Nz):
            moffat = Moffat2D(amplitude[i], 0, 0, alpha[i], beta)
            moffat_kernel = Model2DKernel(moffat, x_size=Nfsf, y_size=Nfsf)
            PSF_Moffat [i,:,:]= moffat_kernel.array
        # Normalization
#         PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat, axis=(1, 2))\
#                     [:, np.newaxis, np.newaxis]
                    
    return PSF_Moffat, fwhm_pix
    
def MOFFAT1(lbda, step_arcsec, Nfsf, beta, a, b):
    """Compute PSF with a Moffat function

    Parameters
    ----------
    lbda        : float or array of float
                  Wavelength values in Angstrom.
    step_arcsec : float
                  Size of the pixel in arcsec
    Nfsf        : int
                  Spatial dimension of the FSF.
    beta        : float
                  Power index of the Moffat.
    a           : float
                  Moffat parameter in arcsec (fwhm=a+b*lbda)
    b           : float
                  Moffat parameter (fwhm=a+b*lbda)

    Returns
    -------
    PSF_Moffat : array (Nz, Nfsf, Nfsf)
                 MUSE PSF
    fwhm_pix   : array (Nz)
                 fwhm of the PSF in pixels
    fwhm_arcsec : array (Nz)
                  fwhm of the PSF in arcsec
    """
    fwhm_arcsec = a + b*lbda
    PSF_Moffat, fwhm_pix = Moffat(step_arcsec, Nfsf, beta, fwhm_arcsec)
    return PSF_Moffat, fwhm_pix, fwhm_arcsec
    
class FSF(object):

    """This class offers Field Spread Function models for MUSE.

    Attributes
    ----------

    typ (string) : FSF type.
    """

    def __init__(self, typ="MOFFAT1"):
        """Manages LSF model.

        Parameters
        ----------
        typ : string
              type of LSF
              
        
        MOFFAT1: Moffat function with a FWHM which varies linearly with the
        wavelength. Parameters:
         - beta (float) Power index of the Moffat.
         - a (float) constant in arcsec which defined the FWHM (fwhm=a+b*lbda)
         - b (float) constant which defined the FWHM (fwhm=a+b*lbda)
        
        """
        self.typ = typ

    def get_FSF(self, lbda, step, size, **kargs):
        """Return an array containing the FSF for a given wavelength.

        Parameters
        ----------
        lbda  : float
                Wavelength value in A.
        step  : float
                Size of the pixel in arcsec.
        size  : integer
                Number of pixels.
        kargs : dict
                kargs can be used to set LSF parameters.
               
        Returns
        -------
        FSF         : array (size, size)
                      MUSE FSF
        fwhm_pix    : float
                      fwhm of the FSF in pixels
        fwhm_arcsec : array
                      fwhm of the FSF in arcsec
        """
        if self.typ == "MOFFAT1":
            return MOFFAT1(lbda, step, size, **kargs)
        else:
            raise IOError('Invalid FSF type')
        
    def get_FSF_cube(self, cube, size, **kargs):
        """Return a cube of FSFs corresponding to the MUSE data cube
        given in input:
        - a FSF per MUSE spectral pixels,
        - the step of the FSF pixel is equal to the spatial step of the
          MUSE data cube.

        Parameters
        ----------
        cube  : `mpdaf.obj.Cube`
                The wavelength coordinates of the MUSE spectral pixels.
                Wavelength value in A.
        size  : integer
                FSF size in pixels.
        kargs : dict
                kargs can be used to set LSF parameters.
               
        Returns
        -------
        FSF         : array (cube.shape[0], size, size)
                      Cube containing MUSE FSF (one per wavelength)
        fwhm_pix    : array(cube.shape[0])
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
        

