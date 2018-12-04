import astropy.units as u
import numpy as np
from abc import ABCMeta, abstractmethod
from astropy.io import fits
from astropy.modeling.models import Moffat2D

from ..obj import Cube


def Moffat(fwhm, beta, shape):
    """Compute Moffat for a value or array of values of FWHM and beta.

    Parameters
    ----------
    fwhm : float or array of float
        Moffat fwhm in pixels.
    beta : float
        Power index of the Moffat.
    shape : tuple
        Spatial dimension of the FSF.

    Returns
    -------
    PSF_Moffat : array (Nz, size, size)
        MUSE PSF

    """
    # alpha coefficient in pixel
    alpha = fwhm / (2 * np.sqrt(2**(1 / beta) - 1))
    amplitude = (beta - 1) * (np.pi * alpha**2)
    x0, y0 = np.array(shape) / 2
    xx, yy = np.mgrid[:shape[0], :shape[1]]

    if np.isscalar(alpha):
        moffat = Moffat2D(amplitude, x0, y0, alpha, beta)
        PSF_Moffat = moffat(xx, yy)
        # Normalization
        # PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat)
    else:
        Nz = alpha.shape[0]
        moffat = Moffat2D(amplitude, [x0] * Nz, [y0] * Nz,
                          alpha, [beta] * Nz, n_models=Nz)
        PSF_Moffat = moffat(xx, yy, model_set_axis=False)
        # Normalization
        # PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat, axis=(1, 2))\
        #     [:, np.newaxis, np.newaxis]

    return PSF_Moffat


class FSFModelABC(metaclass=ABCMeta):
    """Base class for FSF models.

    This defines the interface that FSF models should implement.
    """

    @classmethod
    def from_cube(cls, cube):
        if isinstance(cube, str):
            cube = Cube(cube)
        step = cube.wcs.get_step(unit=u.arcsec)[0]
        return cls.from_header(cube.primary_header, step)

    @classmethod
    @abstractmethod
    def from_header(cls, hdr, pixstep):
        """Read FSF parameters from a FITS header"""

    @abstractmethod
    def to_header(self, hdr):
        """Write FSF parameters to a FITS header"""

    @abstractmethod
    def get_fwhm(self, lbda):
        """Return FWHM for the given wavelengths."""

    @abstractmethod
    def get_image(self, lbda):
        """Return FSF image at the given wavelength."""

    @abstractmethod
    def get_cube(self, lbda):
        """Return FSF cube at the given wavelengths."""


class MoffatFSF(FSFModelABC):
    """Moffat FSF with fixed beta and FWHM varying with wavelength."""

    model = 'MOFFAT1'

    def __init__(self, a, b, beta, pixstep):
        self.a = a
        self.b = b
        self.beta = beta
        self.pixstep = pixstep

    @classmethod
    def from_header(cls, hdr, pixstep):
        if 'FSFMODE' not in hdr:
            raise ValueError('FSFMODE keyword not found')
        for field in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99):
            if 'FSF%02dBET' % field in hdr:
                beta = hdr['FSF%02dBET' % field]
                a = hdr['FSF%02dFWA' % field]
                b = hdr['FSF%02dFWB' % field]
                return cls(a, b, beta, pixstep)

    def to_header(self, field_idx):
        """Write FSF parameters to a FITS header"""
        hdr = fits.header()
        hdr['FSFMODE'] = self.model
        hdr['FSF%02dBET' % field_idx] = np.around(self.beta, decimals=2)
        hdr['FSF%02dFWA' % field_idx] = np.around(self.a, decimals=3)
        hdr['FSF%02dFWB' % field_idx] = float('%.3e' % self.b)
        return hdr

    def get_fwhm(self, lbda, unit='arcsec'):
        fwhm = self.a + self.b * lbda
        if unit == 'pix':
            fwhm /= self.pixstep
        return fwhm

    def get_image(self, lbda, shape):
        """Return FSF image at the given wavelength."""
        return Moffat(self.get_fwhm(lbda, unit='pix'), self.beta, lbda)

    def get_cube(self, lbda, shape):
        """Return FSF cube at the given wavelengths."""
        return Moffat(self.get_fwhm(lbda, unit='pix'), self.beta, lbda)
