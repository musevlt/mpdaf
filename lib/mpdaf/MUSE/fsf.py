import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.modeling.models import Moffat2D as astMoffat2D

from ..obj import Cube, WCS

__all__ = ['Moffat2D', 'FSFModel', 'Moffat1Model']


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def Moffat2D(fwhm, beta, shape):
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
        moffat = astMoffat2D(amplitude, x0, y0, alpha, beta)
        PSF_Moffat = moffat(xx, yy)
        # Normalization
        # PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat)
    else:
        Nz = alpha.shape[0]
        moffat = astMoffat2D(amplitude, [x0] * Nz, [y0] * Nz,
                             alpha, [beta] * Nz, n_models=Nz)
        PSF_Moffat = moffat(xx, yy, model_set_axis=False)
        # Normalization
        # PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat, axis=(1, 2))\
        #     [:, np.newaxis, np.newaxis]

    return PSF_Moffat


class FSFModel:
    """Base class for FSF models."""

    @classmethod
    def read(cls, cube):
        """Read the FSF model from a file, cube, or header.

        Parameters
        ----------
        cube : str, `mpdaf.obj.Cube`, or `astropy.io.fits.Header`
            Must contain a header with a FSF model.

        """
        if isinstance(cube, str):
            # filename
            cube = Cube(cube)

        if isinstance(cube, Cube):
            wcs = cube.wcs
            hdr = cube.primary_header
        elif isinstance(cube, fits.Header):
            hdr = cube
            wcs = WCS(hdr=hdr)

        if 'FSFMODE' not in hdr:
            raise ValueError('FSFMODE keyword not found')

        for klass in all_subclasses(cls):
            if klass.model == hdr['FSFMODE']:
                break
        else:
            raise ValueError('FSFMODE {} is not implemented'
                             .format(hdr['FSFMODE']))

        step = wcs.get_step(unit=u.arcsec)[0]
        return klass.from_header(cube.primary_header, step)

    @classmethod
    def from_header(cls, hdr, pixstep):
        """Read FSF parameters from a FITS header"""
        raise NotImplementedError

    def __repr__(self):
        return "<{}(model={})>".format(self.__class__.__name__, self.model)

    def to_header(self, hdr):
        """Write FSF parameters to a FITS header"""
        raise NotImplementedError

    def get_fwhm(self, lbda):
        """Return FWHM for the given wavelengths."""
        raise NotImplementedError

    def get_image(self, lbda):
        """Return FSF image at the given wavelength."""
        raise NotImplementedError

    def get_cube(self, lbda):
        """Return FSF cube at the given wavelengths."""
        raise NotImplementedError


class Moffat1Model(FSFModel):
    """Moffat FSF with fixed beta and FWHM varying with wavelength."""

    model = 'MOFFAT1'

    def __init__(self, a, b, beta, pixstep):
        self.a = a
        self.b = b
        self.beta = beta
        self.pixstep = pixstep

    @classmethod
    def from_header(cls, hdr, pixstep):
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
        return Moffat2D(self.get_fwhm(lbda, unit='pix'), self.beta, lbda)

    def get_cube(self, lbda, shape):
        """Return FSF cube at the given wavelengths."""
        return Moffat2D(self.get_fwhm(lbda, unit='pix'), self.beta, lbda)


class GaussianModel(FSFModel):

    model = 0
    name = "Circular GAUSS fwhm=poly(lbda)"


class MoffatModel(FSFModel):

    model = 1
    name = "Circular MOFFAT beta=cste fwhm=poly(lbda)"


class MoffatBetaVarModel(FSFModel):

    model = 2
    name = "Circular MOFFAT beta=poly(lbda) fwhm=poly(lbda)"


class EllipticalMoffatModel(FSFModel):

    model = 3
    name = "Elliptical MOFFAT beta=poly(lbda) fwhmx,y=polyx,y(lbda) pa=cste"
