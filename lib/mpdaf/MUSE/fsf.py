"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2018-2019 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2019 Roland Bacon <roland.bacon@univ-lyon1.fr>

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
import warnings
from astropy.io import fits
from astropy.modeling.models import Moffat2D as astMoffat2D
from astropy.stats import sigma_clip

from ..obj import Cube, WCS, Image
from ..tools import all_subclasses

__all__ = ['Moffat2D', 'FSFModel', 'OldMoffatModel', 'MoffatModel2']


def find_model_cls(hdr):
    for cls in all_subclasses(FSFModel):
        if cls.model == hdr['FSFMODE']:
            break
    else:
        raise ValueError('FSFMODE {} is not implemented'
                         .format(hdr['FSFMODE']))

    return cls


def norm_lbda(lbda, lb1, lb2):
    nlbda = (lbda - lb1) / (lb2 - lb1) - 0.5
    return nlbda


def Moffat2D(fwhm, beta, shape, center=None, normalize=True):
    """Compute Moffat for a value or array of values of FWHM and beta.

    Parameters
    ----------
    fwhm : float or array of float
        Moffat fwhm in pixels.
    beta : float or array of float
        Power index of the Moffat.
    shape : tuple
        Spatial dimension of the FSF.
    center : tuple
        Center in pixel (if None the image center is used)
    normalize : bool
        If True, normalize the Moffat.

    Returns
    -------
    PSF_Moffat : array (Nz, size, size)
        MUSE PSF

    """
    # alpha coefficient in pixel
    alpha = fwhm / (2 * np.sqrt(2**(1 / beta) - 1))
    amplitude = (beta - 1) * (np.pi * alpha**2)
    if center is None:
        x0, y0 = np.array(shape) / 2
    else:
        x0, y0 = center
    xx, yy = np.mgrid[:shape[0], :shape[1]]

    if np.isscalar(alpha) and np.isscalar(beta):
        model = astMoffat2D(amplitude, x0, y0, alpha, beta)
        moffat = model(xx, yy)
        # Normalization
        if normalize:
            moffat /= np.sum(moffat)
    else:
        if np.isscalar(beta):
            Nz = alpha.shape[0]
            beta = [beta] * Nz
        elif np.isscalar(alpha):
            Nz = beta.shape[0]
            alpha = [alpha] * Nz
        else:
            Nz = alpha.shape[0]
            if beta.shape[0] != Nz:
                raise ValueError('alpha and beta must have the same dimension')

        model = astMoffat2D(amplitude, [x0] * Nz, [y0] * Nz, alpha, beta,
                            n_models=Nz)
        moffat = model(xx, yy, model_set_axis=False)
        # Normalization
        if normalize:
            moffat /= np.sum(moffat, axis=(1, 2))[:, np.newaxis, np.newaxis]

    return moffat


def get_images(cube, pos, size=5.0, nslice=20):
    # TODO: skip slice with masked value for the notch filter (in AO case)
    logger = logging.getLogger(__name__)
    logger.debug('getting %d images around object ra:%f dec:%f', nslice, *pos)
    l1, l2 = cube.wave.get_range()
    lb1, dl = np.linspace(l1, l2, nslice, endpoint=False, retstep=True)
    subc = cube.subcube(pos, size)
    imalist = [subc.get_image((l1, l1 + dl), method='mean') for l1 in lb1]
    white = subc.mean(axis=0)
    return white, lb1 + 0.5 * dl, imalist


def fit_poly(x, y, deg, reject=3.0):
    logger = logging.getLogger(__name__)
    pol = np.polyfit(x, y, deg)
    yp = np.polyval(pol, x)
    err = yp - y
    if reject > 0:
        err_masked = sigma_clip(err, sigma=reject)
        xx = x[~err_masked.mask]
        if len(xx) < len(x):
            logger.debug('%d points rejected in polynomial fit',
                         len(x) - len(xx))
            yy = y[~err_masked.mask]
            pol = np.polyfit(xx, yy, deg)
            yp = np.polyval(pol, x)
            err = yp - y
    return (pol, yp, err)


class FSFMultiModel(list):
    """Class to manage multiple FSF models."""

    @classmethod
    def from_header(cls, hdr, pixstep, nfields=99):
        self = cls()
        klass = find_model_cls(hdr)
        self.model = klass.model
        for field in range(1, nfields + 1):
            self.append(klass.from_header(hdr, pixstep, field=field))
        return self


class FSFModel:
    """Base class for FSF models."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @classmethod
    def read(cls, cube, field=None, pixstep=None):
        """Read the FSF model from a file, cube, or header.

        Parameters
        ----------
        cube : str, `mpdaf.obj.Cube`, or `astropy.io.fits.Header`
            Must contain a header with a FSF model.
        field : int
            Field number to read, otherwise all models are read.

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

        nfields = 1 if field is not None else hdr.get('NFIELDS', 1)
        if pixstep is None:
            try:
                pixstep = wcs.get_step(unit=u.arcsec)[0]
            except u.core.UnitConversionError:
                warnings.warn('could not find use pixstep from the header',
                              UserWarning)
                pixstep = None

        if nfields > 1:
            return FSFMultiModel.from_header(hdr, pixstep, nfields=nfields)
        else:
            klass = find_model_cls(hdr)
            if field is not None:
                return klass.from_header(hdr, pixstep, field=field)
            else:
                for field in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99):
                    try:
                        return klass.from_header(hdr, pixstep, field=field)
                    except ValueError:
                        pass

    @classmethod
    def from_header(cls, hdr, pixstep, field=0):
        """Read FSF parameters from a FITS header"""
        raise NotImplementedError

    @classmethod
    def from_psfrec(cls, rawfilename):
        """Compute FSF parameters from GLAO MUSE PSF reconstruction"""
        raise NotImplementedError

    @classmethod
    def from_starfit(cls, cube, pos, **kwargs):
        """Compute FSF by fitting a point source on a datacube"""
        raise NotImplementedError

    @classmethod
    def from_hstconv(cls, cube, hstimages, lbrange=(5000, 9000), **kwargs):
        """Compute FSF by convolution of HST images"""
        raise NotImplementedError

    def __repr__(self):
        return "<{}(model={})>".format(self.__class__.__name__, self.model)

    def to_header(self, hdr=None):
        """Write FSF parameters to a FITS header"""
        if hdr is None:
            hdr = fits.Header()
        hdr['FSFMODE'] = (self.model, self.name)
        return hdr

    def get_fwhm(self, lbda):
        """Return FWHM for the given wavelengths."""
        raise NotImplementedError

    def get_beta(self, lbda):
        """Return beta for the given wavelengths."""
        raise NotImplementedError

    def get_2darray(self, lbda, shape, center=None):
        """Return FSF 2D array at the given wavelength."""
        if not np.isscalar(lbda):
            raise ValueError
        return Moffat2D(self.get_fwhm(lbda, unit='pix'), self.get_beta(lbda),
                        shape, center)

    def get_image(self, lbda, wcs, center=None):
        """Return FSF image at the given wavelength."""
        if not np.isscalar(lbda):
            raise ValueError
        data = self.get_2darray(lbda, (wcs.naxis2, wcs.naxis1), center)
        return Image(wcs=wcs, data=data)

    def get_3darray(self, lbda, shape, center=None):
        """Return FSF cube at the given wavelengths."""
        return Moffat2D(self.get_fwhm(lbda, unit='pix'), self.get_beta(lbda),
                        shape, center)

    def get_cube(self, wave, wcs, center=None):
        """Return FSF cube at the given wavelengths."""
        lbda = wave.coord()
        data = self.get_3darray(lbda, (wcs.naxis2, wcs.naxis1), center)
        return Cube(wcs=wcs, wave=wave, data=data)


class OldMoffatModel(FSFModel):
    """Moffat FSF with fixed beta and FWHM varying with wavelength."""

    name = 'Old model with a fixed beta'
    model = 'MOFFAT1'

    def __init__(self, a, b, beta, pixstep, field=0):
        super().__init__()
        self.a = a
        self.b = b
        self.beta = beta
        self.pixstep = pixstep
        self.field = field

    @classmethod
    def from_header(cls, hdr, pixstep, field=0):
        if 'FSF%02dBET' % field not in hdr:
            raise ValueError('FSF%02dBET not found in header' % field)
        beta = hdr['FSF%02dBET' % field]
        a = hdr['FSF%02dFWA' % field]
        b = hdr['FSF%02dFWB' % field]
        return cls(a, b, beta, pixstep, field=field)

    def info(self):
        self.logger.info('Model %s Beta %f FWHM a %f b %f Step %f',
                         self.model, self.beta, self.a, self.b, self.pixstep)

    def to_header(self, hdr=None, field_idx=0):
        hdr = super().to_header(hdr=hdr)
        hdr['FSF%02dBET' % field_idx] = np.around(self.beta, decimals=2)
        hdr['FSF%02dFWA' % field_idx] = np.around(self.a, decimals=3)
        hdr['FSF%02dFWB' % field_idx] = float('%.3e' % self.b)
        return hdr

    def get_fwhm(self, lbda, unit='arcsec'):
        fwhm = self.a + self.b * lbda
        if unit == 'pix':
            fwhm /= self.pixstep
        return fwhm

    def get_beta(self, lbda):
        return self.beta

    def to_model2(self):
        """Convert the model to a model=2 one."""
        l1, l2 = 5000, 9000
        a = self.b * (l2 - l1)
        b = self.a + a * (l1 / (l2 - l1) + 0.5)
        fwhm_pol = [a, b]
        return MoffatModel2(fwhm_pol, [self.beta], (l1, l2), self.pixstep)


class MoffatModel2(FSFModel):

    name = "Circular MOFFAT beta=poly(lbda) fwhm=poly(lbda)"
    model = 2

    def __init__(self, fwhm_pol, beta_pol, lbrange, pixstep, field=0):
        super().__init__()
        self.fwhm_pol = fwhm_pol
        self.beta_pol = beta_pol
        self.lbrange = lbrange
        self.pixstep = pixstep
        self.field = field

    @classmethod
    def from_header(cls, hdr, pixstep, field=0):
        if 'FSFLB1' not in hdr or 'FSFLB2' not in hdr:
            raise ValueError('Missing FSFLB1/FSFLB2 keywords in file header')

        lbrange = (hdr['FSFLB1'], hdr['FSFLB2'])
        if lbrange[1] <= lbrange[0]:
            raise ValueError('Wrong FSF lambda range')

        if 'FSF%02dFNC' % field not in hdr:
            raise ValueError('FSF%02dFNC not found in header' % field)

        ncf = hdr['FSF%02dFNC' % field]
        fwhm_pol = [hdr['FSF%02dF%02d' % (field, k)] for k in range(ncf)]
        ncb = hdr['FSF%02dBNC' % field]
        beta_pol = [hdr['FSF%02dB%02d' % (field, k)] for k in range(ncb)]
        return cls(fwhm_pol, beta_pol, lbrange, pixstep, field=field)

    def to_header(self, hdr=None, field_idx=0):
        hdr = super().to_header(hdr=hdr)
        hdr['FSFLB1'] = (self.lbrange[0], 'FSF Blue Ref Wave (A)')
        hdr['FSFLB2'] = (self.lbrange[1], 'FSF Red Ref Wave (A)')
        hdr['FSF%02dFNC' % field_idx] = (
            len(self.fwhm_pol), 'FSF{:02d} FWHM Poly Ncoef'.format(field_idx))
        for k, coef in enumerate(self.fwhm_pol):
            hdr['FSF%02dF%02d' % (field_idx, k)] = (
                coef, 'FSF{:02d} FWHM Poly C{:02d}'.format(field_idx, k))
        hdr['FSF%02dBNC' % field_idx] = (
            len(self.beta_pol), 'FSF{:02d} BETA Poly Ncoef'.format(field_idx))
        for k, coef in enumerate(self.beta_pol):
            hdr['FSF%02dB%02d' % (field_idx, k)] = (
                coef, 'FSF{:02d} BETA Poly C{:02d}'.format(field_idx, k))
        return hdr

    @classmethod
    def from_psfrec(cls, rawfilename):
        # Try to import muse-psfr, if not available raise an error
        from muse_psfr import psfrec
        logger = logging.getLogger(__name__)
        logger.debug('Computing PSF from Sparta data file %s', rawfilename)
        res = psfrec.compute_psf_from_sparta(rawfilename)
        data = res['FIT_MEAN'].data
        lbda, fwhm, beta = (data['lbda'], data['fwhm'][:, 0], data['n'])
        logger.debug('Fitting polynomial on FWHM (lbda) and Beta(lbda)')
        res = psfrec.fit_psf_with_polynom(lbda, fwhm, beta, output=0)
        fsf = cls(lbrange=(res['lbda_lim'][0] * 10, res['lbda_lim'][1] * 10),
                  fwhm_pol=res['fwhm_pol'], beta_pol=res['beta_pol'],
                  pixstep=0.2)
        return fsf

    @classmethod
    def from_starfit(cls, cube, pos, size=5, nslice=20, fwhmdeg=3, betadeg=3,
                     lbrange=(5000, 9000)):
        """
        Fit a FSF model on a point source
        cube: input datacube
        pos: (dec,ra) location of the source in deg
        size: size of region to extract around the source in arcsec
        nslice: number of wavelength slices to used
        fwhmdeg: degre for polynomial fit of FWHM(lbda)
        betadeg: degre for polynomial fit of Beta(lbda)
        lbdarange: (lbda1,lbda2)  tuple of reference wavelength for normalisation
        return an FSF object and intermediate fitting results as .fit attribute
        """
        dec, ra = pos
        logger = logging.getLogger(__name__)
        logger.info('FSF from star fit at Ra: %.5f Dec: %.5f Size %.1f '
                    'Nslice %d FWHM poly deg %d BETA poly deg %d',
                    pos[1], pos[0], size, nslice, fwhmdeg, betadeg)
        white, lbda, imalist = get_images(cube, pos, size=size, nslice=nslice)
        lbdanorm = norm_lbda(lbda, lbrange[0], lbrange[1])

        logger.debug('-- First fit on white light image')
        fit1 = white.moffat_fit(fwhm=(0.8, 0.8), n=2.5, circular=True,
                                fit_back=True, verbose=False)
        logger.debug('RA: %.5f DEC: %.5f FWHM %.2f BETA %.2f PEAK %.1f '
                     'BACK %.1f', fit1.center[1], fit1.center[0], fit1.fwhm[0],
                     fit1.n, fit1.peak, fit1.cont)

        logger.debug('-- Second fit on all images')
        fit2 = []
        for k, ima in enumerate(imalist):
            f2 = ima.moffat_fit(fwhm=fit1.fwhm[0], n=fit1.n,
                                center=fit1.center, fit_n=True, circular=True,
                                fit_back=True, verbose=False)
            logger.debug('%d RA: %.5f DEC: %.5f FWHM %.2f BETA %.2f PEAK %.1f '
                         'BACK %.1f', k + 1, f2.center[1], f2.center[0],
                         f2.fwhm[0], f2.n, f2.peak, f2.cont)
            fit2.append(f2)

        logger.debug('-- Third fit on all images')
        fit3 = []
        beta_fit = np.array([f.n for f in fit2])
        logger.debug('-- Polynomial fit of BETA(lbda)')
        beta_pol, beta_pval, beta_err = fit_poly(lbdanorm, beta_fit, betadeg)
        logger.debug('BETA poly {}'.format(beta_pol))
        for k, ima in enumerate(imalist):
            f2 = ima.moffat_fit(fwhm=fit1.fwhm[0], n=beta_pval[k],
                                center=fit1.center, fit_n=False, circular=True,
                                fit_back=True, verbose=False)
            logger.debug('RA: %.5f DEC: %.5f FWHM %.2f BETA %.2f PEAK %.1f '
                         'BACK %.1f', f2.center[1], f2.center[0], f2.fwhm[0],
                         f2.n, f2.peak, f2.cont)
            fit3.append(f2)
        fwhm_fit = np.array([f.fwhm[0] for f in fit3])

        logger.debug('-- Polynomial fit of FWHM(lbda)')
        fwhm_pol, fwhm_pval, fwhm_err = fit_poly(lbdanorm, fwhm_fit, fwhmdeg)
        logger.debug('FWHM poly {}'.format(fwhm_pol))

        logger.debug('-- return FSF model')
        fsf = cls(lbrange=lbrange, fwhm_pol=fwhm_pol, beta_pol=beta_pol,
                  pixstep=cube.get_step()[0])
        fsf.fit = {'center': np.array([f.center for f in fit3]),
                   'wave': lbda,
                   'fwhmfit': fwhm_fit,
                   'fwhmpol': fwhm_pval,
                   'fwhmerr': fwhm_err,
                   'betafit': beta_fit,
                   'betapol': beta_pval,
                   'betaerr': beta_err,
                   'center0': fit1.center,
                   'fwhm0': fit1.fwhm[0],
                   'beta0': fit1.n,
                   'ima': imalist}
        return fsf

    def info(self):
        self.logger.info('Wavelength range: %s-%s',
                         self.lbrange[0], self.lbrange[1])
        self.logger.info('FWHM Poly: %s', self.fwhm_pol)
        fwhm = self.get_fwhm(np.array(self.lbrange))
        self.logger.info('FWHM (arcsec): %.2f-%.2f', fwhm[0], fwhm[1])
        self.logger.info('Beta Poly: %s', self.beta_pol)
        beta = self.get_beta(np.array(self.lbrange))
        self.logger.info('Beta values: %.2f-%.2f', beta[0], beta[1])

    def get_fwhm(self, lbda, unit='arcsec'):
        lb = norm_lbda(lbda, self.lbrange[0], self.lbrange[1])
        fwhm = np.polyval(self.fwhm_pol, lb)
        if unit == 'pix':
            fwhm /= self.pixstep
        return fwhm

    def get_beta(self, lbda):
        lb = norm_lbda(lbda, self.lbrange[0], self.lbrange[1])
        return np.polyval(self.beta_pol, lb)


# class EllipticalMoffatModel(FSFModel):

#     model = 3
#     name = "Elliptical MOFFAT beta=poly(lbda) fwhmx,y=polyx,y(lbda) pa=cste"
