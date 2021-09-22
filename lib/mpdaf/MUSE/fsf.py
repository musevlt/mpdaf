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
from astropy.table import Table
from astropy.modeling.models import Moffat2D as astMoffat2D
from astropy.stats import sigma_clip

from ..obj import Cube, WCS, Image, iter_ima
from ..tools import all_subclasses

__all__ = ['Moffat2D', 'FSFModel', 'MoffatModel2', 'combine_fsf']


def find_model_cls(hdr):
    for cls in all_subclasses(FSFModel):
        if cls.model == hdr['FSFMODE']:
            break
    else:
        if hdr['FSFMODE'] != "MOFFAT1":  # old model comptatible with model=2
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
        x0, y0 = np.array(shape) / 2 - np.array([0.5, 0.5])
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
    x = np.array(x)
    y = np.array(y)
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


class MoffatModel2(FSFModel):

    """Circular MOFFAT beta=poly(lbda) fwhm=poly(lbda)."""

    name = "Circular MOFFAT beta=poly(lbda) fwhm=poly(lbda)"
    model = 2

    def __init__(self, fwhm_pol, beta_pol, lbrange, pixstep, field=0):
        """ Create a FSF object

        Parameters
        ----------
        fwhm_pol : list
            list of polynome coefficients for FWHM(l)::

                FWHM(l) = fwhm_pol[0] * l**deg + ... + fwhm_pol[deg]
                l = (lbda - lb1) / (lb2 - lb1) - 0.5

        beta_pol : list
            list of polynome coefficients for beta(l)
        lbrange : tuple
            lb1,lb2 wavelengths used for wavelength normalisation
        pixstep : float
            spaxel value in arcsec
        field : int
            field location in case of multiple FSF

        Returns
        -------
        fsf : `~mpdaf.MUSE.MoffatModel2`
            fsf model

        """
        super().__init__()
        self.fwhm_pol = fwhm_pol
        self.beta_pol = beta_pol
        self.lbrange = lbrange
        self.pixstep = pixstep
        self.field = field

    @classmethod
    def from_header(cls, hdr, pixstep, field=0):
        """ Read FSF from file header

        Parameters
        ----------
        hdr : `astropy.io.fits.Header`
            FITS header
        pixstep : float
            spaxel value in arcsec

        Returns
        -------
        fsf : `~mpdaf.MUSE.MoffatModel2`
            fsf model

        """
        if 'FSFMODE' not in hdr:
            raise ValueError('Missing FSFMODE keyword in file header')
        if hdr['FSFMODE'] == 'MOFFAT1':  # old model
            if 'FSF%02dBET' % field not in hdr:
                raise ValueError('FSF%02dBET not found in header' % field)
            _beta = hdr['FSF%02dBET' % field]
            _a = hdr['FSF%02dFWA' % field]
            _b = hdr['FSF%02dFWB' % field]
            # Convert the model to a model=2 one.
            l1, l2 = 5000, 9000
            a = _b * (l2 - l1)
            b = _a + a * (l1 / (l2 - l1) + 0.5)
            fwhm_pol = [a, b]
            return MoffatModel2(fwhm_pol, [_beta], (l1, l2), pixstep)

        else:
            if 'FSFLB1' not in hdr or 'FSFLB2' not in hdr:
                raise ValueError(
                    'Missing FSFLB1/FSFLB2 keywords in file header')

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
        """ Write FSF in file header

        Parameters
        ----------
        hdr : `astropy.io.fits.Header`
            FITS header
        field_idx : int
            field index

        Returns
        -------
        hdr : `astropy.io.fits.Header`
            FITS header

        """
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
    def from_psfrec(cls, rawfilename, **kwargs):
        """ Compute Reconstructed FSF from AO telemetry
            Need muse_psfrec external python module.

        Parameters
        ----------
        rawfilename : str
            MUSE raw file name with AO telemetry information

        Returns
        -------
        fsf : `~mpdaf.MUSE.MoffatModel2`
            fsf model

        """
        # Try to import muse-psfr, if not available raise an error
        from muse_psfr import psfrec
        logger = logging.getLogger(__name__)
        logger.debug('Computing PSF from Sparta data file %s', rawfilename)
        res = psfrec.compute_psf_from_sparta(rawfilename, **kwargs)
        for k, r in enumerate(Table(res[1].data)):
            logger.debug(
                '%02d: Seeing %.02f,%.02f,%.02f,%.02f '
                'GL %.02f,%.02f,%.02f,%.02f L0 %.02f,%.02f,%.02f,%.02f',
                k + 1,
                r['LGS1_SEEING'], r['LGS2_SEEING'], r[
                    'LGS3_SEEING'], r['LGS4_SEEING'],
                r['LGS1_TUR_GND'], r['LGS2_TUR_GND'], r[
                    'LGS3_TUR_GND'], r['LGS4_TUR_GND'],
                r['LGS1_L0'], r['LGS2_L0'], r['LGS3_L0'], r['LGS4_L0']
            )
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
                     lbrange=(5000, 9000), factor=1):
        """
        Fit a FSF model on a point source

        Parameters
        ----------
        cube : `mpdaf.obj.Cube`
            input datacube
        pos : tuple of float
            (dec,ra) location of the source in deg
        size : float
            size of region to extract around the source in arcsec
        nslice : int
            number of wavelength slices to used
        fwhmdeg : int
            degre for polynomial fit of FWHM(lbda)
        betadeg : int
            degre for polynomial fit of Beta(lbda)
        lbdarange: tuple of float
            (lbda1,lbda2)  reference wavelengths for normalisation
        factor: int
            subsampling factor used in moffat fit

        Returns
        -------
        fsf : `~mpdaf.MUSE.MoffatModel2`
         fsf model with intermediate fitting results as .fit attribute

             fsf.fit : dict
                  center : array of fitted star location
                  wave : array of wavelengths
                  fwhmfit : array of fitted FWHM
                  fwhmerr : array of errors in FWHM returned by the fit
                  fwhmpol : list of FWHM polynomial
                  betafit : array of fitted beta
                  betaerr : array of errors in beta returned by the fit
                  betapol : list of beta polynomial
                  center0 : first iteration of fitted star location
                  fwhm0 : first iteration of fitted FWHM
                  beta0 : first iteration of fitted beta
                  ima : list of images used in the fit
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
                                fit_back=True, verbose=False, factor=factor)
        logger.debug('RA: %.5f DEC: %.5f FWHM %.2f BETA %.2f PEAK %.1f '
                     'BACK %.1f', fit1.center[1], fit1.center[0], fit1.fwhm[0],
                     fit1.n, fit1.peak, fit1.cont)

        logger.debug('-- Second fit on all images')
        fit2 = []
        for k, ima in enumerate(imalist):
            f2 = ima.moffat_fit(fwhm=fit1.fwhm[0], n=fit1.n,
                                center=fit1.center, fit_n=True, circular=True,
                                fit_back=True, verbose=False, factor=factor)
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
                                fit_back=True, verbose=False, factor=factor)
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

    @classmethod
    def from_FSFlist(cls, imalist, lbda, fwhm0, beta0, fwhmdeg=3, betadeg=3,
                     lbrange=(5000, 9000)):
        """
        Fit a FSF model on a point source

        Parameters
        ----------
        imalist : List of `mpdaf.obj.Image`
                  List of FSF images
        lbda : array
               Wavelength vector corresponding to the list of FSFs
        fwhm0 : float
                Value used to initialize the FWHM in the Moffat fit
        beta0 : float
                Value used to initialize the beta parameter in the Moffat fit
        fwhmdeg : int
            degre for polynomial fit of FWHM(lbda)
        betadeg : int
            degre for polynomial fit of Beta(lbda)
        lbdarange: tuple of float
            (lbda1,lbda2)  reference wavelengths for normalisation

        Returns
        -------
        fsf : `~mpdaf.MUSE.MoffatModel2`
         fsf model
        """
        lbdanorm = norm_lbda(lbda, lbrange[0], lbrange[1])

        fit = []
        for k, ima in enumerate(imalist):
            f = ima.moffat_fit(fwhm=fwhm0, n=beta0, fit_n=True, circular=True,
                               fit_back=True, verbose=False)
            fwhm0 = f.fwhm[0]
            beta0 = f.n
            fit.append(f)

        beta_fit = np.array([f.n for f in fit])
        beta_pol, beta_pval, beta_err = fit_poly(lbdanorm, beta_fit, betadeg)

        fwhm_fit = np.array([f.fwhm[0] for f in fit])
        fwhm_pol, fwhm_pval, fwhm_err = fit_poly(lbdanorm, fwhm_fit, fwhmdeg)

        fsf = cls(lbrange=lbrange, fwhm_pol=fwhm_pol, beta_pol=beta_pol,
                  pixstep=imalist[0].get_step()[0])
        return fsf

    def info(self):
        """ Print fsf model information
        """
        self.logger.info('Wavelength range: %s-%s',
                         self.lbrange[0], self.lbrange[1])
        self.logger.info('FWHM Poly: %s', self.fwhm_pol)
        fwhm = self.get_fwhm(np.array(self.lbrange))
        self.logger.info('FWHM (arcsec): %.2f-%.2f', fwhm[0], fwhm[1])
        self.logger.info('Beta Poly: %s', self.beta_pol)
        beta = self.get_beta(np.array(self.lbrange))
        self.logger.info('Beta values: %.2f-%.2f', beta[0], beta[1])

    def get_fwhm(self, lbda, unit='arcsec'):
        """ Return FWHM

        Parameters
        ----------
        lbda : float or array of float
            wavelengths
        unit : str
            arcsec or pix, unit of FWHM

        Returns
        -------
        FWHM : float or array

        """
        lb = norm_lbda(lbda, self.lbrange[0], self.lbrange[1])
        fwhm = np.polyval(self.fwhm_pol, lb)
        if unit == 'pix':
            fwhm /= self.pixstep
        return fwhm

    def get_beta(self, lbda):
        """ Return beta values

        Parameters
        ----------
        lbda : float or array of float
            wavelengths

        Returns
        -------
        beta : float or array

        """

        lb = norm_lbda(lbda, self.lbrange[0], self.lbrange[1])
        return np.polyval(self.beta_pol, lb)

    def _convolve_one(self, lbda, cfwhm, size=21, samp=10):
        """ convolve the FSF by a given kernel """
        shape = (size * samp, size * samp)
        fwhm0 = self.get_fwhm(lbda, unit='pix') * samp
        beta0 = self.get_beta(lbda)
        data = Moffat2D(fwhm0, beta0, shape)
        im = Image(wcs=WCS(shape=shape), data=data)
        cfwhmpix = cfwhm * samp / self.pixstep
        cim = im.fftconvolve_gauss(fwhm=(cfwhmpix, cfwhmpix),
                                   unit_center=None, unit_fwhm=None)
        fit = cim.moffat_fit(fit_back=False, circular=True, unit_fwhm=None,
                             unit_center=None, verbose=False)
        fwhm = fit.fwhm[0] * self.pixstep / samp
        beta = fit.n
        return fwhm, beta

    def convolve(self, cfwhm, samp=10, nlbda=20, size=21, full_output=False):
        """
        Convolve the FSF with a Gaussian kernel

        Parameters
        ----------
        cfwhm : float
             Gaussian FWHM in arcsec
        samp : int
             Resampling factor
        nlbda : int
             Number of wavelengths
        size : int
             Image FSF size in pixel
        full_output: bool
             If True, return an additional dictionary

        Returns
        -------
        fsf : `~mpdaf.MUSE.fsf.MoffatModel2`
             fsf model
        res : dict
             res['lbda']: wavelengths
             res['fwhm0']: initial FWHM values
             res['fwhm1']: FWHM values after convolution
             res['beta0']: initial BETA values
             res['beta1']: BETA values after convolution
        """
        lbda = np.linspace(self.lbrange[0], self.lbrange[1], nlbda)
        fwhm1 = []
        beta1 = []
        for lb in lbda:
            f, b = self._convolve_one(lb, cfwhm, size=size, samp=samp)
            fwhm1.append(f)
            beta1.append(b)

        lbdanorm = norm_lbda(lbda, self.lbrange[0], self.lbrange[1])
        fwhm_pol, _, _ = fit_poly(lbdanorm, fwhm1, len(self.fwhm_pol) - 1)
        beta_pol, _, _ = fit_poly(lbdanorm, beta1, len(self.beta_pol) - 1)
        fsf = MoffatModel2(fwhm_pol, beta_pol, self.lbrange, self.pixstep)

        if full_output:
            fwhm0 = self.get_fwhm(lbda)
            beta0 = self.get_beta(lbda)
            return fsf, dict(fwhm0=fwhm0, fwhm1=fwhm1,
                             beta0=beta0, beta1=beta1, lbda=lbda)
        else:
            return fsf


def fwhm_moffat2gauss(fwhm, beta):
    """
    translate a MOFFAT fwhm,beta in GAUSS equivalent fwhm
    """
    pol = np.array(
        [-1.89848758e-03,  3.37400959e-02, -2.38556527e-01,  8.50778040e-01,
         -1.58670491e+00,  2.39768917e+00])
    gfwhm = fwhm * np.polyval(pol, beta)
    return gfwhm


def combine_fsf(fsflist, nlbda=20, size=21):
    """
    Combine a list of FSF

    Parameters
    ----------
    fsflist : list of `~mpdaf.MUSE.MoffatModel2`
         list of FSF models
    nlbda : int
         Number of wavelengths
    size : int
         Image FSF size in pixel

    Returns
    -------
    fsf : `~mpdaf.MUSE.MoffatModel2`
         fsf model
    cube : `~mpdaf.obj.Cube`
         cube of FSF
    """

    lbda = np.linspace(fsflist[0].lbrange[0], fsflist[0].lbrange[1], nlbda)
    shape = (size, size)

    # compute array
    fsfcube = fsflist[0].get_3darray(lbda, shape)
    for fsf in fsflist[1:]:
        fsfcube += fsf.get_3darray(lbda, shape)
    fsfcube /= fsfcube.sum(axis=(1, 2))[:, None, None]

    # create FSF datacube as average of all FSF for each lbda
    fsfcube = Cube(data=fsfcube, wcs=WCS(), copy=False)

    fwhm = []
    beta = []
    for im in iter_ima(fsfcube):
        # fit a Moffat
        fit = im.moffat_fit(fit_back=False, circular=True, unit_fwhm=None,
                            unit_center=None, verbose=False)
        fwhm.append(fit.fwhm[0] * 0.2)
        beta.append(fit.n)

    # polynomial fit
    lbdanorm = norm_lbda(lbda, fsflist[0].lbrange[0], fsflist[0].lbrange[1])
    fwhm_pol, _, _ = fit_poly(lbdanorm, fwhm, len(fsflist[0].fwhm_pol) - 1)
    beta_pol, _, _ = fit_poly(lbdanorm, beta, len(fsflist[0].beta_pol) - 1)
    fsf = MoffatModel2(fwhm_pol, beta_pol, fsflist[0].lbrange,
                       fsflist[0].pixstep)

    return fsf, fsfcube


# class EllipticalMoffatModel(FSFModel):

#     model = 3
#     name = "Elliptical MOFFAT beta=poly(lbda) fwhmx,y=polyx,y(lbda) pa=cste"
