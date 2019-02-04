import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.modeling.models import Moffat2D as astMoffat2D
from mpdaf.obj import Image

from ..obj import Cube, WCS

__all__ = ['Moffat2D', 'FSFModel', 'OldMoffatModel', 'MoffatModel2']


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

def norm_lbda(lbda, lb1, lb2):
    nlbda = (lbda-lb1)/(lb2-lb1) - 0.5
    return nlbda

def Moffat2D(fwhm, beta, shape):
    """Compute Moffat for a value or array of values of FWHM and beta.

    Parameters
    ----------
    fwhm : float or array of float
        Moffat fwhm in pixels.
    beta : float or array of float
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

    if np.isscalar(alpha) and np.isscalar(beta):
        moffat = astMoffat2D(amplitude, x0, y0, alpha, beta)
        PSF_Moffat = moffat(xx, yy)
        # Normalization
        PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat)
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
                #raise ValueError, 'alpha and beta must have the same dimension' 
                raise ValueError
        moffat = astMoffat2D(amplitude, [x0] * Nz, [y0] * Nz,
                             alpha, beta, n_models=Nz)
        PSF_Moffat = moffat(xx, yy, model_set_axis=False)
        # Normalization
        PSF_Moffat = PSF_Moffat / np.sum(PSF_Moffat, axis=(1, 2))\
             [:, np.newaxis, np.newaxis]

    return PSF_Moffat

def get_images(cube, pos, size=5.0, nslice=20):
    #TODO: skip slice with masked value for the notch filter (in AO case)
    dec, ra = pos
    logger.debug('getting %d images around object ra:%f dec:%f' % (nslice, ra, dec))
    l1, l2 = cube.wave.get_range()
    lb1,dl = np.linspace(l1,l2,nslice,endpoint=False,retstep=True)
    lb2 = lb1+dl
    imalist = []
    for l1,l2 in zip(lb1,lb2):
        scube = cube.subcube(pos, size, lbda=(l1,l2))
        ima = scube.mean(axis=0)
        imalist.append(ima)
    white = cube.subcube(pos, size).mean(axis=0)
    return (white,lb1+0.5*dl,imalist)

def fit_poly(x, y, deg, reject=3.0):
    pol = np.polyfit(x, y, deg)
    yp = np.polyval(pol, x)
    err = yp - y
    if reject > 0:
        err_masked = sigma_clip(err, sigma=reject)
        xx = x[~err_masked.mask]        
        if len(xx) < len(x):
            logger.debug('%d points rejected in polynomial fit',len(x)-len(xx))
            yy = y[~err_masked.mask]
            pol = np.polyfit(xx, yy, deg)
            yp = np.polyval(pol, x)
            err = yp - y                          
    return (pol,yp,err)


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
    
    @classmethod
    def from_psfrec(cls, rawfilename):  
        """Compute FSF parameters from GLAO MUSE PSF reconstruction"""
        raise NotImplementedError  
    
    @classmethod
    def from_starfit(cls, cube, pos, size=5, nslice=20, lbrange=(5000,9000), **kwargs): 
        """Compute FSF by fitting a point source on a datacube"""
        raise NotImplementedError 
    
    @classmethod
    def from_hstconv(cls, cube, hstimages, lbrange=(5000,9000), **kwargs): 
        """Compute FSF by convolution of HST images"""
        raise NotImplementedError     

    def __repr__(self):
        return "<{}(model={})>".format(self.__class__.__name__, self.model)

    def to_header(self, hdr):
        """Write FSF parameters to a FITS header"""
        raise NotImplementedError

    def get_fwhm(self, lbda):
        """Return FWHM for the given wavelengths."""
        raise NotImplementedError

    def get_2darray(self, lbda, shape):
        """Return FSF 2D array at the given wavelength."""
        raise NotImplementedError
    
    def get_image(self, lbda, wcs):
        """Return FSF image at the given wavelength."""
        if not np.isscalar(lbda):
            raise ValueError
        data = self.get_2darray(lbda, (wcs.naxis2,wcs.naxis1)) 
        return Image(wcs=wcs, data=data) 
    
    def get_cube(self, wave, wcs):
        """Return FSF cube at the given wavelengths."""
        lbda = wave.coord()
        data = self.get_3darray(lbda, (wcs.naxis2,wcs.naxis1)) 
        return Cube(wcs=wcs, wave=wave, data=data)        

    def get_3darray(self, lbda, shape):
        """Return FSF cube at the given wavelengths."""
        raise NotImplementedError


class OldMoffatModel(FSFModel):
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
            

    def to_header(self, hdr, field_idx=0):
        """Write FSF parameters to a FITS header"""
        hdr['FSFMODE'] = self.model
        hdr['FSF%02dBET' % field_idx] = np.around(self.beta, decimals=2)
        hdr['FSF%02dFWA' % field_idx] = np.around(self.a, decimals=3)
        hdr['FSF%02dFWB' % field_idx] = float('%.3e' % self.b)
        return 

    def get_fwhm(self, lbda, unit='arcsec'):
        fwhm = self.a + self.b * lbda
        if unit == 'pix':
            fwhm /= self.pixstep
        return fwhm

    def get_2darray(self, lbda, shape):
        """Return FSF 2D array at the given wavelength."""
        if not np.isscalar(lbda):
            raise ValueError
        return Moffat2D(self.get_fwhm(lbda, unit='pix'), self.beta, shape)
    

    def get_3darray(self, lbda, shape):
        """Return FSF 3D array at the given wavelengths."""
        return Moffat2D(self.get_fwhm(lbda, unit='pix'), self.beta, shape)


class MoffatModel1(FSFModel):

    model = 1
    name = "Circular MOFFAT beta=cste fwhm=poly(lbda)"


class MoffatModel2(FSFModel):
   
    name = "Circular MOFFAT beta=poly(lbda) fwhm=poly(lbda)" 
    model = 2

    def __init__(self, fwhm_pol, beta_pol, lbrange, pixstep):
        self.fwhm_pol = fwhm_pol
        self.beta_pol = beta_pol
        self.lbrange = lbrange
        self.pixstep = pixstep

    @classmethod
    def from_header(cls, hdr, pixstep):
        if 'FSFLB1' not in fsfkeys or 'FSFLB2' not in hdr:
            logger.error('Missing FSFLB1 and/or FSFLB2 keywords in file header')
            return None 
        lbrange = (hdr['FSFLB1'],hdr['FSFLB2'])
        if lbrange[1] <= lbrange[0]:
            logger.error('Wrong FSF lambda range')
            return None  
        #TODO check NFIELD and make it more general
        for field in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99):
            if 'FSF%02dFNC' % field in hdr:
                ncf = hdr['FSF%02dFNC' % field]
                fwhm_pol = [hdr['FSF%02dF%02d'%(field,k)] for k in range(ncf)]
                ncb = hdr['FSF%02dBNC' % field]
                beta_pol = [hdr['FSF%02dB%02d'%(field,k)] for k in range(ncb)]
                return cls(fwhm_pol, beta_pol, lbrange, pixstep)

    def to_header(self, field_idx):
        """Write FSF parameters to a FITS header"""
        hdr = fits.header() 
        hdr['FSFMODE'] = (self.model, name)
        hdr['FSFLB1'] = (self.lbrange[0], 'FSF Blue Ref Wave (A)')
        hdr['FSFLB2'] = (self.lbrange[1], 'FSF Red Ref Wave (A)') 
        hdr['FSF%02dFNC' % field_idx] = (len(self.fwhm_pol), f'FSF{field_idx:02d} FWHM Poly Ncoef')
        for k,coef in enumerate(self.fwhmpoly):
            hdr['FSF%02dF%02d'%(field_idx,k)] = (coef, f'FSF{field_idx:02d} FWHM Poly C{k:02d}')        
        hdr['FSF%02dBNC' % field_idx] = (len(self.beta_pol), f'FSF{field_idx:02d} BETA Poly Ncoef')
        for k,coef in enumerate(self.betapoly):
            hdr['FSF%02dB%02d'%(field_idx,k)] = (coef, f'FSF{field_idx:02d} BETA Poly C{k:02d}')            
        return hdr
    
    @classmethod
    def from_psfrec(cls, rawfilename):
        # Try to import muse-psfr, if not available raise an error
        logger.debug('Computing PSF from Sparta data file %s', rawfilename)
        res = psfrec.compute_psf_from_sparta(rawfilename)
        data = res['FIT_MEAN'].data 
        lbda,fwhm,beta = (data['lbda'], data['fwhm'][:, 0], data['n'])
        logger.debug('Fitting polynomial on FWHM (lbda) and Beta(lbda)')
        res = psfrec.fit_psf_with_polynom(lbda, fwhm, beta, output=0) 
        fsf = cls(lbrange=(res['lbda_lim'][0]*10,res['lbda_lim'][1]*10), fwhm_pol=res['fwhm_pol'], beta_pol=res['beta_pol'],
                  pixstep=0.2)
        return fsf   
    
    @classmethod
    def from_starfit(cls, cube, pos, size=5, nslice=20, fwhmdegm=3, betadeg=3, lbrange=(5000,9000)):
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
        logger.info('FSF from star fit at Ra: %.5f Dec: %.5f Size %.1f Nslice %d FWHM poly deg %d BETA poly deg %d',
                            model,pos[1],pos[0],size,nslice,fwhmdeg,betadeg)    
        white,lbda,imalist = get_images(cube, pos, size=size, nslice=nslice)
        lbdanorm = norm_lbda(lbda, lbdarange[0], lbdarange[-1])

        logger.debug('-- First fit on white light image')
        fit1 = white.moffat_fit(fwhm=(0.8, 0.8), n=2.5, circular=True, fit_back=True, verbose=False)
        logger.debug('RA: %.5f DEC: %.5f FWHM %.2f BETA %.2f PEAK %.1f BACK %.1f',fit1.center[1],fit1.center[0],fit1.fwhm[0],fit1.n,fit1.peak,fit1.cont)

        logger.debug('-- Second fit on all images')
        fit2 = []
        for k, ima in enumerate(imalist):
            f2 = ima.moffat_fit(fwhm=fit1.fwhm[0], n=fit1.n, center=fit1.center, fit_n=True, circular=True, fit_back=True, verbose=False)
            logger.debug('%d RA: %.5f DEC: %.5f FWHM %.2f BETA %.2f PEAK %.1f BACK %.1f',k+1,f2.center[1],f2.center[0],f2.fwhm[0],f2.n,f2.peak,f2.cont)
            fit2.append(f2)   
            
        logger.debug('-- Third fit on all images')
        fit3 = []        
        beta_fit = np.array([f.n for f in fit2])
        logger.debug('-- Polynomial fit of BETA(lbda)')
        beta_pol,beta_pval,beta_err = fit_poly(lbdanorm, beta_fit, betadeg)
        logger.debug('BETA poly {}'.format(beta_pol))
        for k, ima in enumerate(imalist):
            f2 = ima.moffat_fit(fwhm=fit1.fwhm[0], n=beta_pval[k], center=fit1.center, fit_n=False, circular=True, fit_back=True, verbose=False)
            logger.debug('RA: %.5f DEC: %.5f FWHM %.2f BETA %.2f PEAK %.1f BACK %.1f',f2.center[1],f2.center[0],f2.fwhm[0],f2.n,f2.peak,f2.cont)
            fit3.append(f2) 
        fwhm_fit = np.array([f.fwhm for f in fit3])     
            
        logger.debug('-- Polynomial fit of FWHM(lbda)')          
        fwhm_pol,fwhm_pval,fwhm_err = fit_poly(lbdanorm, fwhm_fit, fwhmdeg)
        logger.debug('FWHM poly {}'.format(fwhm_pol))

            
        logger.debug('-- return FSF model')
        fsf = cls(model, lbrange=lbdarange, fwhmpoly=fwhm_pol, beta=beta_pol, pixstep=cube.get_step())
        fsf.fit = {'center':np.array([f.center for f in fit3]), 'wave':lbda, 'fwhmfit':fwhm_fit, 'fwhmpol':fwhm_pval,
                   'fwhmerr':fwhm_err, 'center0':fit1.center, 'fwhm0':fit1.fwhm[0], 'beta0':fit1.n,
                   'betafit':np.array([f.n for f in fit2]), 'ima':imalist}        
      
        return fsf        

    def get_fwhm(self, lbda, unit='arcsec'):
        """Return FWHM at the given wavelengths"""
        lb = norm_lbda(lbda,self.lbrange[0],self.lbrange[1])
        fwhm = np.polyval(self.fwhm_pol, lb)        
        if unit == 'pix':
            fwhm /= self.pixstep
        return fwhm
    
    def get_beta(self, lbda):
        """Return BETA at the given wavelengths"""
        lb = norm_lbda(lbda,self.lbrange[0],self.lbrange[1])
        beta = np.polyval(self.beta_pol, lb)        
        return beta    

    def get_image(self, lbda, shape):
        """Return FSF image at the given wavelength."""
        if not np.isscalar(lbda):
            raise ValueError        
        return Moffat2D(self.get_fwhm(lbda, unit='pix'), self.get_beta(lbda), lbda)

    def get_cube(self, lbda, shape):
        """Return FSF cube at the given wavelengths."""
        return Moffat2D(self.get_fwhm(lbda, unit='pix'), self.get_beta(lbda), lbda)    


class EllipticalMoffatModel(FSFModel):

    model = 3
    name = "Elliptical MOFFAT beta=poly(lbda) fwhmx,y=polyx,y(lbda) pa=cste"
