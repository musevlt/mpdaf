"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2016 Jarle Brinchman <jarle@strw.leidenuniv.nl>
Copyright (c) 2015-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2015-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Roland Bacon <roland.bacon@univ-lyon1.fr>

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

# sea.py contains SpectExtractAnd[nothing], the first part of
# SpecExtractAndWeb software developed by Jarle.
#
# This software has been developed by Jarle Brinchmann (University of Leiden)
# and ported to python by Laure Piqueras (CRAL).
#
# It takes a MUSE data cube and a catalogue of objects and extracts small
# sub-cubes around each object. From this it creates narrow-band images and
# eventually spectra. To do the latter it is necessary to run an external
# routine which runs sextractor on the images to define spectrum extraction
# apertures.
#
# Please contact Jarle for more info at jarle@strw.leidenuniv.nl

from astropy.io import fits
import astropy.units as u

import logging
import numpy as np
import os
import shutil
import subprocess
import warnings

from ..obj import Image, Spectrum
from ..tools import broadcast_to_cube, MpdafWarning

__all__ = ('findCentralDetection', 'union', 'intersection', 'findSkyMask',
           'segmentation', 'mask_creation', 'compute_spectrum',
           'compute_optimal_spectrum')

DEFAULT_SEX_FILES = ['default.nnw', 'default.param', 'default.sex',
                     'gauss_5.0_9x9.conv']


def setup_config_files(DIR=None):
    if DIR is None:
        DIR = os.path.dirname(__file__) + '/sea_data/'
        files = DEFAULT_SEX_FILES
    else:
        files = os.listdir(DIR)

    for f in files:
        if not os.path.isfile(f):
            shutil.copy(DIR + '/' + f, './' + f)


def remove_config_files(DIR=None):
    files = DEFAULT_SEX_FILES if DIR is None else os.listdir(DIR)
    for f in files:
        os.remove(f)


def findCentralDetection(images, iyc, ixc, tolerance=1):
    """Determine which image has a detection close to the centre.

    We start with the centre for all. If all have a value zero there we
    continue.
    """
    logger = logging.getLogger(__name__)
    min_distances = {}
    min_values = {}
    global_min = 1e30
    global_ix_min = -1
#     global_iy_min = -1

#     count = 0
    bad = {}
    for key, im in images.items():
        logger.debug('Doing %s', key)
#         if (count == 0):
#             nx, ny = im.shape
#             ixc = nx/2
#             iyc = ny/2

        # Find the parts of the segmentation map where there is an object.
        ix, iy = np.where(im > 0)
        # Find the one closest to the centre.

        if (len(ix) > 0):
            # At least one object detected!
            dist = np.abs(ix - ixc) + np.abs(iy - iyc)
            min_dist = np.min(dist)
            i_min = np.argmin(dist)
            ix_min = ix[i_min]
            iy_min = iy[i_min]
            val_min = im[ix_min, iy_min]

            # Record the essential information
            min_distances[key] = min_dist
            min_values[key] = val_min
            bad[key] = 0

            if (min_dist < global_min):
                global_min = min_dist
                global_ix_min = ix_min
#                 global_iy_min = iy_min
                global_im_index_min = key
                global_value = val_min
        else:
            bad[key] = 1
            min_distances[key] = -1e30
            min_values[key] = -1

#         count = count+1

    # We have now looped through. Time to take stock. First let us check that
    # there was at least one detection.
    n_useful = 0
    segmentation_maps = {}
    isUseful = {}
    if global_ix_min >= 0:
        # Ok, we are good. We have now at least one good segmentation map.
        # So we can make one simple one here.
        ref_map = np.where(images[global_im_index_min] == global_value, 1, 0)

        # Then check the others as well and if they do have a map at this
        # position get another simple segmentation map.
        for key in images:
            if bad[key] == 1:
                logger.warning('Image %s has no objects', key)
                this_map = np.zeros(ref_map.shape, dtype=bool)
            else:
                # Has at least one object - let us see.
                if np.abs(min_distances[key] - global_min) <= tolerance:
                    # Create simple map
                    logger.debug('Image %s has one useful objects', key)
                    this_map = images[key] == min_values[key]
                    n_useful = n_useful + 1
                    isUseful[key] = True
                else:
                    # Ok, this is too far away, I do not want to use this.
                    this_map = np.zeros(ref_map.shape, dtype=bool)

            segmentation_maps[key] = this_map

    else:
        # No objects found. Let us create a list of empty images.
        keys = list(images.keys())
        segmentation_maps = {key: np.zeros(images[keys[0]].shape, dtype=bool)
                             for key in keys}
        isUseful = {key: 0 for key in keys}
        n_useful = 0

    return {'N_useful': n_useful,
            'seg': segmentation_maps,
            'isUseful': isUseful}


def union(seg):
    """Return the union of a list of boolean arrays."""
    mask = np.zeros(seg[0].shape, dtype=bool)
    for im in seg:
        mask |= np.asarray(im, dtype=bool)
    return mask


def intersection(seg):
    """Return the intersection of a list of boolean arrays."""
    mask = np.ones(seg[0].shape, dtype=bool)
    for im in seg:
        mask &= np.asarray(im, dtype=bool)
    return mask


def findSkyMask(images):
    """Loop over all segmentation images and use the region where no object is
    detected in any segmentation map as our sky image."""
    mask = np.ones(images[0].shape, dtype=bool)
    for im in images:
        mask &= (~np.asarray(im, dtype=bool))
    return mask


def segmentation(source, tags, DIR, remove):
    """segmentation by running sextractor"""
    logger = logging.getLogger(__name__)
    # suppose that MUSE_WHITE image exists
    try:
        subprocess.check_call(['sex', '-v'])
        cmd_sex = 'sex'
    except OSError:
        try:
            subprocess.check_call(['sextractor', '-v'])
            cmd_sex = 'sextractor'
        except OSError:
            raise OSError('SExtractor not found')

    dim = source.images['MUSE_WHITE'].shape
    start = source.images['MUSE_WHITE'].wcs.pix2sky([0, 0], unit=u.deg)[0]
    step = source.images['MUSE_WHITE'].get_step(unit=u.arcsec)
    rot = source.images['MUSE_WHITE'].get_rot()
    wcs = source.images['MUSE_WHITE'].wcs

    maps = {}
    setup_config_files(DIR)
    # size in arcsec
    for tag in tags:
        ima = source.images[tag]
        tag2 = tag.replace('[', '').replace(']', '')

        fname = '%04d-%s.fits' % (source.id, tag2)
        start_ima = ima.wcs.pix2sky([0, 0], unit=u.deg)[0]
        step_ima = ima.get_step(unit=u.arcsec)
        rot_ima = ima.get_rot()
        prihdu = fits.PrimaryHDU()
        hdulist = fits.HDUList([prihdu])
        if ima.shape[0] == dim[0] and ima.shape[1] == dim[1] and \
                start_ima[0] == start[0] and start_ima[1] == start[1] and \
                step_ima[0] == step[0] and step_ima[1] == step[1] and \
                rot_ima == rot:
            data_hdu = ima.get_data_hdu(name='DATA', savemask='nan')
        elif rot_ima == rot:
            ima2 = ima.resample(dim, start, step, flux=True)
            data_hdu = ima2.get_data_hdu(name='DATA', savemask='nan')
        else:
            ima2 = ima.rotate(rot - rot_ima, interp='no', reshape=True)
            ima2 = ima2.resample(dim, start, step, flux=True)
            data_hdu = ima2.get_data_hdu(name='DATA', savemask='nan')
        hdulist.append(data_hdu)
        hdulist.writeto(fname, overwrite=True, output_verify='fix')

        catalogFile = 'cat-' + fname
        segFile = 'seg-' + fname

        command = [cmd_sex, "-CHECKIMAGE_NAME", segFile, '-CATALOG_NAME',
                   catalogFile, fname]
        subprocess.call(command)
        # remove source file
        os.remove(fname)
        try:
            hdul = fits.open(segFile)
            maps[tag] = hdul[0].data
            hdul.close()
        except Exception as e:
            logger.error("Something went wrong with sextractor!")
            raise e
        # remove seg file
        os.remove(segFile)
        # remove catalog file
        os.remove(catalogFile)
    if remove:
        remove_config_files(DIR)

    # Save segmentation maps
    if len(maps) > 0:
        for tag, data in maps.items():
            ima = Image(wcs=wcs, data=data, dtype=np.uint8, copy=False)
            source.images['SEG_' + tag] = ima


def mask_creation(source, maps):
    wcs = source.images['MUSE_WHITE'].wcs
    yc, xc = wcs.sky2pix((source.DEC, source.RA), unit=u.deg)[0]
    r = findCentralDetection(maps, yc, xc, tolerance=3)

    segmaps = list(r['seg'].values())
    source.images['MASK_UNION'] = Image(wcs=wcs, dtype=np.uint8, copy=False,
                                        data=union(segmaps))
    source.images['MASK_SKY'] = Image(wcs=wcs, dtype=np.uint8, copy=False,
                                      data=findSkyMask(list(maps.values())))
    source.images['MASK_INTER'] = Image(wcs=wcs, dtype=np.uint8, copy=False,
                                        data=intersection(segmaps))


def compute_spectrum(cube, weights):
    """Compute a spectrum for a cube by summing along the spatial axis.

    The method conserves the flux by using the algorithm from Jarle Brinchmann
    (jarle@strw.leidenuniv.nl): It takes into account bad pixels in the
    addition, and normalize for flux conservation.

    Parameters
    ----------
    cube : `~mpdaf.obj.Cube`
        Input data cube.
    weights : array-like
        An array of weights associated with the data values.

    """
    w = np.asarray(weights, dtype=float)

    # First ensure that we have a 3D mask
    w = broadcast_to_cube(w, cube.shape)

    # mask of the non-zero weights and its surface
    npix = np.sum(w != 0, axis=(1, 2))

    # Normalize weights
    w = w / np.sum(w, axis=(1, 2))[:, np.newaxis, np.newaxis]

    data = np.nansum(cube.data.filled(np.nan) * w, axis=(1, 2)) * npix

    if cube._var is not None:
        var = np.nansum(cube.var.filled(np.nan) * w, axis=(1, 2)) * npix
    else:
        var = None

    return Spectrum(wave=cube.wave, unit=cube.unit, data=data, var=var,
                    copy=False)


def compute_optimal_spectrum(cube, mask, psf):
    """Compute a spectrum for a cube by summing along the spatial axis.

    An optimal extraction algorithm for CCD spectroscopy Horne, K. 1986
    http://adsabs.harvard.edu/abs/1986PASP...98..609H

    Parameters
    ----------
    cube : `~mpdaf.obj.Cube`
        Input data cube.
    mask : np.ndarray
        Aperture mask.
    psf : np.ndarray
        PSF, 2D or 3D.

    """
    # First ensure that we have a 3D psf
    if psf.ndim != 3:
        psf = broadcast_to_cube(psf, cube.shape)

    # Normalize weights
    psf = psf * mask
    psf /= np.sum(psf, axis=(1, 2))[:, np.newaxis, np.newaxis]

    data = cube.data.filled(np.nan)

    if cube._var is not None:
        var = cube.var.filled(np.nan)
        d = np.nansum(psf**2 / var, axis=(1, 2))
        newdata = np.nansum(psf * data / var, axis=(1, 2)) / d
        newvar = np.nansum(psf, axis=(1, 2)) / d
    else:
        warnings.warn('Extracting spectrum from a cube without variance',
                      MpdafWarning)
        d = np.nansum(psf**2, axis=(1, 2))
        newdata = np.nansum(psf * data, axis=(1, 2)) / d
        newvar = None

    return Spectrum(wave=cube.wave, unit=cube.unit, data=newdata, var=newvar,
                    copy=False)
