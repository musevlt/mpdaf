"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2016 Jarle Brinchman <jarle@strw.leidenuniv.nl>
Copyright (c) 2015-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2015-2016 Simon Conseil <simon.conseil@univ-lyon1.fr>
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

from __future__ import absolute_import, division

from astropy.io import fits as pyfits
import astropy.units as u

import logging
import numpy as np
import os
import shutil
import six
import subprocess

from ..obj import Image, Spectrum

__version__ = 1.0


def setup_config_files(DIR=None):
    if DIR is None:
        DIR = os.path.dirname(__file__) + '/sea_data/'
        files = ['default.nnw', 'default.param', 'default.sex', 'gauss_5.0_9x9.conv']
    else:
        files = os.listdir(DIR)
    for f in files:
        if not os.path.isfile(f):
            shutil.copy(DIR + '/' + f, './' + f)


def remove_config_files(DIR=None):
    if DIR is None:
        files = ['default.nnw', 'default.param', 'default.sex', 'gauss_5.0_9x9.conv']
    else:
        files = os.listdir(DIR)
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
        logger.debug('Doing %s' % key)
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
    mask = np.ones(images[0].shape, dtype=np.bool)
    for im in images:
        mask &= (~np.asarray(im, dtype=bool))
    return mask


def segmentation(source, tags, DIR, remove):
    """segmentation by running sextractor"""
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
        prihdu = pyfits.PrimaryHDU()
        hdulist = [prihdu]
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
        hdu = pyfits.HDUList(hdulist)
        hdu.writeto(fname, clobber=True, output_verify='fix')

        catalogFile = 'cat-' + fname
        segFile = 'seg-' + fname

        command = [cmd_sex, "-CHECKIMAGE_NAME", segFile, '-CATALOG_NAME',
                   catalogFile, fname]
        subprocess.call(command)
        # remove source file
        os.remove(fname)
        try:
            hdul = pyfits.open(segFile)
            maps[tag] = hdul[0].data
            hdul.close()
        except:
            raise Exception("Something went wrong with sextractor!")
        # remove seg file
        os.remove(segFile)
        # remove catalog file
        os.remove(catalogFile)
    if remove:
        remove_config_files(DIR)

    # Save segmentation maps
    if len(maps) > 0:
        for tag, data in six.iteritems(maps):
            ima = Image(wcs=wcs, data=data, dtype=np.uint8, copy=False)
            source.images['SEG_' + tag] = ima


def mask_creation(source, maps):
    wcs = source.images['MUSE_WHITE'].wcs
    yc, xc = wcs.sky2pix((source.DEC, source.RA), unit=u.deg)[0]
    r = findCentralDetection(maps, yc, xc, tolerance=3)
    source.images['MASK_UNION'] = Image(wcs=wcs, dtype=np.uint8, copy=False,
                                        data=union(list(r['seg'].values())))
    source.images['MASK_SKY'] = Image(wcs=wcs, dtype=np.uint8, copy=False,
                                      data=findSkyMask(list(maps.values())))
    source.images['MASK_INTER'] = Image(wcs=wcs, dtype=np.uint8, copy=False,
                                        data=intersection(list(r['seg'].values())))

def compute_spectrum(cube, weights):
    """Compute a spectrum for a cube by summing along the spatial axis.
    The method conserves the flux by using the algorithm
    from Jarle Brinchmann (jarle@strw.leidenuniv.nl):
    It takes into account bad pixels in the addition.
    It normalizes with the median value of weighting sum/no-weighting sum

    Parameters
    ----------

    cube : `~mpdaf.obj.Cube`
           Data cube
    weights : array
              An array of weights associated with the data values.
    """
    w = np.array(weights, dtype=np.float)
    excmsg = 'Incorrect dimensions for the weights (%s) (it must be (%s))'

    if len(w.shape) == 3:
        if not np.array_equal(w.shape, cube.shape):
            raise IOError(excmsg % (w.shape, cube.shape))
    elif len(w.shape) == 2:
        if w.shape[0] != cube.shape[1] or w.shape[1] != cube.shape[2]:
            raise IOError(excmsg % (w.shape, cube.shape[1:]))
        else:
            w = np.tile(w, (cube.shape[0], 1, 1))
    elif len(w.shape) == 1:
        if w.shape[0] != cube.shape[0]:
            raise IOError(excmsg % (w.shape[0], cube.shape[0]))
        else:
            w = np.ones_like(cube._data) * w[:, np.newaxis, np.newaxis]
    else:
        raise IOError(excmsg % (None, cube.shape))

    # weights mask
    wmask = np.ma.masked_where(cube._mask, np.ma.masked_where(w == 0, w))

    data = cube.data * w
    npix = np.sum(np.sum(~cube.mask, axis=1), axis=1)
    data = np.ma.sum(np.ma.sum(data, axis=1), axis=1) / npix
    orig_data = cube.data * ~wmask.mask
    orig_data = np.ma.sum(np.ma.sum(orig_data, axis=1), axis=1)
    rr = data / orig_data
    med_rr = np.ma.median(rr)
    if med_rr > 0:
        data /= med_rr
    if cube._var is not None:
        var = np.ma.sum(np.ma.sum(cube.var * w, axis=1), axis=1) / npix
        dspec = np.ma.sqrt(var)
        if med_rr > 0:
            dspec /= med_rr
        orig_var = cube._var * ~wmask.mask
        orig_var = np.ma.masked_where(cube.mask,
                                   np.ma.masked_invalid(orig_var))
        orig_var = np.ma.sum(np.ma.sum(orig_var, axis=1), axis=1)
        sn_orig = orig_data / np.ma.sqrt(orig_var)
        sn_now = data / dspec
        sn_ratio = np.ma.median(sn_orig / sn_now)
        dspec /= sn_ratio
        var = dspec * dspec
        var = var.filled(np.inf)
    else:
        var = None

    return Spectrum(wave=cube.wave, unit=cube.unit, data=data, var=var,
                            copy=False)
