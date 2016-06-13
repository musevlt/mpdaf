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


sea.py contains SpectExtractAnd[nothing], the first part of
SpecExtractAndWeb software developed by Jarle.

This software has been developed by Jarle Brinchmann (University of Leiden)
and ported to python by Laure Piqueras (CRAL).

It takes a MUSE data cube and a catalogue of objects and extracts small
sub-cubes around each object. From this it creates narrow-band images and
eventually spectra. To do the latter it is necessary to run an external routine
which runs sextractor on the images to define spectrum extraction apertures.

Please contact Jarle for more info at jarle@strw.leidenuniv.nl

"""
from __future__ import absolute_import, division

from astropy.io import fits as pyfits
import astropy.units as u

import logging
import numpy as np
import os
import shutil
import six
import subprocess

from ..obj import Image
from ..sdetect import Source, SourceList

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
    # suppose that MUSE_WHITE image exists
    try:
        subprocess.check_call(['sex'])
        cmd_sex = 'sex'
    except OSError:
        try:
            subprocess.check_call(['sextractor'])
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


def SEA(cat, cube, images=None, size=10, eml=None, width=8, margin=10.,
        fband=3., DIR=None, psf=None, path=None):
    """

    Parameters
    ----------
    cat : astropy.Table
        Tables containing positions and names of the objects.
        It needs to have at minimum these columns: ID, Z, RA, DEC
        for the name, redshift & position of the object.
    cube : `~mpdaf.obj.Cube`
        Data cube.
    images : `dict`
        Dictionary containing one or more external images of the field
        which you want to extract stamps.

        Keys gives the filter ('HST_F814' for example)

        Values are `~mpdaf.obj.Image` objects.
    size : float
        The total size to extract images in arcseconds.
        By default 10x10 arcsec
    eml  : dict{float: string}
        Full catalog of lines used to extract narrow band images.
        Dictionary: key is the wavelength value in Angstrom,
        value is the name of the line.
        If None, the following catalog is used:

            eml = {1216 : 'Lyalpha1216', 1909: 'CIII]1909', 3727: '[OII]3727',
                   4861 : 'Hbeta4861' , 5007: '[OIII]5007', 6563: 'Halpha6563',
                   6724 : '[SII]6724'}

    width : float
        Angstrom total width used to extract narrow band images.
    margin : float
        Parameter used to extract narrow band images.
        This off-band is offseted by margin wrt narrow-band limit.
    fband : float
        Parameter used to extract narrow band images.
        The size of the off-band is fband*narrow-band width.
    DIR   : string
        Directory that contains the configuration files of sextractor
    psf  : np.array
        The PSF to use for PSF-weighted extraction.
        This can be a vector of length equal to the wavelength
        axis to give the FWHM of the Gaussian PSF at each
        wavelength (in arcsec) or a cube with the PSF to use.
    path : path where the source file will be saved.
        This option should be used to avoid memory problem
        (source are saved as we go along)

    Returns
    -------
    out : `mpdaf.sdetect.SourceList` if path is None
    """
    logger = logging.getLogger(__name__)

    if images is None:
        images = {}

    # create source objects
    sources = []
    origin = ('sea', __version__, os.path.basename(cube.filename))

    ntot = len(cat)
    n = 1

    write = True
    if path is None:
        write = False

    for obj in cat:

        logger.info('%d/%d Doing Source %d' % (n, ntot, obj['ID']))

        cen = cube.wcs.sky2pix([obj['DEC'], obj['RA']], unit=u.deg)[0]
        if cen[0] >= 0 and cen[0] <= cube.wcs.naxis1 and \
                cen[1] >= 0 and cen[1] <= cube.wcs.naxis2:

            source = Source.from_data(obj['ID'], obj['RA'], obj['DEC'], origin)
            try:
                z = obj['Z']
                try:
                    errz = obj['Z_ERR']
                except:
                    try:
                        errz = (obj['Z_MAX'] - obj['Z_MIN']) / 2.0
                    except:
                        errz = np.nan
                source.add_z('CAT', z, errz)
            except:
                z = -9999

            # create white image
            source.add_white_image(cube, size, unit_size=u.arcsec)

            # create narrow band images
            source.add_narrow_band_images(cube=cube, z_desc='CAT', eml=eml,
                                          size=None, unit_size=u.arcsec,
                                          width=width, margin=margin,
                                          fband=fband, is_sum=False)

            # extract images stamps
            for tag, ima in six.iteritems(images):
                source.add_image(ima, 'HST_' + tag)

            # segmentation maps
            source.add_seg_images(DIR=DIR)
            source.add_masks()

            # extract spectra
            source.extract_spectra(cube, skysub=True, psf=psf)
            source.extract_spectra(cube, skysub=False, psf=psf)

            if write:
                if not os.path.exists(path):
                    os.makedirs(path)
                name = os.path.basename(path)
                source.write('%s/%s-%04d.fits' % (path, name, source.ID))
            else:
                sources.append(source)

        n += 1

    # return list of sources
    if not write:
        return SourceList(sources)
