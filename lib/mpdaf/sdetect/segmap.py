"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2019 Simon Conseil <simon.conseil@univ-lyon1.fr>

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
from collections import defaultdict
from os.path import exists
from scipy import ndimage as ndi

from ..obj import Image, moffat_image
from ..sdetect import Catalog
from ..tools import isiter, progressbar

__all__ = ('Segmap', 'create_masks_from_segmap')


class Segmap:
    """
    Handle segmentation maps, where pixel values are sources ids.
    """

    def __init__(self, file_or_image, cut_header_after='D001VER'):
        if isinstance(file_or_image, str):
            self.img = Image(file_or_image)
        elif isinstance(file_or_image, Image):
            self.img = file_or_image
        elif isinstance(file_or_image, np.ndarray):
            self.img = Image(data=file_or_image, copy=False, mask=np.ma.nomask)
        else:
            raise TypeError('unknown input')

        if cut_header_after:
            if cut_header_after in self.img.data_header:
                idx = self.img.data_header.index(cut_header_after)
                self.img.data_header = self.img.data_header[:idx]
            if cut_header_after in self.img.primary_header:
                idx = self.img.primary_header.index(cut_header_after)
                self.img.primary_header = self.img.primary_header[:idx]

    def copy(self):
        im = self.__class__(self.img.copy())
        im._mask = np.ma.nomask
        return im

    def get_mask(self, value, dtype=np.uint8, dilate=None, inverse=False,
                 struct=None, regrid_to=None, outname=None):
        if inverse:
            data = (self.img._data != value)
        else:
            data = (self.img._data == value)
        if dilate:
            data = dilate_mask(data, niter=dilate, struct=struct)

        im = Image.new_from_obj(self.img, data)
        if regrid_to:
            im = regrid_to_image(im, regrid_to, inplace=True, order=0,
                                 antialias=False)
            np.around(im._data, out=im._data)

        im._data = im._data.astype(dtype)
        im._mask = np.ma.nomask
        if inverse:
            np.logical_not(im._data, out=im._data)

        if outname:
            im.write(outname, savemask='none')

        return im

    def get_source_mask(self, iden, center, size, minsize=None, dilate=None,
                        dtype=np.uint8, struct=None, unit_center=u.deg,
                        unit_size=u.arcsec, regrid_to=None, outname=None):
        if minsize is None:
            minsize = size

        im = self.img.subimage(center, size, minsize=minsize,
                               unit_center=unit_center, unit_size=unit_size)

        if isiter(iden):
            # combine the masks for multiple ids
            data = np.logical_or.reduce([(im._data == i) for i in iden])
        else:
            data = (im._data == iden)

        if dilate:
            data = dilate_mask(data, niter=dilate, struct=struct)

        if regrid_to:
            other = regrid_to.subimage(center, size, minsize=0.,
                                       unit_center=unit_center,
                                       unit_size=unit_size)
            im._data = data.astype(float)
            im = regrid_to_image(im, other, size=size, order=0,
                                 inplace=True, antialias=False)
            data = np.around(im._data, out=im._data)

        im._data = data.astype(dtype)
        im._mask = np.ma.nomask

        logger = logging.getLogger(__name__)
        logger.debug('source %s (%.5f, %.5f), extract mask (%d masked pixels)',
                     iden, center[1], center[0], np.count_nonzero(im._data))
        if outname:
            im.write(outname, savemask='none')
        else:
            return im

    def align_with_image(self, other, inplace=False, truncate=False, margin=0):
        """Rotate and truncate the segmap to match 'other'."""
        out = self if inplace else self.copy()
        rot = other.wcs.get_rot() - self.img.wcs.get_rot()
        if np.abs(rot) > 1.e-3:
            out.img = self.img.rotate(rot, reshape=True, regrid=True,
                                      flux=False, order=0, inplace=inplace)

        if truncate:
            y0 = margin - 1
            y1 = other.shape[0] - margin
            x0 = margin - 1
            x1 = other.shape[1] - margin
            pixsky = other.wcs.pix2sky([[y0, x0],
                                        [y1, x0],
                                        [y0, x1],
                                        [y1, x1]],
                                       unit=u.deg)
            pixcrd = out.img.wcs.sky2pix(pixsky)
            ymin, xmin = pixcrd.min(axis=0)
            ymax, xmax = pixcrd.max(axis=0)
            out.img.truncate(ymin, ymax, xmin, xmax, mask=False,
                             unit=None, inplace=True)

        out.img._data = np.around(out.img._data).astype(int)
        # FIXME: temporary workaround to make sure that the data_header is
        # up-to-date when pickling the segmap. This should be detected direclty
        # in MPDAF.
        out.img.data_header = out.img.get_wcs_header()
        return out

    def cmap(self, background_color='#000000'):
        """matplotlib colormap with random colors.
        (taken from photutils' segmentation map class)"""
        return get_cmap(self.img.data.max() + 1,
                        background_color=background_color)


def dilate_mask(data, thres=0.5, niter=1, struct=None):
    if struct is None:
        struct = ndi.generate_binary_structure(2, 1)
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(0)
    maxval = data.max()
    if maxval != 1:
        data /= maxval
        data = data > 0.5
    return ndi.binary_dilation(data, structure=struct, iterations=niter)


def get_cmap(ncolors, background_color='#000000'):
    from matplotlib import colors
    prng = np.random.RandomState(42)
    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)
    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))
    cmap = colors.ListedColormap(rgb)

    if background_color is not None:
        cmap.colors[0] = colors.hex2color(background_color)

    return cmap


def regrid_to_image(im, other, order=1, inplace=False, antialias=True,
                    size=None, unit_size=u.arcsec, **kwargs):
    im.data = im.data.astype(float)
    refpos = other.wcs.pix2sky([0, 0])[0]
    if size is not None:
        newdim = size / other.wcs.get_step(unit=unit_size)
    else:
        newdim = other.shape
    inc = other.wcs.get_axis_increments(unit=unit_size)
    im = im.regrid(newdim, refpos, [0, 0], inc, order=order,
                   unit_inc=unit_size, inplace=inplace, antialias=antialias)
    return im


def struct_from_moffat_fwhm(wcs, fwhm, psf_threshold=0.5, beta=2.5):
    """Compute a structuring element for the dilatation, to simulate
    a convolution with a psf."""
    # image size will be twice the full-width, to account for
    # psf_threshold < 0.5
    size = int(round(fwhm / wcs.get_step(u.arcsec)[0])) * 2 + 1

    psf = moffat_image(fwhm=(fwhm, fwhm), n=beta, peak=True,
                       wcs=wcs[:size, :size])

    # remove useless zeros on the edges.
    psf.mask_selection(psf._data < psf_threshold)
    psf.crop()
    assert tuple(np.array(psf.shape) % 2) == (1, 1)
    return ~psf.mask


def _get_psf_convolution_params(convolve_fwhm, segmap, psf_threshold):
    if convolve_fwhm:
        # compute a structuring element for the dilatation, to simulate
        # a convolution with a psf, but faster.
        dilateit = 1
        struct = struct_from_moffat_fwhm(segmap.img.wcs, convolve_fwhm,
                                         psf_threshold=psf_threshold)
    else:
        dilateit = 0
        struct = None
    return dilateit, struct


def create_masks_from_segmap(
        segmap, catalog, ref_image, n_jobs=1, skip_existing=True,
        masksky_name='mask-sky.fits', maskobj_name='mask-source-%05d.fits',
        idname='ID', raname='RA', decname='DEC', margin=0, mask_size=(20, 20),
        convolve_fwhm=0, psf_threshold=0.5, verbose=0):
    """Create binary masks from a segmentation map.

    For each source from the catalog, extract the segmap region, align with
    ref_image and regrid to the resolution of ref_image.

    Parameters
    ----------
    segmap : str or `mpdaf.obj.Image`
        The segmentation map.
    catalog : str or `mpdaf.sdetect.Catalog` or `astropy.table.Table`
        The catalog with sources id and position.
    ref_image : str or `mpdaf.obj.Image`
        The reference image, with which the segmap is aligned.
    n_jobs : int
        Number of parallel processes (for joblib).
    skip_existing : bool
        If True, skip sources for which the mask file exists.
    masksky_name : str or callable
        The filename for the sky mask.
    maskobj_name : str or callable
        The filename for the source masks, with a format string that will be
        substituted with the ID, e.g. ``%05d``.
    idname, raname, decname : str
        Name of the 'id', 'ra' and 'dec' columns.
    margin : float
        Margin used for the segmap alignment (pixels).
    mask_size : tuple
        Size of the source masks (arcsec).
    convolve_fwhm : float
        FWHM for the PSF convolution (arcsec).
    psf_threshold : float
        Threshold applied to the PSF to get a binary image.
    verbose: int
        Verbosity level for joblib.Parallel.

    """
    from joblib import delayed, Parallel

    logger = logging.getLogger(__name__)

    if isinstance(ref_image, str):
        ref_image = Image(ref_image)
    if isinstance(catalog, str):
        catalog = Catalog.read(catalog)
    if not isinstance(segmap, Segmap):
        segmap = Segmap(segmap)

    logger.info('Aligning segmap with reference image')
    segm = segmap.align_with_image(ref_image, truncate=True, margin=margin)

    dilateit, struct = _get_psf_convolution_params(convolve_fwhm, segm,
                                                   psf_threshold)

    # create sky mask
    masksky = masksky_name() if callable(masksky_name) else masksky_name
    if exists(masksky) and skip_existing:
        logger.debug('sky mask exists, skipping')
    else:
        logger.debug('creating sky mask')
        segm.get_mask(0, inverse=True, dilate=dilateit, struct=struct,
                      regrid_to=ref_image, outname=masksky)

    # extract source masks
    minsize = 0.
    to_compute = []
    stats = defaultdict(list)

    for row in catalog:
        id_ = int(row[idname])  # need int, not np.int64
        source_path = (maskobj_name(id_) if callable(maskobj_name)
                       else maskobj_name % id_)
        if skip_existing and exists(source_path):
            stats['skipped'].append(id_)
        else:
            center = (row[decname], row[raname])
            stats['computed'].append(id_)
            to_compute.append(delayed(segm.get_source_mask)(
                id_, center, mask_size, minsize=minsize, struct=struct,
                dilate=dilateit, outname=source_path, regrid_to=ref_image))

    # FIXME: check which value to use for max_nbytes
    if to_compute:
        logger.info('computing masks for %d sources', len(to_compute))
        Parallel(n_jobs=n_jobs, verbose=verbose)(progressbar(to_compute))
    else:
        logger.info('nothing to compute')
