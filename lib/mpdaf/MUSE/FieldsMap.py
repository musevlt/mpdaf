"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2016-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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

import numpy as np
import os

from astropy.io import fits
from scipy.interpolate import griddata
from scipy.signal import fftconvolve

from ..obj import Image

__all__ = ['create_fields_map', 'FieldsMap']


def create_fields_map(imglist, refimg, outfile):
    """Create a "Field map" image.

    Parameters
    ----------
    imglist : list of str
        List of image filenames.
    refimg : `mpdaf.obj.Image`
        Reference image, used to create the field map image.
    outfile : str
        Output filename.

    """
    maskim = Image(data=np.zeros(refimg.shape, dtype=np.uint),
                   wcs=refimg.wcs.copy(), dtype=None, copy=False)
    nimg = len(imglist)

    for i, f in enumerate(imglist, 1):
        img = Image(f)
        field = int(img.primary_header['OBJECT'][-2:])
        img = Image(f)
        offset = (- img.wcs.wcs.wcs.crpix[::-1]).astype(int)
        sy, sx = list(zip(offset, offset + np.array(img.shape)))
        print('{:03d}/{}'.format(i, nimg), os.path.basename(f), field, sy, sx)
        maskim.data[slice(*sy), slice(*sx)] |= (2**field *
                                                (~img.mask).astype(np.uint))

    maskim.write(outfile, savemask='none')


class FieldsMap(object):

    def __init__(self, filename=None, nfields=None, **kwargs):
        """Class to work with the mosaic field map.

        Parameters
        ----------
        filename : str
            Name of the FITS file containing the field map. Use
            ``extname='FIELDMAP'`` to read the field map from an
            extension the MUSE data cube.
        nfields : int
            Number of fields.

        """
        if filename is None:
            self.nfields = 0
            self.data = None
        else:
            if nfields is None:
                self.nfields = fits.getval(filename, 'NFIELDS')
            else:
                self.nfields = nfields
            self.data = fits.getdata(filename, **kwargs)

    def __getitem__(self, item):
        """Return a sliced object.
        """
        res = self.__class__()
        res.data = self.data[item]
        res.nfields = self.nfields
        return res

    def get_field_mask(self, field_name):
        """Return an array with non-zeros values for pixels matching a field.

        ``field_name`` can be an integer (between 1 and nfields+1) or a string
        (e.g. UDF-03).

        """
        if isinstance(field_name, str):
            field_name = int(field_name[-2:])
        return (self.data & 2**field_name).astype(bool)

    def get_pixel_fields(self, y, x):
        """Return a list of fields that cover a given pixel (y, x)."""
        ind = reversed("{0:010b}".format(self.data[y, x])[:-1])
        fields = ('UDF-%02d' % i for i in range(1, self.nfields + 1))
        return [field for field, i in zip(fields, ind) if i == '1']

    def get_pixel_fields_indexes(self, y, x):
        """Return a list of fields indexes (between 0 and nfields)
        that cover a given pixel (y, x)."""
        ind = reversed("{0:010b}".format(self.data[y, x])[:-1])
        indexes = (i for i in range(self.nfields))
        return [index for index, i in zip(indexes, ind) if i == '1']

    def compute_weights(self):
        """Return a list of weight maps (one per fields).

        The weight gives the influence of the field for each pixel.  In the
        overlap area the weight changes linearly to have a smooth transition.

        """
        p, q = np.mgrid[:self.data.shape[0], :self.data.shape[1]]

        # compute the mask for each field
        fmaps = [self.get_field_mask(i).astype(int)
                 for i in range(1, self.nfields + 1)]

        several = (np.sum(fmaps, axis=0) > 1)
        weights = []
        s = None
        for m in fmaps:
            # pixels just in one field
            ksel = np.where(np.logical_or(~m, ~several))
            # pixels just in this field
            pd = list(ksel[0])
            qd = list(ksel[1])
            z = m[ksel].astype(float)
            wmap = griddata((pd, qd), z, (p, q), method='linear')
            weights.append(wmap)
            if s is None:
                s = wmap.copy()
            else:
                s += wmap

        ksel = np.where((s != 0) & (s != 1))
        for wmap in weights:
            wmap[ksel] /= s[ksel]

        return weights

    def get_FSF(self, y, x, kernels, weights=None):
        """Return the local FSF.

        Parameters
        ----------
        y       : integer
                  Pixel coordinate along the y-axis.
        x       : integer
                  Pixel coordinate along the x-axis.
        kernels : list of np.array
                  List of FSF.
        weights : list of np.array
                  List of corresponding weights maps.
                  Computed by compute_weights by default.
        """
        if weights is None:
            weights = self.compute_weights()
        fields = self.get_pixel_fields_indexes(y, x)
        if len(fields) == 0:
            return None
        elif len(fields) == 1:
            return kernels[fields[0]]
        else:
            i0 = fields[0]
            FSF = weights[i0] * kernels[i0]
            for i in fields[1:]:
                FSF += weights[i] * kernels[i]
            return FSF

    def variable_PSF_convolution(self, img, kernels, weights=None):
        """ Function used for the convolution of an image by a set of PSF.

        We use shift-variant blur techniques to model the variation of the PSF.

        Reference: Denis, L. Thiebaut E., Soulez F., Becker J.-M. and Mourya R.
               'Fast approximations of shift-variant blur',
               International Journal of Computer Vision,
               Springer Verlag, 115(3), 253-278 (2015)

        Parameters
        ----------
        img     : np.array
                  Image to convolve.
        kernels : list(np.array)
                  List of convolution kernels.
        weights : list of np.array
                  List of corresponding weights maps.
                  Computed by compute_weights by default.
        """
        if weights is None:
            weights = self.compute_weights()
        # kernels and weights shall have the same length
        if len(kernels) != len(weights):
            raise IOError('kernels and weights shall have the same length')
        # img and weights shall have the same shape
        if img.shape != weights[0].shape:
            raise IOError('img and weights shall have the same shape')

        convolved_img = np.zeros_like(img)

        # build a weighting map per PSF and convolve
        for i in range(self.nfields):
            convolved_img = convolved_img \
                + fftconvolve(weights[i] * img,
                              kernels[i] / np.sum(kernels[i]),
                              mode='same')

        return convolved_img
