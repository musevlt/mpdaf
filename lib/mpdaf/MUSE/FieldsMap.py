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
"""
from __future__ import absolute_import, division

from astropy.io import fits
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import fftconvolve
import six
from six.moves import range
from six.moves import zip

class FieldsMap(object):
    """Class to work with the mosaic field map.
    """

    def __init__(self, filename=None, nfields=None, **kwargs):
        """Class to work with the mosaic field map.
        
        Parameters
        ----------
        filename : FITS file name
                   Name of the file containing the field map.
                   Use ext='FIELDMAP' to read the field map from an
                   extension the MUSE data cube.
        nfields : integer
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
            indexes = (i for i in range(self.nfields))
        
    def __getitem__(self, item):
        """Return a sliced object.
        """
        res = self.__class__()
        res.data = self.data[item]
        res.nfields = self.nfields
        return res

    def get_field_mask(self, field_name):
        """Return an array with non-zeros values for pixels matching a field.

        ``field_name`` can be an integer (between 1 and nfields+1) or a string (e.g.
        UDF-03).

        """
        if isinstance(field_name, six.string_types):
            field_name = int(field_name)
        return (self.data & 2**field_name).astype(bool)

    def get_pixel_fields(self, y, x):
        """Return a list of fields that cover a given pixel (y, x)."""
        ind = reversed("{0:010b}".format(self.data[y, x])[:-1])
        fields = ('UDF-%02d' % i for i in range(1, self.nfields+1))
        return [field for field, i in zip(fields, ind) if i == '1']
        
    def get_pixel_fields_indexes(self, y, x):
        """Return a list of fields indexes (between 0 and nfields) 
        that cover a given pixel (y, x)."""
        ind = reversed("{0:010b}".format(self.data[y, x])[:-1])
        indexes = (i for i in range(self.nfields))
        return [index for index, i in zip(indexes, ind) if i == '1']
    
    def compute_weights(self):
        """Return a list of weight maps (one per fields).
        The weight gives the influence of the field for each pixel.
        In the overlap area the weight changes linearly to have a
        smooth transition.
        """
        p, q = np.meshgrid(range(self.data.shape[0]),
                           range(self.data.shape[1]), 
                           sparse=False, indexing='ij')
    
        # compute the mask for each field
        fmaps = []
        for i in range(1, self.nfields+1):
            fmaps.append(self.get_field_mask(i).astype(np.int))
       
        several = (np.sum(fmaps, axis=0) > 1)

        w = []
        s = None
        for m in fmaps:
            # pixels just in one field
            ksel = np.where(np.logical_or(~m, ~several))
            # pixels just in this field
            pd = list(ksel[0])
            qd = list(ksel[1])
            z = m[ksel].astype(np.float)
            wmap = griddata((pd, qd), z, (p, q), method='linear')
            w.append(wmap)
            if s is None:
                s = wmap.copy()
            else:
                s += wmap
            
        ksel = np.where((s!=0) & (s!=1))
        for wmap in w:
            wmap[ksel] /= s[ksel]

        return w
    
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
                FSF *= weights[i] * kernels[i]
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
        #img and weights shall have the same shape
        if img.shape != weights[0].shape:
            raise IOError('img and weights shall have the same shape')

        convolved_img = np.zeros_like(img)

        # build a weighting map per PSF and convolve
        for i in range(self.nfields):
            convolved_img = convolved_img \
                        + fftconvolve(weights[i]*img,
                                      kernels[i]/np.sum(kernels[i]),
                                      mode='same')

        return convolved_img
    

