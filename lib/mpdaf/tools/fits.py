# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2015-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Martin Shepherd <martin.shepherd@univ-lyon1.fr>

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
import fnmatch
import logging
import numpy as np
import os

from astropy.io import fits

from ..version import __version__

__all__ = ('add_mpdaf_method_keywords', 'add_mpdaf_keywords_to_file',
           'fix_unit_read', 'fix_unit_write', 'copy_header', 'copy_keywords',
           'is_valid_fits_file', 'read_slice_from_fits')

FITS_EXTENSIONS = ("fits", "fits.gz", "fits.fz")


def add_mpdaf_method_keywords(header, method, params, values, comments):
    """Add keywords in a FITS header to describe the method and the
    corresponding parameters.

    Parameters
    ----------
    header : pyfits.Header
        FITS header.
    method : string
        MPDAF method identifier
    params : list of strings
        Names of parameters
    values : list
        Values of parameters
    comments : list of strings
        parameters description

    """
    i = 1
    while 'MPDAF METH%d ID' % i in header:
        i += 1
    header['HIERARCH MPDAF METH%d VERSION' % i] = (__version__,
                                                   'MPDAF version')
    header['HIERARCH MPDAF METH%d ID' % i] = (method,
                                              'MPDAF method identifier')
    n = len(params)
    for p in range(n):
        header['HIERARCH MPDAF METH%d PARAM%d NAME' % (i, p + 1)] = (
            params[p], comments[p])
        keyword = 'HIERARCH MPDAF METH%d PARAM%d VALUE' % (i, p + 1)
        if isinstance(values[p], str):
            n = 80 - len(keyword) - 14
            s = values[p][0:n]
            header[keyword] = s
        else:
            header[keyword] = values[p]


def add_mpdaf_keywords_to_file(path, method, params, values, comments, ext=0):
    """Add keywords in a FITS file header to describe the method and the
    corresponding parameters.

    Parameters
    ----------
    path : str
        File path.
    method : string
        MPDAF method identifier
    params : list of strings
        Names of parameters
    values : list
        Values of parameters
    comments : list of strings
        parameters description

    """
    with fits.open(path, mode='update') as hdul:
        add_mpdaf_method_keywords(hdul[ext].header, method,
                                  params, values, comments)
        hdul.flush()


def fix_unit_read(x):
    x = x.replace('10**(-20)', '1e-20')
    x = x.replace('*', ' ')
    x = x.replace('mum', 'micron')
    return x


def fix_unit_write(x):
    x = x.replace('1e-20', '10**(-20)')
    return x


def copy_header(srchdr, dsthdr=None, exclude=(), unit=None):
    """Copy all keywords from a FITS header to another.

    Parameters
    ----------
    srchdr : `astropy.io.fits.Header`
        Source header
    dsthdr : `astropy.io.fits.Header`
        Destination header, created if needed.
    exclude : list
        List of glob patterns to exclude keywords.
    unit : str or `astropy.units.Unit`
        Unit

    """
    logger = logging.getLogger(__name__)
    if dsthdr is None:
        dsthdr = fits.Header()

    keys = set(srchdr.keys()) - set(dsthdr.keys())
    if exclude:
        for pat in exclude:
            keys -= set(fnmatch.filter(keys, pat))

    for card in srchdr.cards:
        if card.keyword not in keys:
            continue
        try:
            card.verify('fix')
            dsthdr.append(card, end=True)
        except Exception:
            try:
                if isinstance(card.value, str):
                    n = 80 - len(card.keyword) - 14
                    s = card.value[0:n]
                else:
                    s = card.value
                dsthdr['hierarch %s' % card.keyword] = (s, card.comment)
            except Exception:
                logger.warning("%s not copied in data header", card.keyword)

    if unit is not None:
        try:
            dsthdr['BUNIT'] = (unit.to_string('fits'), 'data unit type')
        except u.format.fits.UnitScaleError:
            dsthdr['BUNIT'] = (fix_unit_write(str(unit)), 'data unit type')

    return dsthdr


def copy_keywords(srchdr, dsthdr, keys):
    """Copy a list of FITS keywords from one header to another.

    Parameters
    ----------
    srchdr : `astropy.io.fits.Header`
        Source header
    dsthdr : `astropy.io.fits.Header`
        Destination header
    keys : list
        List of keys

    """
    for key in keys:
        if key in srchdr:
            dsthdr[key] = (srchdr[key], srchdr.comments[key])


def is_valid_fits_file(filename):
    """Return True is a file exist and is a valid FITS file (based on its
    extension)."""
    return os.path.isfile(filename) and filename.endswith(FITS_EXTENSIONS)


def read_slice_from_fits(filename_or_hdu, item=None, ext='DATA', mask_ext=None,
                         dtype=None, convert_float64=True):
    """Read data from a FITS file."""

    try:
        if isinstance(filename_or_hdu, fits.HDUList):
            close_hdu = False
            hdulist = filename_or_hdu
        else:
            hdulist = fits.open(filename_or_hdu)
            close_hdu = True

        data = hdulist[ext].data
        if item is not None:
            data = data[item]
        data = np.asarray(data, dtype=dtype)
        # Force data to be in double instead of float
        if convert_float64 and data.dtype.type == np.float32:
            data = data.astype(np.float64)

        # mask extension
        if mask_ext is not None and mask_ext in hdulist:
            mask = hdulist[mask_ext].data
            if item is not None:
                mask = mask[item]
            mask = np.asarray(mask, dtype=bool)
        else:
            mask = None
    finally:
        if close_hdu:
            hdulist.close()

    return data, mask
