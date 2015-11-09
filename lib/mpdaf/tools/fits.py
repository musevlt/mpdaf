# -*- coding: utf-8 -*-

import numpy as np
import os
from astropy.io import fits
from numpy import ma


def add_mpdaf_method_keywords(header, method, params, values, comments):
    """Add keywords in a FITS header to describe the method and the
    corresponding parameters.

    Parameters
    ----------
    header   : pyfits.Header
               FITS header.
    method   : string
               MPDAF method identifier
    params   : list of strings
               Names of parameters
    values   : list
               Values of parameters
    comments : list of strings
               parameters description
    """
    i = 1
    while 'MPDAF METH%d ID' % i in header:
        i += 1
    import mpdaf
    header['HIERARCH MPDAF METH%d VERSION' % i] = (mpdaf.__version__,
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
    path     : str
               File path.
    method   : string
               MPDAF method identifier
    params   : list of strings
               Names of parameters
    values   : list
               Values of parameters
    comments : list of strings
               parameters description
    """
    hdu = fits.open(path, mode='update')
    add_mpdaf_method_keywords(hdu[ext].header, method,
                              params, values, comments)
    hdu.flush()
    hdu.close()


def copy_keywords(srchdr, dsthdr, keys):
    """Copy a list of FITS keywords from one header to another.

    Parameters
    ----------
    srchdr : astropy.io.fits.Header
             Source header
    dsthdr : astropy.io.fits.Header
             Destination header
    keys   : list
             List of keys
    """
    for key in keys:
        if key in srchdr:
            dsthdr[key] = (srchdr[key], srchdr.comments[key])


def is_valid_fits_file(filename):
    """Return True is a file exist and is a valid FITS file (based on its
    extension)."""
    return os.path.isfile(filename) and filename.endswith(("fits", "fits.gz"))


def read_slice_from_fits(filename, item=None, ext='DATA', mask_ext=None,
                         dtype=None):
    """Read data from a FITS file."""
    hdulist = fits.open(filename)
    if item is None:
        data = np.asarray(hdulist[ext].data, dtype=dtype)
    else:
        data = np.asarray(hdulist[ext].data[item], dtype=dtype)

    # mask extension
    if mask_ext is not None and mask_ext in hdulist:
        mask = ma.make_mask(hdulist[mask_ext].data[item])
        data = ma.MaskedArray(data, mask=mask)

    hdulist.close()
    return data
