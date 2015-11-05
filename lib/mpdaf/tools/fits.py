from astropy.io import fits


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
