def add_mpdaf_method_keywords(header, method, params, values, comments):
    """adds keywords in a FITS header to describe the method
    and the corresponding parameters.

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
    while 'MPDAF METH%d ID'%i in header:
        i += 1
    import mpdaf
    header['HIERARCH MPDAF METH%d VERSION'%i] = (mpdaf.__version__,'MPDAF version')
    header['HIERARCH MPDAF METH%d ID'%i] = (method,'MPDAF method identifier')
    n = len(params)
    for p in range(n):
        header['HIERARCH MPDAF METH%d PARAM%d NAME'%(i,p+1)] = (params[p], comments[p])
        header['HIERARCH MPDAF METH%d PARAM%d VALUE'%(i,p+1)] = values[p]