*****************
World coordinates
*****************

The `~mpdaf.obj.WCS` class manages spatial world-coordinates (A
2-dimensional WCS object of the pywcs package is used).

The `~mpdaf.obj.WaveCoord` class manages spectral world-coordinates
(A 1-dimensional WCS object of the pywcs package is used).

Note that by convention python reverses x, y indices : ``(dec,ra)`` order is
used.

Degree / sexagesimal conversion
===============================

- `~mpdaf.obj.deg2sexa` transforms the values of n coordinates from
  degrees to sexagesimal.

- `~mpdaf.obj.sexa2deg` transforms the values of n coordinates from
  sexagesimal to degrees.

- `~mpdaf.obj.deg2hms` transforms a degree value to a string
  representation of the coordinate as hours:minutes:seconds.

- `~mpdaf.obj.hms2deg` transforms a string representation of the
  coordinate as hours:minutes:seconds to a float degree value.

- `~mpdaf.obj.deg2dms` transforms a degree value to a string
  representation of the coordinate as degrees:arcminutes:arcseconds.

- `~mpdaf.obj.dms2deg` transforms a string representation of the
  coordinate as degrees:arcminutes:arcseconds to a float degree value.

Examples of conversions::

    >>> import numpy as np
    >>> from mpdaf.obj import deg2sexa,sexa2deg
    >>> ac = np.array([2.5,2.5])
    >>> ac2 = [ac,ac*2,ac*4]
    >>> print ac2
    [array([ 2.5,  2.5]), array([ 5.,  5.]), array([ 10.,  10.])]
    >>> ac3 = deg2sexa(ac2)
    >>> print ac3
    [['02:30:00' '00:10:00']
    ['05:00:00' '00:20:00']
    ['10:00:00' '00:40:00']]
    >>> ac = sexa2deg(ac3)
    >>> print ac
    [[  2.5   2.5]
    [  5.    5. ]
    [ 10.   10. ]]

Spatial world coordinates
=========================

The `~mpdaf.obj.WCS` class manages spatial world coordinates.

Some examples::

    >>> from mpdaf.obj import WCS
    >>> from astropy.io import fits
    >>> hdr = fits.Header()
    >>> # creates a WCS object from data header
    >>> wcs = WCS(hdr)
    >>> wcs.info()
    [INFO] spatial coord (): min:(1.0,1.0) max:(0.0,0.0) step:(1.0,1.0) rot:-0.0 deg
    >>> # the reference point is the center of the image
    >>> wcs = WCS(shape=(300,300))
    >>> wcs.info()
    [INFO] spatial coord (pix): min:(-148.5,-148.5) max:(150.5,150.5) step:(1.0,1.0) rot:-0.0 deg
    >>> # the reference point is in decimal degree
    >>> wcs = WCS(crval=(-3.11E+01,1.46E+02),cdelt=4E-04, deg=True, rot = 20, shape=(300,300))
    >>> wcs.info()
    [INFO] center:(-31:06:00,09:44:00) size in arcsec:(432.000,432.000) step in arcsec:(1.440,1.440) rot:20.0 deg

Spectral world coordinates
==========================

The `~mpdaf.obj.WaveCoord` class manages spectral world coordinates.
