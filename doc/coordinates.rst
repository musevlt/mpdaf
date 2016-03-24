World coordinates
=================

The :class:`~mpdaf.obj.WCS` class manages world coordinates in spatial
direction (2-dimensions WCS object of pywcs package is used).

The :class:`~mpdaf.obj.WaveCoord` class manages world coordinates in
spectral direction (1-dimension WCS object of pywcs package is used).

deg2sexa and sexa2deg methods transforms coordinates from degree/sexagesimal to
sexagesimal/degree.

Note that by convention python reverse x,y indices : ``(dec,ra)`` order is
used.

Degree / sexagesimal conversion
-------------------------------

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

Example of conversion::

    from mpdaf.obj import deg2sexa,sexa2deg
    ac = np.array([2.5,2.5])
    ac2 = [ac,ac*2,ac*4]
    print ac2
    ac3 = deg2sexa(ac2)
    print ac3
    ac = sexa2deg(ac3)
    print ac

World coordinates in spatial direction
--------------------------------------

The `~mpdaf.obj.WCS` class manages spatial world coordinates.

Example::

    from mpdaf.obj import WCS

    # creates a WCS object from data header
    wcs = WCS(hdr)

    # the reference point is the center of the image
    wcs = WCS(shape=(300,300))

    # the reference point is in decimal degree
    wcs = WCS(crval=(-3.11E+01,1.46E+02),cdelt=4E-04, deg=True, rot = 20, shape=(300,300))


World coordinates in spectral direction
---------------------------------------

The `~mpdaf.obj.WaveCoord` class manages world coordinates in spectral
direction.


Reference/API
-------------

.. automodapi:: mpdaf.obj.coords
