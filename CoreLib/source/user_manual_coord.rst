Interface for world coordinates
*******************************


    WCS class manages world coordinates in spatial direction (pywcs package is used) .

    WaveCoord class manages world coordinates in spectral direction.

    deg2sexa and sexa2deg methods transforms coordinates from degree/sexagesimal to sexagesimal/degree.
    
    Note that by convention python reverse x,y indices : (dec,ra) order is used.
    

Degree / sexagesimal conversion
===============================

:func:`mpdaf.obj.deg2sexa` transforms the values of n coordinates from degrees to sexagesimal.

:func:`mpdaf.obj.sexa2deg` transforms the values of n coordinates from sexagesimal to degrees.

:func:`mpdaf.obj.deg2hms` transforms a degree value to a string representation of the coordinate as hours:minutes:seconds.

:func:`mpdaf.obj.hms2deg` transforms a string representation of the coordinate as hours:minutes:seconds to a float degree value.

:func:`mpdaf.obj.deg2dms` transforms a degree value to a string representation of the coordinate as degrees:arcminutes:arcseconds.

:func:`mpdaf.obj.dms2deg` transforms a string representation of the coordinate as degrees:arcminutes:arcseconds to a float degree value.


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
======================================

WCS class manages spatial world coordinates.


References
----------

:func:`mpdaf.obj.WCS.copy` copies WCS object in a new one and returns it.

:func:`mpdaf.obj.WCS.info` prints information.

:func:`mpdaf.obj.WCS.to_header` generates a pyfits header object with the WCS information.

:func:`mpdaf.obj.WCS.sky2pix` converts world coordinates to pixel coordinates.

:func:`mpdaf.obj.WCS.pix2sky` converts pixel coordinates to world coordinates.

:func:`mpdaf.obj.WCS.isEqual` tests if 2 WCS objects have the same attributes.

:func:`mpdaf.obj.WCS.get_step` returns the steps along the Y and X axis.

:func:`mpdaf.obj.WCS.get_range` returns the minimum and maximum coordinates values.

:func:`mpdaf.obj.WCS.get_start` returns coordinates corresponding to the pixel (0,0).

:func:`mpdaf.obj.WCS.get_end` returns coordinates corresponding to the pixel (-1,-1).

:func:`mpdaf.obj.WCS.get_rot` returns the rotation angle.

:func:`mpdaf.obj.WCS.rotate` rotates WCS coordinates to a new orientation.

:func:`mpdaf.obj.WCS.rebin` rebins to a new coordinate system.

:func:`mpdaf.obj.WCS.is_deg` returns True if world coordinates are in decimal degrees.


World coordinates in spectral direction
=======================================

WaveCoord class manages world coordinates in spectral direction.


References
----------

:func:`mpdaf.obj.WaveCoord.copy` copies WaveCoord object in a new one and returns it.

:func:`mpdaf.obj.WaveCoord.info` prints information.

:func:`mpdaf.obj.WaveCoord.isEqual` tests if 2 WaveCoords objects have the same attributes.

:func:`mpdaf.obj.WaveCoord.coord` returns coordinate(s) corresponding to pixel value(s).

:func:`mpdaf.obj.WaveCoord.pixel` returns pixel value(s) corresponding the coordinate(s).

:func:`mpdaf.obj.WaveCoord.rebin` rebins to a new coordinate system.

:func:`mpdaf.obj.WaveCoord.get_step` returns the step in wavelength.

:func:`mpdaf.obj.WaveCoord.get_start` returns the wavelength value of the first pixel.

:func:`mpdaf.obj.WaveCoord.get_end` returns the wavelength value of the last pixel.

:func:`mpdaf.obj.WaveCoord.get_range` returns the wavelength range.
