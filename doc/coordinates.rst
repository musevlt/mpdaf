World coordinates
=================

Examples::

  from mpdaf.obj import WCS
  wcs = WCS(hdr) # creates a WCS object from data header
  wcs = WCS(shape=(300,300)) # the reference point is the center of the image
  wcs = WCS(crval=(-3.11E+01,1.46E+02),cdelt=4E-04, deg=True, rot = 20, shape=(300,300)) # the reference point is in decimal degree


.. automodapi:: mpdaf.obj.coords
    :no-heading:
