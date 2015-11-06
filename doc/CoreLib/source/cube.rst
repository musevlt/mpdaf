Cube class
==========

Examples::

  import astropy.units as u
  import numpy as np
  from mpdaf.obj import Cube
  from mpdaf.obj import WCS
  from mpdaf.obj import WaveCoord

  wcs1 = WCS(crval=0,cdelt=0.2)
  wcs2 = WCS(crval=0,cdelt=0.2, shape=40)
  wave1 = WaveCoord(cdelt=1.25, crval=4000.0, cunit=u.angstrom)
  wave2 = WaveCoord(cdelt=1.25, crval=4000.0, cunit=u.angstrom, shape=300)
  MyData = np.ones((400,30,30))

  cub = Cube(filename="cube.fits",ext=1) # cube from file without variance (extension number is 1)
  cub = Cube(filename="cube.fits",ext=(1,2)) # cube from file with variance (extension numbers are 1 and 2)
  cub = Cube(wcs=wcs1, wave=wave1, data=MyData) # cube filled with MyData
  # warning: wavelength coordinates and data have not the same dimensions.
  # Shape of WaveCoord object is modified.
  # cub.wave.shape = 400
  cub = Cube(wcs=wcs2, wave=wave1, data=MyData) # warning: world coordinates and data have not the same dimensions.
  # Shape of WCS object is modified.
  # cub.wcs.naxis1 = 30
  # cub.wcs.naxis2 = 30

.. autoclass:: mpdaf.obj.Cube
    :members:
    :special-members:




