Cube class
==========

Examples::

  import numpy as np
  from mpdaf.obj import Cube
  from mpdaf.obj import WCS
  from mpdaf.obj import WaveCoord

  wcs1 = WCS(crval=0,cdelt=0.2)
  wcs2 = WCS(crval=0,cdelt=0.2, shape=400)
  wave1 = WaveCoord(cdelt=1.25, crval=4000.0, cunit='Angstrom')
  wave2 = WaveCoord(cdelt=1.25, crval=4000.0, cunit='Angstrom', shape=3000)
  MyData = np.ones((4000,300,300))

  cub = Cube(filename="cube.fits",ext=1) # cube from file without vaiance (extension number is 1)
  cub = Cube(filename="cube.fits",ext=(1,2)) # cube from file with vaiance (extension numbers are 1 and 2)
  cub = Cube(shape=(4000,300,300), wcs=wcs1, wave=wave1) # cube 4000x300x300 filled with zeros
  cub = Cube(wcs=wcs1, wave=wave1, data=MyData) # cube filled with MyData
  cub = Cube(shape=(4000,300,300), wcs=wcs1, wave=wave2) # warning: wavelength coordinates and data have not the same dimensions.
  # Shape of WaveCoord object is modified.
  # cub.wave.shape = 4000
  cub = Cube(wcs=wcs1, wave=wave2, data=MyData) # warning: wavelength coordinates and data have not the same dimensions.
  # Shape of WaveCoord object is modified.
  # cub.wave.shape = 4000
  cub = Cube(shape=(4000,300,300), wcs=wcs2, wave=wave1) # warning: world coordinates and data have not the same dimensions.
  # Shape of WCS object is modified.
  # cub.wcs = (300,300)
  cub = Cube(wcs=wcs2, wave=wave1, data=MyData) # warning: world coordinates and data have not the same dimensions.
  # Shape of WCS object is modified.
  # cub.wcs = (300,300)

.. autoclass:: mpdaf.obj.Cube
    :members:
    :special-members:

.. autoclass:: mpdaf.obj.CubeDisk
	:members:
	:special-members:





