Spectrum class
==============

Examples::

  import numpy as np
  from mpdaf.obj import Spectrum
  from mpdaf.obj import WaveCoord

  spe = Spectrum(filename="spectrum.fits",ext=1) # spectrum from file (extension number is 1)

  wave1 = WaveCoord(cdelt=1.25, crval=4000.0, cunit='Angstrom')
  wave2 = WaveCoord(cdelt=1.25, crval=4000.0, cunit='Angstrom', shape=3000)
  MyData = np.ones(4000)

  spe = Spectrum(wave=wave1, data=MyData) # spectrum filled with MyData
  spe = Spectrum(wave=wave2, data=MyData) # warning: wavelength coordinates and data have not the same dimensions. Shape of WaveCoord object is modified.
					  # Shape of WaveCoord object is modified.
					  # spe.wave = 4000
