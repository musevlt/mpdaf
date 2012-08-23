Spectrum object
***************

The Spectrum object handles a 1D data array containing flux values, associated with a WCS 
object (WaveCoord) containing the wavelength information. Optionally, a variance data array 
can be attached and used for weighting the flux values. A bad pixel mask is used to ignore 
some of the pixel values.


Tutorial
========

Preliminary imports::

  import numpy as np
  from mpdaf.obj import Spectrum
  from mpdaf.obj import WaveCoord

A Spectrum object is created: 

- either from one or two numpy data arrays (containing flux values and variance), 
using the following command::

  MyData=np.ones(4000) # numpy data array
  MyVariance=np.ones(4000) # numpy variance array
  spe = Spectrum(data=MyData) # spectrum filled with MyData 
  spe = Spectrum(data=MyData,variance=MyVariance) # spectrum filled with MyData and MyVariance


- or from a FITS file (in which case the flux and variance data are read from specific 
extensions), using the following commands::

  spe = Spectrum(filename="spectrum.fits",ext=1) # data array read from file (extension number 1)
  spe = Spectrum(filename="spectrum.fits",ext=[1,2]) # data and variance arrays read from file (extension numbers 1 and 2)

The WaveCoord object is created using a linear scale, copied from another Spectrum, or 
using the information from the FITS header:



Indexing
--------

``Spectrum[i]`` returns the corresponding value of pixel i.

``Spectrum[i1:i2]`` returns the sub-spectrum between pixels i1 and i2

``Spectrum[i] = value`` sets the value of Spectrum.data[i]

``Spectrum[i1:i2] = array`` sets the values in the corresponding part of Spectrum.data.


Operators
---------

+------+------------------------------------------------------------------------------------+
| <=   | Masks data array where greater than a given value.                                 |
+------+------------------------------------------------------------------------------------+
| <    | Masks data array where greater or equal than a given value.                        |
+------+------------------------------------------------------------------------------------+
| >=   | Masks data array where less than a given value.                                    |
+------+------------------------------------------------------------------------------------+
| >    | Masks data array where less or equal than a given value.                           |
+------+------------------------------------------------------------------------------------+
| \+   | - addition                                                                         |
|      | - spectrum1 + number = spectrum2 (spectrum2[k] = spectrum1[k] + number)            |
|      | - spectrum1 + spectrum2 = spectrum3 (spectrum3[k] = spectrum1[k] + spectrum2[k])   |
|      | - spectrum + cube1 = cube2 (cube2[k,j,i] = cube1[k,j,i] + spectrum[k])             |
+------+------------------------------------------------------------------------------------+	  
| \-   | - substraction                                                                     |
|      | - spectrum1 - number = spectrum2 (spectrum2[k] = spectrum1[k] - number)            |
|      | - spectrum1 - spectrum2 = spectrum3 (spectrum3[k] = spectrum1[k] - spectrum2[k])   |
|      | - spectrum - cube1 = cube2 (cube2[k,j,i] = spectrum[k] - cube1[k,j,i])             |
+------+------------------------------------------------------------------------------------+
| \*   | - multiplication                                                                   |
|      | - spectrum1 \* number = spectrum2 (spectrum2[k] = spectrum1[k] \* number)          |
|      | - spectrum1 \* spectrum2 = spectrum3 (spectrum3[k] = spectrum1[k] \* spectrum2[k]) |
|      | - spectrum \* cube1 = cube2 (cube2[k,j,i] = spectrum[k] \* cube1[k,j,i])           |
|      | - spectrum \* image = cube (cube[k,j,i]=image[j,i] \* spectrum[k]                  |
+------+------------------------------------------------------------------------------------+
| /    | - division                                                                         |
|      | - spectrum1 / number = spectrum2 (spectrum2[k] = spectrum1[k] / number)            |
|      | - spectrum1 / spectrum2 = spectrum3 (spectrum3[k] = spectrum1[k] / spectrum2[k])   |
|      | - spectrum / cube1 = cube2 (cube2[k,j,i] = spectrum[k] / cube1[k,j,i])             |
+------+------------------------------------------------------------------------------------+	  
| \*\* | Computes the power exponent of data extensions                                     |
+------+------------------------------------------------------------------------------------+


Reference
=========


:func:`mpdaf.obj.Spectrum.copy` copies the Spectrum object in a new one and returns it.

:func:`mpdaf.obj.Spectrum.info` prints information.

:func:`mpdaf.obj.Spectrum.write` saves the Spectrum object in a FITS file.

:func:`mpdaf.obj.Spectrum.mean` computes the mean flux value over a wavelength range.

:func:`mpdaf.obj.Spectrum.sum` computes the total flux value over a wavelength range.



Getters and setters
-------------------

:func:`mpdaf.obj.Spectrum.get_lambda` returns the flux value corresponding to a wavelength, or returns the sub-spectrum corresponding to a wavelength range.
 
:func:`mpdaf.obj.Spectrum.get_step` returns the wavelength step.
 
:func:`mpdaf.obj.Spectrum.get_start` returns the wavelength value of the first pixel.

:func:`mpdaf.obj.Spectrum.get_end` returns the wavelength value of the last pixel.

:func:`mpdaf.obj.Spectrum.get_range` returns the wavelength range [Lambda_min,Lambda_max]

:func:`mpdaf.obj.Spectrum.set_wcs` sets the world coordinates.

:func:`mpdaf.obj.Spectrum.set_var` sets the variance array.


Mask
----

:func:`mpdaf.obj.Spectrum.mask` masks the spectrum.

:func:`mpdaf.obj.Spectrum.unmask` unmasks the spectrum.

:func:`mpdaf.obj.Spectrum.mask_variance` masks pixels with a variance upper than threshold value.

:func:`mpdaf.obj.Spectrum.interp_mask` interpolates masked pixels.



Transformation
--------------

:func:`mpdaf.obj.Spectrum.resize` resizes the spectrum to have a minimum number of masked values.

:func:`mpdaf.obj.Spectrum.sqrt` computes the positive square-root of data extension.

:func:`mpdaf.obj.Spectrum.abs` computes the absolute value of data extension.

:func:`mpdaf.obj.Spectrum.rebin_factor` shrinks the size of the spectrum by factor.

:func:`mpdaf.obj.Spectrum.rebin` rebins spectrum to different wavelength step size.

:func:`mpdaf.obj.Spectrum.truncate` truncates a spectrum.

:func:`mpdaf.obj.Spectrum.median_filter` performs a median filter on the spectrum.

:func:`mpdaf.obj.Spectrum.convolve` convolves the spectrum with a other spectrum or an array.

:func:`mpdaf.obj.Spectrum.fftconvolve` convolves the spectrum with a other spectrum or an array using fft.

:func:`mpdaf.obj.Spectrum.correlate` cross-correlates the spectrum with a other spectrum or an array.

:func:`mpdaf.obj.Spectrum.fftconvolve_gauss` convolves the spectrum with a Gaussian using fft.



Fit
---

:func:`mpdaf.obj.Spectrum.poly_fit` returns polynomial fit on spectrum.
 
:func:`mpdaf.obj.Spectrum.poly_val` performs polynomial fit on spectrum.

:func:`mpdaf.obj.Spectrum.poly_spec` performs polynomial fit on spectrum.

:func:`mpdaf.obj.Spectrum.fwhm` returns the fwhm of a peak.

:func:`mpdaf.obj.Spectrum.gauss_fit` performs polynomial fit on spectrum.

:func:`mpdaf.obj.Spectrum.add_gaussian` adds a gaussian on spectrum.


Filter
------

:func:`mpdaf.obj.Spectrum.abmag_band` computes AB magnitude corresponding to the wavelength band.

:func:`mpdaf.obj.Spectrum.abmag_filter_name` computes AB magnitude using the filter name.

:func:`mpdaf.obj.Spectrum.abmag_filter` computes AB magnitude using array filter.


Plotting
--------

:func:`mpdaf.obj.Spectrum.plot` plots the spectrum.

:func:`mpdaf.obj.Spectrum.log_plot` plots the spectrum with y logarithmic scale.

:func:`mpdaf.obj.Spectrum.ipos` prints cursor position in interactive mode.

:func:`mpdaf.obj.Spectrum.idist` gets distance and center from 2 cursor positions (interactive mode).

:func:`mpdaf.obj.Spectrum.imask` over-plots masked values (interactive mode).

:func:`mpdaf.obj.Spectrum.igauss_fit` performs an plots a polynomial fit on spectrum.
  
        
  
