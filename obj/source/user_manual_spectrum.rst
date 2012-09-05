Spectrum object
***************

The Spectrum object handles a 1D data array (basically a numpy masked array) containing flux values, associated with a WCS 
object (WaveCoord) containing the wavelength information. Optionally, a variance data array 
can be attached and used for weighting the flux values. Array masking is used to ignore 
some of the pixel values in the calculation.

Spectrum object format
======================

A Spectrum object O consists of:

+------------+---------------------------------------------------------+
| Component  | Description                                             |
+============+=========================================================+
| O.wave     | world coordinate spectral information (WaveCoord object)|
+------------+---------------------------------------------------------+
| O.data     | masked numpy 1D array with data values                  |
+------------+---------------------------------------------------------+
| O.var      | (optionally) masked numpy 1D array with variance values |
+------------+---------------------------------------------------------+


Tutorial
========

Preliminary imports for all tutorials::

  import numpy as np
  from mpdaf.obj import Spectrum
  from mpdaf.obj import WaveCoord

Tutorial 1: Spectrum Creation
-----------------------------

A Spectrum object is created: 

- either from one or two numpy data arrays (containing flux values and variance), 
using the following command::

  MyData=np.ones(4000) # numpy data array
  MyVariance=np.ones(4000) # numpy variance array
  spe = Spectrum(data=MyData) # spectrum filled with MyData 
  spe = Spectrum(data=MyData,variance=MyVariance) # spectrum filled with MyData and MyVariance

- or from a FITS file (in which case the flux and variance data are read from specific extensions), 
using the following commands::

  spe = Spectrum(filename="spectrum.fits",ext=1) # data array read from file (extension number 1)
  spe = Spectrum(filename="spectrum.fits",ext=[1,2]) # data and variance arrays read from file (extension numbers 1 and 2)

The WaveCoord object is either created using a linear scale, copied from another Spectrum, or 
using the information from the FITS header::

  wave1 = WaveCoord(crval=4000.0, cdelt=1.25, cunit='Angstrom')
  wave2 = spe.wave

  spe2=Spectrum(data=MyData,wave=wave1)


Tutorial 2: Spectrum masking and interpolating
----------------------------------------------

Here we describe how we can mask noisy parts in a spectrum, and do a polynomial 
interpolation taking into account the variance.

We start from the original spectrum and its variance::
  spvar=Spectrum('Spectrum_Variance.fits',ext=[0,1])
  
We mask the residuals from the strong sky emission line arround 5577 Angstroms::

  spvar.mask(5575,5580)

We select (in wavelengths) the clean spectrum region we want to interpolate::

  spvarcut=spvar.get_lambda(4000,6250)

We can then choose to perform a linear interpolation of the masked values::

  spvarcut.interp_mask()

The other option is to perform an interpolation with a spline::

  spvarcut.interp_mask(spline=True)
  

The results of the interpolations are shown below:

.. insert image here::


Tutorial 3: Gaussian Line fitting
---------------------------------



Indexing
--------

``Spectrum[k]`` returns the corresponding value of pixel k.

``Spectrum[k1:k2]`` returns the sub-spectrum between pixels k1 and k2

``Spectrum[k] = value`` sets the value of Spectrum.data[k]

``Spectrum[k1:k2] = array`` sets the values in the corresponding part of Spectrum.data.


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
|      | - spectrum + cube1 = cube2 (cube2[k,p,q] = cube1[k,p,q] + spectrum[k])             |
+------+------------------------------------------------------------------------------------+	  
| \-   | - substraction                                                                     |
|      | - spectrum1 - number = spectrum2 (spectrum2[k] = spectrum1[k] - number)            |
|      | - spectrum1 - spectrum2 = spectrum3 (spectrum3[k] = spectrum1[k] - spectrum2[k])   |
|      | - spectrum - cube1 = cube2 (cube2[k,p,q] = spectrum[k] - cube1[k,p,q])             |
+------+------------------------------------------------------------------------------------+
| \*   | - multiplication                                                                   |
|      | - spectrum1 \* number = spectrum2 (spectrum2[k] = spectrum1[k] \* number)          |
|      | - spectrum1 \* spectrum2 = spectrum3 (spectrum3[k] = spectrum1[k] \* spectrum2[k]) |
|      | - spectrum \* cube1 = cube2 (cube2[k,p,q] = spectrum[k] \* cube1[k,p,q])           |
|      | - spectrum \* image = cube (cube[k,p,q]=image[p,q] \* spectrum[k]                  |
+------+------------------------------------------------------------------------------------+
| /    | - division                                                                         |
|      | - spectrum1 / number = spectrum2 (spectrum2[k] = spectrum1[k] / number)            |
|      | - spectrum1 / spectrum2 = spectrum3 (spectrum3[k] = spectrum1[k] / spectrum2[k])   |
|      | - spectrum / cube1 = cube2 (cube2[k,p,q] = spectrum[k] / cube1[k,p,q])             |
+------+------------------------------------------------------------------------------------+	  
| \*\* | Computes the power exponent of data extensions                                     |
+------+------------------------------------------------------------------------------------+


Reference
=========


:func:`mpdaf.obj.Spectrum.copy` returns a new copy of a Spectrum object.

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

:func:`mpdaf.obj.Spectrum.mask` masks the spectrum (in place).

:func:`mpdaf.obj.Spectrum.unmask` unmasks the spectrum (in place).

:func:`mpdaf.obj.Spectrum.mask_variance` masks pixels with a variance upper than threshold value (in place).

:func:`mpdaf.obj.Spectrum.interp_mask` interpolates masked pixels (in place).



Transformation
--------------

:func:`mpdaf.obj.Spectrum.resize` resizes the spectrum to have a minimum number of masked values (in place).

:func:`mpdaf.obj.Spectrum.sqrt` computes the positive square-root of data extension.

:func:`mpdaf.obj.Spectrum.abs` computes the absolute value of data extension.

:func:`mpdaf.obj.Spectrum.rebin_factor` shrinks the size of the spectrum by factor.

:func:`mpdaf.obj.Spectrum.rebin` rebins spectrum to different wavelength step size.

:func:`mpdaf.obj.Spectrum.truncate` truncates a spectrum (in place).

:func:`mpdaf.obj.Spectrum.median_filter` performs a median filter on the spectrum.

:func:`mpdaf.obj.Spectrum.convolve` convolves the spectrum with a other spectrum or an array.

:func:`mpdaf.obj.Spectrum.fftconvolve` convolves the spectrum with a other spectrum or an array using fft.

:func:`mpdaf.obj.Spectrum.correlate` cross-correlates the spectrum with a other spectrum or an array.

:func:`mpdaf.obj.Spectrum.fftconvolve_gauss` convolves the spectrum with a Gaussian using fft.



Fit
---

:func:`mpdaf.obj.Spectrum.poly_fit` returns coefficients of the polynomial fit on spectrum.
 
:func:`mpdaf.obj.Spectrum.poly_val` updates in place the spectrum data from polynomial fit coefficients.

:func:`mpdaf.obj.Spectrum.poly_spec` performs polynomial fit on spectrum.

:func:`mpdaf.obj.Spectrum.fwhm` returns the fwhm of a peak.

:func:`mpdaf.obj.Spectrum.gauss_fit` performs Gaussian fit on spectrum.

:func:`mpdaf.obj.Spectrum.add_gaussian` adds a Gaussian on spectrum (in place).


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

:func:`mpdaf.obj.Spectrum.igauss_fit` performs and plots a Gaussian fit on spectrum.
  
        
  
