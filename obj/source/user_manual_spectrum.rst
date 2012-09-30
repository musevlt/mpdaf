Spectrum object
***************

The Spectrum object handles a 1D data array (basically a numpy masked array) containing flux values, associated with a :class:`WaveCoord <mpdaf.obj.WaveCoord>` object containing the wavelength information. Optionally, a variance data array 
can be attached and used for weighting the flux values. Array masking is used to ignore 
some of the pixel values in the calculations.

Note that virtually all numpy and scipy functions are available.

Spectrum object format
======================

A Spectrum object O consists of:

+------------+----------------------------------------------------------------------------------------+
| Component  | Description                                                                            |
+============+========================================================================================+
| O.wave     | world coordinate spectral information (:class:`WaveCoord <mpdaf.obj.WaveCoord>` object)|
+------------+----------------------------------------------------------------------------------------+
| O.data     | masked numpy 1D array with data values                                                 |
+------------+----------------------------------------------------------------------------------------+
| O.var      | (optionally) masked numpy 1D array with variance values                                |
+------------+----------------------------------------------------------------------------------------+
| O.shape    | number of elements in O.data and O.var (int)                                           |
+------------+----------------------------------------------------------------------------------------+
| O.fscale   | Scaling factor for the flux and variance values                                          |
+------------+----------------------------------------------------------------------------------------+


Tutorial
========

We can load the tutorial files with the command::

 > git clone http://urania1.univ-lyon1.fr/git/mpdaf_data.git

Preliminary imports for all tutorials::

  >>> import numpy as np
  >>> from mpdaf.obj import Spectrum
  >>> from mpdaf.obj.coords import WaveCoord

Tutorial 1: Spectrum Creation
-----------------------------

A Spectrum object is created: 

- either from one or two numpy data arrays (containing flux values and variance), using the following command::

  >>> MyData=np.ones(4000) # numpy data array
  >>> MyVariance=np.ones(4000) # numpy variance array
  >>> spe = Spectrum(data=MyData) # spectrum filled with MyData 
  >>> spe = Spectrum(data=MyData,var=MyVariance) # spectrum filled with MyData and MyVariance

- or from a FITS file (in which case the flux and variance values are read from specific extensions), using the following commands::

  >>> spe = Spectrum(filename="spectrum.fits",ext=1) # data array is read from the file (extension number 1)
  >>> spe = Spectrum(filename="spectrum.fits",ext=[1,2]) # data and variance arrays read from the file (extension numbers 1 and 2)

If the FITS file contains a single extension (spectrum fluxes), or when the FITS extension are specifically named 'DATA' (for flux values) and 'STAT' (for variance  values), the keyword "ext=" is unnecessary.

The :class:`WaveCoord <mpdaf.obj.WaveCoord>` object is either created using a linear scale, copied from another Spectrum, or 
using the information from the FITS header::

  >>> wave1 = WaveCoord(crval=4000.0, cdelt=1.25, cunit='Angstrom')
  >>> wave2 = spe.wave

  >>> spe2=Spectrum(data=MyData,wave=wave1)

In the first case, the wavelength solution is linear with the array index k: the first array value (k=0) corresponds to a wavelength of 4000 Angstroms, and the next array values (k=1,2 ...) are spaced by 1.25 Angstroms.


Tutorial 2: Spectrum manipulation: masking, interpolating, rebinning
--------------------------------------------------------------------

Here we describe how we can mask noisy parts in a spectrum, and do a polynomial 
interpolation taking into account the variance.

We start from the original spectrum and its variance::

  >>> spvar=Spectrum('Spectrum_Variance.fits',ext=[0,1])
  
We mask the residuals from the strong sky emission line arround 5577 Angstroms::

  >>> spvar.mask(5575,5590)

We select (in wavelengths) the clean spectrum region we want to interpolate::

  >>> spvarcut=spvar.get_lambda(4000,6250)

We can then choose to perform a linear interpolation of the masked values::

  >>> spvarcut.interp_mask()

The other option is to perform an interpolation with a spline::

  >>> spvarcut.interp_mask(spline=True)
  

The results of the interpolations are shown below::

  >>> spvar.unmask()
  >>> spvar.plot(lmin=4600, lmax=6200, title='Spectrum before interpolation')
  >>> spvarcut.plot(lmin=4600, lmax=6200, title='Spectrum after interpolation')
  
  
.. image:: user_manual_spectrum_images/Spectrum_before_interp_mask.png

.. image:: user_manual_spectrum_images/Spectrum_after_interp_mask.png 


Last, we will rebin the extracted spectrum using the 2 dedicated functions (rebin_factor and rebin). 
The function :func:`rebin_factor <mpdaf.obj.Spectrum.rebin_factor>` rebins the Spectrum using an integer number of pixels per bin. The corresponding variance is updated accordingly. We can overplot the rebinned Spectrum and show the corresponding variance as follows::

  >>> sprebin1=spvarcut.rebin_factor(5)
  >>> spvarcut.plot()
  >>> (sprebin1+10).plot(noise=True)

.. figure:: user_manual_spectrum_images/Spectrum_rebin.png
  :align:   center

The function :func:`rebin <mpdaf.obj.Spectrum.rebin>` rebins the Spectrum 
with a specific numbers of wavelength units per pixel. The Variance is not 
updated::

  >>> sprebin2=spvarcut.rebin(4.2) # 4.2 Angstroms / pixel
  >>> spvarcut.plot()
  >>> (sprebin2+10).plot(noise=True)

.. figure:: user_manual_spectrum_images/Spectrum_rebin2.png
  :align:   center

Tutorial 3: Gaussian Line fitting
---------------------------------

We want to fit the emission lines in a z=0.6758 galaxy (Hbeta and [OIII]).
We open the spectrum and associated variance::

  >>> specline=Spectrum('Spectrum_lines.fits')

We plot the spectrum around the [OIII] line::

  >>> specline.plot(lmin=8350,lmax=8420)

We do an interactive line fitting on the plot, by selecting with the mouse the left and right 
continuum (2 positions) and the peak of the line. Variance weighting is used in the fit::

  >>> specline.igauss_fit()
  Use the 2 first mouse clicks to get the wavelength range to compute the gaussian left value.
  Use the next click to get the peak wavelength.
  Use the 2 last mouse clicks to get the wavelength range to compute the gaussian rigth value.
  To quit the interactive mode, click on the right mouse button.
  The parameters of the last gaussian are saved in self.gauss.

The result of the fit is overploted in red:

.. figure:: user_manual_spectrum_images/specline1.png
  :align:   center

  Interactive Gaussian line fitting result

and the result is given on the console::

  Gaussian center = 8390.53 (error:0.209096)
  Gaussian integrated flux = 650.329 (error:68.2009)
  Gaussian peak value = 150.279 (error:2.43122)
  Gaussian fwhm = 4.06538 (error:0.492112)
  Gaussian continuum = 3.27427


Now, we move to the fainter line (Hbeta) and we perform the same analysis, again using variance weighting::

  >>> specline.plot(lmin=8090,lmax=8210)
  >>> specline.igauss_fit()

The result of the fit is given below:

.. figure:: user_manual_spectrum_images/specline2.png
  :align:   center

  Interactive Gaussian line fitting on a faint line


The results from the fit can be retrieved in the :class:`Gauss1D <mpdaf.obj.Gauss1D>` object associated 
with the spectrum (self.gauss). For example we can measure the equivalent width of the line like this::

  >>> specline.gauss.flux/specline.gauss.cont
  198.618

Reference
=========

:func:`mpdaf.obj.Spectrum <mpdaf.obj.Spectrum>` is the Spectrum constructor.

:func:`mpdaf.obj.Spectrum.copy <mpdaf.obj.Spectrum.copy>` returns a new copy of a Spectrum object.

:func:`mpdaf.obj.Spectrum.clone <mpdaf.obj.Spectrum.clone>` returns a new spectrum of the same shape and coordinates, filled with zeros.

:func:`mpdaf.obj.Spectrum.info <mpdaf.obj.Spectrum.info>` prints information.

:func:`mpdaf.obj.Spectrum.write <mpdaf.obj.Spectrum.write>` saves the Spectrum object in a FITS file.

:func:`mpdaf.obj.Spectrum.peak_detection <mpdaf.obj.Spectrum.peak_detection>` returns a list of peak locations.


Indexing
--------

:func:`Spectrum[k] <mpdaf.obj.Spectrum.__getitem__>` returns the corresponding value of pixel k.

:func:`Spectrum[k1:k2] <mpdaf.obj.Spectrum.__getitem__>` returns the sub-spectrum between pixels k1 and k2

:func:`Spectrum[k] = value <mpdaf.obj.Spectrum.__setitem__>` sets the value of Spectrum.data[k]

:func:`Spectrum[k1:k2] = array <mpdaf.obj.Spectrum.__setitem__>` sets the values in the corresponding part of Spectrum.data.


Getters and setters
-------------------

:func:`mpdaf.obj.Spectrum.get_lambda <mpdaf.obj.Spectrum.get_lambda>` returns the flux value corresponding to a wavelength, or returns the sub-spectrum corresponding to a wavelength range.
 
:func:`mpdaf.obj.Spectrum.get_step <mpdaf.obj.Spectrum.get_step>` returns the wavelength step.
 
:func:`mpdaf.obj.Spectrum.get_start <mpdaf.obj.Spectrum.get_start>` returns the wavelength value of the first pixel.

:func:`mpdaf.obj.Spectrum.get_end <mpdaf.obj.Spectrum.get_end>` returns the wavelength value of the last pixel.

:func:`mpdaf.obj.Spectrum.get_range <mpdaf.obj.Spectrum.get_range>` returns the wavelength range [Lambda_min,Lambda_max]

:func:`mpdaf.obj.Spectrum.set_wcs <mpdaf.obj.Spectrum.set_wcs>` sets the world coordinates.

:func:`mpdaf.obj.Spectrum.set_var <mpdaf.obj.Spectrum.set_var>` sets the variance array.


Mask
----

:func:`<= <mpdaf.obj.Spectrum.__le__>` masks data array where greater than a given value.                                 

:func:`< <mpdaf.obj.Spectrum.__lt__>` masks data array where greater or equal than a given value. 

:func:`>= <mpdaf.obj.Spectrum.__ge__>` masks data array where less than a given value.

:func:`> <mpdaf.obj.Spectrum.__gt__>` masks data array where less or equal than a given value.  

:func:`mpdaf.obj.Spectrum.mask <mpdaf.obj.Spectrum.mask>` masks the spectrum (in place).

:func:`mpdaf.obj.Spectrum.unmask <mpdaf.obj.Spectrum.unmask>` unmasks the spectrum (in place).

:func:`mpdaf.obj.Spectrum.mask_variance <mpdaf.obj.Spectrum.mask_variance>` masks pixels with a variance upper than threshold value (in place).

:func:`mpdaf.obj.Spectrum.interp_mask <mpdaf.obj.Spectrum.interp_mask>` interpolates masked pixels (in place).

:func:`mpdaf.obj.Spectrum.mask_selection <mpdaf.obj.Spectrum.mask_selection>` masks pixels corresponding to a selection.


Arithmetic
----------

:func:`\+ <mpdaf.obj.Spectrum.__add__>` makes a addition.

:func:`\- <mpdaf.obj.Spectrum.__sub__>` makes a substraction .

:func:`\* <mpdaf.obj.Spectrum.__mul__>` makes a multiplication.

:func:`/ <mpdaf.obj.Spectrum.__div__>` makes a division.

:func:`\*\* <mpdaf.obj.Spectrum.__pow__>`  computes the power exponent of data extensions.

:func:`mpdaf.obj.Spectrum.mean <mpdaf.obj.Spectrum.mean>` computes the mean flux value over a wavelength range.

:func:`mpdaf.obj.Spectrum.sum <mpdaf.obj.Spectrum.sum>` computes the total flux value over a wavelength range.

:func:`mpdaf.obj.Spectrum.sqrt <mpdaf.obj.Spectrum.sqrt>` computes the positive square-root of data extension.

:func:`mpdaf.obj.Spectrum.abs <mpdaf.obj.Spectrum.abs>` computes the absolute value of data extension.



Transformation
--------------

:func:`mpdaf.obj.Spectrum.resize <mpdaf.obj.Spectrum.resize>` resizes the spectrum to have a minimum number of masked values (in place).

:func:`mpdaf.obj.Spectrum.rebin_factor <mpdaf.obj.Spectrum.rebin_factor>` shrinks the size of the spectrum by factor.

:func:`mpdaf.obj.Spectrum.rebin <mpdaf.obj.Spectrum.rebin>` rebins spectrum to different wavelength step size.

:func:`mpdaf.obj.Spectrum.truncate <mpdaf.obj.Spectrum.truncate>` truncates a spectrum (in place).

:func:`mpdaf.obj.Spectrum.median_filter <mpdaf.obj.Spectrum.median_filter>` performs a median filter on the spectrum.

:func:`mpdaf.obj.Spectrum.convolve <mpdaf.obj.Spectrum.convolve>` convolves the spectrum with a other spectrum or an array.

:func:`mpdaf.obj.Spectrum.fftconvolve <mpdaf.obj.Spectrum.fftconvolve>` convolves the spectrum with a other spectrum or an array using fft.

:func:`mpdaf.obj.Spectrum.correlate <mpdaf.obj.Spectrum.correlate>` cross-correlates the spectrum with a other spectrum or an array.

:func:`mpdaf.obj.Spectrum.fftconvolve_gauss <mpdaf.obj.Spectrum.fftconvolve_gauss>` convolves the spectrum with a Gaussian using fft.



Fit
---

:func:`mpdaf.obj.Spectrum.poly_fit <mpdaf.obj.Spectrum.poly_fit>` returns coefficients of the polynomial fit on spectrum.
 
:func:`mpdaf.obj.Spectrum.poly_val <mpdaf.obj.Spectrum.poly_val>` updates in place the spectrum data from polynomial fit coefficients.

:func:`mpdaf.obj.Spectrum.poly_spec <mpdaf.obj.Spectrum.poly_spec>` performs polynomial fit on spectrum.

:func:`mpdaf.obj.Spectrum.fwhm <mpdaf.obj.Spectrum.fwhm>` returns the fwhm of a peak.

:func:`mpdaf.obj.Spectrum.gauss_fit <mpdaf.obj.Spectrum.gauss_fit>` performs Gaussian fit on spectrum.

:func:`mpdaf.obj.Spectrum.add_gaussian <mpdaf.obj.Spectrum.add_gaussian>` adds a Gaussian on spectrum (in place).


Photometry
----------

:func:`mpdaf.obj.Spectrum.abmag_band <mpdaf.obj.Spectrum.abmag_band>` computes AB magnitude corresponding to the wavelength band.

:func:`mpdaf.obj.Spectrum.abmag_filter_name <mpdaf.obj.Spectrum.abmag_filter_name>` computes AB magnitude using the filter name.

:func:`mpdaf.obj.Spectrum.abmag_filter <mpdaf.obj.Spectrum.abmag_filter>` computes AB magnitude using array filter.


Plotting
--------

:func:`mpdaf.obj.Spectrum.plot <mpdaf.obj.Spectrum.plot>` plots the spectrum.

:func:`mpdaf.obj.Spectrum.log_plot <mpdaf.obj.Spectrum.log_plot>` plots the spectrum with a logarithmic scale on the y-axis.

:func:`mpdaf.obj.Spectrum.ipos <mpdaf.obj.Spectrum.ipos>` prints cursor position in interactive mode.

:func:`mpdaf.obj.Spectrum.idist <mpdaf.obj.Spectrum.idist>` gets distance and center from 2 cursor positions (interactive mode).

:func:`mpdaf.obj.Spectrum.imask <mpdaf.obj.Spectrum.imask>` over-plots masked values (interactive mode).

:func:`mpdaf.obj.Spectrum.igauss_fit <mpdaf.obj.Spectrum.igauss_fit>` performs and plots a Gaussian fit on spectrum.
  
        
  
