.. _spectrum:


***************
Spectrum object
***************

The Spectrum object handles a 1D data array (basically a numpy masked array)
containing flux values, associated with a `WaveCoord <mpdaf.obj.WaveCoord>`
object containing the wavelength information. Optionally, a variance data array
can be attached and used for weighting the flux values. Array masking is used
to ignore some of the pixel values in the calculations.

Note that virtually all numpy and scipy functions are available.

.. ipython::
   :suppress:

   In [4]: import sys

   In [4]: from mpdaf import setup_logging

Preliminary imports:

.. ipython::

   In [1]: import numpy as np

   In [1]: import matplotlib.pyplot as plt

   In [1]: import astropy.units as u

   In [2]: from mpdaf.obj import Spectrum, WaveCoord


Spectrum Creation
=================

A `Spectrum <mpdaf.obj.Spectrum>` object is created:

- either from one or two numpy data arrays (containing flux values and variance):

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  In [6]: wave1 = WaveCoord(cdelt=1.25, crval=4000.0, cunit= u.angstrom)

  # numpy data array
  In [8]: MyData = np.ones(4000)

  # numpy variance array
  In [8]: MyVariance=np.ones(4000)

  # spectrum filled with MyData
  In [9]: spe = Spectrum(wave=wave1, data=MyData)

  In [10]: spe.info()

  # spectrum filled with MyData and MyVariance
  In [9]: spe = Spectrum(wave=wave1, data=MyData, var=MyVariance)

  In [10]: spe.info()


- or from a FITS file (in which case the flux and variance values are read from specific extensions):

.. ipython::
  :okwarning:

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  # data array is read from the file (extension number 0)
  In [4]: spe = Spectrum(filename='../data/obj/Spectrum_Variance.fits', ext=0)

  In [10]: spe.info()

  # data and variance arrays read from the file (extension numbers 1 and 2)
  In [4]: spe = Spectrum(filename='../data/obj/Spectrum_Variance.fits', ext=[0, 1])

  In [10]: spe.info()


If the FITS file contains a single extension (spectrum fluxes), or when the FITS extension are specifically named 'DATA' (for flux values) and 'STAT' (for variance  values), the keyword "ext=" is unnecessary.

The `WaveCoord <mpdaf.obj.WaveCoord>` object is either created using a linear scale, copied from another Spectrum, or
using the information from the FITS header. The wavelength solution is linear with the array index k: in the first example, the first array value (k=0) corresponds to a wavelength of 4000 Angstroms, and the next array values (k=1,2 ...) are spaced by 1.25 Angstroms.

Information are printed by using the `info <mpdaf.obj.Spectrum.info>` method.

The `plot <mpdaf.obj.Spectrum.plot>` method is based on `matplotlib.pyplot.plot <http://matplotlib.org/api/pyplot_api.html>`_ and accepts all matplotlib arguments:

.. ipython::

   In [4]: plt.figure()

   @savefig Spectrum.png width=4in
   In [5]: spe.plot(color='g')

The spectrum could also be plotted with a logarithmic scale on the y-axis
(by using `log_plot <mpdaf.obj.Spectrum.log_plot>` in place of `plot <mpdaf.obj.Spectrum.plot>`).


Spectrum manipulation: masking, interpolating, rebinning
========================================================

Here we describe how we can mask noisy parts in a spectrum, and do a polynomial
interpolation taking into account the variance.

We start from the original spectrum and its variance:

.. ipython::
  :okwarning:

  In [5]: spvar = Spectrum('../data/obj/Spectrum_Variance.fits',ext=[0,1])

By using the `mask_region <mpdaf.obj.Spectrum.mask_region>` method, we mask the residuals from the strong sky emission line around 5577 Angstroms:

.. ipython::

  In [5]: spvar.mask_region(lmin=5575, lmax=5590, unit=spvar.wave.unit)

We select (in wavelengths - `~mpdaf.obj.Spectrum.subspec` method) the clean spectrum region we want to interpolate:

.. ipython::

  In [5]: spvarcut = spvar.subspec(lmin=4000, lmax=6250, unit=spvar.wave.unit)

We can then choose to apply `interp_mask <mpdaf.obj.Spectrum.interp_mask>` and perform a linear interpolation of the masked values:

.. ipython::

  In [5]: spvarcut.interp_mask()

The other option is to perform an interpolation with a spline:

.. ipython::

  In [5]: spvarcut.interp_mask(spline=True)

The results of the interpolations are shown below:

.. ipython::

  In [5]: spvar.unmask()

  In [7]: plt.figure()

  @savefig Spectrum_before_interp_mask.png width=3.5in
  In [6]: spvar.plot(lmin=4600, lmax=6200, title='Spectrum before interpolation', unit=spvar.wave.unit)

  In [7]: plt.figure()

  @savefig Spectrum_after_interp_mask.png width=3.5in
  In [6]: spvarcut.plot(lmin=4600, lmax=6200, title='Spectrum after interpolation', unit=spvar.wave.unit)

Last, we will resample the extracted spectrum using the 2 dedicated functions
(rebin and resample).  The function `rebin
<mpdaf.obj.Spectrum.rebin>` rebins the Spectrum using an integer number of
pixels per bin. The corresponding variance is updated accordingly. We can
overplot the rebinned Spectrum and show the corresponding variance as follows:

.. ipython::
  :okwarning:

  In [5]: plt.figure()

  In [6]: sprebin1 = spvarcut.rebin(5)

  In [7]: spvarcut.plot()

  @savefig Spectrum_rebin.png width=4in
  In [8]: (sprebin1 + 10).plot(noise=True)

The function `resample <mpdaf.obj.Spectrum.resample>` resamples the Spectrum
with a specific numbers of wavelength units per pixel. The variance is not
updated:

.. ipython::

  In [5]: plt.figure()

  In [5]: sp = spvarcut[1500:2000]

  # 4.2 Angstroms / pixel
  In [6]: sprebin2 = sp.resample(4.2, unit=sp.wave.unit)

  In [7]: sp.plot()

  @savefig Spectrum_rebin2.png width=4in
  In [8]: (sprebin2 + 10).plot(noise=True)


Continuum and line fitting
==========================

Line fitting
------------

We want to fit the emission lines in a z=0.6758 galaxy (Hbeta and [OIII]).
We open the spectrum and associated variance:

.. ipython::
  :okwarning:

  In [1]: specline = Spectrum('../data/obj/Spectrum_lines.fits')

We plot the spectrum around the [OIII] line:

.. ipython::

  In [2]: plt.figure()

  In [2]: specline.plot(lmin=8350, lmax=8420, unit=specline.wave.unit, title = '[OIII] line')

`gauss_fit <mpdaf.obj.Spectrum.gauss_fit>` performs a Gaussian fit on spectrum. Variance weighting is used in the fit:

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  @savefig Spectrum_specline1.png width=4in
  In [3]: OIII = specline.gauss_fit(lmin=8350, lmax=8420, unit=specline.wave.unit, plot=True)

  In [4]: OIII.print_param()

The result of the fit is overploted in red.

Now, we move to the fainter line (Hbeta) and we perform the same analysis, again using variance weighting:

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  In [5]: plt.figure()

  In [6]: specline.plot(lmin=8090,lmax=8210, unit=specline.wave.unit, title = 'Hbeta line')

  @savefig Spectrum_specline2.png width=4in
  In [7]: Hbeta = specline.gauss_fit(lmin=8090,lmax=8210, unit=specline.wave.unit, plot=True)

  In [8]: Hbeta.print_param()


The results from the fit can be retrieved in the returned `Gauss1D <mpdaf.obj.Gauss1D>` object. For example we can measure the equivalent width of the line like this:

.. ipython::

  In [8]: Hbeta.flux/Hbeta.cont

If the wavelength of the line is already known, `line_gauss_fit <mpdaf.obj.Spectrum.line_gauss_fit>` could perform an better Gaussian fit on the line by fixing the Gaussian center:

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  In [5]: plt.figure()

  In [6]: specline.plot(lmin=8090,lmax=8210, unit=specline.wave.unit, title = 'Hbeta line')

  @savefig Spectrum_specline2.png width=4in
  In [7]: Hbeta2 = specline.line_gauss_fit(lmin=8090,lmax=8210, lpeak=Hbeta.lpeak, unit=specline.wave.unit, plot=True)

  In [8]: Hbeta2.print_param()


In the same way:
 - `gauss_dfit <mpdaf.obj.Spectrum.gauss_dfit>` performs a double Gaussian fit on spectrum.

 - `gauss_asymfit <mpdaf.obj.Spectrum.gauss_asymfit>` performs an asymetric Gaussian fit on spectrum.


Continuum fitting
-----------------

`poly_spec <mpdaf.obj.Spectrum.poly_spec>` performs a polynomial fit on spectrum and it can be used to fit the continuum:

.. ipython::

  In [1]: plt.figure()

  In [2]: cont = spe.poly_spec(5)

  In [3]: spe.plot()

  @savefig Spectrum_cont.png width=4in
  In [4]: cont.plot(color='r')


.. ipython::
   :suppress:

   In [4]: plt.close("all")

   In [4]: %reset -f
