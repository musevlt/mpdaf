.. _spectrum:


***************
Spectrum object
***************

Spectrum objects contain a 1D data array of flux values, and a `WaveCoord
<mpdaf.obj.WaveCoord>` object that describes the wavelength scale of the
spectrum. Optionally, an array of variances can also be provided to give the
statistical uncertainties of the fluxes. These can be used for weighting the
flux values and for computing the uncertainties of least-squares fits and other
calculations. Finally a mask array is provided for indicating bad pixels.

The fluxes and their variances are stored in numpy masked arrays, so
virtually all numpy and scipy functions can be applied to them.

Preliminary imports:

.. ipython::

   In [1]: import numpy as np

   In [1]: import matplotlib.pyplot as plt

   In [1]: import astropy.units as u

   In [2]: from mpdaf.obj import Spectrum, WaveCoord


Spectrum Creation
=================

There are two common ways to obtain a `Spectrum <mpdaf.obj.Spectrum>` object:

- A spectrum can be created from a user-provided array of the flux values at
  each wavelength of the spectrum, or from both an array of flux values and a
  corresponding array of variances. These arrays can be simple numpy arrays, or
  they can be numpy masked arrays in which bad pixels have been masked.

.. ipython::

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


- Alternatively, a spectrum can be read from a FITS file. In this case the flux
  and variance values are read from specific extensions:

.. ipython::
  :okwarning:

  # data array is read from the file (extension number 0)
  In [4]: spe = Spectrum(filename='obj/Spectrum_Variance.fits', ext=0)

  In [10]: spe.info()

  # data and variance arrays read from the file (extension numbers 1 and 2)
  In [4]: spe = Spectrum(filename='obj/Spectrum_Variance.fits', ext=[0, 1])

  In [10]: spe.info()

By default, if a FITS file has more than one extension, then it is expected to
have a 'DATA' extension that contains the pixel data, and possibly a 'STAT'
extension that contains the corresponding variances. If the file doesn't contain
extensions of these names, the "ext=" keyword can be used to indicate the
appropriate extension or extensions, as shown in the example above.

The `WaveCoord <mpdaf.obj.WaveCoord>` object of a spectrum describes the
wavelength scale of the spectrum. When a spectrum is read from a FITS file, this
is automatically generated based on FITS header keywords. Alternatively, when a
spectrum is extracted from a cube or another spectrum, the wavelength object is
derived from the wavelength object of the original object. In the first example
on this page, the wavelength scale of the spectrum increases linearly with array
index, k. The wavelength of the first pixel (k=0) is 4000 Angstrom, and the
subsequent pixels (k=1,2 ...) are spaced by 1.25 Angstroms.

Information about a spectrum can be printed using the `info
<mpdaf.obj.Spectrum.info>` method.

Spectrum objects also have a `plot <mpdaf.obj.Spectrum.plot>` method, which is
based on `matplotlib.pyplot.plot <http://matplotlib.org/api/pyplot_api.html>`_
and accepts all matplotlib arguments:

.. ipython::

   In [4]: plt.figure()

   @savefig Spectrum.png width=4in
   In [5]: spe.plot(color='g')

This spectrum could also be plotted with a logarithmic scale on the y-axis
(by using `log_plot <mpdaf.obj.Spectrum.log_plot>` in place of `plot <mpdaf.obj.Spectrum.plot>`).


Spectrum masking and interpolation
==================================

This section demonstrates how one can mask a sky line in a spectrum, and
replace it with a linear or spline interpolation over the resulting gap.

The original spectrum and its variance is first loaded:

.. ipython::
  :okwarning:

  In [5]: spvar = Spectrum('obj/Spectrum_Variance.fits',ext=[0,1])

Next the `mask_region <mpdaf.obj.Spectrum.mask_region>` method is used to mask a
strong sky emission line around 5577 Angstroms:

.. ipython::
  :okwarning:

  In [5]: spvar.mask_region(lmin=5575, lmax=5590, unit=u.angstrom)

Then the `~mpdaf.obj.Spectrum.subspec` method is used to select the sub-set of
the spectrum that we are interested in, including the masked region:

.. ipython::
  :okwarning:

  In [5]: spvarcut = spvar.subspec(lmin=4000, lmax=6250, unit=u.angstrom)

The `interp_mask <mpdaf.obj.Spectrum.interp_mask>` method can then be used to
replace the masked pixels with values that are interpolated from pixels on
either side of the masked region. By default, this method uses linear
interpolation:

.. ipython::

  In [5]: spvarcut.interp_mask()

However it can also be told to use a spline interpolation:

.. ipython::

  In [5]: spvarcut.interp_mask(spline=True)

The results of the interpolations are shown below:

.. ipython::

  In [5]: spvar.unmask()

  In [7]: plt.figure()

  @savefig Spectrum_before_interp_mask.png width=3.5in
  In [6]: spvar.plot(lmin=4600, lmax=6200, title='Spectrum before interpolation', unit=u.angstrom)

  In [7]: plt.figure()

  @savefig Spectrum_after_interp_mask.png width=3.5in
  In [6]: spvarcut.plot(lmin=4600, lmax=6200, title='Spectrum after interpolation', unit=u.angstrom)

Spectrum rebinning and resampling
=================================

Two methods are provided for resampling spectra.  The `rebin
<mpdaf.obj.Spectrum.rebin>` method reduces the resolution of a spectrum by
integer factors. If the integer factor is n, then the pixels of the new spectrum
are calculated from the mean of n neighboring pixels. If the spectrum has
variances, the variances of the averaged pixels are updated accordingly.

In the example below, the spectrum of the previous section is rebinned to reduce
its resolution by a factor of 5. In a plot of the original spectrum, the
rebinned spectrum is drawn vertically offset from it by 10. The grey areas above
and below the line of the rebinned spectrum indicate the standard deviation
computed from the rebinned variances. The standard deviations clearly don't
reflect the actual noise level, but this is because the variances in the FITS
file are incorrect.

.. ipython::
  :okwarning:

  In [5]: plt.figure()

  In [6]: sprebin1 = spvarcut.rebin(5)

  In [7]: spvarcut.plot()

  @savefig Spectrum_rebin.png width=4in
  In [8]: (sprebin1 + 10).plot(noise=True)

Whereas the rebin method is restricted to decreasing the resolution by integer
factors, the `resample <mpdaf.obj.Spectrum.resample>` method can resample a
Spectrum to any resolution. The desired pixel size is specified in wavelength
units. At the current time the variances are not updated, but this will be
remedied in the near future.

.. ipython::

  In [5]: plt.figure()

  In [5]: sp = spvarcut[1500:2000]

  # 4.2 Angstroms / pixel
  In [6]: sprebin2 = sp.resample(4.2, unit=u.angstrom)

  In [7]: sp.plot()

  @savefig Spectrum_rebin2.png width=4in
  In [8]: (sprebin2 + 10).plot(noise=True)


Continuum and line fitting
==========================

Line fitting
------------

In this section, the Hbeta and [OIII] emission lines of a z=0.6758 galaxy are
fitted. The spectrum and associated variances are first loaded:

.. ipython::
  :okwarning:

  In [1]: specline = Spectrum('obj/Spectrum_lines.fits')

The spectrum around the [OIII] line is then plotted:

.. ipython::

  In [2]: plt.figure()

  In [2]: specline.plot(lmin=8350, lmax=8420, unit=u.angstrom, title = '[OIII] line')

Next the `gauss_fit <mpdaf.obj.Spectrum.gauss_fit>` method is used to perform a
Gaussian fit to the section of the spectrum that contains the line. The fit is
automatically weighted by the variances of the spectrum:

.. ipython::

  @savefig Spectrum_specline1.png width=4in
  In [3]: OIII = specline.gauss_fit(lmin=8350, lmax=8420, unit=u.angstrom, plot=True)

  In [4]: OIII.print_param()

The result of the fit plotted in red over the spectrum.

Next a fit is performed to the fainter Hbeta line, again using the variances
to weight the least-squares Gaussian fit:

.. ipython::

  In [5]: plt.figure()

  In [6]: specline.plot(lmin=8090,lmax=8210, unit=u.angstrom, title = 'Hbeta line')

  @savefig Spectrum_specline2.png width=4in
  In [7]: Hbeta = specline.gauss_fit(lmin=8090,lmax=8210, unit=u.angstrom, plot=True)

  In [8]: Hbeta.print_param()


The results from the fit can be retrieved in the returned `Gauss1D
<mpdaf.obj.Gauss1D>` object. For example the equivalent width of the line can be
estimated as follows:

.. ipython::

  In [8]: Hbeta.flux/Hbeta.cont

If the wavelength of the line is already known, using `fix_lpeak=True` can
perform an better Gaussian fit on the line by fixing the Gaussian center:

.. ipython::

  In [5]: plt.figure()

  In [6]: specline.plot(lmin=8090,lmax=8210, unit=u.angstrom, title = 'Hbeta line')

  @savefig Spectrum_specline2.png width=4in
  In [7]: Hbeta2 = specline.gauss_fit(lmin=8090,lmax=8210, lpeak=Hbeta.lpeak, unit=u.angstrom, plot=True, fix_lpeak=True)

  In [8]: Hbeta2.print_param()


In the same way:
 - `gauss_dfit <mpdaf.obj.Spectrum.gauss_dfit>` performs a double Gaussian fit on spectrum.

 - `gauss_asymfit <mpdaf.obj.Spectrum.gauss_asymfit>` performs an asymetric Gaussian fit on spectrum.


Continuum fitting
-----------------

The `poly_spec <mpdaf.obj.Spectrum.poly_spec>` method performs a polynomial fit
to a spectrum. This can be used to fit the continuum:

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

In the plot, the polynomial fit to the continuum is the red line drawn over the
spectrum.
