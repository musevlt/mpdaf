***********
Rearranging
***********

`crop <mpdaf.obj.DataArray.crop>` reduce the size of the array to the smallest sub-array that keeps all unmasked pixels:

.. ipython::
   :okwarning:

   In [1]: import numpy as np

   In [1]: import matplotlib.pyplot as plt

   In [2]: from mpdaf.obj import Image

   In [3]: ima = Image('../data/obj/a370II.fits')

   In [2]: center=[-1.5642, 39.9620]

   In [2]: ima.mask_ellipse(center=center, radius=(110, 60), posangle=40, inside=False)

   In [2]: ima.shape

   In [3]: plt.figure()

   @savefig Obj_transfo1.png width=3.5in
   In [3]: ima.plot()

   In [4]: ima.crop()

   In [2]: ima.shape

   In [3]: plt.figure()

   @savefig Obj_transfo2.png width=3.5in
   In [3]: ima.plot()


`Spectrum.truncate <mpdaf.obj.Spectrum.truncate>`, `Image.truncate <mpdaf.obj.Image.truncate>` and
`Cube.truncate <mpdaf.obj.Cube.truncate>` return a sub-object bounded by specified wavelength or/and spatial world-coordinates:

For example, we can plot a specified area of the sky:

.. ipython::
   :okwarning:

   In [43]: ymin, xmin = np.array(center) - 50./3600

   In [44]: ymax, xmax = np.array(center) + 50./3600

   In [45]: ima2 = ima.truncate(ymin, xmin, ymax, xmax)

   In [44]: plt.figure()

   @savefig Obj_transfo3.png width=4in
   In [1]: ima2.plot()

   In [48]: ima2.get_rot()

   @suppress
   In [5]: ima = None ; ima2 = None


The ranges x_min to x_max and y_min to y_max, specify a rectangular region of the sky in world coordinates. The
truncate function returns the sub-image that just encloses this region. In this case, the world coordinate axes are not
parallel to the array axes, the region appears to be a rotated rectangle within the sub-image.

The methods `Spectrum.rebin <mpdaf.obj.Spectrum.rebin>`, `Image.rebin <mpdaf.obj.Image.rebin>` and `Cube.rebin <mpdaf.obj.Cube.rebin>`
return a object that shrinks the size of the current object by an integer division factor:

.. ipython::
   :suppress:

   In [4]: import sys

   In [4]: from mpdaf import setup_logging

.. ipython::
  :okwarning:

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  In [3]: ima = Image('../data/obj/a370II.fits')

  In [8]: ima.info()

  In [3]: plt.figure()

  @savefig Obj_transfo4.png width=3.5in
  In [3]: ima.plot(zscale=True)

  In [4]: ima2 = ima.rebin(factor=10)

  In [9]: ima2.info()

  In [6]: plt.figure()

  @savefig Obj_transfo5.png width=3.5in
  In [7]: ima2.plot(zscale=True)

  @suppress
  In [5]: ima = None

The methods `Spectrum.resample <mpdaf.obj.Spectrum.resample>` and `Image.resample <mpdaf.obj.Image.resample>` resamples the spectrum/image to a new coordinate system.
We will resample our image to select its angular resolution and to specify which sky position appears at the center of pixel [0,0]:

.. ipython::
  :okwarning:

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  In [3]: ima = Image('../data/obj/a370II.fits')

  In [8]: ima.info()

  In [3]: plt.figure()

  @savefig Obj_transfo6.png width=3.5in
  In [3]: ima.plot(zscale=True)

  In [4]: newdim = (np.array(ima.shape)/4.5).astype(np.int)

  In [18]: import astropy.units as u

  In [19]: newstep = ima.get_step(unit=u.arcsec) * 4.5

  In [4]: newstart =  np.array(center) + 50./3600

  In [4]: ima2 = ima.resample(newdim, newstart, newstep)

  In [9]: ima2.info()

  In [6]: plt.figure()

  @savefig Obj_transfo7.png width=3.5in
  In [7]: ima2.plot(zscale=True)

  @suppress
  In [5]: ima = None ; ima2 = None

.. ipython::
   :suppress:

   In [4]: plt.close("all")

   In [4]: %reset -f
