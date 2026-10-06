***********
Rearranging
***********

The `~mpdaf.obj.DataArray.crop` method can be used to reduce the size of a
Cube, Image or Spectrum to the smallest sub-array that retains all unmasked
pixels. In the following example, the pixels outside of an elliptical region of
an image are first masked, and then the crop method is used to select the
sub-image that just contains the unmasked elliptical region:

.. ipython::
   :okwarning:

   In [1]: import numpy as np

   In [1]: import matplotlib.pyplot as plt

   In [2]: from mpdaf.obj import Image

   In [3]: ima = Image('obj/a370II.fits')

   In [2]: center=[-1.5642, 39.9620]

   In [2]: ima.mask_ellipse(center=center, radius=(80, 110), posangle=ima.get_rot(), inside=False)

   In [2]: ima.shape

   In [3]: plt.figure()

   @savefig Obj_transfo1.png width=3.5in
   In [3]: ima.plot()

   In [4]: ima.crop()

   In [2]: ima.shape

   In [3]: plt.figure()

   @savefig Obj_transfo2.png width=3.5in
   In [3]: ima.plot()


The `Spectrum.truncate <mpdaf.obj.Spectrum.truncate>`, `Image.truncate
<mpdaf.obj.Image.truncate>` and `Cube.truncate <mpdaf.obj.Cube.truncate>`
methods return a sub-object that is bounded by specified wavelength or/and
spatial world-coordinates:

In the following example, the image from the previous example is truncated to
just enclose a region of the sky whose width in right ascension is 150
arc-seconds, and whose height in declination is also 150 arc-seconds. Since the
ellipse of the previous example was deliberately aligned with the declination
and right ascension axes, this effectively truncates the axes of the ellipse.

.. ipython::
   :okwarning:

   In [43]: ymin, xmin = np.array(center) - 75./3600

   In [44]: ymax, xmax = np.array(center) + 75./3600

   In [45]: ima2 = ima.truncate(ymin, ymax, xmin, xmax)

   In [44]: plt.figure()

   @savefig Obj_transfo3.png width=4in
   In [1]: ima2.plot()

   In [48]: ima2.get_rot()

   @suppress
   In [5]: ima = None ; ima2 = None


The ranges x_min to x_max and y_min to y_max, specify a rectangular region of
the sky in world coordinates. The truncate function returns the sub-image that
just encloses this region. In the above example, the world coordinate axes are
not parallel to the array axes, so there are some pixels in the image that are
outside the specified world-coordinate region. These pixels are masked.

The methods `Spectrum.rebin <mpdaf.obj.Spectrum.rebin>`, `Image.rebin
<mpdaf.obj.Image.rebin>` and `Cube.rebin <mpdaf.obj.Cube.rebin>` reduce the
array dimensions of these objects by integer factors, without changing the area
of sky that they cover. They do this by creating a new object whose pixels are
the mean of several neighboring pixels of the input object.

.. ipython::
  :okwarning:

  In [3]: ima = Image('obj/a370II.fits')

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

The methods `Spectrum.resample <mpdaf.obj.Spectrum.resample>` and
`Image.resample <mpdaf.obj.Image.resample>` resample a spectrum or image to a
new world-coordinate grid. The following example resamples an image to change
its angular resolution and also to change which sky position appears at the
center of pixel [0,0]:

.. ipython::
  :okwarning:

  In [3]: ima = Image('obj/a370II.fits')

  In [8]: ima.info()

  In [3]: plt.figure()

  @savefig Obj_transfo6.png width=3.5in
  In [3]: ima.plot(zscale=True)

  In [4]: newdim = (np.array(ima.shape)/4.5).astype(int)

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
