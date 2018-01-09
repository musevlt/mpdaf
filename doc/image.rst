.. _image:


************
Image object
************

Image objects contain a 2D data array of flux values, and a `WCS
<mpdaf.obj.WCS>` object that describes the spatial coordinates of the
image. Optionally, an array of variances can also be provided to give the
statistical uncertainties of the fluxes. These can be used for weighting the
flux values and for computing the uncertainties of least-squares fits and other
calculations. Finally a mask array is provided for indicating bad pixels.

The fluxes and their variances are stored in numpy masked arrays, so virtually
all numpy and scipy functions can be applied to them.

Preliminary imports:

.. ipython::

   In [1]: import numpy as np

   In [1]: import matplotlib.pyplot as plt

   In [1]: import astropy.units as u

   In [2]: from mpdaf.obj import Image, WCS


Image creation
==============

There are two common ways to obtain an `~mpdaf.obj.Image` object:

- An image can be created from a user-provided 2D array of the pixel values, or
  from both a 2D array of pixel values, and a corresponding 2D array of
  variances. These can be simple numpy arrays, or they can be numpy masked
  arrays in which bad pixels have been masked. For example:

.. ipython::

  In [6]: wcs1 = WCS(crval=0, cdelt=0.2)

  # numpy data array
  In [8]: MyData = np.ones((300,300))

  # image filled with MyData
  In [9]: ima = Image(data=MyData, wcs=wcs1) # image 300x300 filled with data

  In [10]: ima.info()

- Alternatively, an image can be read from a FITS file. In this case the flux
  and variance values are read from specific extensions:

.. ipython::
  :okwarning:

  # data and variance arrays read from the file (extension DATA and STAT)
  In [2]: ima = Image('obj/IMAGE-HDFS-1.34.fits')

  In [10]: ima.info()

By default, if a FITS file has more than one extension, then it is expected to
have a 'DATA' extension that contains the pixel data, and possibly a 'STAT'
extension that contains the corresponding variances. If the file doesn't contain
extensions of these names, the "ext=" keyword can be used to indicate the
appropriate extension or extensions.

The world-coordinate grid of an Image is described by a `~mpdaf.obj.WCS`
object. When an image is read from a FITS file, this is automatically generated
based on FITS header keywords. Alternatively, when an image is extracted from a
cube or another image, the WCS object is derived from the WCS object of the
original object.

As shown in the above example, information about an image can be printed using
the `~mpdaf.obj.Image.info` method.

Image objects provide a `~mpdaf.obj.Image.plot` method that is based on
`matplotlib.pyplot.plot <http://matplotlib.org/api/pyplot_api.html>`_ and
accepts all matplotlib arguments.  The colors used to plot an image are
distributed between a minimum and a maximum pixel value. By default these are
the minimum and maximum pixel values in the image, but different thresholds can
be specified via the vmin and vmax arguments, as shown below.

.. ipython::

   In [4]: plt.figure()

   @savefig Image1a.png width=4in
   In [5]: ima.plot(vmin=0, vmax=10, colorbar='v')

The `~mpdaf.obj.Image.plot` method has many options to customize the plot, for
instance:

.. ipython::

   In [4]: plt.figure()

   @savefig Image1b.png width=4in
   In [5]: ima.plot(zscale=True, colorbar='v', use_wcs=True, scale='sqrt')

The indexing of the image arrays follows the Python conventions for indexing a
2D array. For an MPDAF image im, the pixel in the lower-left corner is
referenced as im[0,0] and the pixel im[p,q] refers to the horizontal pixel q
and the vertical pixel p, as follows:

.. figure:: _static/image/grid.jpg
  :align: center

In total, this image im contains nq pixels in the horizontal direction and
np pixels in the vertical direction (see :ref:`objformat` for more information).


Image Geometrical manipulation
==============================

In the following example, the sky is rotated within the image by 40 degrees
anticlockwise, then re-sampled to change its pixel size from 0.2 arcseconds to
0.4 arcseconds.

.. ipython::

  In [2]: import astropy.units as u

  In [1]: fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)

  In [5]: ima.plot(ax=ax1, colorbar='v')

  In [1]: ima2 = ima.rotate(40) #this rotation uses an interpolation of the pixels

  In [5]: ima2.plot(ax=ax2, colorbar='v')

  In [3]: ima3 = ima2.resample(newdim=(150,150), newstart=None, newstep=(0.4,0.4), unit_step=u.arcsec, flux=True)

  @savefig Image4.png width=8in
  In [5]: ima3.plot(ax=ax3, colorbar='v')


The `~mpdaf.obj.Image.rotate` method interpolates the image onto a
rotated coordinate grid.

The `~mpdaf.obj.Image.resample` method also interpolates the image
onto a new grid, but before doing this it applies a decimation filter to remove
high spatial frequencies that would otherwise be undersampled by the pixel
spacing.

The ``newstart=None`` argument indicates that the sky position that appears at
the center of pixel [0,0] should also be at the center of pixel [0,0] of the
resampled image.  This argument can alternatively be used to move the sky within
the image.

The `~mpdaf.obj.Image.resample` method is a simplified interface to
the `~mpdaf.obj.Image.regrid` function, which provides more options.

The following example shows how images from different telescopes can be
resampled onto the same coordinate grid, then how the coordinate offsets of the
pixels can be adjusted to account for relative pointing errors:

.. ipython::
  :okwarning:

  # Read a small part of an HST image
  In [2]: imahst = Image('obj/HST-HDFS.fits')

  # Resample the HST image onto the coordinate grid of the MUSE image
  In [3]: ima2hst = imahst.align_with_image(ima)

  # Adjust the relative pointing of the MUSE image.
  In [4]: ima2hst = ima2hst.adjust_coordinates(ima)

  In [5]: plt.figure()

  @savefig Image5.png width=3.5in
  In [6]: ima.plot(colorbar='v', vmin=0.0, vmax=20.0, title='MUSE image')

  In [7]: plt.figure()

  @savefig Image6.png width=3.5in
  In [8]: ima2hst.plot(colorbar='v', title='Part of the HST image')


In the example shown above, the `align_with_image
<mpdaf.obj.Image.align_with_image>` method resamples an HST image onto the same
coordinate grid as a MUSE image. The resampled HST image then has the same
number of pixels, and the same pixel coordinates as the MUSE image.

The `~mpdaf.obj.Image.adjust_coordinates` method then uses
an enhanced form of cross-correlation to estimate and correct for any relative
pointing errors between the two images. Note that, to see the estimated
correction without applying it, the `estimate_coordinate_offset
<mpdaf.obj.Image.estimate_coordinate_offset>` method could have been used.

In the following example, the aligned HST and MUSE images are combined to
produce a higher S/N image. Note the use of the addition operator to add the two
images:

.. ipython::

  In [1]: ima2hst[ima2hst.mask] = 0

  In [1]: ima2hst.unmask()

  In [1]: imacomb = ima + ima2hst

  In [1]: plt.figure()

  @savefig Image7.png width=3.5in
  In [5]: ima[200:, 30:150].plot(colorbar='v', title='original image')

  In [1]: plt.figure()

  @savefig Image8.png width=3.5in
  In [5]: imacomb[200:, 30:150].plot(colorbar='v', title='combined image')

The `~mpdaf.obj.Image.subimage` method can be used to extract a square
or rectangular sub-image of given world-coordinate dimensions from an image. In
the following example it is used used to extract a 20 arcsecond square sub-image
from the center of the HST image.

.. ipython::

  In [1]: dec, ra = imahst.wcs.pix2sky(np.array(imahst.shape)/2)[0]

  In [25]: subima = ima.subimage(center=(dec,ra), size=20.0)

  In [1]: plt.figure()

  @savefig Image9.png width=4in
  In [26]: subima.plot()

The `~mpdaf.obj.Image.inside` method lets the user test whether a given
coordinate is inside an image. In the following example, dec and ra are the
coordinates of the center of the image that were calculated in the preceding
example.

.. ipython::

  In [29]: subima.inside([dec, ra])

  In [30]: subima.inside(ima.get_start())


Object analysis: image segmentation, peak measurement, profile fitting
======================================================================

The following demonstration will show some examples of extracting and analyzing
images of individual objects within an image. The first example segments the
image into several cutout images using the (`~mpdaf.obj.Image.segment`)
method:

.. ipython::
  :okwarning:

  In [1]: im = Image('obj/a370II.fits')

  In [1]: seg = im.segment(minsize=10, background=2100)

The `~mpdaf.obj.Image.segment` method returns a list of images of the
detected sources. In the following example, we extract one of these for further
analysis:

.. ipython::

  In [1]: source = seg[8]

  In [1]: plt.figure()

  @savefig Image10.png width=4in
  In [2]: source.plot(colorbar='v')

  @suppress
  In [5]: im = None

For a first approximation, some simple analysis methods are applied:

 - `~mpdaf.obj.Image.background` to estimate the background level,
 - `~mpdaf.obj.Image.peak` to locate the peak of the source,
 - `~mpdaf.obj.Image.fwhm` to estimate the FWHM of the source.

.. ipython::

  # background value and its standard deviation
  In [1]: source.background()

  # peak position and intensity
  In [2]: source.peak()

  # fwhm in arcsec
  In [3]: source.fwhm()

Then, for greater accuracy we fit a 2D Gaussian to the source, and plot the
isocontours (`~mpdaf.obj.Image.gauss_fit`):

.. ipython::

  In [1]: gfit = source.gauss_fit(plot=False)

  @savefig Image11.png width=4in
  In [2]: gfit = source.gauss_fit(maxiter=150, plot=True)

In general, Moffat profiles provide a better representation of the point-spread
functions of ground-based telescope observations, so next we perform a 2D MOFFAT
fit to the same source (`~mpdaf.obj.Image.moffat_fit`):

.. ipython::

  In [1]: mfit = source.moffat_fit(plot=True)

We then subtract the fitted Gaussian and Moffat models of from the original
source to see the residuals. Note the use of `~mpdaf.obj.gauss_image` and
`~mpdaf.obj.moffat_image` to create MPDAF images of the 2D Gaussian and Moffat
functions:

.. ipython::

  In [1]: from mpdaf.obj import gauss_image, moffat_image

  In [2]: gfitim = gauss_image(wcs=source.wcs, gauss=gfit)

  In [3]: mfitim = moffat_image(wcs=source.wcs, moffat=mfit)

  In [4]: gresiduals = source-gfitim

  In [5]: mresiduals = source-mfitim

  In [1]: plt.figure()

  @savefig Image12.png width=3.5in
  In [1]: mresiduals.plot(colorbar='v', title='Residuals from 2D Moffat profile fitting')

  In [1]: plt.figure()

  @savefig Image13.png width=3.5in
  In [1]: gresiduals.plot(colorbar='v', title='Residuals from 2D Gaussian profile fitting')

Finally we estimate the energy received from the source:

 - The `~mpdaf.obj.Image.ee` method computes ensquared or encircled energy, which is the sum of the flux within a given radius of the center of the source.
 - The `~mpdaf.obj.Image.ee_size` method computes the size of a square centered on the source that contains a given fraction of the total flux of the source,
 - The `~mpdaf.obj.Image.eer_curve` method returns the normalized enclosed energy as a function radius.

.. ipython::

  # Obtain the encircled flux within a radius of one FWHM of the source
  In [4]: source.ee(radius=source.fwhm(), cont=source.background()[0])

  # Get the enclosed energy normalized by the total energy as a function of radius (ERR)
  In [6]: radius, ee = source.eer_curve(cont=source.background()[0])

  # The size of the square centered on the source that contains 90% of the energy (in arcsec)
  In [6]: source.ee_size()

  In [7]: plt.figure()

  In [7]: plt.plot(radius, ee)

  In [8]: plt.xlabel('radius')

  @savefig Image14.png width=4in
  In [9]: plt.ylabel('ERR')


.. ipython::
   :suppress:

   In [4]: plt.close("all")

   In [4]: %reset -f
