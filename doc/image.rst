************
Image object
************

The Image object handles a 2D data array (basically a numpy masked array) containing flux values, associated with a `WCS <mpdaf.obj.WCS>`
object containing the spatial coordinated information (alpha,delta). Optionally, a variance data array
can be attached and used for weighting the flux values. Array masking is used to ignore
some of the pixel values in the calculations.

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
   
   In [2]: from mpdaf.obj import Image, WCS


Image creation
==============

A `Image <mpdaf.obj.Image>` object is created:

- either from one or two 2D numpy arrays containing the flux and variance values (optionally, the data array can be a numpy masked array to deal with bad pixel values):

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)
  
  In [6]: wcs1 = WCS(crval=0, cdelt=0.2)
  
  # numpy data array
  In [8]: MyData = np.ones((300,300))
  
  # image filled with MyData
  In [9]: ima = Image(data=MyData, wcs=wcs1) # image 300x300 filled with data
  
  In [10]: ima.info()

- or from a FITS file (in which case the flux and variance values are read from specific extensions):

.. ipython::
  :okwarning:

  @suppress
  In [5]: setup_logging(stream=sys.stdout)
  
  # data and variance arrays read from the file (extension DATA and STAT)
  In [2]: ima = Image('../data/obj/IMAGE-HDFS-1.34.fits')
  
  In [10]: ima.info()


If the FITS file contains more than one extension and when the FITS extension are not named 'DATA' (for flux values) and 'STAT' (for variance  values), the keyword "ext=" is necessary to give the number of the extensions.

The `WCS <mpdaf.obj.WCS>` object is either created using a linear scale, copied from another Image, or
using the information from the FITS header. 

Information are printed by using the `info <mpdaf.obj.Image.info>` method.

The `plot <mpdaf.obj.Image.plot>` method is based on `matplotlib.pyplot.plot <http://matplotlib.org/api/pyplot_api.html>`_ and accepts all matplotlib arguments.
We display the image with lower / upper scale values:

.. ipython::

   In [4]: plt.figure()

   @savefig Image1.png width=4in
   In [5]: ima.plot(vmin=0, vmax=10, colorbar='v')
   
The format of each numpy array follows the indexing used by Python to
handle images. For an MPDAF image im, the pixel in the lower-left corner is
referenced as im[0,0] and the pixel im[p,q] refers to the horizontal position
q and the vertical position p, as follows:

.. figure:: _static/image/grid.jpg
  :align: center

In total, this image im contains nq pixels in the horizontal direction and
np pixels in the vertical direction (see :ref:`objformat` for more information).


Image Geometrical manipulation
==============================

We `rotate <mpdaf.obj.Image.rotate>` the image by 40 degrees and rebin it onto a 0.4"/pixel scale (conserving flux):

.. ipython::
  
  In [1]: plt.figure()
  
  @savefig Image2.png width=2in
  In [5]: ima.plot(colorbar='v')
  
  In [1]: ima2 = ima.rotate(40) #this rotation uses an interpolation of the pixels
  
  In [1]: plt.figure()
  
  @savefig Image3.png width=2in
  In [5]: ima2.plot(colorbar='v')
  
  In [2]: import astropy.units as u
  
  In [3]: ima3 = ima2.resample(newdim=(150,150), newstart=None, newstep=(0.4,0.4), unit_step=u.arcsec, flux=True)

  In [1]: plt.figure()
  
  @savefig Image4.png width=2in
  In [5]: ima3.plot(colorbar='v')


`rotate <mpdaf.obj.Image.rotate>` rotates the image using an interpolation of the pixels.

``newstart=None`` in `resample <mpdaf.obj.Image.resample>` indicates that we we want that 
the sky position that appears at the center of pixel [0,0] is unchanged by the resampling operation.

`resample <mpdaf.obj.Image.resample>` is a simplified interface to the `regrid <mpdaf.obj.Image.regrid>`
function, which it calls with the more arguments.

Then, we load an external image of the same field (observed with a different instrument) and align it to the previous image in WCS coordinates using the `align_with_image <mpdaf.obj.Image.align_with_image>`:

.. ipython::
  :okwarning:

  # this is a small part of an HST image
  In [1]: imahst = Image('../data/obj/HST-HDFS.fits')
  
  # pixel offsets
  In [1]: imahst.estimate_coordinate_offset(ima)
  
  # align it like the MUSE image
  In [2]: ima2hst = imahst.align_with_image(ima)
  
  In [1]: plt.figure()
  
  @savefig Image5.png width=3.5in
  In [5]: ima.plot(colorbar='v', title='MUSE image')
  
  In [1]: plt.figure()
  
  @savefig Image6.png width=3.5in
  In [5]: ima2hst.plot(colorbar='v', title='part of the HST image')

`estimate_coordinate_offset <mpdaf.obj.Image.estimate_coordinate_offset>` computes the pixels offset between the two image.

`align_with_image <mpdaf.obj.Image.align_with_image>` aligns the two images (at the end they have the same world coordinates).

`adjust_coordinates <mpdaf.obj.Image.adjust_coordinates>` would just adjust the coordinate reference pixel of the HST image to bring its coordinates into line with
those of the reference image.

We combine both datasets to produce a higher S/N image:

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

`subimage <mpdaf.obj.Image.subimage>` extracts a sub-image around a given position:

.. ipython::

  In [1]: dec, ra = imahst.wcs.pix2sky(np.array(imahst.shape)/2)[0]
  
  In [25]: subima = ima.subimage(center=(dec,ra), size=20.0)
  
  In [1]: plt.figure()
  
  @savefig Image9.png width=4in
  In [26]: subima.plot()

`inside <mpdaf.obj.Image.inside>` lets the user to test if coordinates are or not inside the image:

.. ipython::

  In [29]: subima.inside([dec, ra])

  In [30]: subima.inside(ima.get_start())


Object analysis: image segmentation, peak measurement, profile fitting
======================================================================

We will analyse the 2D images of specific objects detected in the image.
We start by segmenting the original image into several cutout images (`segment <mpdaf.obj.Image.segment>`):

.. ipython::
  :okwarning:

  In [1]: im = Image('../data/obj/a370II.fits')

  In [1]: seg = im.segment(minsize=10, background=2100)

We plot one of the sub-images to analyse the corresponding source:

.. ipython::

  In [1]: source = seg[8]
  
  In [1]: plt.figure()
  
  @savefig Image10.png width=4in
  In [2]: source.plot(colorbar='v')
  
At first approximation, we apply wimple methods:
 - `background <mpdaf.obj.Image.background>` to estimate background value,
 - `peak <mpdaf.obj.Image.peak>` to locate the peak of the source,
 - `fwhm <mpdaf.obj.Image.fwhm>` to compute the fwhm of the source.
  
.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  # background value and its standard deviation
  In [1]: source.background()
  
  # peak position and intensity
  In [2]: source.peak()
  
  # fwhm in arcsec
  In [3]: source.fwhm()

Then, For greater accuracy we perform a 2D Gaussian fitting of the source, and plot the isocontours (`gauss_fit <mpdaf.obj.Image.gauss_fit>`):

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)
  
  In [1]: gfit = source.gauss_fit(plot=False)
  
  @savefig Image11.png width=4in
  In [2]: gfit = source.gauss_fit(maxiter=150, plot=True)

Alternatively, we perform a 2D MOFFAT fitting of the same source (`moffat_fit <mpdaf.obj.Image.moffat_fit>`):

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)
  
  In [1]: mfit = source.moffat_fit(plot=True)

We can then subtract each modelled image from the original source and plot the residuals. Note the use of `gauss_image <mpdaf.obj.gauss_image>` and 
`moffat_image <mpdaf.obj.moffat_image>` thaht create a new MPDAF image from a 2D Gaussian/Moffat function.:

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

We try now to estimate the energy of the source:
- `ee <mpdaf.obj.Image.ee>` computes ensquaredencircled energy,
- `ee_size <mpdaf.obj.Image.ee_size>` computes the size of the square centered on the source containing the fraction of the energy,
- `eer_curve <mpdaf.obj.Image.eer_curve>` returns the enclosed ratio energy as function of radius.

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)
  
  # encircled flux
  In [4]: source.ee(radius=source.fwhm(), cont=source.background()[0])
  
  # enclosed energy ratio (ERR)
  In [6]: radius, ee = source.eer_curve(cont=source.background()[0])
  
  # size of the square centered on the source containing 90% of the energy (in arcsec)
  In [6]: source.ee_size()
  
  In [7]: plt.figure()

  In [7]: plt.plot(radius, ee)

  In [8]: plt.xlabel('radius')

  @savefig Image14.png width=4in
  In [9]: plt.ylabel('ERR')


.. ipython::
   :suppress:
   
   In [4]: plt.close("all")