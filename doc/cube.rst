***********
Cube object
***********

Cube python object can handle datacubes which have a regular grid format in
both spatial and spectral axis.  Variance information can also be taken into
account as well as bad pixels.  Cube object can be read and written to disk as
a multi-extension FITS file.

Object is build as a set or numpy masked arrays and world coordinate
information. A number of transformation have been developed  as object
properties. Note that virtually all numpy and scipy functions are available.

.. ipython::
   :suppress:

   In [4]: import sys

   In [4]: from mpdaf import setup_logging

Preliminary imports:

.. ipython::

   In [1]: import numpy as np

   In [1]: import matplotlib.pyplot as plt

   In [1]: import astropy.units as u

   In [2]: from mpdaf.obj import Cube, WCS, WaveCoord

Cube creation
=============

A `Cube <mpdaf.obj.Cube>` object is created:

- either from one or two 3D numpy arrays containing the flux and variance
  values (optionally, the data array can be a numpy masked array to deal with
  bad pixel values):

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  In [6]: wcs1 = WCS(crval=0, cdelt=0.2)

  In [7]: wave1 = WaveCoord(cdelt=1.25, crval=4000.0, cunit=u.angstrom)

  # numpy data array
  In [8]: MyData = np.ones((400, 30, 30))

  # cube filled with MyData
  In [9]: cube = Cube(data=MyData, wcs=wcs1, wave=wave1) # cube 400X30x30 filled with data

  In [10]: cube.info()

  @suppress
  In [5]: cube = None ; data = None

- or from a FITS file (in which case the flux and variance values are read from specific extensions):

.. ipython::
  :okwarning:

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  # data and variance arrays read from the file (extension DATA and STAT)
  In [2]: obj1 = Cube('../data/obj/CUBE.fits')

  In [10]: obj1.info()


If the FITS file contains more than one extension and when the FITS extension are not named 'DATA' (for flux values) and 'STAT' (for variance  values), the keyword "ext=" is necessary to give the number of the extensions.

The `WCS <mpdaf.obj.WCS>` object is either created using a linear scale, copied from another Image, or
using the information from the FITS header.

The `WaveCoord <mpdaf.obj.WaveCoord>` object is either created using a linear scale, copied from another Spectrum, or
using the information from the FITS header.

The `info <mpdaf.obj.Cube.info>` directive gives us already some important informations:

- The cube format 1595 x 10 x 20 has 10 x 20 spatial pixels and 1595 spectral pixels
- In addition to the data extension (.data(1595 x 10 x 20)) a variance extension is also present (.var(1595 x 10 x 20))
- The flux data unit is erg/s/cm\ :sup:`2`/Angstrom and the scale factor is 10\ :sup:`-20`
- The center of the field of view is at DEC: -30° 0' 0.45" and RA: 1°20'0.437" and its size is 2x4 arcsec\ :sup:`2`. The spaxel dimension is 0.2x0.2 arcsec\ :sup:`2`. The rotation angle is 0° with respect to the North.
- The wavelength range is 7300-9292.5 Angstrom with a step of 1.25 Angstrom

The format follows the indexing used by Python to
handle 3D arrays. The pixel in the bottom-lower-left corner is
referenced as [0,0,0] and the pixel [k,p,q] refers to the horizontal position
q, the vertical position p, and the spectral position k, as follows:

.. figure:: _static/cube/gridcube.jpg
  :align: center

(see :ref:`objformat` for more information).


Let's compute the reconstructed white light image and display it:

.. ipython::

  In [1]: ima1 = obj1.sum(axis=0)

  In [2]: plt.figure()

  @savefig Cube1.png width=4in
  In [3]: ima1.plot(scale='arcsinh', colorbar='v')

Let's now compute the total spectrum of the object:

.. ipython::

  In [1]: sp1 = obj1.sum(axis=(1,2))

  In [2]: plt.figure()

  @savefig Cube2.png width=4in
  In [3]: sp1.plot()


Loop over all spectra
=====================

We will create the continuum subtracted datacube of the previously extracted object.

We start by fitting the continuum on sp1:

.. ipython::

  In [1]: plt.figure()

  In [2]: cont1 = sp1.poly_spec(5)

  In [3]: sp1.plot()

  @savefig Cube3.png width=4in
  In [4]: cont1.plot(color='r')

Let's try also on a single spectrum at the edge of the galaxy:

.. ipython::

  In [1]: plt.figure()

  In [2]: sp1 = obj1[:,5,2]

  In [2]: sp1.plot()

  @savefig Cube4.png width=4in
  In [3]: sp1.poly_spec(5).plot(color='r')


Fine, now let's do this for all spectrum of the input datacube. We are going to use the spectra iterator
to loop over all spectra. Let's see how `iter_spe <mpdaf.obj.iter_spe>` works:

.. ipython::

  In [1]: from mpdaf.obj import iter_spe

  In [2]: small = obj1[:,0:2,0:3]

  In [3]: small.shape

  @verbatim
  In [7]: for sp in iter_spe(small):
     ...:     print sp.data.max()
     ...:

In this example, we have extracted successively all six spectra of the small datacube and printed their peak value.

Now let's use it to perform the computation of the continuum datacube.  We start
by creating an empty datacube with the same dimensions than the original one,
but without variance information (using the `colne <mpdaf.obj.DataArray.clone>`
function). Using two spectrum iterators we extract iteratively all input spectra
(sp) and (still empty) continuum spectrum (co). For each extracted spectrum we
just fit the continuum and save it to the continuum datacube.:

.. ipython::
  :okwarning:

  In [1]: cont1 = obj1.clone(data_init=np.empty, var_init=np.zeros)

  In [2]: for sp, co in zip(iter_spe(obj1), iter_spe(cont1)):
     ...:     co[:] = sp.poly_spec(5)

And that's it, we have now the continuum datacube. Note that we have used the co[:] = sp.poly_spec(5)
assignment rather than the more intuitive co = sp.poly_spec(5) assignment. The use of co[:] is mandatory
otherwise the continuum spectra co is created but not written into the cont1 datacube.

But, the better way to compute the continuum datacube is to use the `loop_spe_multiprocessing <mpdaf.obj.Cube.loop_spe_multiprocessing>` that automatically loop on spectrum using multiprocessing:

.. ipython::
  :okwarning:

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  In [1]: from mpdaf.obj import Spectrum

  In [2]: cont2 = obj1.loop_spe_multiprocessing(f=Spectrum.poly_spec, deg=5)

Let's check the results and display the continuum reconstructed images:

.. ipython::

  In [1]: rec1 = cont1.sum(axis=0)

  In [2]: plt.figure()

  @savefig Cube5.png width=3.5in
  In [3]: rec1.plot(scale='arcsinh', colorbar='v', title='method 1')

  In [1]: rec2 = cont2.sum(axis=0)

  In [2]: plt.figure()

  @savefig Cube6.png width=3.5in
  In [3]: rec2.plot(scale='arcsinh', colorbar='v', title='method2')

  @suppress
  In [5]: cont2 = None

We can also compute the line emission datacube:

.. ipython::

  In [1]: line1 = obj1 - cont1

  In [2]: plt.figure()

  @savefig Cube7.png width=4in
  In [2]: line1.sum(axis=0).plot(scale='arcsinh', colorbar='v')

Then, we will compute equivalent width of the Ha emission in the galaxy.
First let's isolate the emission line by truncating the object datacube in wavelength:

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  In [1]: plt.figure()

  In [2]: sp1.plot()

  In [3]: k1,k2 = sp1.wave.pixel([9000,9200], nearest=True)

  In [4]: emi1 = obj1[k1:k2+1,:,:]

  In [4]: emi1.info()

  In [5]: sp1 = emi1.sum(axis=(1,2))

  @savefig Cube8.png width=4in
  In [6]: sp1.plot(color='r')

  @suppress
  In [1]: obj1 = None ; cont1 = None ; line1 = None

We first fit and subtract the continuum. Before doing the polynomial fit we mask the region of
the emission lines (sp1.mask) and then we perform the linear fit. Then the spectrum is unmasked
and the continuum subtracted:

.. ipython::

  In [1]: plt.figure()

  In [2]: sp1.mask_region(9050, 9125)

  In [3]: cont1 = sp1.poly_spec(1)

  In [4]: sp1.unmask()

  In [5]: plt.figure()

  In [6]: cont1.plot()

  In [7]: line1 = sp1 - cont1

  @savefig Cube9.png width=4in
  In [8]: line1.plot(color='r')

We then compute the Ha line total flux by simple integration (taking into account the pixel size in A)
over the wavelength range centered around Halfa and the continuum mean flux at the same location:

.. ipython::

  In [1]: plt.figure()

  In [2]: k = line1.data.argmax()

  @savefig Cube10.png width=4in
  In [3]: line1[55-10:55+11].plot(color='r')

  In [4]: fline = (line1[55-10:55+11].sum()*line1.unit) * (line1.get_step(unit=line1.wave.unit)*line1.wave.unit)

  In [5]: cline = cont1[55-10:55+11].mean()*cont1.unit

  In [6]: ew = fline/cline

  In [7]: print fline, cline, ew

Now we repeat this for all datacube spectra, and we  save Ha flux and equivalent width in two images.
We start creating two images with identical shape and wcs as the reconstructed image and then use
the spectrum iterator `iter_spe <mpdaf.obj.iter_spe>`:

.. ipython::

  In [1]: ha_flux = ima1.clone(data_init=np.empty)

  In [2]: cont_flux = ima1.clone(data_init=np.empty)

  In [3]: ha_ew = ima1.clone(data_init=np.empty)

  In [4]: for sp,pos in iter_spe(emi1, index=True):
     ...:     p,q = pos
     ...:     sp.mask_region(9050, 9125)
     ...:     cont = sp.poly_spec(1)
     ...:     sp.unmask()
     ...:     line = sp - cont
     ...:     fline = line[55-10:55+11].sum() * line.get_step(unit=line.wave.unit)
     ...:     cline = cont[55-10:55+11].mean()
     ...:     ew = fline/cline
     ...:     cont_flux[p,q] = cline
     ...:     ha_flux[p,q] = fline
     ...:     ha_ew[p,q] = ew

  In [1]: plt.figure()

  @savefig Cube11.png width=2in
  In [5]: cont_flux.plot(title="continuum mean flux (%s)"%cont_flux.unit, colorbar='v')

  In [6]: ha_flux.unit = sp.unit * sp.wave.unit

  In [1]: plt.figure()

  @savefig Cube12.png width=2in
  In [7]: ha_flux.plot(title="Ha line total flux (%s)"%ha_flux.unit, colorbar='v')

  In [8]: ha_ew.mask_selection(np.where((ima1.data)<4000))

  In [9]: ha_ew.unit = ha_flux.unit / cont_flux.unit

  In [1]: plt.figure()

  @savefig Cube13.png width=2in
  In [10]: ha_ew.plot(title="Ha line ew (%s)"%ha_ew.unit, colorbar='v')

  @suppress
  In [1]: ha_flux = None ; cont_flux = None ; ha_ew = None


Loop over all images
====================

In this section, we are going to process our datacube in spatial direction. We
consider the datacube as a collection of monochromatic images and we process
each of them. For each monochromatic image we apply a convolution by a gaussian
kernel.

.. ipython::
  :okwarning:

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  # data and variance arrays read from the file (extension DATA and STAT)
  In [6]: cube = Cube('../data/obj/Central_Datacube_bkg.fits')

First, we use the image iterator `iter_ima <mpdaf.obj.iter_ima>`:

.. ipython::
  :verbatim:

  In [1]: from mpdaf.obj import iter_ima

  In [2]: cube2 = cube.clone(data_init=np.empty, var_init=np.empty)

  In [3]: for ima,k in iter_ima(cube, index=True):
     ...:     cube2[k,:,:] = ima.gaussian_filter(sigma=3)

We can also use the `loop_ima_multiprocessing <mpdaf.obj.Cube.loop_ima_multiprocessing>` method that automatically loops over all images to apply the convolution:

.. ipython::

  In [1]: from mpdaf.obj import Image

  In [2]: cube2 = cube.loop_ima_multiprocessing(f=Image.gaussian_filter, sigma=3)

We then plot the result:

.. ipython::

  In [1]: plt.figure()

  @savefig Cube14.png width=3.5in
  In [2]: cube.sum(axis=0).plot(title='before Gaussian filter')

  In [1]: plt.figure()

  @savefig Cube15.png width=3.5in
  In [3]: cube2.sum(axis=0).plot(title='after Gaussian filter')

  @suppress
  In [5]: cube2 = None

Then, we will use the `loop_ima_multiprocessing
<mpdaf.obj.Cube.loop_ima_multiprocessing>` method to fit and remove
a background gradient from a simulated datacube.  For each image of the cube,
we fit a 2nd order polynomial to the background values (selected here by simply
applying a flux threshold to mask all bright objects). We do so by doing
a chi^2 minimization over the polynomial coefficients using the numpy recipe
``np.linalg.lstsq()``. For this, we define a function that takes an image as
parameter and returns the background-subtracted image:

.. ipython::

  In [1]: def remove_background_gradient(ima):
     ...:     ksel = np.where(ima.data.data<2.5)
     ...:     pval = ksel[0]
     ...:     qval = ksel[1]
     ...:     zval = ima.data.data[ksel]
     ...:     degree = 2
     ...:     Ap = np.vander(pval,degree)
     ...:     Aq = np.vander(qval,degree)
     ...:     A = np.hstack((Ap,Aq))
     ...:     (coeffs,residuals,rank,sing_vals) = np.linalg.lstsq(A,zval)
     ...:     fp = np.poly1d(coeffs[0:degree])
     ...:     fq = np.poly1d(coeffs[degree:2*degree])
     ...:     X,Y = np.meshgrid(xrange(ima.shape[0]),xrange(ima.shape[1]))
     ...:     ima2 = ima - np.array(map(lambda q,p: fp(p)+fq(q),Y,X))
     ...:     return ima2
     ...:

We can then create the background-subtracted cube:

.. ipython::

  In [1]: cube2 = cube.loop_ima_multiprocessing(f=remove_background_gradient)

Finally, we compare the results for one of the slices:

.. ipython::

  In [1]: plt.figure()

  @savefig Cube16.png width=3.5in
  In [2]: cube[5,:,:].plot(vmin=-1, vmax=4)

  In [1]: plt.figure()

  @savefig Cube17.png width=3.5in
  In [2]: cube2[5,:,:].plot(vmin=-1, vmax=4)

  @suppress
  In [5]: cube2 = None ; cube = None

Sub-cube extraction
===================

.. warning::

  To be written


`mpdaf.obj.Cube.get_lambda <mpdaf.obj.Cube.get_lambda>` returns the sub-cube corresponding to a wavelength range.

`mpdaf.obj.Cube.get_image <mpdaf.obj.Cube.get_image>` extracts an image around a position from the datacube.

`mpdaf.obj.Cube.bandpass_image <mpdaf.obj.Cube.bandpass_image>`

`mpdaf.obj.Cube.subcube <mpdaf.obj.Cube.subcube>` extracts a sub-cube around a position.

`mpdaf.obj.Cube.aperture <mpdaf.obj.aperture>`

`mpdaf.obj.Cube.subcube_circle_aperture <mpdaf.obj.Cube.subcube_circle_aperture>` extracts a sub-cube from an circle aperture of fixed radius.

.. ipython::
   :suppress:

   In [4]: plt.close("all")

   In [4]: %reset -f
