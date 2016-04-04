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

Example::

  import astropy.units as u
  import numpy as np
  from mpdaf.obj import Cube
  from mpdaf.obj import WCS
  from mpdaf.obj import WaveCoord

  wcs1 = WCS(crval=0,cdelt=0.2)
  wcs2 = WCS(crval=0,cdelt=0.2, shape=40)
  wave1 = WaveCoord(cdelt=1.25, crval=4000.0, cunit=u.angstrom)
  wave2 = WaveCoord(cdelt=1.25, crval=4000.0, cunit=u.angstrom, shape=300)
  MyData = np.ones((400,30,30))

  cub = Cube(filename="cube.fits",ext=1) # cube from file without variance (extension number is 1)
  cub = Cube(filename="cube.fits",ext=(1,2)) # cube from file with variance (extension numbers are 1 and 2)
  cub = Cube(wcs=wcs1, wave=wave1, data=MyData) # cube filled with MyData
  # warning: wavelength coordinates and data have not the same dimensions.
  # Shape of WaveCoord object is modified.
  # cub.wave.shape = 400
  cub = Cube(wcs=wcs2, wave=wave1, data=MyData) # warning: world coordinates and data have not the same dimensions.
  # Shape of WCS object is modified.
  # cub.wcs.naxis1 = 30
  # cub.wcs.naxis2 = 30

Cube object format
==================

A cube object O consist of:

+------------------+--------------------------------------------------------------------------------------------------+
| Component        | Description                                                                                      |
+==================+==================================================================================================+
| O.filename       | Possible FITS filename                                                                           |
+------------------+--------------------------------------------------------------------------------------------------+
| O.primary_header | FITS primary header instance                                                                     |
+------------------+--------------------------------------------------------------------------------------------------+
| O.wcs            | World coordinate spatial information (`WCS <mpdaf.obj.WCS>` object)                              |
+------------------+--------------------------------------------------------------------------------------------------+
| O.wave           | World coordinate spectral information  (`WaveCoord <mpdaf.obj.WaveCoord>` object)                |
+------------------+--------------------------------------------------------------------------------------------------+
| O.shape          | Array containing the 3 dimensions [nk,np,nq] of the cube: nk channels and np x nq spatial pixels |
+------------------+--------------------------------------------------------------------------------------------------+
| O.data           | Masked numpy array with data values                                                              |
+------------------+--------------------------------------------------------------------------------------------------+
| O.data_header    | FITS data header instance                                                                        |
+------------------+--------------------------------------------------------------------------------------------------+
| O.unit           | Physical units of the data values                                                                |
+------------------+--------------------------------------------------------------------------------------------------+
| O.dtype          | Type of the data (integer, float)                                                                |
+------------------+--------------------------------------------------------------------------------------------------+
| O.var            | (optionally) Numpy array with variance values                                                    |
+------------------+--------------------------------------------------------------------------------------------------+

Masked arrays are arrays that may have missing or invalid entries.
The `numpy.ma <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_ module provides a nearly work-alike replacement for numpy that supports data arrays with masks.

Each numpy masked array has 3 dimensions: Array[k,p,q] with k the spectral axis, p and q the spatial axes

The format of each numpy array follows the indexing used by Python to
handle 3D arrays. For an MPDAF cube C, the pixel in the bottom-lower-left corner is
referenced as C[0,0,0] and the pixel C[k,p,q] refers to the horizontal position
q, the vertical position p, and the spectral position k, as follows:

.. figure:: _static/cube/gridcube.jpg
  :align: center

In total, this cube C contains nq pixels in the horizontal direction,
np pixels in the vertical direction and nk channels in the spectral direction.


Reference
=========

`mpdaf.obj.Cube <mpdaf.obj.Cube>` is the classic cube constructor.

`mpdaf.obj.Cube.copy <mpdaf.obj.DataArray.copy>` copies Cube object in a new one and returns it.

`mpdaf.obj.Cube.clone <mpdaf.obj.DataArray.clone>` returns a shallow copy with the same shape and coordinates, filled with zeros.

`mpdaf.obj.cube.info <mpdaf.obj.DataArray.info>` prints information.

`mpdaf.obj.Cube.write <mpdaf.obj.Cube.write>` saves the Cube in a FITS file.


Indexing
--------

`Cube[k,p,q] <mpdaf.obj.Cube.__getitem__>` returns the corresponding value.

`Cube[k1:k2,p1:p2,q1:q2] <mpdaf.obj.Cube.__getitem__>` returns the sub-cube.

`Cube[k,:,:] <mpdaf.obj.Cube.__getitem__>` returns an Image.

`Cube[:,p,q] <mpdaf.obj.Cube.__getitem__>` returns a Spectrum.

`Cube[k,p,q] = value <mpdaf.obj.Cube.__setitem__>` sets value in Cube.data[k,p,q]

`Cube[k1:k2,p1:p2,q1:q2] = array <mpdaf.obj.Cube.__setitem__>` sets the corresponding part of Cube.data.


Getters and setters
-------------------

`mpdaf.obj.Cube.get_lambda <mpdaf.obj.Cube.get_lambda>` returns the sub-cube corresponding to a wavelength range.

`mpdaf.obj.Cube.get_step <mpdaf.obj.Cube.get_step>` returns the cube steps.

`mpdaf.obj.Cube.get_range <mpdaf.obj.Cube.get_range>` returns minimum and maximum values of cube coordiantes.

`mpdaf.obj.Cube.get_start <mpdaf.obj.Cube.get_start>` returns coordinates values corresponding to pixel (0,0,0).

`mpdaf.obj.Cube.get_end <mpdaf.obj.Cube.get_end>` returns coordinates values corresponding to pixel (-1,-1,-1).

`mpdaf.obj.Cube.get_rot <mpdaf.obj.Cube.get_rot>` returns the rotation angle.

`mpdaf.obj.Cube.get_data_hdu <mpdaf.obj.Cube.get_data_hdu>` returns astropy.io.fits.ImageHDU corresponding to the DATA extension.

`mpdaf.obj.Cube.get_stat_hdu <mpdaf.obj.Cube.get_stat_hdu>` returns astropy.io.fits.ImageHDU corresponding to the STAT extension.

`mpdaf.obj.Cube.set_wcs <mpdaf.obj.Cube.set_wcs>` sets the world coordinates.


Mask
----

`<= <mpdaf.obj.DataArray.__le__>` masks data array where greater than a given value.

`< <mpdaf.obj.DataArray.__lt__>` masks data array where greater or equal than a given value.

`>= <mpdaf.obj.DataArray.__ge__>` masks data array where less than a given value.

`> <mpdaf.obj.DataArray.__gt__>` masks data array where less or equal than a given value.

`mpdaf.obj.cube.unmask <mpdaf.obj.DataArray.unmask>` unmasks the cube (just invalid data (nan,inf) are masked) (in place).

`mpdaf.obj.Cube.mask <mpdaf.obj.Cube.mask>` masks values inside/outside the described region (in place).

`mpdaf.obj.Cube.mask_ellipse <mpdaf.obj.Cube.mask_ellipse>` masks values inside/outside the described region. Uses an elliptical shape.

`mpdaf.obj.Cube.mask_variance <mpdaf.obj.DataArray.mask_variance>` masks pixels with a variance upper than threshold value.

`mpdaf.obj.Cube.mask_selection <mpdaf.obj.DataArray.mask_selection>` masks pixels corresponding to a selection.


Arithmetic
----------

`\+ <mpdaf.obj.Cube.__add__>` makes a addition.

`\- <mpdaf.obj.Cube.__sub__>` makes a substraction .

`\* <mpdaf.obj.Cube.__mul__>` makes a multiplication.

`/ <mpdaf.obj.Cube.__div__>` makes a division.

`mpdaf.obj.Cube.sqrt <mpdaf.obj.DataArray.sqrt>` computes the positive square-root of data extension.

`mpdaf.obj.Cube.abs <mpdaf.obj.DataArray.abs>` computes the absolute value of data extension.

`mpdaf.obj.Cube.sum <mpdaf.obj.Cube.sum>` returns the sum over the given axis.

`mpdaf.obj.Cube.mean <mpdaf.obj.Cube.mean>` returns the mean over the given axis.

`mpdaf.obj.Cube.median <mpdaf.obj.Cube.median>` returns the median over the given axis.


Transformation
--------------

`mpdaf.obj.Cube.resize <mpdaf.obj.Cube.resize>` resizes the cube to have a minimum number of masked values (in place).

`mpdaf.obj.Cube.truncate <mpdaf.obj.Cube.truncate>` extracts a sub-cube.

`mpdaf.obj.Cube.get_image <mpdaf.obj.Cube.get_image>` extracts an image around a position from the datacube.

`mpdaf.obj.Cube.subcube <mpdaf.obj.Cube.subcube>` extracts a sub-cube around a position.

`mpdaf.obj.Cube.subcube <mpdaf.obj.Cube.subcube_circle_aperture>` extracts a sub-cube from an circle aperture of fixed radius.

`mpdaf.obj.Cube.rebin_mean <mpdaf.obj.Cube.rebin_mean>` shrinks the size of the cube by factor using mean values.

`mpdaf.obj.Cube.rebin_median <mpdaf.obj.Cube.rebin_median>` shrinks the size of the cube by factor using median values.

`mpdaf.obj.Cube.loop_ima_multiprocessing <mpdaf.obj.Cube.loop_ima_multiprocessing>` loops over all images to apply a function/method.

`mpdaf.obj.Cube.loop_ima_multiprocessing <mpdaf.obj.Cube.loop_ima_multiprocessing>` loops over all images to apply a function/method.


Tutorials
=========

We can load the tutorial files with the command::

 > git clone http://urania1.univ-lyon1.fr/git/mpdaf_data.git

Tutorial 1
----------

In this tutorial we learn how to play with an existing datacube, extract a small cube centered around an object and compute its spectrum.

We read the datacube from disk and display basic information::

 >>> from mpdaf.obj import Cube
 >>> cube = Cube('Central_DATACUBE_FINAL_11to20_2012-05-16.fits')
 >>> cube.info()
 [INFO] 3601 x 101 x 101 Cube (Central_DATACUBE_FINAL_11to20_2012-05-16.fits)
 [INFO] .data(3601,101,101) (1e-20 erg / (Angstrom cm2 s)), .var(3601,101,101)
 [INFO] center:(-30:00:01.3494,01:20:00.1373) size in arcsec:(20.200,20.200) step in arcsec:(0.200,0.200) rot:0.0 deg
 [INFO] wavelength: min:4800.00 max:9300.00 step:1.25 angstrom

The info directive gives us already some important informations:

- The cube format 3601 x 101 x 101 has 101 x 101 spatial pixels and 3601 spectral pixels
- In addition to the data extension (.data(3601,101,101) a variance extension is also present (.var(3601,101,101))
- The flux data unit is erg/s/cm\ :sup:`2`/Angstrom and the scale factor is 10\ :sup:`-20`
- The center of the field of view is at DEC: -30° 0' 1.35" and RA: 1°20'0.137" and its size is 20.2x20.2 arcsec\ :sup:`2`. The spaxel dimension is 0.2x0.2 arcsec\ :sup:`2`. The rotation angle is 0° with respect to the North.
- The wavelength range is 4800-9300 Angstrom with a step of 1.25 Angstrom

Let's compute the reconstructed white light image and display it::

 >>> ima = cube.sum(axis=0)
 >>> ima.plot(scale='arcsinh', colorbar='v')

.. figure::  _static/cube/recima1.png
   :align:   center

We extract the cube corresponding to the object centered at x=31 y=55 spaxels::

 >>> obj1 = cube[:,55-5:55+5,31-10:31+10]
 >>> ima1 = obj1.mean(axis=0)
 >>> ima1.plot(colorbar='v')

.. figure::  _static/cube/recima2.png
   :align:   center

Let's now compute the total spectrum of the object::

 >>> import matplotlib.pyplot as plt
 >>> plt.figure()
 >>> sp1 = obj1.sum(axis=(1,2))
 >>> sp1.plot()

.. figure::  _static/cube/spec1.png
   :align:   center

Tutorial 2
----------

In this second tutorial we create the continuum subtracted datacube of the previously extracted object.

We start by fitting the continuum on sp1 (see tutorial 1)::

 >>> plt.figure()
 >>> cont1 = sp1.poly_spec(5)
 >>> sp1.plot()
 >>> cont1.plot(color='r')

.. figure::  _static/cube/spec2.png
   :align:   center

Let's try also on a single spectrum at the edge of the galaxy::

 >>> plt.figure()
 >>> obj1[:,5,2].plot()
 >>> obj1[:,5,2].poly_spec(5).plot(color='r')

.. figure::  _static/cube/spec3.png
   :align:   center

Fine, now let's do this for all spectrum of the input datacube. We are going to use the spectra iterator
to loop over all spectra.
Let's see how the spectrum iterator works::

 >>> from mpdaf.obj import iter_spe
 >>> small = obj1[:,0:2,0:3]
 >>> small.shape
 (3601, 2, 3)
 >>> for sp in iter_spe(small):
 >>> 	print sp.data.max()
 2.06232500076
 1.98103439808
 1.90471208096
 1.92691171169
 1.94003844261
 1.57908594608

In this example, we have extracted sucessively all six spectra of the small datacube and printed their peak value.

Now let's use it to perform the computation of the continuum datacube.
We start by creating an empty datacube with the same dimensions than the original one, but without variance
information (using the clone function). Using two spectrum iterors we extract iteratively
all input spectra (sp) and (still
empty) continuum spectrum (co). For each extracted spectrum we just fit the continuum and save it to the
continuum datacube.::

 >>> cont1 = obj1.clone()
 >>> for sp,co in zip(iter_spe(obj1), iter_spe(cont1)):
 >>>   co[:] = sp.poly_spec(5)

And that's it, we have now the continuum datacube. Note that we have used the co[:] = sp.poly_spec(5)
assignment rather than the more intuitive co = sp.poly_spec(5) assignment. The use of co[:] is mandatory
otherwise the continnum spectra co is created but not written into the cont1 datacube.

But, the better way to compute the continuum datacube is to use the `mpdaf.obj.Cube.loop_spe_multiprocessing <mpdaf.obj.Cube.loop_spe_multiprocessing>` that automatically loop on spectrum using multiprocessing::

 >>> from mpdaf.obj import Spectrum
 >>> cont2 = obj1.loop_spe_multiprocessing(f=Spectrum.poly_spec, deg=5)
 [INFO] loop_spe_multiprocessing (poly_spec): 200 tasks

Let's check the result and display the continuum reconstructed image::

 >>> rec2 = cont2.sum(axis=0)
 >>> rec2.plot(scale='arcsinh', colorbar='v')

.. figure::  _static/cube/recima4.png
   :align:   center

We can also compute the line emission datacube::

 >>> line1 = obj1 - cont1
 >>> line1.sum(axis=0).plot(scale='arcsinh', colorbar='v')

.. figure::  _static/cube/recima5.png
   :align:   center


Tutorial 3
----------

In this tutorial we will compute equivalent width of the Ha emission in the galaxy.
First let's isolate the emission line by truncating the object datacube in wavelength.::

 >>> plt.figure()
 >>> sp1.plot()
 >>> k1,k2 = sp1.wave.pixel([9000,9200], nearest=True)
 >>> emi1 = obj1[k1+1:k2+1,:,:]
 >>> emi1.info()
 [INFO] 160 x 10 x 20 Cube (Central_DATACUBE_FINAL_11to20_2012-05-16.fits)
 [INFO] .data(160,10,20) (1e-20 erg / (Angstrom cm2 s)), .var(160,10,20)
 [INFO] center:(-30:00:00.4494,01:20:00.4376) size in arcsec:(2.000,4.000) step in arcsec:(0.200,0.200) rot:0.0 deg
 [INFO] wavelength: min:9001.25 max:9200.00 step:1.25 angstrom
 >>> sp1 = emi1.sum(axis=(1,2))
 >>> sp1.plot(color='r')

.. figure::  _static/cube/spec4.png
   :align:   center

We first fit and subtract the continuum. Before doing the polynomial fit we mask the region of
the emission lines (sp1.mask) and then we perform the linear fit. Then the spectrum is unmasked
and the continnum subtracted::

 >>> plt.figure()
 >>> sp1.mask(9050, 9125)
 >>> cont1 = sp1.poly_spec(1)
 >>> sp1.unmask()
 >>> cont1.plot()
 >>> line1 = sp1 - cont1
 >>> line1.plot(color='r')

.. figure::  _static/cube/spec5.png
   :align:   center

We then compute the Ha line total flux by simple integration (taking into account the pixel size in A)
over the wavelength range centered around Halfa and the continuum mean flux at the same location::

 >>> plt.figure()
 >>> k = line1.data.argmax()
 >>> line1[55-10:55+11].plot(color='r')
 >>> fline = (line1[55-10:55+11].sum()*line1.unit) * (line1.get_step()*line1.wave.unit)
 >>> cline = cont1[55-10:55+11].mean()*cont1.unit
 >>> ew = fline/cline
 >>> print fline, cline, ew
 8352.08991389 1e-20 erg / (cm2 s) 1932.61993433 1e-20 erg / (Angstrom cm2 s) 4.32164119056 Angstrom

.. figure::  _static/cube/spec6.png
   :align:   center

Now we repeat this for all datacube spectra, and we  save Ha flux and equivalent width in two images.
We start creating two images with identical shape and wcs as the reconstructed image and then use
the spectrum iterator.::

 >>> ha_flux = ima1.clone()
 >>> cont_flux = ima1.clone()
 >>> ha_ew = ima1.clone()
 >>> for sp,pos in iter_spe(emi1, index=True):
 >>>   p,q = pos
 >>>   sp.mask(9050, 9125)
 >>>   cont = sp.poly_spec(1)
 >>>   sp.unmask()
 >>>   line = sp - cont
 >>>   fline = line[55-10:55+11].sum()*line.get_step()
 >>>   cline = cont[55-10:55+11].mean()
 >>>   ew = fline/cline
 >>>   cont_flux[p,q] = cline
 >>>   ha_flux[p,q] = fline
 >>>   ha_ew[p,q] = ew
 >>> cont_flux.plot(title="continuum mean flux (%s)"%cont_flux.unit, colorbar='v')
 >>> ha_flux.unit = sp.unit * sp.wave.unit
 >>> ha_flux.plot(title="Ha line total flux (%s)"%ha_flux.unit, colorbar='v')
 >>> import numpy as np
 >>> ha_ew.mask_selection(np.where((ima1.data)<40))
 >>> ha_ew.unit = ha_flux.unit / cont_flux.unit
 >>> ha_ew.plot(title="Ha line ew (%s)"%ha_ew.unit, colorbar='v')

.. image::  _static/cube/recima6.png

.. image::  _static/cube/recima7.png

.. image::  _static/cube/recima8.png


Tutorial 4
----------

In this tutorial we are going to process our datacube in spatial direction. We consider the datacube as a collection of
monochromatic images and we process each of them. For each monochromatic image we apply a convolution by a gaussian kernel.

First, we use the image iterator::

 >>> from mpdaf.obj import iter_ima
 >>> cube2 = cube.clone()
 >>> for ima,k in iter_ima(cube, index=True):
 >>>   cube2[k,:,:] = ima.gaussian_filter(sigma=3)

We can also use the `mpdaf.obj.Cube.loop_ima_multiprocessing <mpdaf.obj.Cube.loop_ima_multiprocessing>` method that automatically loops over all images to apply the convolution::

 >>> from mpdaf.obj import Image
 >>> cube2 = cube.loop_ima_multiprocessing(f=Image.gaussian_filter, sigma=3)
 [INFO] loop_ima_multiprocessing (gaussian_filter): 3601 tasks

We then plot the result::

 >>> cube.sum(axis=0).plot(title='before Gaussian filter')
 >>> cube2.sum(axis=0).plot(title='after Gaussian filter')

.. image::  _static/cube/recima9.png

.. image::  _static/cube/recima10.png

Tutorial 5
----------

In this tutorial, we will use the spectrum iterator (Tutorial 3) to compute the
emission line velocity field in one of the objects. We start by extracting the object
from Tutorial 1 and computing the total spectrum to retrieve the central peak of the
emission line::

 >>> from mpdaf.obj import Cube
 >>> from mpdaf.obj import iter_spe
 >>> import numpy as np
 >>> cube = Cube('Central_DATACUBE_FINAL_11to20_2012-05-16.fits')
 >>> obj1 = cube[:,55-5:55+5,31-10:31+10]
 >>> sp1 = obj1.sum(axis=(1,2))
 >>> ltotal = sp1.gauss_fit(9000.0,9200.0).lpeak

We then create three maps by cloning the continuum image and computing the
line fit parameters spectrum by spectrum on the datacube::

 >>> im1 = obj1.mean(axis=0)
 >>> lfield = im1.clone()
 >>> sfield = im1.clone()
 >>> ffield = im1.clone()

 >>> for sp,pos in iter_spe(obj1, index=True):
 >>>     p,q = pos
 >>>     g = sp.gauss_fit(9000.0,9200.0)
 >>>     lfield[p,q] = (g.lpeak - ltotal) * 300000 / ltotal    # velocity shift from the mean
 >>>     sfield[p,q] = (g.fwhm / 2.35) * 300000 / g.lpeak      # velocity dispersion
 >>>     ffield[p,q] = g.flux                            # line flux

We then plot the resulting velocity field, masking the outliers::

 >>> lfield2=lfield>-200
 >>> lfield3=lfield2<200
 >>> lfield3.plot()

.. image::  _static/cube/vfield.png

Tutorial 6
----------

In this tutorial, we will use the `mpdaf.obj.Cube.loop_ima_multiprocessing <mpdaf.obj.Cube.loop_ima_multiprocessing>` method (Tutorial 4) to fit and remove a background
gradient from the simulated datacube Central_DATACUBE_bkg.fits. We start by loading this cube::

 >>> from mpdaf.obj import Cube
 >>> import numpy as np
 >>> cube = Cube('Central_DATACUBE_bkg.fits.gz')

For each image of the cube, we fit a 2nd order polynomial to the background values
(selected here by simply applying a flux threshold to mask all bright objects). We
do so by doing a chi^2 minimization over the polynomial coefficients using the
numpy recipe np.linalg.lstsq(). for this, we define a function that takes an image as parameter
and returns the background-subtracted image::

 >>> def remove_background_gradient(ima):
 >>>     ksel = np.where(ima.data.data<2.5)
 >>>     pval = ksel[0]
 >>>     qval = ksel[1]
 >>>     zval = ima.data.data[ksel]
 >>>     degree = 2
 >>>     Ap = np.vander(pval,degree)
 >>>     Aq = np.vander(qval,degree)
 >>>     A = np.hstack((Ap,Aq))
 >>>     (coeffs,residuals,rank,sing_vals) = np.linalg.lstsq(A,zval)
 >>>     fp = np.poly1d(coeffs[0:degree])
 >>>     fq = np.poly1d(coeffs[degree:2*degree])
 >>>     X,Y = np.meshgrid(xrange(ima.shape[0]),xrange(ima.shape[1]))
 >>>     ima2 = ima-np.array(map(lambda q,p: fp(p)+fq(q),Y,X))
 >>>     return ima2

We can then create the background-subtracted cube:::

 >>> cube2 = cube.loop_ima_multiprocessing(f=remove_background_gradient)

Finally, we write the output datacube and compare the results for one of the slices::

 >>> cube2.write('Central_DATACUBE_bkgsub.fits.gz')
 >>> cube[1000,:,:].plot(vmin=-1, vmax=4)
 >>> cube2[1000,:,:].plot(vmin=-1, vmax=4)

.. image::  _static/cube/cube1.png
.. image::  _static/cube/cube2.png