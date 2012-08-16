Cube object
***********

.. warning::

   Draft version, still incomplete

Cube python object can handle datacubes which have a regular grid format in both spatail and spectral axis.
Variance information can also be taken into account as well as bad pixels. 
Cube object can be read and written to disk as a multi-extension FITS file.

Object is build as a set or numpy masked arrays and world coordinate information. A number of transformation
have been developed  as object properties. Note that virtually all numpy and scipy functions are available.

Cube object format
==================

A cube object O consist of:

+------------+--------------------------------------------------------+
| Component  | Description                                            |
+============+========================================================+
| O.wcs      | world coordinate spatail information                   |
+------------+--------------------------------------------------------+
| O.wave     | world coordinate spectral information                  |
+------------+--------------------------------------------------------+
| O.data     | masked numpy array with data values                    |
+------------+--------------------------------------------------------+
| O.var      | (optionally) masked numpy array with variance values   |
+------------+--------------------------------------------------------+

Each numpy masked array has 3 dimensions: Array[k,l,m] with k the spectral axis, l and m the spatail axes

Tutorials
=========

Tutorial 1
----------

In this tutorial we learn how to play with an existing datacube, extract a small cube centered around an object and compute its spectrum.

We read the datacube from disk and display basic information::

 >>> from mpdaf.obj import Cube
 >>> cube = Cube('Central_DATACUBE_FINAL_11to20_2012-05-16.fits')
 >>> cube.info()
 101 X 101 X 3601 cube (Central_DATACUBE_FINAL_11to20_2012-05-16.fits)
 .data(3601,101,101) (erg/s/cm**2/Angstrom) fscale=1e-20, .var(3601,101,101)
 center:(-30:00:01.35,01:20:00.137) size in arcsec:(20.200,20.200) step in arcsec:(0.200,0.200) rot:0.0
 wavelength: min:4800.00 max:9300.00 step:1.25 Angstrom

The info directive gives us already some important informations:

- The cube format 101 X 101 X 3601 has 101 x 101 spatial pixels and 3601 spectral pixels
- In addition to the data extension (.data(3601,101,101) a variance extension is also present (.var(3601,101,101)
- The flux data unit is erg/s/cm**2/Angstrom and the scale factor is 10**-20
- The center of the field of view is at RA: -30° 0' 1.35" and DEC: 1°20'0.137" and its size is 20.2x20.2 arcsec**2. The spaxel dimension is 0.2x0.2 arcsec**2. The rotation angle is 0° with respect to the North.
- The wavelength range is 4800-9300 Angstrom with a step of 1.25 Angstroem

Let's compute the reconstructed white light image and display it::

 >>> ima = cube.sum(axis=0)
 >>> ima.plot(scale='arcsinh')

.. figure::  user_manual_cube_images/recima1.png
   :align:   center

We extract the cube corresponding to the object centered at x=31 y=55 spaxels::

 >>> obj1 = cube[:,55-5:55+5,31-10:31+10]
 >>> ima1 = obj1.mean(axis=0)
 >>> ima1.plot()

.. figure::  user_manual_cube_images/recima2.png
   :align:   center

Let's now compute the total spectrum of the object::

 >>> sp1 = obj1.sum(axis=(1,2))
 >>> sp1.plot()

.. figure::  user_manual_cube_images/spec1.png
   :align:   center

Tutorial 2
----------

In this second tutorial we create the continuum subtracted datacube of the previously extracted object.

We start by building a function which fit the continuum of a spectra::

 >>> from scipy import polyfit, polyval
 >>> def fitcont(spec, cont, deg):
 >>>       """ fit the continuum with a polynome
 >>>       spec: input spectrum
 >>>       cont: output spectrum
 >>>       deg: polynomila degre
 >>>       """
 >>>       pol = polyfit(range(spec.shape), spec.data, deg)
 >>>       cont.data = polyval(pol, range(spec.shape))
 >>>       return

Let's try it on the total spectrum sp1 computed in the previous tutorial::

 >>> cont1 = sp1.copy()
 >>> fitcont(sp1, cont1, 5)
 >>> sp1.plot()
 >>> cont1.plot()

.. figure::  user_manual_cube_images/spec2.png
   :align:   center

Let's try also on a single spectrum at the edge of the galaxy::

 >>> sp1 = obj1[:,5,2]
 >>> cont1 = sp1.copy()
 >>> fitcont(sp1, cont1, 5)
 >>> sp1.plot()
 >>> cont1.plot()

.. figure::  user_manual_cube_images/spec3.png
   :align:   center

Fine, now let's do this for all spectrum of the input datacube.
We start by doing a copy of the input datacube (obj1). Note that we set the variance of the
continuum datacube to None. We then loop over all spectra. At each spatial location
we extract the corresponding spectra (sp), we create a copy (co) and use our function
fitcont to get the polynomial approximation of the continuum. Then we save the continumm values
(co.data) into the corresponding continuum datacube::

 >>> cont1 = obj1.copy()
 >>> cont1.var = None
 >>> m,n = obj1.shape[1:]
 >>> for i in range(m):
 >>>        for j in range(n):
 >>>                sp = obj1[:,i,j]
 >>>                co = sp.copy()
 >>>                fitcont(sp, co, 5)
 >>>                cont1[:,i,j] = co.data

Let's check the result and display the continuum reconstructed image::

 >>> rec2 = cont1.sum(axis=0)
 >>> rec2.plot(scale='arcsinh')

.. figure::  user_manual_cube_images/recima4.png
   :align:   center

We can also compute the line emission datacube::

 >>> line1 = obj1 - cont1
 >>> line1.sum(axis=0).plot('arcsinh')

.. figure::  user_manual_cube_images/recima5.png
   :align:   center


Tutorial 3
----------

In this tutorial we are going to process our datacube in spatial direction. We consider the datacube as a collection of
monochromatic images and we process each of them. For each monochromatic image we apply a convolution by a gaussian kernel.


Indexing
--------

``Cube[k,i,j]`` returns the corresponding value.

``Cube[k1:k2,i1:i2,j1:j2]`` returns the sub-cube.

``Cube[k,:,:]`` returns an Image.

``Cube[:,i,j]`` returns a Spectrum.

``Cube[k,i,j] = value`` sets value in Cube.data[k,i,j]

``Cube[k1:k2,i1:i2,j1:j2] = array`` sets the corresponding part of Cube.data.


Operators
---------

+------+--------------------------------------------------------------------------+
| <=   | Masks data array where greater than a given value.                       |
+------+--------------------------------------------------------------------------+
| <    | Masks data array where greater or equal than a given value.              |
+------+--------------------------------------------------------------------------+
| >=   | Masks data array where less than a given value.                          |
+------+--------------------------------------------------------------------------+
| >    | Masks data array where less or equal than a given value.                 |
+------+--------------------------------------------------------------------------+
| \+   | - addition                                                               |
|      | - cube1 + number = cube2 (cube2[k,j,i] = cube1[k,j,i] + number)          |
|      | - cube1 + cube2 = cube3 (cube3[k,j,i] = cube1[k,j,i] + cube2[k,j,i])     |
|      | - cube1 + image = cube2 (cube2[k,j,i] = cube1[k,j,i] + image[j,i])       |
|      | - cube1 + spectrum = cube2 (cube2[k,j,i] = cube1[k,j,i] + spectrum[k])   |
+------+--------------------------------------------------------------------------+	  
| \-   | - substraction                                                           |
|      | - cube1 - number = cube2 (cube2[k,j,i] = cube1[k,j,i] - number)          |
|      | - cube1 - cube2 = cube3 (cube3[k,j,i] = cube1[k,j,i] - cube2[k,j,i])     |
|      | - cube1 - image = cube2 (cube2[k,j,i] = cube1[k,j,i] - image[j,i])       |
|      | - cube1 - spectrum = cube2 (cube2[k,j,i] = cube1[k,j,i] - spectrum[k])   |
+------+--------------------------------------------------------------------------+
| \*   | - multiplication                                                         |
|      | - cube1 \* number = cube2 (cube2[k,j,i] = cube1[k,j,i] \* number)        |
|      | - cube1 \* cube2 = cube3 (cube3[k,j,i] = cube1[k,j,i] \* cube2[k,j,i])   |
|      | - cube1 \* image = cube2 (cube2[k,j,i] = cube1[k,j,i] \* image[j,i])     |
|      | - cube1 \* spectrum = cube2 (cube2[k,j,i] = cube1[k,j,i] \* spectrum[k]) |
+------+--------------------------------------------------------------------------+
| /    | - division                                                               |
|      | - cube1 / number = cube2 (cube2[k,j,i] = cube1[k,j,i] / number)          |
|      | - cube1 / cube2 = cube3 (cube3[k,j,i] = cube1[k,j,i] / cube2[k,j,i])     |
|      | - cube1 / image = cube2 (cube2[k,j,i] = cube1[k,j,i] / image[j,i])       |
|      | - cube1 / spectrum = cube2 (cube2[k,j,i] = cube1[k,j,i] / spectrum[k])   |
+------+--------------------------------------------------------------------------+	  
| \*\* | Computes the power exponent of data extensions                           |
+------+--------------------------------------------------------------------------+


Reference
=========


:func:`mpdaf.obj.Cube.copy` copies Cube object in a new one and returns it.

:func:`mpdaf.obj.Cube.info` prints information.

:func:`mpdaf.obj.Cube.write` saves the Cube in a FITS file.

:func:`mpdaf.obj.Cube.resize` resizes the cube to have a minimum number of masked values.

:func:`mpdaf.obj.Cube.sqrt` computes the positive square-root of data extension.

:func:`mpdaf.obj.Cube.abs` computes the absolute value of data extension.

:func:`mpdaf.obj.Cube.get_lambda` returns the sub-cube corresponding to a wavelength range.

:func:`mpdaf.obj.Cube.get_step` returns the cube steps.

:func:`mpdaf.obj.Cube.get_range` returns minimum and maximum values of cube coordiantes.

:func:`mpdaf.obj.Cube.get_start` returns coordinates values corresponding to pixel (0,0,0).

:func:`mpdaf.obj.Cube.get_end` returns coordinates values corresponding to pixel (-1,-1,-1).

:func:`mpdaf.obj.Cube.get_rot` returns the rotation angle.

:func:`mpdaf.obj.Cube.set_wcs` sets the world coordinates.

:func:`mpdaf.obj.Cube.set_var` sets the variance array.

:func:`mpdaf.obj.Cube.sum` returns the sum over the given axis.

:func:`mpdaf.obj.Cube.mean` returns the mean over the given axis.



