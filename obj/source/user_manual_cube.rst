Cube class
**********

This class manages cubes, optionally including a variance and a bad pixel mask.

.. class:: obj.Cube([filename=None, ext = None, notnoise=False, shape=(101,101,101), wcs = None, wave = None, unit=None, data=None, var=None,fscale=1.0])

  :param filename: Possible FITS filename.
  :type filename: string
  :param ext: Number/name of the data extension or numbers/names of the data and variance extensions.
  :type ext: integer or (integer,integer) or string or (string,string)
  :param notnoise: True if the noise Variance cube is not read (if it exists).
  
		   Use notnoise=True to create cube without variance extension.
  :type notnoise: boolean
  :param shape: Lengths of data in Z, Y and X. Python notation is used (nz,ny,nx). (101,101,101) by default.
  :type shape: integer or (integer,integer,integer)
  :param wcs: World coordinates.
  :type wcs: WCS
  :param wave: Wavelength coordinates.
  :type wave: WaveCoord
  :param unit: Possible data unit type. None by default.
  :type unit: string
  :param data: Array containing the pixel values of the cube. None by default.
  :type data: float array
  :param var: Array containing the variance. None by default.
  :type var: float array
  :param fscale: Flux scaling factor (1 by default).
  :type fscale: float
  
Examples::
 
  import numpy as np
  from mpdaf.obj import Cube
  from mpdaf.obj import WCS
  from mpdaf.obj import WaveCoord
  
  wcs1 = WCS(crval=0,cdelt=0.2)
  wcs2 = WCS(crval=0,cdelt=0.2, shape=400)
  wave1 = WaveCoord(cdelt=1.25, crval=4000.0, cunit='Angstrom')
  wave2 = WaveCoord(cdelt=1.25, crval=4000.0, cunit='Angstrom', shape=3000)
  MyData = np.ones((4000,300,300))
  
  cub = Cube(filename="cube.fits",ext=1) # cube from file without vaiance (extension number is 1)
  cub = Cube(filename="cube.fits",ext=(1,2)) # cube from file with vaiance (extension numbers are 1 and 2)
  cub = Cube(shape=(4000,300,300), wcs=wcs1, wave=wave1) # cube 4000x300x300 filled with zeros
  cub = Cube(wcs=wcs1, wave=wave1, data=MyData) # cube filled with MyData
  cub = Cube(shape=(4000,300,300), wcs=wcs1, wave=wave2) # warning: dimensions of wavelength coordinates and data are incompatible
							 # cub.wave = None
  cub = Cube(wcs=wcs1, wave=wave2, data=MyData) # warning: dimensions of wavelength coordinates and data are incompatible
					          # cub.wave = None
  cub = Cube(shape=(4000,300,300), wcs=wcs2, wave=wave1) # warning: dimensions of world coordinates and data are incompatible
							 # cub.wcs = None
  cub = Cube(wcs=wcs2, wave=wave1, data=MyData) # warning: dimensions of world coordinates and data are incompatible
					          # cub.wcs = None
  


Attributes
==========

+---------+-----------------------+-------------------------------------------------------------+
|filename | string                | Possible FITS filename.                                     |
+---------+-----------------------+-------------------------------------------------------------+
| unit    | string                | Possible data unit type.                                    |
+---------+-----------------------+-------------------------------------------------------------+
| cards   | pyfits.CardList       | Possible FITS header instance.                              |
+---------+-----------------------+-------------------------------------------------------------+
| data    | masked array          | Pixel values or masked pixels of the cube.                  |
+---------+-----------------------+-------------------------------------------------------------+
| shape   | array of 3 integers   | Lengths of data in Z, Y and X (python notation: (nz,ny,nx)) |
+---------+-----------------------+-------------------------------------------------------------+
| var     | array                 | Array containing the variance.                              |
+---------+-----------------------+-------------------------------------------------------------+
| fscale  | float                 | Flux scaling factor (1 by default).                         |
+---------+-----------------------+-------------------------------------------------------------+
| wcs     | WCS                   | World coordinates.                                          |
+---------+-----------------------+-------------------------------------------------------------+
| wave    | WaveCoord             | Wavelength coordinates.                                     |
+---------+-----------------------+-------------------------------------------------------------+


Tutorial
========


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



