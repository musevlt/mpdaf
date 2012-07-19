Cube object
***********

Cubes, optionally including a variance and a bad pixel mask.


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



