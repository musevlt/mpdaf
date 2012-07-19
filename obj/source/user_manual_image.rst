Image object
************

Image, optionally including a variance and a bad pixel mask.


Tutorial
========


Indexing
--------

``Image[i,j]`` returns the corresponding value.

``Image[i1:i2,j1:j2]`` returns the sub-image.

``Image[i,j] = value`` sets value in Image.data[i,j]

``Image[i1:i2,j1:j2] = array`` sets the corresponding part of Image.data.


Operators
---------

+------+------------------------------------------------------------------------+
| <=   | Masks data array where greater than a given value.                     |
+------+------------------------------------------------------------------------+
| <    | Masks data array where greater or equal than a given value.            |
+------+------------------------------------------------------------------------+
| >=   | Masks data array where less than a given value.                        |
+------+------------------------------------------------------------------------+
| >    | Masks data array where less or equal than a given value.               |
+------+------------------------------------------------------------------------+
| \+   | - addition                                                             |
|      | - image1 + number = image2 (image2[i,i] = image1[j,i] + number)        |
|      | - image1 + image2 = image3 (image3[j,i] = image1[j,i] + image2[j,i])   |
|      | - image + cube1 = cube2 (cube2[k,j,i] = cube1[k,j,i] + image[j,i])     |
+------+------------------------------------------------------------------------+	  
| \-   | - substraction                                                         |
|      | - image1 - number = image2 (image2[j,i] = image1[j,i] - number)        |
|      | - image1 - image2 = image3 (image3[j,i] = image1[j,i] - image2[j,i])   |
|      | - image - cube1 = cube2 (cube2[k,j,i] = image[j,i] - cube1[k,j,i])     |
+------+------------------------------------------------------------------------+
| \*   | - multiplication                                                       |
|      | - image1 \* number = image2 (image2[j,i] = image1[j,i] \* number)      |
|      | - image1 \* image2 = image3 (image3[j,i] = image1[j,i] \* image2[j,i]) |
|      | - image \* cube1 = cube2 (cube2[k,j,i] = image[j,i] \* cube1[k,j,i])   |
|      | - image \* spectrum = cube (cube[k,j,i] = image[j,i] \* spectrum[k]    |
+------+------------------------------------------------------------------------+
| /    | - division                                                             |
|      | - image1 / number = image2 (image2[j,i] = image1[j,i] / number)        |
|      | - image1 / image2 = image3 (image3[j,i] = image1[j,i] / image2[j,i])   |
|      | - image / cube1 = cube2 (cube2[k,j,i] = image[j,i] / cube1[k,j,i])     |
+------+------------------------------------------------------------------------+	  
| \*\* | Computes the power exponent of data extensions                         |
+------+------------------------------------------------------------------------+


Reference
=========

:func:`mpdaf.obj.Image.copy` copies Image object in a new one and returns it.

:func:`mpdaf.obj.Image.write` saves Image object in a FITS file.

:func:`mpdaf.obj.Image.info` prints information.

:func:`mpdaf.obj.Image.resize` resizes the image to have a minimum number of masked values.

:func:`mpdaf.obj.Image.sqrt` computes the positive square-root of data extension.

:func:`mpdaf.obj.Image.abs` computes the absolute value of data extension.
        
:func:`mpdaf.obj.Image.get_step` returns the image steps [dDec, dRa].

:func:`mpdaf.obj.Image.get_range` returns [ [dec_min,ra_min], [dec_max,ra_max] ]

:func:`mpdaf.obj.Image.get_start` returns [dec,ra] corresponding to pixel (0,0).

:func:`mpdaf.obj.Image.get_end` returns [dec,ra] corresponding to pixel (-1,-1).

:func:`mpdaf.obj.Image.get_rot` returns the angle of rotation.

:func:`mpdaf.obj.Image.set_wcs` sets the world coordinates.

:func:`mpdaf.obj.Image.set_var` sets the variance array.

:func:`mpdaf.obj.Image.mask` masks values inside/outside the described region.

:func:`mpdaf.obj.Image.unmask` unmasks the image (just invalid data (nan,inf) are masked).

:func:`mpdaf.obj.Image.truncate` truncates the image.

:func:`mpdaf.obj.Image.rotate_wcs` rotates WCS coordinates to new orientation given by theta.

:func:`mpdaf.obj.Image.rotate` rotates the image using spline interpolation.

:func:`mpdaf.obj.Image.sum` returns the sum over the given axis.

:func:`mpdaf.obj.Image.norm` normalizes total flux to value (default 1).

:func:`mpdaf.obj.Image.background` computes the image background.

:func:`mpdaf.obj.Image.peak` finds image peak location.

:func:`mpdaf.obj.Image.fwhm` computes the fwhm center. 

:func:`mpdaf.obj.Image.ee` computes ensquared energy.

:func:`mpdaf.obj.Image.ee_curve` returns Spectrum object containing enclosed energy as function of radius.

:func:`mpdaf.obj.Image.ee_size` computes the size of the square center on (dec,ra) containing the fraction of the energy.

:func:`mpdaf.obj.Image.moments` returns [width_dec, width_ra] first moments of the 2D gaussian.

:func:`mpdaf.obj.Image.gauss_fit` performs Gaussian fit on image.

:func:`mpdaf.obj.Image.moffat_fit` performs moffat fit on image.

:func:`mpdaf.obj.Image.rebin_factor` shrinks the size of the image by factor.

:func:`mpdaf.obj.Image.rebin` rebins the image to a new coordinate system.

:func:`mpdaf.obj.Image.gaussian_filter` applies gaussian filter to the image.

:func:`mpdaf.obj.Image.median_filter` applies median filter to the image.

:func:`mpdaf.obj.Image.maximum_filter` applies maximum filter to the image.

:func:`mpdaf.obj.Image.minimum_filter` applies minimum filter to the image.

:func:`mpdaf.obj.Image.add` adds an other image to the current image.

:func:`mpdaf.obj.Image.segment` segments the image in a number of smaller images.

:func:`mpdaf.obj.Image.add_gaussian_noise` adds Gaussian noise to image.

:func:`mpdaf.obj.Image.add_poisson_noise` adds Poisson noise to image.

:func:`mpdaf.obj.Image.inside` returns True if coord is inside image.

:func:`mpdaf.obj.Image.fftconvolve` convolves the image with an other image using fft.

:func:`mpdaf.obj.Image.fftconvolve_gauss` convolves the image with a 2D gaussian.

:func:`mpdaf.obj.Image.fftconvolve_moffat` convolves the image with a 2D moffat.


Plotting
--------

:func:`mpdaf.obj.Image.plot` plots the image.

:func:`mpdaf.obj.Image.ipos` prints cursor position in interactive mode.

:func:`mpdaf.obj.Image.idist` gets distance and center from 2 cursor positions on the plot.

:func:`mpdaf.obj.Image.istat` computes image statistics from windows defined on the plot.

:func:`mpdaf.obj.Image.ipeak` finds peak location in windows defined on the plot.

:func:`mpdaf.obj.Image.ifwhm` computes fwhm in windows defined on the plot.

:func:`mpdaf.obj.Image.imask` over-plots masked values.
 

Functions to create a new image
===============================
            
:func:`mpdaf.obj.gauss_image` creates a new image from a 2D gaussian.
      
:func:`mpdaf.obj.moffat_image` creates a new image from a 2D Moffat function.

:func:`mpdaf.obj.make_image` interpolates z(x,y) and returns an image.

:func:`mpdaf.obj.composite_image` builds composite image from a list of image and colors.
