
*******************
Operations on masks
*******************

Given a Spectrum, Image or Cube object, O, the pixel values and variances of
this object can be accessed via the O.data and O.var arrays.  These are masked
arrays, which are arrays that can have missing or invalid entries. Masked
arrays, which are provided by the `numpy.ma
<http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_ module, provide a
nearly work-alike replacement for numpy arrays.  The O.data and O.var arrays
share a single array of boolean masking elements. This is a simple boolean
array, and it can be accessed directly via the O.mask property. The elements of
this array can be modified implicitly or explicitly via operations on the data,
variance or mask arrays. For example:

.. ipython::
   :okwarning:

   In [1]: import numpy as np

   In [1]: import matplotlib.pyplot as plt

   In [2]: from mpdaf.obj import Image

   In [3]: ima = Image('obj/a370II.fits')

   In [3]: ima.data[1000:1500, 800:1800] = np.ma.masked

   # is equivalent to
   In [3]: ima.mask[1000:1500, 800:1800] = True

   @savefig ObjMask1.png width=4in
   In [4]: ima.plot(vmin=1950, vmax=2400, colorbar='v')

   @suppress
   In [4]: plt.close()

The comparison operators can also be used to mask all data that are not less
than or equal (`<= <mpdaf.obj.DataArray.__le__>`) to a value, are not less than
(`< <mpdaf.obj.DataArray.__lt__>`) a value, are not greater than or equal (`>=
<mpdaf.obj.DataArray.__ge__>`) to a value, or that are not greater than (`>
<mpdaf.obj.DataArray.__gt__>`) a given value. In the following example, the
``<`` operator is used to select data that have values over 2000, and mask
everything below this threshold. In the plot of the resulting image, the masked
values below 2000 are drawn in white. These values would otherwise be dark blue
if they weren't masked.

.. ipython::

  In [1]: ima.unmask()

  In [2]: ima2 = ima > 2000

  In [3]: plt.figure()

  @savefig ObjMask2.png width=4in
  In [4]: ima2.plot(vmin=1950, vmax=2400, colorbar='v')

  @suppress
  In [4]: plt.close()


Note the use of the `unmask <mpdaf.obj.DataArray.unmask>` method to clear the
current mask.

It is also possible to mask pixels that correspond to a selection, by using the
`mask_selection <mpdaf.obj.DataArray.mask_selection>` method, or to mask pixels
whose variance exceeds a given threshold value, by using the `mask_variance
<mpdaf.obj.DataArray.mask_variance>` method:

.. ipython::

  In [1]: ima.unmask()

  In [2]: ksel = np.where(ima.data < 2000)

  In [3]: ima.mask_selection(ksel)

  In [3]: plt.figure()

  @savefig ObjMask3.png width=4in
  In [4]: ima.plot(vmin=1950, vmax=2400, colorbar='v')

On Image or Cube objects, there are additional methods to mask inside or outside:

 - a circular or rectangular region (`Image.mask_region <mpdaf.obj.Image.mask_region>` and `Cube.mask_region <mpdaf.obj.Cube.mask_region>`)

 - an elliptical region (`Image.mask_ellipse <mpdaf.obj.Image.mask_ellipse>` and `Cube.mask_ellipse <mpdaf.obj.Cube.mask_ellipse>`)

 - a polygonal region (`Image.mask_polygon <mpdaf.obj.Image.mask_polygon>` and `Cube.mask_polygon <mpdaf.obj.Cube.mask_polygon>`)

For example:

.. ipython::

  In [1]: ima.unmask()

  In [2]: ima.mask_region(center=[800.,600.], radius=500., unit_center=None, unit_radius=None, inside=False)

  In [3]: plt.figure()

  @savefig ObjMask4.png width=4in
  In [4]: ima.plot(vmin=1950, vmax=2400, colorbar='v')

.. ipython::
   :suppress:

   In [4]: plt.close("all")

   In [4]: %reset -f
