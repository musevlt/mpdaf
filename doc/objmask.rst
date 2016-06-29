*******************
Operations on masks
*******************

Given an object, the data and variance arrays are accessible via the properties O.data and O.var.
These two arrays are in fact masked arrays (arrays that may have missing or invalid entries, the `numpy.ma <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_ module provides a nearly work-alike replacement for numpy that supports data arrays with masks).
These O.data and 0.var share a single array of boolean masking elements, which is also accessible as a simple boolean array via the 0.mask property. The shared mask can be modified through any of the three properties:

.. ipython::
   :suppress:

   In [4]: import sys

   In [4]: from mpdaf import setup_logging


.. ipython::
   :okwarning:

   In [1]: import numpy as np

   In [1]: import matplotlib.pyplot as plt

   In [2]: from mpdaf.obj import Image

   In [3]: ima = Image('../data/obj/a370II.fits')

   In [3]: ima.data[1000:1500, 800:1800] = np.ma.masked

   # is equivalent to
   In [3]: ima.mask[1000:1500, 800:1800] = True

   @savefig ObjMask1.png width=4in
   In [4]: ima.plot(vmin=1950, vmax=2400, colorbar='v')

   @suppress
   In [4]: plt.close()

The inequality symbols could also be used to mask data array where greater (`<= <mpdaf.obj.DataArray.__le__>`),
greater or equal (`< <mpdaf.obj.DataArray.__lt__>`), less (`>= <mpdaf.obj.DataArray.__ge__>`), less or equal (`> <mpdaf.obj.DataArray.__gt__>`) than a given value:

.. ipython::

  In [1]: ima.unmask()

  In [2]: ima2 = ima > 2000

  In [3]: plt.figure()

  @savefig ObjMask2.png width=4in
  In [4]: ima2.plot(vmin=1950, vmax=2400, colorbar='v')

  @suppress
  In [4]: plt.close()


Note the use of `unmask <mpdaf.obj.DataArray.unmask>` to clear the current mask.

It is also possible to mask pixels corresponding to a selection (`mask_selection <mpdaf.obj.DataArray.mask_selection>`)
or to mask pixels with a variance upper than threshold value (`mask_variance <mpdaf.obj.DataArray.mask_variance>`):

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
