.. _objformat:

*******************************
Spectrum, Image and Cube format
*******************************

Attributes
----------

A Spectrum/Image/Cube object O consists of:

+------------------+-----------------------------------------------------------------------------------+
| Component        | Description                                                                       |
+==================+===================================================================================+
| O.filename       | Possible FITS filename                                                            |
+------------------+-----------------------------------------------------------------------------------+
| O.primary_header | FITS primary header instance                                                      |
+------------------+-----------------------------------------------------------------------------------+
| O.wcs            | World coordinate spatial information (`WCS <mpdaf.obj.WCS>` object)               |
+------------------+-----------------------------------------------------------------------------------+
| O.wave           | World coordinate spectral information  (`WaveCoord <mpdaf.obj.WaveCoord>` object) |
+------------------+-----------------------------------------------------------------------------------+
| O.shape          | Array containing the dimension of the object                                      |
+------------------+-----------------------------------------------------------------------------------+
| O.data           | Masked numpy array with data values                                               |
+------------------+-----------------------------------------------------------------------------------+
| O.data_header    | FITS data header instance                                                         |
+------------------+-----------------------------------------------------------------------------------+
| O.unit           | Physical units of the data values                                                 |
+------------------+-----------------------------------------------------------------------------------+
| O.dtype          | Type of the data (integer, float)                                                 |
+------------------+-----------------------------------------------------------------------------------+
| O.var            | (optionally) Masked numpy array with variance values                              |
+------------------+-----------------------------------------------------------------------------------+
| O.mask           | The shared masking array of the data and variance arrays                          |
+------------------+-----------------------------------------------------------------------------------+
| O.ndim           | The number of dimensions in the data and variance arrays                          |
+------------------+-----------------------------------------------------------------------------------+

Masked arrays are arrays that may have missing or invalid entries.
The `numpy.ma <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_ module provides a nearly work-alike replacement for numpy that supports data arrays with masks.

See `DataArray <mpdaf.obj.DataArray>` documentation for more details.

Indexing
--------

The format of each numpy array follows the indexing used by Python to
handle 2D/3D arrays. For an MPDAF cube C, the pixel in the bottom-lower-left corner is
referenced as C[0,0,0] and the pixel C[k,p,q] refers to the horizontal position
q, the vertical position p, and the spectral position k, as follows:

.. figure:: _static/cube/gridcube.jpg
  :align: center

In total, this cube C contains nq pixels in the horizontal direction,
np pixels in the vertical direction and nk channels in the spectral direction.
Each numpy masked array has 3 dimensions: Array[k,p,q] with k the spectral axis, p and q the spatial axes:

.. ipython::
   :suppress:
   
   In [4]: import sys
   
   In [4]: from mpdaf import setup_logging
   
   In [2]: import matplotlib.pyplot as plt

.. ipython::
  :okwarning:

  @suppress
  In [5]: setup_logging(stream=sys.stdout)
  
  In [2]: from mpdaf.obj import Cube
  
  # data array is read from the file (extension number 0)
  In [1]: cube = Cube(filename='../data/sdetect/minicube.fits')
  
  In [2]: cube.shape
  
  In [2]: cube.data.shape
  
  In [2]: cube.var.shape
  
  In [2]: cube.mask.shape

`Cube[k,p,q] <mpdaf.obj.Cube.__getitem__>` returns the corresponding value:

.. ipython::

  In [2]: cube[3659, 8, 28]
  
In the same way `Cube[k,p,q] = value <mpdaf.obj.Cube.__setitem__>` sets value in Cube.data[k,p,q] and `Cube[k1:k2,p1:p2,q1:q2] = array <mpdaf.obj.Cube.__setitem__>` sets the corresponding part of Cube.data.

`Cube[k1:k2,p1:p2,q1:q2] <mpdaf.obj.Cube.__getitem__>` returns the sub-cube:

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  In [2]: cube.info()
  
  In [2]: cube[3000:4000,10:20,25:40].info()

`Cube[k,:,:] <mpdaf.obj.Cube.__getitem__>` returns an Image:

.. ipython::
  
  In [3]: ima1 = cube[1000, :, :]
  
  In [4]: plt.figure()
  
  @savefig ObjFormatIma1.png width=2.3in
  In [5]: ima1.plot(colorbar='v', title = '$\lambda$ = %.1f (%s)' %(cube.wave.coord(1000), cube.wave.unit))
  
  In [6]: ima2 = cube[3000, :, :]
  
  In [7]: plt.figure()
  
  @savefig ObjFormatIma2.png width=2.3in
  In [8]: ima2.plot(colorbar='v', title = '$\lambda$ = %.1f (%s)' %(cube.wave.coord(3000), cube.wave.unit))
  
  In [7]: plt.figure()
  
  @savefig ObjFormatZommIma2.png width=2.3in
  In [8]: ima2[5:25, 15:35].plot(colorbar='v',title = 'Zoom $\lambda$ = %.1f (%s)' %(cube.wave.coord(3000), cube.wave.unit))

We can see that `Image[p1:p2,q1:q2] <mpdaf.obj.Image.__getitem__>` returns a sub-image.
In the same ways as cube indexing, `Image[p,q] <mpdaf.obj.Image.__getitem__>` returns the value of pixel (p,q),
`Image[p,q] = value <mpdaf.obj.Image.__setitem__>` sets value in Image.data[p,q],
and `Image[p1:p2,q1:q2] = array <mpdaf.obj.Image.__setitem__>` sets the corresponding part of Image.data.


Then, `Cube[:,p,q] <mpdaf.obj.Cube.__getitem__>` returns a Spectrum:

.. ipython::

  In [5]: spe = cube[:, 8, 28]
  
  In [5]: import astropy.units as u
  
  In [5]: from mpdaf.obj import deg2sexa
  
  In [5]: coord_sky = cube.wcs.pix2sky([8, 28], unit=u.deg)
  
  In [6]: dec, ra = deg2sexa(coord_sky)[0]
  
  In [6]: plt.figure()
  
  @savefig ObjFormatSpe.png width=3.5in
  In [8]: spe.plot(title = 'Spectrum ra=%s dec=%s' %(ra, dec))
  
  In [6]: plt.figure()
  
  @savefig ObjFormatZoomSpe.png width=3.5in
  In [8]: spe[1640:2440].plot(title = 'Zoom Spectrum ra=%s dec=%s' %(ra, dec))

  
Getters and setters
-------------------

`Cube.get_step <mpdaf.obj.Cube.get_step>`, `Image.get_step <mpdaf.obj.Image.get_step>` and `Spectrum.get_step <mpdaf.obj.Spectrum.get_step>`  returns the cube/image/spectrum steps:

.. ipython::

  In [1]: cube.get_step(unit_wave=u.nm, unit_wcs=u.deg)

  In [1]: ima1.get_step(unit=u.deg)
  
  In [1]: spe.get_step(unit=u.angstrom)
  
`Cube.get_range <mpdaf.obj.Cube.get_range>`, `Image.get_range <mpdaf.obj.Image.get_range>` and `Spectrum.get_range <mpdaf.obj.Spectrum.get_range>` return the range of wavelengths, declinations and right ascensions:

.. ipython::

  In [1]: cube.get_range(unit_wave=u.nm, unit_wcs=u.deg)

  In [1]: ima1.get_range(unit=u.deg)
  
  In [1]: spe.get_range(unit=u.angstrom)
  
`get_start <mpdaf.obj.Cube.get_start>` and `get_end <mpdaf.obj.Cube.get_end>` return coordinates values corresponding to pixels 0 and -1 in all directions:

.. ipython::

  In [1]: print cube.get_start(unit_wave=u.nm, unit_wcs=u.deg), cube.get_end(unit_wave=u.nm, unit_wcs=u.deg)

  In [1]: print ima1.get_start(unit=u.deg), ima2.get_end(unit=u.deg)
  
  In [1]: print spe.get_start(unit=u.angstrom), spe.get_end(unit=u.angstrom)
  
Note that when the rotation angle of the image on the sky is not zero, `get_range <mpdaf.obj.Image.get_range>` is not at the corners of the image
and is different to `get_start <mpdaf.obj.Image.get_start>` and `get_end <mpdaf.obj.Image.get_end>`.

`Cube.get_rot <mpdaf.obj.Cube.get_rot>` and `Image.get_rot <mpdaf.obj.Image.get_rot>` return the rotation angle:

.. ipython::

  In [1]: cube.get_rot(unit=u.deg)

  In [1]: ima1.get_rot(unit=u.rad)
  

Set a flux/variance value is done directly on the ``O.data`` and ``O.var`` attributes.
In the same way, mask/unmasked a part of the object could be done by changing the value of the ``O.mask``.
However, the world coordinates must be set by using `set_wcs <mpdaf.obj.Cube.set_wcs>` method:

.. ipython::

  In [1]: ima2.data[0:10,0:10] = 0

  In [1]: ima2.mask[0:10,0:10] = True
  
  In [1]: plt.figure()
  
  @savefig ObjFormatMaskedIma2.png width=4in
  In [8]: ima2.plot()
  
 .. ipython::
   :suppress:
   
   In [4]: plt.close("all")