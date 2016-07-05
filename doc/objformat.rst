.. _objformat:

*******************************
Spectrum, Image and Cube format
*******************************

Attributes
----------

Spectrum, Image and Cube objects all have the following items, where
O denotes the name of the object:

+------------------+-----------------------------------------------------------------------------------+
| Component        | Description                                                                       |
+==================+===================================================================================+
| O.filename       | A FITS filename if the data were loaded from a FITS file                          |
+------------------+-----------------------------------------------------------------------------------+
| O.primary_header | A FITS primary header instance                                                    |
+------------------+-----------------------------------------------------------------------------------+
| O.wcs            | World coordinate spatial information (`WCS <mpdaf.obj.WCS>` object)               |
+------------------+-----------------------------------------------------------------------------------+
| O.wave           | World coordinate spectral information  (`WaveCoord <mpdaf.obj.WaveCoord>` object) |
+------------------+-----------------------------------------------------------------------------------+
| O.shape          | An array of the dimensions of the cube                                            |
+------------------+-----------------------------------------------------------------------------------+
| O.data           | A numpy masked array of pixel values                                              |
+------------------+-----------------------------------------------------------------------------------+
| O.data_header    | A FITS data header instance                                                       |
+------------------+-----------------------------------------------------------------------------------+
| O.unit           | The physical units of the data values                                             |
+------------------+-----------------------------------------------------------------------------------+
| O.dtype          | The data-type of the data array (int, float)                                      |
+------------------+-----------------------------------------------------------------------------------+
| O.var            | An optional numpy masked array of pixel variances                                 |
+------------------+-----------------------------------------------------------------------------------+
| O.mask           | An array of the masked state of each pixel                                        |
+------------------+-----------------------------------------------------------------------------------+
| O.ndim           | The number of dimensions in the data, variance and mask arrays                    |
+------------------+-----------------------------------------------------------------------------------+

Masked arrays are arrays that can have missing or invalid entries.  The
`numpy.ma <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_ module
provides a nearly work-alike replacement for numpy that supports data arrays
with masks. See the `DataArray <mpdaf.obj.DataArray>` documentation for more
details.

When an object is constructed from a MUSE FITS file, ``O.data`` will contain the DATA extension,
``O.var`` will contain the STAT extension and ``O.mask`` will contain the DQ extension if it exists.
A DQ extension contains the pixel data quality. By default all bad pixels are masked.
But it is possible for the user to create his mask by using the :ref:`euro3D`.

Indexing
--------

The format of each numpy array follows the indexing used by Python to handle 2D
and 3D arrays. For an MPDAF cube C, the pixel in the bottom-lower-left corner is
C[0,0,0] and pixel C[k,p,q] refers to the horizontal position q, the
vertical position p, and the spectral position k, as follows:

.. figure:: _static/cube/gridcube.jpg
  :align: center

In total, this cube C contains nq pixels in the horizontal direction,
np pixels in the vertical direction and nk channels in the spectral
direction.  In the cube, each numpy masked array has 3 dimensions,
array[k,p,q], where k is the spectral axis, and p and q are the
spatial axes:

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

  # The 3 dimensions of the cube:
  In [2]: cube.shape

  In [2]: cube.data.shape

  In [2]: cube.var.shape

  In [2]: cube.mask.shape

`Cube[k,p,q] <mpdaf.obj.Cube.__getitem__>` returns the value of Cube.data[k,p,q]:

.. ipython::

  In [2]: cube[3659, 8, 28]

Similarly `Cube[k,p,q] = value <mpdaf.obj.Cube.__setitem__>` sets the
value of Cube.data[k,p,q], and `Cube[k1:k2,p1:p2,q1:q2] = array
<mpdaf.obj.Cube.__setitem__>` sets the corresponding subset of
Cube.data.  Finally `Cube[k1:k2,p1:p2,q1:q2] <mpdaf.obj.Cube.__getitem__>`
returns a sub-cube, as demonstrated in the following example:

.. ipython::

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  In [2]: cube.info()

  In [2]: cube[3000:4000,10:20,25:40].info()

Likewise, `Cube[k,:,:] <mpdaf.obj.Cube.__getitem__>` returns an Image, as
demonstrated below:

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

In the Image objects extracted from the cube, `Image[p1:p2,q1:q2]
<mpdaf.obj.Image.__getitem__>` returns a sub-image, `Image[p,q]
<mpdaf.obj.Image.__getitem__>` returns the value of pixel (p,q), `Image[p,q] =
value <mpdaf.obj.Image.__setitem__>` sets value in Image.data[p,q], and
`Image[p1:p2,q1:q2] = array <mpdaf.obj.Image.__setitem__>` sets the
corresponding part of Image.data.


Finally, `Cube[:,p,q] <mpdaf.obj.Cube.__getitem__>` returns a Spectrum:

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

`Cube.get_step <mpdaf.obj.Cube.get_step>`, `Image.get_step <mpdaf.obj.Image.get_step>` and `Spectrum.get_step <mpdaf.obj.Spectrum.get_step>` return the world-coordinate separations between pixels along each axis of a cube, image, or spectrum, respectively:

.. ipython::

  In [1]: cube.get_step(unit_wave=u.nm, unit_wcs=u.deg)

  In [1]: ima1.get_step(unit=u.deg)

  In [1]: spe.get_step(unit=u.angstrom)

`Cube.get_range <mpdaf.obj.Cube.get_range>` returns the range of wavelengths,
declinations and right ascensions in a cube. Similarly, `Image.get_range
<mpdaf.obj.Image.get_range>` returns the range of declinations and right
ascensions in an image, and `Spectrum.get_range <mpdaf.obj.Spectrum.get_range>`
returns the range of wavelengths in a spectrum, as demonstrated below:

.. ipython::

  In [1]: cube.get_range(unit_wave=u.nm, unit_wcs=u.deg)

  In [1]: ima1.get_range(unit=u.deg)

  In [1]: spe.get_range(unit=u.angstrom)

The `get_start <mpdaf.obj.Cube.get_start>` and `get_end
<mpdaf.obj.Cube.get_end>` methods of cube, image and spectrum objects, return
the world-coordinate values of the first and last pixels of each axis:

.. ipython::

  In [1]: print cube.get_start(unit_wave=u.nm, unit_wcs=u.deg), cube.get_end(unit_wave=u.nm, unit_wcs=u.deg)

  In [1]: print ima1.get_start(unit=u.deg), ima2.get_end(unit=u.deg)

  In [1]: print spe.get_start(unit=u.angstrom), spe.get_end(unit=u.angstrom)

Note that when the declination axis is rotated away from the vertical axis of
the image, the coordinates returned by `get_start <mpdaf.obj.Image.get_start>`
and `get_end <mpdaf.obj.Image.get_end>` are not the minimum and maximum
coordinate values within the image, so they differ from the values returned by
`get_range <mpdaf.obj.Image.get_range>`.

`Cube.get_rot <mpdaf.obj.Cube.get_rot>` and `Image.get_rot
<mpdaf.obj.Image.get_rot>` return the rotation angle of the declination axis to
the vertical axis of the images within these objects:

.. ipython::

  In [1]: cube.get_rot(unit=u.deg)

  In [1]: ima1.get_rot(unit=u.rad)


Updated flux and variance values can be assigned directly to the ``O.data`` and
``O.var`` attributes, respectively.  Similarly, elements of the data can be
masked or unmasked by assigning True or False values to the corresponding
elements of the ``O.mask`` attribute.  Changes to the spatial world coordinates
must be performed using the `set_wcs <mpdaf.obj.Cube.set_wcs>` method:

.. ipython::

  In [1]: ima2.data[0:10,0:10] = 0

  In [1]: ima2.mask[0:10,0:10] = True

  In [1]: plt.figure()

  @savefig ObjFormatMaskedIma2.png width=4in
  In [8]: ima2.plot()

 .. ipython::
   :suppress:

   In [4]: plt.close("all")

   In [4]: %reset -f
