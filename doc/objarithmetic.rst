**********
Arithmetic
**********

Operations along one or more axes
---------------------------------

In Cube objects, the `~mpdaf.obj.Cube.sum`, `~mpdaf.obj.Cube.mean` and
`~mpdaf.obj.Cube.median` methods return the sum, mean and median values along
a given axis or axes of the cube:

 - If ``axis=0``, the operation is performed along the wavelength axis and an
   `~mpdaf.obj.Image` is returned,

 - If ``axis=(1, 2)``, the operation is performed over the two spatial axes
   and a `~mpdaf.obj.Spectrum` is returned.

.. ipython::
  :okwarning:

  In [1]: import matplotlib.pyplot as plt

  In [1]: from mpdaf.obj import Cube

  In [2]: cube = Cube('sdetect/minicube.fits')

  In [3]: cube.info()

  In [4]: spe = cube.mean(axis=(1,2))

  In [6]: spe.info()

  @savefig Obj_arithm1.png width=3.5in
  In [5]: spe.plot()

  # white image
  In [7]: ima = cube.sum(axis=0)

  In [9]: ima.info()

  In [8]: plt.figure()

  @savefig Obj_arithm2.png width=3.5in
  In [10]: ima.plot()

  In [13]: spe.sum(7100,7200)

  In [21]: spe.integrate(7100, 7200)

  @suppress
  In [1]: cube = None


The `Spectrum.mean <mpdaf.obj.Spectrum.mean>` and `Spectrum.sum
<mpdaf.obj.Spectrum.sum>` methods compute the mean and total flux
value over a given wavelength range.  Similarly, `Spectrum.integrate
<mpdaf.obj.Spectrum.integrate>` integrates the flux value over a given
wavelength range.


Arithmetic Operations between objects
-------------------------------------

Arithmetic operations can be performed between MPDAF objects and
scalar numbers, numpy arrays, masked arrays or other MPDAF
objects. When an operation is performed between two MPDAF objects or
between an MPDAF object and an array, their dimensions must either
match, or be broadcastable to the same dimensions via the usual numpy
rules. The broadcasting rules make it possible to perform arithmetic
operations between a `~mpdaf.obj.Cube` and an `~mpdaf.obj.Image` or a
`~mpdaf.obj.Spectrum`, assuming that they have compatible spatial and
spectral world-coordinates. The following demonstration shows a cube
being multiplied by an image that has the same spatial dimensions and
world-coordinates as the cube, and also being divided by a spectrum
that has the same spectral dimensions and wavelengths as the spectral
axis of the cube.

.. ipython::
  :okwarning  :
  :verbatim:

  In [16]: cube2 = cube * ima / spe

  In [17]: cube2.info()
  [INFO] 3681 x 40 x 40 Cube (sdetect/minicube.fits)
  [INFO] .data(3681 x 40 x 40) (1e-20 erg / (Angstrom cm2 s)), .var(3681 x 40 x 40)
  [INFO] center:(10:27:56.3962,04:13:25.3588) size in arcsec:(8.000,8.000) step in arcsec:(0.200,0.200) rot:-0.0 deg
  [INFO] wavelength: min:4749.89 max:9349.89 step:1.25 Angstrom



Generic object arithmetic:
--------------------------

Cube, Image and Spectrum objects are all derived from a base class
called `~mpdaf.obj.DataArray`. This class implements a couple of
arithmetic functions that operate on the data and variance arrays of
these objects:

 - `~mpdaf.obj.DataArray.sqrt` returns a new object with positive data square-rooted and negative data masked.

 - `~mpdaf.obj.DataArray.abs` returns a new object containing the absolute values of the data.

.. ipython::
  :okwarning:

  In [18]: ima2 = ima.sqrt()

  @savefig Obj_arithm3.png width=3.5in
  In [10]: ima2.plot()

.. ipython::
   :suppress:

   In [4]: plt.close("all")

   In [4]: %reset -f
