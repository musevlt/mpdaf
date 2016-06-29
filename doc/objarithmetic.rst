**********
Arithmetic
**********

Operations over a given axis
----------------------------

On Cube object, `sum <mpdaf.obj.Cube.sum>`, `mean <mpdaf.obj.Cube.mean>`,  `median <mpdaf.obj.Cube.median>` return the sum/mean/median over the given axis:

 - if ``axis=0``, the operation is done over the wavelength axe and an `Image <mpdaf.obj.Image>` is returned,

 - if ``axis=(1, 2)``, the operation is done over the spatial axis and a `Spectrum <mpdaf.obj.Spectrum>` is returned.

.. ipython::
   :suppress:

   In [4]: import sys

   In [4]: from mpdaf import setup_logging

.. ipython::
  :okwarning:

  @suppress
  In [1]: setup_logging(stream=sys.stdout)

  In [1]: import matplotlib.pyplot as plt

  In [1]: from mpdaf.obj import Cube

  In [2]: cube = Cube('../data/sdetect/minicube.fits')

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


`Spectrum.mean <mpdaf.obj.Spectrum.mean>` and `Spectrum.sum <mpdaf.obj.Spectrum.sum>` computes the mean/total flux value over a wavelength range.
`Spectrum.integrate <mpdaf.obj.Spectrum.integrate>` integrates the flux value over a wavelength range.


Operations with an other object
-------------------------------

Operation can be performed with a scalar number, a Numpy ndarray or masked array, or a Mpdaf object.
The dimensions must be equal, or, if ``self`` and ``operand`` have compatible shapes, they will be broadcasted together.
So it is possible to perfom an operation between a `~mpdaf.obj.Cube` and an a `~mpdaf.obj.Image` or a `~mpdaf.obj.Spectrum`.
For Mpdaf objects, they must also have compatible coordinates (world and wavelength):

.. ipython::
  :okwarning  :
  :verbatim:

  In [16]: cube2 = cube * ima / spe

  In [17]: cube2.info()
  [INFO] 3681 x 40 x 40 Cube (../data/sdetect/minicube.fits)
  [INFO] .data(3681 x 40 x 40) (1e-20 erg / (Angstrom cm2 s)), .var(3681 x 40 x 40)
  [INFO] center:(10:27:56.3962,04:13:25.3588) size in arcsec:(8.000,8.000) step in arcsec:(0.200,0.200) rot:-0.0 deg
  [INFO] wavelength: min:4749.89 max:9349.89 step:1.25 Angstrom



Operations on the data extension
--------------------------------

 - `sqrt <mpdaf.obj.DataArray.sqrt>` returns a new object with positive data square-rooted and negative data masked.

 - `abs <mpdaf.obj.DataArray.abs>` returns a new object with the absolute value of the data.

.. ipython::
  :okwarning:

  In [18]: ima2 = ima.sqrt()

  @savefig Obj_arithm3.png width=3.5in
  In [10]: ima2.plot()

.. ipython::
   :suppress:

   In [4]: plt.close("all")

   In [4]: %reset -f
