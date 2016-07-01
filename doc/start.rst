***************
Getting Started
***************

.. ipython::
   :suppress:

   In [4]: import sys

   In [4]: from mpdaf import setup_logging

Importing MPDAF
---------------

MPDAF is divided into sub-packages, each of which is composed of several
classes.  The following example shows how to import the `~mpdaf.obj.Cube` and
`~mpdaf.drs.PixTable` classes:

.. ipython::

   In [1]: from mpdaf.obj import Cube

   In [2]: from mpdaf.drs import PixTable


All of the examples in the MPDAF web pages are shown being typed into
an interactive IPython shell. This shell is the origin of the prompts
like ``In [1]:`` in the above example. The examples can also be entered
in other shells, such as the native Python shell.

Loading your first MUSE datacube
--------------------------------

MUSE datacubes are generally loaded from FITS files. In these files
the fluxes and variances are stored in separate FITS extensions. For
example:

.. ipython::
  :okwarning:

  @suppress
  In [5]: setup_logging(stream=sys.stdout)

  # data and variance arrays are read from DATA and STAT extensions of the file
  In [2]: cube = Cube('../data/obj/CUBE.fits')

  In [10]: cube.info()


The listed dimensions of the cube, 1595 x 10 x 20, indicate that the
cube has 1595 spectral pixels and 10 x 20 spatial pixels.  The order
in which these dimensions are listed, follows the indexing conventions
used by Python to handle 3D arrays (see :ref:`objformat` for more
information).

Let's compute the reconstructed white-light image and display it. The
white-light image is obtained by summing each spatial pixel of the
cube along the wavelength axis. This converts the 3D cube into a 2D
image.

.. ipython::

  In [1]: ima = cube.sum(axis=0)

  In [1]: type(ima)

  In [2]: plt.figure()

  @savefig Cube1.png width=4in
  In [3]: ima.plot(scale='arcsinh', colorbar='v')


Let's now compute the overall spectrum of the cube by taking the cube
and summing along the X and Y axes of the image plane. This yields the
total flux per spectral pixel.

.. ipython::

  In [1]: sp = cube.sum(axis=(1,2))

  In [1]: type(sp)

  In [2]: plt.figure()

  @savefig Cube2.png width=4in
  In [3]: sp.plot()


Online Help
-----------

Because different sub-packages have very different functionality,
further suggestions for getting started are provided in the online
documentation of these sub-packages. For example, click on :ref:`cube`,
:ref:`image`, or :ref:`spectrum` for help with the 3 main classes of
the ``mpdaf.obj`` package.

Alternatively, if you use the IPython interactive python shell, then you can
look at the docstrings of classes, objects and functions by following them with
the magic ``?`` of IPython. Examples of this are shown below. A more general
way to see these docstrings, which works in all Python shells, is to use the
built-in ``help()`` function:

.. ipython::

   In [7]: Cube.sum?

.. ipython::

   In [2]: help(ima.plot)

.. ipython::
   :suppress:

   In [4]: cube = None ; ima = None ; sp = None

   In [4]: plt.close("all")
