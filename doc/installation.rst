************
Installation
************

Requirements
============

MPDAF has the following strict requirements:

- Python_ version 2.7 or 3.3+
- Numpy_ version 1.8 or above
- Scipy_ version 0.14 or above
- Matplotlib_ version 1.4 or above
- Astropy_ version 1.0 or above
- Six_

Several additional packages can be installed for optional features:

- Pytest_, to run the unit tests.
- Numexpr_, to optimize some computations with pixtables.
- fitsio_, a Python wrapper for cfitsio, used in `~mpdaf.obj.CubeMosaic`.
- adjustText_, for catalogs plotting.
- `pkg-config`_, helper tool used when compiling C libraries.
- CFITSIO_, C library for reading and writing FITS files.
- C OpenMP_ library, to get parallelization.
- SExtractor_ for several methods of `~mpdaf.sdetect.Source` and for
  :doc:`muselet`.

Installing with pip
===================

MPDAF can be installed with pip::

    pip install mpdaf

.. note::

   If binary wheels are not available for your OS/Python version, you will need
   a C compiler (e.g. gcc or clang) to be installed for the installation to
   succeed (see below).

   If you want to use specific compilation options (compiler, OpenMP, cfitsio
   path), you can do so by using ``pip install --no-binary mpdaf mpdaf`` and
   prepending environment variables (CC, CFLAGS, etc.).

MPDAF can also be installed with extra dependencies (Numexpr, fitsio) with::

    pip install mpdaf[all]

Installing with conda
=====================

An old version of MPDAF can be installed with the OpenAstronomy_ channel for
conda_, but this package is not maintained currently. It has to be moved to
conda-forge, which has not been done yet. Instead we now produce binary wheels,
which achieve the same goal.

.. This will install a compiled version of MPDAF, with CFITSIO_ and the other
.. dependencies.

Building from source
====================

C extensions
------------

MPDAF contains a few C extensions that must be built during the installation,
and these require optional dependencies:

- The first extension needs ``pkg-config``, to find the correct compiler and
  library flags, and CFITSIO_. If not available, the extension is not
  built, and a few things will not work (`~mpdaf.obj.CubeList`, and several
  PixTable methods: `~mpdaf.drs.PixTable.sky_ref` and
  `~mpdaf.drs.PixTable.selfcalibrate`).

  This extension can also use OpenMP_ if available.  Note that on Mac OS,
  OpenMP is deactivated by default and not supported yet.

  To force the use of OpenMP, the ``USEOPENMP`` environment variable can be set
  to 1::

      USEOPENMP=1 CC=<local path of gcc> pip install --no-binary mpdaf mpdaf

- The second extension is used for `~mpdaf.obj.CubeMosaic` and uses Cython, but
  it is only required for the development version. The distributed package
  includes directly the C files.

Tips for Mac OS users
---------------------

- First, XCode is needed to get some developer tools (compiler, ...). On
  recent Mac OS versions, this can be done with ``$ xcode-select --install``.

- A great package manager can be used to install packages like cfitsio or
  pkg-config: `Homebrew <http://brew.sh/>`_. Then, ``brew install cfitsio
  pkgconfig``.

- It is also possible to install and use gcc to compile MPDAF with OpenMP
  support (``brew install gcc --without-multilib``). Otherwise clang is used.

- `Anaconda <http://continuum.io/downloads>`_ is a great scientific python
  distribution, it comes with up-to-date and pre-compiled versions of Numpy,
  Scipy, Astropy and more.


Unit tests
==========

To run the unit tests, you need to install the Pytest_ package, then run::

    $ python setup.py test

To run the unit tests on an installed version of MPDAF::

    $ pytest $(dirname $(python -c 'import mpdaf; print(mpdaf.__file__)'))


.. _Python: http://python.org/
.. _Numpy: http://www.numpy.org/
.. _Scipy: http://www.scipy.org/
.. _Matplotlib: http://matplotlib.org/
.. _Astropy: http://www.astropy.org/
.. _Six: https://pypi.python.org/pypi/six
.. _Pytest: http://pytest.org/
.. _Numexpr: http://pypi.python.org/pypi/numexpr
.. _fitsio: https://pypi.python.org/pypi/fitsio
.. _pkg-config: https://pkgconfig.freedesktop.org/
.. _CFITSIO: http://heasarc.gsfc.nasa.gov/fitsio/
.. _OpenMP: http://openmp.org
.. _SExtractor: http://www.astromatic.net/software/sextractor
.. _OpenAstronomy: https://anaconda.org/openastronomy
.. _conda: http://conda.pydata.org/docs/
.. _adjustText: https://github.com/Phlya/adjustText
