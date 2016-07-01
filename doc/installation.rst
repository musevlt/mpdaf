************
Installation
************

Requirements
============

MPDAF has the following strict requirements:

- `Python <http://python.org/>`_ version 2.7 or 3.3+
- `Numpy`_ version 1.8 or above
- `Scipy <http://www.scipy.org/>`_ version 0.14 or above
- `Matplotlib <http://matplotlib.org/>`_ version 1.4 or above
- `Astropy <http://www.astropy.org/>`_ version 1.0 or above
- `Six <https://pypi.python.org/pypi/six>`_

Several additional packages can be installed for optional features:

- `Nose <http://pypi.python.org/pypi/nose/>`_, to run the unit tests.
- `Numexpr <http://pypi.python.org/pypi/numexpr>`_, to optimize some
  computations with pixtables.
- `fitsio <https://pypi.python.org/pypi/fitsio>`_, a Python wrapper for
  cfitsio, used in `~mpdaf.obj.CubeMosaic`.
- `pkg-config <https://pkgconfig.freedesktop.org/>`_, helper tool used when
  compiling C libraries.
- `CFITSIO <http://heasarc.gsfc.nasa.gov/fitsio/>`_ (C library for reading and
  writing FITS files).
- `C OpenMP library <http://openmp.org>`_, to get parallelization.
- `SExtractor <http://www.astromatic.net/software/sextractor>`_ for several
  methods of `~mpdaf.sdetect.Source` and for :doc:`muselet`.

.. _Numpy: http://www.numpy.org/

Installing MPDAF
================

MPDAF can be installed with pip::

    pip install mpdaf

.. note::

  - `Numpy`_ must be installed before MPDAF as it is needed by the setup
    script.

  - You will need a C compiler (e.g. gcc or clang) to be installed for the
    installation to succeed (see below).

MPDAF can also be installed with extra dependencies (Numexpr, fitsio) with::

    pip install mpdaf[all]

C extensions
============

MPDAF contains a few C extensions that must be built during the installation,
and these require optional dependencies:

- The first extension needs ``pkg-config``, to find the correct compiler and
  library flags, and ``cfitsio``. If not available, the extensions is not
  built, and a few things will not work (`~mpdaf.obj.CubeList`, and several
  PixTable methods: `~mpdaf.drs.PixTable.sky_ref`,
  `~mpdaf.drs.PixTable.subtract_slice_median` and
  `~mpdaf.drs.PixTable.divide_slice_median`).

  This extension can also use OpenMP if available.  Note that on Mac OS, OpenMP
  is not used by default because clang doesn't support OpenMP. To force it, the
  ``USEOPENMP`` environment variable can be set to anything except an empty
  string::

      USEOPENMP=1 CC=<local path of gcc> pip install mpdaf

- The second extension is used for `~mpdaf.obj.CubeMosaic` and uses Cython, but
  it is only required for the development version. The distributed package
  includes directly the C files.

Tips for Mac OS users
=====================

- First, XCode is needed to get some developer tools (compiler, ...). On
  recent Mac OS versions, this can be done with ``$ xcode-select --install``.

- A great package manager can be used to install packages like cfitsio or
  pkg-config: `Homebrew <http://brew.sh/>`_. Then, ``brew install cfitsio
  pkgconfig``.

- It is also possible to install and use gcc to compile MPDAF
  with OpenMP support (for parallelized functions). Otherwise clang is used.

- `Anaconda <http://continuum.io/downloads>`_ is a great scientific python
  distribution, it comes with up-to-date and pre-compiled versions of Numpy,
  Scipy, Astropy and more.


Unit tests
==========

To run the unit tests, you need to install the *nose* package, then run::

    $ python setup.py test
