Download and install mpdaf
**************************


Download the code
=================

Trac system
-----------

The `Trac repository browser <http://urania1.univ-lyon1.fr/mpdaf/browser>`_ can
be used to navigate through the directory structure.


Wiki page
---------

You can browse tarball of specific revisions in the wiki page `CoreLib
<http://urania1.univ-lyon1.fr/mpdaf/wiki/WikiCoreLib>`_.


Prerequisites
=============

The various software required are:

 * `Python <http://python.org/>`_ version 2.7
 * `IPython <http://ipython.org/>`_  (enhanced interactive console)
 * `Numpy <http://www.numpy.org/>`_ version 1.6.2 or above (base N-dimensional array Python package)
 * `Scipy <http://www.scipy.org/>`_ version 0.12 or above (fundamental Python library for scientific computing)
 * `Matplotlib <http://matplotlib.org/>`_ version 1.1.0 or above (Python 2D plotting library)
 * `Astropy <http://www.astropy.org/>`_ version 1.0 or above (Python package for Astronomy)

Some additional libraries can be installed for optional features:

 * `Nose <http://pypi.python.org/pypi/nose/>`_, to run the unit tests.
 * `Pillow <http://pypi.python.org/pypi/Pillow>`_, Python imaging library, to read jpg or png images.
 * `Numexpr <http://pypi.python.org/pypi/numexpr>`_, to optimize some computations with pixtables.
 * ``pkg-config`` tool (helper tool used when compiling C libraries)
 * `CFITSIO <http://heasarc.gsfc.nasa.gov/fitsio/>`_ (C library for reading and writing FITS files)
 * `C OpenMP library <http://openmp.org>`_, to get parallelization.

.. _installation-label:

Installation
============

To install the Mpdaf package::

    $ python setup.py install

The setup script requires ``pkg-config`` to find the correct compiler flags and
library flags. ``cfitsio`` is also required.

Note that on Mac OS, OpenMP is not used by default because clang doesn't
support OpenMP. To force it, the ``USEOPENMP`` environment variable can be set
to anything except an empty string::

    USEOPENMP=1 CC=<local path of gcc> python setup.py install

Tips for Mac OS users
---------------------

- First, XCode is needed to get some developper tools (compiler, ...). On
  recent Mac OS versions, this can be done with ``$ xcode-select --install``.

- A great package manager can be used to install packages like cfitsio or
  pkg-config: `Homebrew <http://brew.sh/>`_. Then, ``brew install cfitsio
  pkgconfig``.  - It is also possible to install and use gcc to compile MPDAF
  with OpenMP support (for parallelized functions). Otherwise clang is used.

- `Anaconda <http://continuum.io/downloads>`_ is a great scientific python
  distribution, it comes with up-to-date and precompiled versions of numpy,
  scipy, astropy and more.


Unit tests
==========

To run the unit tests, you need to install the *nose* package, then run::

    $ python setup.py test
