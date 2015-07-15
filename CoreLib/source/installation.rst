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

You can browse tarball of specific revisions in the wiki page `CoreLib <http://urania1.univ-lyon1.fr/mpdaf/wiki/WikiCoreLib>`_.


MPDAF sub-modules
-----------------

mpdaf contains user packages:

+-------------------+--------------------+-----------------------------------------------------------------------+
| User packages     | Path               |                                                                       |
+===================+====================+=======================================================================+
| mpdaf_user.fsf    | mpdaf_user/fsf     | Estimation of the Field Spread Function                               |
|                   |                    | `fsf doc <http://urania1.univ-lyon1.fr/mpdaf/wiki/FsfModelWiki>`_     |
+-------------------+--------------------+-----------------------------------------------------------------------+
| mpdaf_user.zap    | mpdaf_user/zap     | sky subtraction tool                                                  |
|                   |                    | `zap doc <http://urania1.univ-lyon1.fr/mpdaf/wiki/ZapWiki>`_          |
+-------------------+--------------------+-----------------------------------------------------------------------+
| mpdaf_user.galpak | mpdaf_user/galpak  | galaxy parameters and kinematics extraction tool                      |
|                   |                    | `galpak doc <http://galpak.irap.omp.eu>`_                             |
+-------------------+--------------------+-----------------------------------------------------------------------+

These user packages are included in the mpdaf tarball.

However mpdaf contains also a large package that is not present in tarball:

+-------------------+--------------------+-----------------------------------------------------------------------+
| Large packages    | Path               | Description                                                           |
+===================+====================+=======================================================================+
| quickViz          | lib/mpdaf/quickViz | vizualisation tool for MUSE cubes                                     |
|                   |                    | `quickViz doc <http://urania1.univ-lyon1.fr/mpdaf/wiki/DocQuickViz>`_ |
+-------------------+--------------------+-----------------------------------------------------------------------+


The user can download this package from the wiki page `CoreLib <http://urania1.univ-lyon1.fr/mpdaf/wiki/WikiCoreLib>`_.



Prerequisites
=============

The various software required are:

 * `python <http://python.org/>`_ version 2.6 or 2.7
 * `ipython <http://ipython.org/>`_  (enhanced interactive console)
 * `numpy <http://www.numpy.org/>`_ version 1.6.2 or above (base N-dimensional array Python package)
 * `scipy <http://www.scipy.org/>`_ version 0.12 or above (fundamental Python library for scientific computing)
 * `matplotlib <http://matplotlib.org/>`_ version 1.1.0 or above (Python 2D plotting library)
 * `astropy <http://www.astropy.org/>`_ version 1.0 or above (Python package for Astronomy)
 * `nose <http://pypi.python.org/pypi/nose/>`_ (testing for python)
 * `PIL <http://pypi.python.org/pypi/PIL>`_  (Python imaging library)
 * `numexpr <http://pypi.python.org/pypi/numexpr>`_ (fast numerical expression evaluator for NumPy)
 * pkg-config tool (helper tool used when compiling C libraries)
 * `CFITSIO <http://heasarc.gsfc.nasa.gov/fitsio/>`_ (C library for reading and writing FITS files)
 * optional: `C OpenMP library <http://openmp.org>`_ (parallel programming)


.. _installation-label:

Installation
============

To install the mpdaf package, you must first run the *setup.py build* command
to build everything needed to install::

  /mpdaf$ python setup.py build

The setup script requires ``pkg-config`` to find the correct compiler flags and
library flags. ``cfitsio`` is also required.

Note that on Mac OS, OpenMP is not used by default because clang doesn't
support OpenMP. To force it, the ``USEOPENMP`` environment variable can be set
to anything except an empty string::

 /mpdaf$ sudo USEOPENMP=1 CC=<local path of gcc> python setup.py build

After building everything, you log as root and install everything from build
directory::

  root:/mpdaf$ python setup.py install

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

The command *setup.py test* runs unit tests after in-place build::

  /mpdaf$ python setup.py test
