=======
 MPDAF
=======

MPDAF is the *MUSE Python Data Analysis Framework*. Its goal is to develop
a python framework in view of the analysis of MUSE data in the context of the
GTO.

Installation
------------

The various software required are:

 * Python (version 2.7, not compatible yet with python 3)
 * numpy (version 1.6.2 or above)
 * scipy (version 0.10.1 or above)
 * matplotlib (version 1.1.0 or above)
 * astropy (version 1.0 or above)
 * pkg-config tool
 * C numerics library
 * C CFITSIO library

Some additional libraries can be installed for optional features:

 * C OpenMP library, to get parallelization.
 * nose, to run the unit tests.
 * Pillow, to read jpg or png images.
 * numexpr, to optimize some computations with pixtables.

To install the mpdaf package::

    python setup.py install

The ``setup.py`` tries to use ``pkg-config`` to find the correct compiler and
library flags. Note that on MAC OS, OpenMP is not used by default because
clang doesn't support OpenMP. To force it, the ``USEOPENMP`` environment variable
can be set to anything except an empty string::

    USEOPENMP=1 CC=<local path of gcc> python setup.py install

Unit tests
----------

To run the unit tests, you need to install the nosetests package, then run::

    python setup.py test
