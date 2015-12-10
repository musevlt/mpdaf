.. mpdaf documentation master file, created by
   sphinx-quickstart on Fri Jun 22 10:03:09 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MPDAF CoreLib |release| documentation!
======================================

:Release: |release|
:Date: |today|

.. ifconfig:: 'dev' in release

    .. warning::

        This documentation is for the version of MPDAF currently under
        development.

`Back to the mpdaf wiki <http://urania1.univ-lyon1.fr/mpdaf/>`_

MPDAF, the *MUSE Python Data Analysis Framework*, is a set of Python packages
in view of the analysis of MUSE data in the context of the GTO.

The main library, labeled *CoreLib*, is developed and maintained  by CRAL. It
contains a list of Python packages to play with MUSE specific objects (raw
data, pixel tables, etc.) and spectra, images and data cubes.

A user library, labeled *UserLib*, will contains user developed packages that
will be available for the consortium.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   obj
   drs
   muse
   tools
   sdetect
   dvper_manual
   api
   changelog


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

