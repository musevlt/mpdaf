Download and install mpdaf
**************************


Download the code
=================

Trac system
-----------

The Trac repository browser `Browse Source <http://urania1.univ-lyon1.fr/mpdaf/browser>`_ can be used to navigate through the directory structure.


Wiki page
---------

You can browse tarball of specific revisions in the wiki page `CoreLib <http://urania1.univ-lyon1.fr/mpdaf/wiki/WikiCoreLib>`_.


MPDAF sub-modules
-----------------

mpdaf contains user packages:

+-------------------+--------------------+-----------------------------------------------------------------------+
| User packages     |                    |                                                                       |
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

 * Python (version 2.6 or 2.7)
 * IPython
 * numpy (version 1.6.2 or above)
 * scipy (version 0.12 or above)
 * matplotlib (version 1.1.0 or above)
 * astropy (version 1.0 or above)
 * nose
 * PIL
 * numexpr
 * python-development package
 * pkg-config tool
 * C numerics library
 * C CFITSIO library
 * C OpenMP library (optional)


.. _installation-label:

Installation
============

To install the mpdaf package, you first run the *setup.py build* command to build everything needed to install::

  /mpdaf$ python setup.py build
  
The setup script tries to use pkg-config to find the correct compiler flags and library flags.

Note that on MAC OS, openmp is not used by default because clang doesn't support OpenMp.
To force it, the USEOPENMP environment variable can be set to anything except an empty string::

 /mpdaf$ sudo USEOPENMP=0 CC=<local path of gcc> python setup.py build
 

After building everything, you log as root and install everything from build directory::

  root:/mpdaf$ python setup.py install


Unit tests
==========

The command *setup.py test* runs unit tests after in-place build::

  /mpdaf$ python setup.py test
