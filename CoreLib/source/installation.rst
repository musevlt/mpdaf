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

These user packages are included in the mpdaf tarball.

However mpdaf contains also large packages that are not present in tarball:

+-------------------+--------------------+-----------------------------------------------------------------------+
| Large packages    | Path               | Description                                                           |
+===================+====================+=======================================================================+
| fusion            | lib/mpdaf/fusion   | C++ code for the Bayesian fusion of hyperspectral astronomical images |
|                   |                    | `fusion doc <user_manual_fusion.html>`_                               |
+-------------------+--------------------+-----------------------------------------------------------------------+
| quickViz          | lib/mpdaf/quickViz | vizualisation tool for MUSE cubes                                     |
|                   |                    | `quickViz doc <http://urania1.univ-lyon1.fr/mpdaf/wiki/DocQuickViz>`_ |                                        
+-------------------+--------------------+-----------------------------------------------------------------------+


The user has the choice to download or not download these packages, the sub-modules directories are there, but they're empty. Pulling down the submodules is a two-step process.

First download the submodules that you want used via the wiki page `CoreLib <http://urania1.univ-lyon1.fr/mpdaf/wiki/WikiCoreLib>`_.

Then, untar and move the repository::

  $ tar -xzf fusion.tar.gz
  $ mv fusion lib/mpdaf/


Prerequisites
=============

The various software required are:

 * Python (version 2.6 or 2.7)
 * IPython
 * numpy (version 1.6.2 or above)
 * scipy (version 0.10.1 or above)
 * matplotlib (version 1.1.0 or above)
 * pyfits (version 3.1 or astropy)
 * pywcs (version 1.11-4.7 or astropy)
 * nose
 * PIL

.. _installation-label:

Installation
============

To install the mpdaf package, you first run the *setup.py build* command to build everything needed to install::

  /mpdaf$ python setup.py build


Then, you lof as root and install everything from build directory::


  root:/mpdaf$ python setup.py install


setup.py informs you that the fusion package is not found. But it's just a warning, it's not blocking and you can continue to install mpdaf.

To install the fusion submodule, log as root and run the *setup.py fusion* command::

  root:/mpdaf$ python setup.py fusion



Unit tests
==========

The command *setup.py test* runs unit tests after in-place build::

  /mpdaf$ python setup.py test
