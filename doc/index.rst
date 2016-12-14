******************************
MPDAF |release| documentation!
******************************

:Release: |release|
:Date: |today|

.. ifconfig:: 'dev' in release

    .. warning::

        This documentation is for the version of MPDAF currently under
        development.

.. include:: ../README.rst

Sub-packages status
-------------------

.. role:: stable(raw)
   :format: html

.. role:: dev(raw)
   :format: html

.. raw:: html

   <style>
      .stable {font-weight: bold;color:green;}
      .dev {font-weight: bold;color:red;}
   </style>

The currently existing sub-packages are:

+------------------+-------------------------------------------------+------------------+
|  sub-packages    | description                                     |  status          |
+==================+=================================================+==================+
| |mpdaf.obj|_     | Interface for spectra, images and cubes         | :stable:`Stable` |
+------------------+-------------------------------------------------+------------------+
| |mpdaf.drs|_     | Interface for the MUSE raw file and pixel table | :stable:`Stable` |
+------------------+-------------------------------------------------+------------------+
| |mpdaf.sdetect|_ | Source: Creates single-source FITS files        | :dev:`Dev`       |
|                  +-------------------------------------------------+------------------+
|                  | Catalog: Creates source catalogs                | :dev:`Dev`       |
|                  +-------------------------------------------------+------------------+
|                  | MUSELET: MUSE Line Emission Tracker             | :stable:`Stable` |
+------------------+-------------------------------------------------+------------------+
| |mpdaf.MUSE|_    | slicer: MUSE slicer numbering scheme            | :stable:`Stable` |
|                  +-------------------------------------------------+------------------+
|                  | PSF: MUSE PSF models                            | :dev:`Dev`       |
+------------------+-------------------------------------------------+------------------+

Where :dev:`Dev` means *Actively developed*, so be prepared for potentially
significant changes, and :stable:`Stable` means *Reasonably stable*, so any
significant changes or additions will generally be backwards-compatible.


.. |mpdaf.obj| replace:: ``mpdaf.obj``
.. |mpdaf.drs| replace:: ``mpdaf.drs``
.. |mpdaf.MUSE| replace:: ``mpdaf.MUSE``
.. |mpdaf.sdetect| replace:: ``mpdaf.sdetect``

.. _mpdaf.drs: http://mpdaf.readthedocs.io/en/stable/drs.html
.. _mpdaf.obj: http://mpdaf.readthedocs.io/en/stable/obj.html
.. _mpdaf.MUSE: http://mpdaf.readthedocs.io/en/stable/muse.html
.. _mpdaf.sdetect: http://mpdaf.readthedocs.io/en/stable/sdetect.html


Contents
========

.. toctree::
   :maxdepth: 1

   installation
   start
   obj
   drs
   sdetect
   muse
   logging
   tools
   contribute
   changelog
   credits


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

