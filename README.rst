MPDAF, the *MUSE Python Data Analysis Framework*, is an open-source (BSD
licensed) Python package, developed and maintained by `CRAL
<https://cral.univ-lyon1.fr/>`_ and partially funded by the ERC advanced grant 339659-MUSICOS
(see `Authors and Credits <http://mpdaf.readthedocs.io/en/latest/credits.html>`_ for more details).

It has been developed and used in the `MUSE
Consortium <http://muse-vlt.eu/science/>`_ for several years, and is now
available freely for the community.

It provides tools to work with MUSE-specific data (raw data, pixel tables,
etc.), and with more general data like spectra, images and data cubes. Although
its main use is to work with MUSE data, it is also possible to use it other
data, for example HST images. MPDAF also provides MUSELET, a SExtractor-based
tool to detect emission lines in a datacube, and a format to gather all the
informations on a source in one FITS file.

Bug reports, comments, and help with development are very welcome.

MPDAF is compatible with Python 2.7 and 3.3+.

Links :

- `Documentation <http://mpdaf.readthedocs.io/>`_
- Source, issues and pull requests on a
  `Gitlab <https://git-cral.univ-lyon1.fr/MUSE/mpdaf>`_ instance
- Releases on `PyPI <http://pypi.python.org/pypi/mpdaf>`_
- `Mailing list <mpdaf-support@osulistes.univ-lyon1.fr>`_ to get help or
  discuss issues

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

Reporting Issues
----------------

If you have found a bug in MPDAF please report it.

The preferred way is to create a new issue on `the MPDAF gitlab issue page
<https://git-cral.univ-lyon1.fr/MUSE/mpdaf/issues>`_ .  This requires creating
a account on `git-cral <https://git-cral.univ-lyon1.fr>`_ if you don't have
one.  To create an account, please send email to
`mpdaf-support@osulistes.univ-lyon1.fr
<mailto:mpdaf-support@osulistes.univ-lyon1.fr?subject=Account%20creation>`_


.. |mpdaf.obj| replace:: ``mpdaf.obj``
.. |mpdaf.drs| replace:: ``mpdaf.drs``
.. |mpdaf.MUSE| replace:: ``mpdaf.MUSE``
.. |mpdaf.sdetect| replace:: ``mpdaf.sdetect``

.. _mpdaf.drs: http://mpdaf.readthedocs.io/en/latest/drs.html
.. _mpdaf.obj: http://mpdaf.readthedocs.io/en/latest/obj.html
.. _mpdaf.MUSE: http://mpdaf.readthedocs.io/en/latest/muse.html
.. _mpdaf.sdetect: http://mpdaf.readthedocs.io/en/latest/sdetect.html
