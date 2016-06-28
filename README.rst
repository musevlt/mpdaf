MPDAF, the *MUSE Python Data Analysis Framework*, is a Python package developed
and maintained by `CRAL <https://cral.univ-lyon1.fr/>`_, for the analysis of
MUSE data in the context of the GTO.

It contains a list of Python classes and functions to play with MUSE specific
objects (raw data, pixel tables, etc.) and spectra, images and data cubes.

The current existing sub-packages are:

+---------------+-------------------------------------------------+---------+
|  sub-packages | description                                     |  status |
+===============+=================================================+=========+
| mpdaf.obj     | Interface for the MUSE datacube                 | Mature  |
+---------------+-------------------------------------------------+---------+
| mpdaf.drs     | Interface for the MUSE raw file and pixel table | Stable  |
+---------------+-------------------------------------------------+---------+
| mpdaf.MUSE    | slicer: MUSE slicer numbering scheme            | Mature  |
|               +-------------------------------------------------+---------+
|               | PSF: MUSE PSF models                            | Dev     |
+---------------+-------------------------------------------------+---------+
| mpdaf.sdetect | Source: storing source file                     | Dev     |
|               +-------------------------------------------------+---------+
|               | Catalog: storing catalog file                   | Dev     | 
|               +-------------------------------------------------+---------+
|               | MUSELET: MUSE Line Emission Tracker             | Mature  |
+---------------+-------------------------------------------------+---------+

The classification is as follows:

 - Dev: Actively developed, be prepared for possible significant changes.
 - Stable: Reasonably stable, any significant changes/additions will generally include backwards-compatiblity.
 - Mature: Additions/improvements possible, but no major changes planned.


A `Gitlab <https://git-cral.univ-lyon1.fr/MUSE/mpdaf>`_ instance is used for
development, download and tickets.

Getting help
------------

If you want to get help or discuss issues, you can send an email to mpdaf-support@osulistes.univ-lyon1.fr


Reporting Issues
----------------

If you have found a bug in MPDAF please report it.

The preferred way is to create a new issue on `the MPDAF gitlab issue page <https://git-cral.univ-lyon1.fr/MUSE/mpdaf/issues>`_ ;
that requires creating a account on `git-cral <https://git-cral.univ-lyon1.fr>`_ if you do not have one.

To create an account, please send email to `mpdaf-support@osulistes.univ-lyon1.fr <mailto:mpdaf-support@osulistes.univ-lyon1.fr?subject=Account%20creation>`_



MPDAF documentation
-------------------

`MPDAF v1.2 <http://urania1.univ-lyon1.fr/mpdaf/chrome/site/DocCoreLib/index.html>`_

`MPDAF dev <http://urania1.univ-lyon1.fr/mpdaf/chrome/site/DocCoreLib_dev/index.html>`_