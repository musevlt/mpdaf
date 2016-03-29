Catalog object
**************

Catalog is a subclass of Table class from astropy.table.

A catalog object can be read from a file (by using the astropy.table constructor Table.read) or it can be created from a list of :class:`mpdaf.sdetect.Source` objects.


Reference
=========

:func:`mpdaf.sdetect.Catalog.from_sources <mpdaf.sdetect.Catalog.from_sources>` constructs a catalog from a list of source objects.

:func:`mpdaf.sdetect.Catalog.from_path <mpdaf.sdetect.Catalog.from_path>` constructs a catalog from the path of a directory containing source files.

:func:`mpdaf.sdetect.Catalog.masked_invalid <mpdaf.sdetect.Catalog.masked_invalid>` masks where invalid values occur.

:func:`mpdaf.sdetect.Catalog.match <mpdaf.sdetect.Catalog.match>` matchs elements of the current catalog with an other (in RA, DEC).

:func:`mpdaf.sdetect.Catalog.select <mpdaf.sdetect.Catalog.select>` selects all sources from catalog which are inside the WCS of an image.

:func:`mpdaf.sdetect.Catalog.plot_symb <mpdaf.sdetect.Catalog.plot_symb>` plots the sources location from the catalog.

:func:`mpdaf.sdetect.Catalog.plot_id <mpdaf.sdetect.Catalog.plot_id>` displays the id of the catalog.


Examples
========

Preliminary import::

 >>> from mpdaf.sdetect import Catalog

First, we create our catalog from a list of sources.
The new catalog will contain all data stored in the primary headers and in the tables extensions of the sources:

- a column per header fits,
- two columns per magnitude band "[BAND]" and "[BAND]_ERR",
- two or three columns per redshfit: "Z_[Z_DESC]", "Z_[Z_DESC]_ERR" or "Z_[Z_DESC]_MIN" and "Z_[Z_DESC]_MAX"
- several columns per line.

The lines columns depend of the format. By default the columns names are created around unique LINE names ( "[LINE]_[lines_colname]").
But it is also possible to use a working format "[lines_colname]_xxx" where xxx is the number of lines present in each source.
In the following example, sources is the output of a detection code (focus or muselet or sea or selfi)::

 >>> cat = Catalog.from_sources(sources, fmt='default')
 >>> print cat
  ID     RA        DEC      ORIGIN ORIGIN_V      CUBE     ... [SII]2_LBDA_OBS_ERR   [SII]_FLUX  [SII]_FLUX_ERR [SII]_LBDA_OBS [SII]_LBDA_OBS_ERR
        deg        deg                                    ...       Angstrom                                      Angstrom         Angstrom
  --- ---------- ---------- ------- -------- ------------- ... ------------------- ------------- -------------- -------------- ------------------
   6 63.3561058 10.4661665 muselet      2.1 minicube.fits ...                  -- 469.158943905  32.4582923751    7289.890625               1.25
   4 63.3554039 10.4647026 muselet      2.1 minicube.fits ...                1.25            --             --             --                 --
   1 63.3559265 10.4653692 muselet      2.1 minicube.fits ...                  -- 768.280835505  55.7679692876    7292.390625               1.25
   2 63.3556900 10.4646378 muselet      2.1 minicube.fits ...                  -- 872.328376229  50.5601526862    7293.640625               1.25
   3 63.3552666 10.4657850 muselet      2.1 minicube.fits ...                  -- 755.092227665  46.8217588531    7294.890625               1.25
   5 63.3567162 10.4656963 muselet      2.1 minicube.fits ...                1.25 581.031931012  45.6505167557    7293.640625               1.25
   7 63.3559914 10.4646139 muselet      2.1 minicube.fits ...                  --            --             --             --                 --
   8 63.3549309 10.4656277 muselet      2.1 minicube.fits ...                  --            --             --             --                 --
 >>> cat.colnames
  ['ID',
   'RA',
   'DEC',
   'ORIGIN',
   'ORIGIN_V',
   'CUBE',
   'Z_EMI',
   'Z_EMI_MAX',
   'Z_EMI_MIN',
   'Halpha_FLUX',
   'Halpha_FLUX_ERR',
   'Halpha_LBDA_OBS',
   'Halpha_LBDA_OBS_ERR',
   'Lya/[OII]_FLUX',
   'Lya/[OII]_FLUX_ERR',
   'Lya/[OII]_LBDA_OBS',
   'Lya/[OII]_LBDA_OBS_ERR',
   '[NII]_FLUX',
   '[NII]_FLUX_ERR',
   '[NII]_LBDA_OBS',
   '[NII]_LBDA_OBS_ERR',
   '[SII]2_FLUX',
   '[SII]2_FLUX_ERR',
   '[SII]2_LBDA_OBS',
   '[SII]2_LBDA_OBS_ERR',
   '[SII]_FLUX',
   '[SII]_FLUX_ERR',
   '[SII]_LBDA_OBS',
   '[SII]_LBDA_OBS_ERR']
 >>> cat = Catalog.from_sources(sources, fmt='working')
 >>> print cat
  ID     RA        DEC      ORIGIN ORIGIN_V      CUBE      Z_EMI   ... LINE005 FLUX006  FLUX_ERR006 LBDA_OBS006 LBDA_OBS_ERR006 LINE006
        deg        deg                                             ...                                Angstrom      Angstrom
  --- ---------- ---------- ------- -------- ------------- -------- ... ------- -------- ----------- ----------- --------------- -------
   6 63.3561058 10.4661665 muselet      2.1 minicube.fits 0.085600 ...               --          --          --              --
   4 63.3554039 10.4647026 muselet      2.1 minicube.fits 0.086000 ...               --          --          --              --
   1 63.3559265 10.4653692 muselet      2.1 minicube.fits 0.086000 ...     0.0 442.4661     29.5532     7121.14            1.25     0.0
   2 63.3556900 10.4646378 muselet      2.1 minicube.fits 0.086200 ...     0.0       --          --          --              --
   3 63.3552666 10.4657850 muselet      2.1 minicube.fits 0.086400 ...     0.0 332.8741     30.2875     6843.64            1.25     0.0
   5 63.3567162 10.4656963 muselet      2.1 minicube.fits 0.086200 ...  [SII]2       --          --          --              --
   7 63.3559914 10.4646139 muselet      2.1 minicube.fits       -- ...               --          --          --              --
   8 63.3549309 10.4656277 muselet      2.1 minicube.fits       -- ...               --          --          --              --
 >>> cat.colnames
 ['ID',
  'RA',
  'DEC',
  'ORIGIN',
  'ORIGIN_V',
  'CUBE',
  'Z_EMI',
  'Z_EMI_MAX',
  'Z_EMI_MIN',
  'FLUX001',
  'FLUX_ERR001',
  'LBDA_OBS001',
  'LBDA_OBS_ERR001',
  'LINE001',
  'FLUX002',
  'FLUX_ERR002',
  'LBDA_OBS002',
  'LBDA_OBS_ERR002',
  'LINE002',
  'FLUX003',
  'FLUX_ERR003',
  'LBDA_OBS003',
  'LBDA_OBS_ERR003',
  'LINE003',
  'FLUX004',
  'FLUX_ERR004',
  'LBDA_OBS004',
  'LBDA_OBS_ERR004',
  'LINE004',
  'FLUX005',
  'FLUX_ERR005',
  'LBDA_OBS005',
  'LBDA_OBS_ERR005',
  'LINE005',
  'FLUX006',
  'FLUX_ERR006',
  'LBDA_OBS006',
  'LBDA_OBS_ERR006',
  'LINE006']

Then, we visualize these sources on our white image::

 >>> from mpdaf.obj import Image
 >>> ima = Image('white.fits')
 >>> fig = plt.figure()
 >>> ax = fig.add_subplot(1,1,1)
 >>> ima.plot(scale='log')
 >>> cat.plot_id(ax, ima.wcs)

.. image::  _static/sources/catalog_id.png
