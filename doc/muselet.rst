*******
MUSELET
*******

.. figure:: _static/muselet/muselet_logo.jpg
  :align: center

Description
===========

MUSELET (for MUSE Line Emission Tracker) is a simple SExtractor-based python
tool to detect emission lines in a datacube. It has been developed by Johan
Richard (johan.richard@univ-lyon1.fr), with help from Johany Martinez (CRAL)
and Laure Piqueras (CRAL).

MUSELET uses SExtractor (Bertin & Arnouts 1996,
http://www.astromatic.net/software/sextractor) to detect line emission in
narrow-band images created from the cube. It then merges all detections in
a single catalog, separating the emission lines linked with continuum sources
detected in the white light images from the isolated emission lines. It then
tries to estimate the redshift from multiple emission lines.

MUSELET takes as an input a MUSE DATACUBE (fits format), and works in 3 steps:

- STEP 1: creation of white light, color and narrow band images.

  MUSELET will first create a variance-weighted white light image as well as
  R,G,B images based on 1/3 of the wavelength range each.  The code will then
  go through the wavelength axis and create one narrow band image at each
  wavelength plane.  The narrow band image is based on a line-weighted
  (spectrally) average of 5 wavelength planes in the cube (so 5x1.25 Angstroms
  wide). The continuum is estimated from 2 spectral medians of ~ 25 Angstroms
  each on the blue and red side of the narrow band region. The size of the
  continuum region can be adjusted with the optional parameter delta (in
  number of wavelength planes, default=20).

  These narrow band images are created in the ``nb/`` directory. If not present
  it will be created.

- STEP 2: MUSELET will run SExtractor using the ``default.sex``,
  ``default.param``, ``default.conv`` and ``default.nnw`` parameter files in
  the current and ``nb/`` directory. If not present default parameter files are
  created. These files can be used to control SExtractor.  
  Alternatively, it is possible to supply SExtractor command line options via
  the "sex_config=" and "sex_config_nb" keywords.

- STEP 3: The code will merge all SExtractor catalogs and will return separate
  emission lines linked with continuum objects from the rest.  For each of
  these catalogs, MUSELET will estimate a redshift based on multiple emission
  lines. Emission lines are merged spatially to the same source based on the
  "radius" parameter (in pixels, default radius=4).  The redshifts are
  estimated from emission line catalogs ``emlines`` (all emission lines) and
  ``emlines_small`` (list of brightest emission lines). These files are
  2 columns (name and wavelength) and can be adjusted to one's needs.

The code will produce:
  - a `~mpdaf.sdetect.Catalog` containing detected lines,
  - a directory containing the corresponding `~mpdaf.sdetect.Source` files,
  - a `~mpdaf.sdetect.Catalog` containing detected objects (merged groups of
    lines),
  - a directory containing the corresponding `~mpdaf.sdetect.Source` files.

These are written to the current directory (or the working directory if
specified via the "workdir" keyword)

Requirements:

- SExtractor ("sex" or "sextractor" binary file in your $PATH).

Tutorials
=========

MUSELET is run through the following commands in MPDAF::

  >>> from mpdaf.sdetect import muselet
  >>> muselet('DATACUBE.fits')

Optionally, one can provide the starting step (2 or 3) in order to only redo
one part of the script. For instance, assuming the narrow-band images are
already created::

  >>> muselet('DATACUBE.fits', step=2)

Optionally, one can provide the size of the continuum region to subtract on
each side of the narrow-band images::

  >>> # only 15 wavelength planes in continuum estimate
  >>> muselet('DATACUBE.fits', delta=15)

The output detection catalogs can be opened as a `mpdaf.sdetect.Catalog`
object::
  
  >>> from mpdaf.sdetect import Catalog
  >>> cat_lines = Catalog.read('lines.fit')
  >>> cat_objects = Catalog.read('objects.fit')

and or as a `mpdaf.sdetect.SourceList` object::

  >>> from mpdaf.sdetect import SourceList
  >>> sources_lines = SourceList.from_path('lines')
  >>> sources_objects = SourceList.from_path('objects')

Individual `mpdaf.sdetect.Source` files can be open::

  >>> from mpdaf.sdetect import Source
  >>> src_line = Source.from_file('lines/lines-0001.fits')
  >>> src_object = Source.from_file('objects/objects-0001.fits')

Finally it is possible to interact easily in topcat between the muselet catalog
and a MUSE datacube opened in ds9. To do so one has to select the "Activation
Action" menu and put the following custom code:

>>> exec("topcat_show_ds9",toString(RA),toString(DEC),toString(LBDA_OBS))
for the catalog of lines
or 
>>> exec("topcat_show_ds9",toString(RA),toString(DEC),toString(LBDA_OBS001))
for the catalog of objects

.. figure:: _static/muselet/topcat_muselet_catalogs.png

The shell script ``topcat_show_ds9`` (inspired by Benjamin Clement) is
installed with MPDAF and should be in your path.
