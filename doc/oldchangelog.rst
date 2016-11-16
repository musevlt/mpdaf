:orphan:

v2.0 (05/07/2016)
-----------------

This is the first public release, as an open source project (BSD
licensed).  MPDAF is also now compatible with Python 3.3+.

New Features
~~~~~~~~~~~~

* `Image.resample <mpdaf.obj.Image.resample>`: now applies by default an
  anti-aliasing filter when asked to lower the resolution of an image.
  The variance is now computed for the re-sampled image.

* `Image.regrid <mpdaf.obj.Image.regrid>`: **new method** similar to
  `Image.resample`, but more flexible in the positioning of the sky
  in the re-sampled image, and it accepts signed pixel increments.
  `resample` now uses `regrid` and was conserved for compatibility reasons.

* `Image.rotate <mpdaf.obj.Image.rotate>`: now works correctly on images
  whose pixels have different angular sizes along the X and Y axes.

* `Image.align_with_image <mpdaf.obj.Image.align_with_image>`:
  **new method** to resample the image of the object to give it the same
  orientation, position, resolution and size as another image.

* `Image.estimate_coordinate_offset <mpdaf.obj.Image.estimate_coordinate_offset>`:
  **new method** which uses a full-image auto-correlation to measure the
  average offsets between the world coordinates of common astronomical
  features in two images. This was written primarily to determine coordinate
  offsets between MUSE images and HST reference images.

* `Image.adjust_coordinates <mpdaf.obj.Image.adjust_coordinates>`:
  **new method** which determines the world coordinate offsets between an
  image and a reference image, then applies this shift to the reference
  pixel index of the image.

* `Image.mask_region <mpdaf.obj.Image.mask_region>` and
  `Cube.mask_region <mpdaf.obj.Cube.mask_region>` methods have been revised
  to support rectangular and square regions that are rotated.

* `Cube.mask_polygon <mpdaf.obj.Cube.mask_polygon>`: **new method** similar
  to `Image.mask_polygon <mpdaf.obj.Image.mask_polygon>`.

* `Cube.bandpass_image <mpdaf.obj.Cube.bandpass_image>`: **new method**
  for creating images with a given filter response curve.

* `WCS.get_axis_increments <mpdaf.obj.WCS.get_axis_increments>`:
  **new method** which returns the displacements on the sky per pixel
  increment along the Y and X axes.
  There is also a new method `WCS.set_axis_increments <mpdaf.obj.WCS.set_axis_increments>`
  to update the signed displacements on the sky.

* `~mpdaf.MUSE.FieldsMap`: **new class** which reads the possible FIELDMAP
  extension of the MUSE data cube.

* `~mpdaf.MUSE.FSF`: **new class** which proposes a simple FSF model
  (Moffat function with a FWHM which varies linearly with the wavelength).


Breaking changes
~~~~~~~~~~~~~~~~

* Remove unused methods:

   - Remove the peak detection on a spectrum (``peak_detection``).
   - Remove interactive plotting methods of Spectrum and Image classes.
   - Remove filter methods on images (``minimum_filter``, ``maximum_filter``,
     ``median_filter``) and spectra (``median_filter``).
   - Remove ``Image.add_poisson_noise``.
   - Remove ``Image.sum``.
   - Remove methods to create images (``mask_image``, ``composite_image``, ``make_image``).
   - Remove ``PixTable.get_slices``.
   - Remove the SEA script (each step remains available as library).

* Remove the `drs.calibobj` module.

* Remove submodules *ZAP* and *GALPAK*.

  - `ZAP <https://github.com/ktsoto/zap>`_ is now publicly available.

  - `GALPAK <http://galpak.irap.omp.eu/downloads.html>`_ is also publicly
    available.

* The `resize` methods of Cube, Image and Spectrum have been renamed
  `crop` to better indicate their purpose. The new methods are also faster
  and use less memory, and return the list of slices that have been used to
  crop the data.

* The `rebin_median` methods of Cube, Image and Spectrum have been removed
  because its effect on the variances couldn't be computed, and because it
  didn't seem scientifically useful.

* The `rebin_factor` and `rebin_mean` methods of Cube, Image and Spectrum
  have been replace by the `rebin` method. This method has been simplified
  to make it more maintainable. The simplification to this function was to
  effectively truncate the input data to be an integer multiple of the
  re-binning reduction factor, rather than computing partially sampled
  output pixels for the edge of the re-binned data.

* The method `Cube.get_lambda` has been renamed `~mpdaf.obj.Cube.select_lambda`.

* The method `Spectrum.get_lambda` has been renamed `~mpdaf.obj.Spectrum.subspec`.

* The weighting options of `Cube.sum <mpdaf.obj.Cube.sum>` has been reimplemented.
  The old method with the flux conservation has been moved in the SEA library.

* `WCS.get_rot` has been corrected to always return the angle between north
  and the Y axis of the image, in the sense of a rotation of north eastwards
  of Y.  The updated versions of `Image.get_rot <mpdaf.obj.Image.get_rot>`
  and `Image.rotate <mpdaf.obj.Image.rotate>` resolve this discrepancy,
  so any software that has been using these functions may see changes.

* The return value of the `get_range` methods of Cube, Image and DataArray
  have been changed to a flat array, rather than an array of two coordinate
  tuples that only denoted image corners for non-rotated images.

* `WCS.get_naxis1` and `WCS.get_naxis2` have been removed, because the
  underlying `WCS.naxis1` and `WCS.naxis2` values can be queried directly.

* `WCS.resample` and `WCS.rotate` methods has been removed, because
  they are no more needed by the equivalent methods in `Image`.

* The method `Image.add` has been removed because it is easy for a user
  to do the equivalent by using align_with_image() followed by a simple
  addition of two images.

* PIL dependency has been removed: now MPDAF doesn't read .PNG, .JPEG or
  .BMP images.

WCS
~~~

* When an `~mpdaf.obj.WCS` object is initialized via its cdelt1, cdelt2
  and rot parameters, the corresponding coordinate transformation matrix
  is now calculated in the way recommended in equation 189
  of FITS paper *II*
  (Calabretta, M. R. & Greisen, E. W. 2002 paper II, A&A, 395, 1077-1122).

* The `~mpdaf.obj.WCS` initializer now accepts a `cd` argument, which may
  be used to set the coordinate transformation directly.

* `~mpdaf.obj.WCS.set_step` now changes the pixel scaling correctly for all
  FITS files. The previous version, which worked for MUSE FITS files,
  failed on FITS files whose coordinate transform matrix included any shear
  terms.

* A couple of issues have been resolved in `~mpdaf.obj.WCS.get_step`.
  Incorrect values were returned for FITS files with pixels that were
  rectangular on the sky, rather than square. This didn't affect typical
  MUSE FITS files.

* Don't write WCS headers with both `CDELT1` and `CD1_1`.

Data classes (Cube, Image, Spectrum)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Methods which transform the object return now by default an other object.
  The optional copy argument can be used to request that the input data be
  transformed in-place.

* Fix `clone` to update the NAXIS keywords in the data header

* Change management of the data type (by default Cube object have the type of the DATA extension)

* The variance calculation of `~mpdaf.obj.DataArray.sqrt` has been
  corrected.

* Fix `Cube.median <mpdaf.obj.Cube.median>` when used with `axis=None`.

* `Cube.get_image <mpdaf.obj.Cube.get_image>`: add input parameters in the
  FITS header.

* Allow to overwrite `BUNIT` for data without unit.

* Fix `EXPTIME` of combined cubes when cubes overlap (using a median of the
  exposure map give a more realistic estimate).

* The `mean` method of Cube, Image and Spectrum have been rewrote to
  optionally perform weighted means.

* In the `sum` methods of Cube, Image and Spectrum, the weighted mean of
  the spectral pixels was being multiplied by the total number of input
  pixels instead of the number of unmasked pixels. This will have resulted
  in sums that were too small wherever there were masked spectral pixels.

* Propagate mask in `Cube.subcube <mpdaf.obj.Cube.subcube>`.

* Refactor `Cube.subcube_circle_aperture <mpdaf.obj.Cube.subcube_circle_aperture>`
  to use `Cube.subcube` and `Cube.mask`.

* Make `Cube.subcube <mpdaf.obj.Cube.subcube>` and
  `Image.subimage <mpdaf.obj.Image.subimage>` as similar as possible.
  These methods extract rectangular as well as square areas, and they now
  work correctly for data with rectangular pixels.

* A couple of problems have been fixed in the code that multiplies an
  image by a spectrum. The original code didn't handle variances
  correctly and crashed due to a non-existent variable.

* Correct the pixel value shown when moving the mouse over pixels of
  `Image.plot <mpdaf.obj.Image.plot>`.

Sources
~~~~~~~

* Correct bug in `Catalog.from_sources <mpdaf.sdetect.Catalog.from_sources>`

* Rewrite `Catalog.match <mpdaf.sdetect.Catalog.match>` which was not working
  properly, now it returns a dictionary.

* Tell which keyword is missing when creating a source.

* Allow to deactivate the masking of invalid values when loading a source.
  This part (`~mpdaf.sdetect.Catalog.masked_invalid`) takes ~40% of the
  total load time for sources with a lot of tables and columns.

Muselet
~~~~~~~

* Optimize `matchlines` function

* Split the main function in several functions

* Remove Numpy warning

* Use a Numpy array and not a masked array for the weights array

* Use masked median of inv_variance

* Make Muselet less verbose when running SExtractor

Pixtable
~~~~~~~~

* Convert pixtable values to double to avoid precision errors.  This avoids
  discrepancies between `_get_pos_sky` and `_get_pos_sky_numexpr` (Numexpr
  seems to use double precision by default) and probably also with other
  methods.

* Fix initial value for `~mpdaf.drs.PixTable.subtract_slice_median` correction.
  This value was set to 1 which seems wrong for an additive correction. To
  emphasize the fact that some combination of slices and quadrants are not
  valid, and allow to filter these values later, this commit changes the init
  value for the count to -1, and the correction to NaN.

* Fix `~mpdaf.drs.PixTable.select_sky` in the case of positioned pixel table.

RawFile
~~~~~~~

* Simplify `~mpdaf.drs.RawFile` and `~mpdaf.drs.Channel` classes


v1.2 (13/01/2016)
-----------------

* Optimize ``Cube.subcube`` and use ``__getitem__`` in ``subcube``/``subimage``
  and other methods to speed up things (avoid reading the full cube/image).
* Add missing units in image methods.
* Fill data with NaNs only for float arrays, otherwise raise exception.
* Use a new ``MpdafUnitsWarning`` to allow filtering the unit warnings. It can
  be used this way::

      # filter only MPDAF's warnings
      from mpdaf.tools import MpdafUnitsWarning
      warnings.simplefilter('ignore', category=MpdafUnitsWarning)

      # or filter both MPDAF and Astropy warnings
      import astropy.units as u
      warnings.simplefilter('ignore', category=u.UnitsWarning)

* CUNIT FITS keyword: patch to read ``mum`` as micron.
* Correct ``cube.get_step`` that returned nothing.
* Use setuptools for the ``setup.py``:

  - Allow to use develop mode (``python setup.py develop``).
  - Install dependencies automatically.
  - Use optional dependencies.

* Remove unmaintained submodules: *quickViz* and *fsf*. *quickViz* is still
  available `here <http://lsiit-miv.u-strasbg.fr/paseo/cubevisualization.php>`_
  but maybe not compatible with the latest Aladin version.
* Remove the ``displaypixtable`` module.
* Avoid a huge memory peak when creating masked arrays with ``mask=True``.
* Add some tools to print execution times.
* Added scaling option in ``Cubelist.combine()``.
* Fix ``cube.var = None`` to remove the variance part of the Cube.
* Revert ZAP version to the same as before 1.2b1 (was updated by mistake).
* Add a new method ``Image.find_wcs_offsets`` to find the WCS offset with a
  reference image.

PixTable
~~~~~~~~

* Use ``CRVAL1/CRVAL2`` instead of ``RA/DEC`` as reference point for positioned
  pixtables.
* Remove ``cos(delta)`` correction for positioned pixtables.
* Use directly the binary mask in ``extract_from_mask``.
* Allow to use a boolean mask for pixtable selections.

Sources
~~~~~~~

* ``Source.add_image``: the order of the rotation is set to 0 in case of an
  image of 0 and 1.
* Add methods to manage a history in the sources headers.
* Use ``savemask='none'`` for MASK and SEG extensions.
* Correct bug in ``source.write`` when a column has no unit.
* Allow to pass the lambda range and wave unit to ``Source.extract_spectra``.
* Correct bug in Catalog initialization due to units.
* ``Catalog.from_sources``: update the default format.
* Split ``Source.add_masks`` in 3 methods: ``find_sky_mask``,
  ``find_union_mask`` and ``find_intersection_mask``.
* Isolate comments and history in source information.

Muselet
~~~~~~~

* Limit the memory usage.
* Added option to clean detections on skylines.
* Added exposure map cube.
* Remove automatic narrow-band images cleaning in muselet.

v1.2b1 (05/11/2015)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~

* Add a new base class for the :class:`~mpdaf.obj.Cube`,
  :class:`~mpdaf.obj.Image` and :class:`~mpdaf.obj.Spectrum` classes.  This
  allows to fix some inconsistencies between these classes and to bring more
  easily new common features.

* FITS files are now read only when the data is needed: when creating an object
  the data is not loaded into memory. The data is loaded at the first access of
  the ``.data`` attribute, and the same goes for the variance (and ``.var``).
  A consequence of these optimization is that the ``CubeDisk`` class has
  been removed.

* Shape of objects:

  - Remove the ``shape`` parameter in constructors. Instead the shape is derived
    from the datasets.
  - Spectrum's shape is now a tuple, which is consistent with the Cube and Image
    classes, and with Numpy arrays.

* Allow to specify the data type of Cube/Image/Spectrum in the constructor (and
  read an extension as an integer array).

* Change the behavior of the ``.clone`` method: now by default it returns an
  object with the data attribute set to None. This was changed as an
  optimization, because in most cases (at least in MPDAF's code) a Numpy array
  is set to the cloned object, just after the clone, so the Numpy array that was
  created by clone was discarded. You can get the previous behavior with::

    sp = sptot.clone(data_init=np.zeros)

  Or you can set directly a Numpy array to the cloned object::

    sp = sptot.clone()
    sp.data = np.zeros(sptot.shape)

* The ``fscale`` attribute of a Cube/Image/Spectrum object has disappeared.
  MUSE units are now read from the FITS header (it takes into account possible
  ``FSCALE`` keyword). The ``.unit`` attribute of Cube/Image/Spectrum saves
  physical units of the data values and the scale value as an ``astropy.units``
  object.

* When a method of MPDAF objects requires a physical value as input, the unit of
  this value is also given ``(x=, x_unit=)``. By default coordinates are in
  degrees and wavelengths are in angstroms.

* Results of ``Source.subcube`` methods are always centered on the source given
  in input (columns/row of NaN are added when the source is on the border).

* Source/Catalog object write and read masked values.

* From Johan and Benjamin: shell script to interact in Topcat between the
  muselet catalog and a MUSE datacube opened in ds9.

Changes that should be imperceptible to users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use ``astropy.wcs`` for handling the wavelength coordinates.
* Simplify logging configuration.
* Cube/Image/Spectrum constructors: allow to pass a hdulist object in place of
  the filename (this option should reduce the time when the FITS file is used
  several times because of the big time spent reading the FITS headers).

v1.1.18.1 (31/07/2015)
----------------------

* Full correction of ``mask_polygon`` function.
* Correct a bug in ``source.show_spec``.
* ``Source.add_white_image`` now compute mean(cube) instead of sum(cube).
* Workaround bug in GCC 5.1 & OpenMP.
* Add prints for the number of threads in the merging c code.
* Change redshift table format to have z,zmin,zmax.
* Use ``astropy.constants`` for the c value.
* Update wcs info method.
* Correct bug to compute the size of images that are added in source objects.
* New method ``Source.add_narrow_band_image_lbdaobs``.
* Default size of 5 arcsec in ``Source.add_white method``.
* Still have the same type of WCS matrix(CD/PC).
* Update muselet package to be compatible with new source object.
* Correct bug in catalog initialization.

v1.1.18 (08/07/2015)
--------------------

* Update ``CubeList.save_combined_cube`` to be more generic.
* Optimize C libraries using openmp (cubes combination).
* Update WCS according to FITS standard.
* Modify ``Spectrum.log_plot`` to be the same as plot with a log stretch.
* Allow to create a cube object with a masked array.
* Correct bug in ``mask_polygon`` function of Image object.
* Possibility to use MAD (median absolute deviation) statistics for
  sigma-clipping during cube combination.
* Take into account cos(delta) in ``source.info``.
* Split ``mpdaf.logging`` method in 2 methods (steam_handler/file_handler).
* Update mask computation of source:

  - option to give a directory containing default files of sextractor.
  - option to remove or not the sextractor files.
  - split add_masks method in two methods (add_seg_images and add_masks).

* Update ``source.info`` method.
* Correct bug in ``Cube.aperture``.
* Spectrum extraction code from Jarle (SEA code).
* Print info in ``source.add_narrow_band_images()``.
* Update Source class:
  - add_line method.
  - add_attr/remove_attr methods.
  - dictionary of tables.
* Add CubeMosaic class for the merging of a mosaic.
* Update Source class:
  - add image rotation in ``source.info``.
  - rebin mask before applying weight in ``source.extract_spectra``.
* Initialize a SourceList object from a path name.
* Image/Cube truncate methods: update computation of boundaries.
* Correct bug in muselet/setup_files_n.
* Take into account quadrant in pixtable autocalibration.
* Fix merged cube headers so that the cube can be ingested by MuseWise.

  - Add needed keywords: RA, DEC, MJD-OBS, DATE-OBS, PI-COI, OBSERVER, OBJECT,
    ESO INS DROT POSANG, ESO INS MODE
  - Allow to override OBJECT name
  - Compute a correct EXPTIME for the mosaic case
  - Put the list of merged files in comments, otherwise the keyword value can be
    too long for MuseWise

* Update mask computation of source (SEA):

  - take into account rotation of the image
  - replace central detection by detection around the source center.

v1.1.17.1
---------

* Update ``CubeList.save_combined_cube`` to be more generic.
* Optimize c libraries using openmp.
* Update WCS according to FITS standard.
* Modify ``Spectrum.log_plot`` to be the same as plot with a log stretch.
* Allow to create a cube object with a masked array.
* Corrected bug in ``mask_polygon`` function of Image object.

v1.1.17 (16/06/2015)
--------------------

* Correct bug concerning .var attribute of Spectrum/Image/Cube.
  It must be an array and not a masked array.
* PixTable: Optimize origin2xoffset and origin2coords
* Remove tuples in parameters of np.sum/mean/median
* Update write method of Cube/Image/Spectrum objects
* Update write method of PixTable
* Add matplotlib.Axes in plot parameters
* Update arithmetic methods of Cube/Image in order to accept array as input
* Add mask_polygon method in image
* Correct bug in add_mpdaf_method_keywords (MPDAF #365)
* Make a copy of wcs object during the initialization if Cube/Image/Spectrum objects
* Update merging of data cubes:

  - method returns a cube object
  - option to compute the variance of the merged cube as the the mean of the variances
    of the N individual exposures divided by N**2
  - method returns more pixels statistics

* Source and Catalog classes
* correct bug in Cube.aperture method
* Fix numexpr not used when installed.
* Refactor common part of PixTable.extract
* Remove 'ESO PRO' keywords writing in PixTable.
  This was changed a long time ago and is not useful anymore.
* Allow to extract data from a PixTable with stack numbers.
* Add a param to PixTable.extract to choose if multiple selection are combined
  with logical_and (default) or logical_or.
* Refactor ``get_*`` methods of PixTable.
* Split PixTable.extract in several methods for selecting values.
  Make a method for each selection type (lambda, slices, ifus, position, ...), so
  that it will be more flexible.
* Pass units to the extracted PixTable, this avoids muse_exp_combine rejecting
  pixtables because of different units.
* Update inputs of fftconvolve_moffat method
* Add some basic tests for PixTable
* Refactor PixTable column setters.
* Correct bug in WCS.__getitem__
* Add snr option in spectrum.plot to plot data/sqrt(var)
* ListSource class
* Update FOCUS detection code to be compatible with new Source object
* Fixes and enhancements for cubelist:

  - Save MPDAF keywords with comments in the correct order.
  - Save the unit in the output cubes.
  - Fix unit checking, and use the unit/fscale from the first cube if these are
    not consistent, with a warning.

* Improve saving of combined cube.

  - FILES list is too long to be both a HIERARCH and CONTINUE keyword. So use
    a CONTINUE keyword instead.
  - Refactor the saving, and put the saved keywords in the good order.
  - Copy several useful keywords from the source cubes: ORIGIN, TELESCOP,
    INSTRUME, EQUINOX, RADECSYS, EXPTIME, OBJECT
  - Update EXPTIME, assuming that all files have the same EXPTIME value (to be
    improved later).

* Refactor the pixtable extraction from a mask.
* Subtract_slice_median: don't correct when all pixels are masked.
* Change precision in the equality test of two WCSs.
* Always initialize CubeList.wcs. If there are not equal, just raise a warning.
* Open raw file without memory mapping
* Fix flux conservation in rebin methods
* Cube.subcube method to extract sub-cube
* Correct Cube.mean
* Add weights in Cube.sum
* subtract_slice_median: indent, remove useless stat var, add check for mpdaf_median
* Add a PixTable.select_stacks method
* Simplify CubeDisk.truncate
* Cube.get_image method
* Cube.subcube_aperture method
* Corrected median for even-sized tables in merging
* Source display methods
* Catalog display methods
* Correct wcs.info
* galpak v 1.6.0
* Spectrum: add gauss_dfit, gauss_asymfit, igauss_asymfit methods
* Update muselet detection code to be compatible with new Source object

v1.1.16.1
---------

* Correct bug concerning .var attribute of Spectrum/Image/Cube. It must be an
  array and not a masked array.
* PixTable: Optimize origin2xoffset and origin2coords
* Remove tuples in parameters of np.sum/mean/median
* Update write method of Cube/Image/Spectrum objects
* Update write method of PixTable

v1.1.16 (16/03/2015)
--------------------

* correct bug in Image.resize method
* add a script to create a white-light image from a cube
* correct bug in pixtable.set_lambda method (mpdaf#358)
* correct bug in pixtable.copy method (mpdaf#359)
* change method to get the path directory under which mpdaf is installed
* remove fusion submodule
* add muselet module

v1.1.15.1 (20/02/2015)
----------------------

* Don't print the msg about Focus each time mpdaf is imported.
* Don't load/write the data when only the header must be updated.
* Add an option to not show the x/y labels in Image.plot
* Cube merging: Save the list of files that have been merged in the FITS header.
* Take correctly into account the mask to compute the resulted variance values
  in cube.sum/mean/median methods.
* If data are scaled by a constant, variance is scaled by the square of that constant.
* Correct weight values in least squares fit
* Replace pyfits by astropy.io.fits in fsf module

v1.1.15 (02/02/2015)
--------------------

* update multiprocess methods to be compatible with logger
* correct bug in Image.mask methods
* Cube.mask methods
* Optimize a bit Image.background
* Update autocalibration methods on pixtable:

  - apply multiplicative correction to stat column
  - PixTableAutoCalib class to store pixtables auto calibration results

* update cubes merging:

  - cubelist.merging returns cube object
  - cubelist.merging manages BUNIT

* mpdaf_user.galpak version 1.4.5
* Spectrum.integrate method
* Handle float precision in the WCS comparison
* correct wave unit of pixtable object
* Source detection package
* update savemask option in Cube/Image/Spectrum write methods

v1.1.14 (21/01/2015)
--------------------

* correct bug in variance computation during CubeDisk.get_white_image method
* when merging cubes, replace the single sigma clipping parameter into two
  lower/upper clipping parameters
* gzip raw file MUSE mask named PAE_July2013.fits
* restructure C code (tools.c)
* compute the reference sky spectrum from a pixel table
* method mask_image that creates a new image from a table of apertures.
* update Image.mask and Image.mask_ellipse methods
* allow to apply a slice on all the cubes of a CubeList.
* Image/Cube/CubeDisk: correct truncate methods
* PixTable: new methods to bring all slices to the same median value
    (using sky reference spectrum)
* update mpdaf logging
* simplify sky2pix and pix2sky and add a test.
* replace use of the deprecated commands module with subprocess.
* update setup.py for MAC
* add keywords in a FITS header to describe what is done on pixtable

v1.1.13 (17/12/2014)
--------------------

* Spectrum/Image/Cube: save mask in DQ extension
* add setter to pixtable object
* use numpy methods to convert angles from radians/degrees to degrees/radians
* add mask_ellipse function in Image object to mask elliptical regions
* correct bug in world coordinates
* subtract_slice_median method of PixTable
* CubeList object to manage merging of cubes
* pyfits replaced by astropy.io.fits and pywcs replaced by astropy.wcs
* add inside=T/F parameter for the mask function of Spectrum

v1.1.12 (03/10/2014)
--------------------

* the flux scale attribute of Cube/Image/spectrum objects is now never changed
  by methods.
* sanity check on wavelength coordinates.
* new Cube.get_image method that extracts an image from the datacube.
* write cube/image/spectrum in float32
* add nearest option for WCS.sky2pix method
* pixtable: write data/xpos/ypos/lbda column in float32
* spectrum: oversampling factor for the overplotted Gaussian fit
* pixtable: code optimization with numexpr
* zap v0.6
* galpak v1.1.3
* correct MOFFAT fit error

v1.1.11 (26/09/2014)
--------------------

* Spectrum.GaussFit : update continuum computation
* Spectrum/Image/Cube
  - add get_np_data method that returns flux*fscale
  - add fscale parameter in write methods
* update docstrings
* option to overplotted inverse of variance on image
* Cube.sum/mean methods: mask nan variance values
* astropy.io.fits.EXTENSION_NAME_CASE_SENSITIVE deprecated -> astropy.io.fits.conf.extension_name_case_sensitive
* replace "slice" parameter by "sl"
* add Cube.median and Cube.aperture methods
* ignore warnings of pyfits.writeto
* zap v 0.5.1

v1.1.10 (26/08/2014)
--------------------

* zap v 0.5.
* correction of minor bugs in core library

v1.1.9 (31/07/2014)
-------------------

* update gitmodules path
* use astropy to sexa/deg coordinates transforms
* zap v 0.4.
* update PixTable documentation

v1.1.8 (09/07/2014)
-------------------

* read spheric coordinates of pixel tables.
* zap v 0.3.

v1.1.7 (26/06/2014)
-------------------

* set case sensitive for pixtable extension name.
* update pixtable coordinates types.
* correct bug in PixTable.extract method.
* update pixtable world coordinates.
* correct PixTable.write method.
* update documentation of mpdaf installation.

v1.1.6 (02/06/2014)
-------------------

* correct error in CalibFile.getImage() method
* zap update, including the new methods for the offset sky/saturated field case

v1.1.5 (20/04/2014)
-------------------

* correct bug in spectrum.write
* correct bug due to Nan in variance array
* correct bug in loop_ima
* support both pyfits and astropy in test_spectrum.py

v1.1.4 (04/02/2014)
-------------------

* correct bug in cube.resize method
* correct typo on right
* replace print by loggings or errors
* replace pyfits.setExtensionNameCaseSensitive which is deprecated
* PEP-8 coding conventions
* Cube.rebin in the case of naxis < factor
* autodetect noise extension during Spectrum/Image/Cube creation
* insert submodule zap
* replace deprecated methods of pywcs/pyfits
  replace pywcs by astropy.wcs and pyfits by astropy.fits
* correct test failures
* correct bug in Spectrum.fftconvolve_moffat method
* update wavelength range of Spectrum.rebin() method
* correct bug in Cube.__getitem__
* correct bug (typo) in spectrum.write

v1.1.3 (17/01/2014)
-------------------

* Image : check if the file exists during the initialization
* correct bug in the copy of masked array
* correct bug in cube.rebin_median
* pixel table visualization
* fast reconstruction of the white image from RawFile object
* add check in Spectrum.rebin method
* correct bug in sub-pixtable extraction

v1.1.2 (11/09/2013)
-------------------

* correct coordinates unit in pixtable header
* pixtable: rename OCS.IPS.PIXSCALE keyword

v1.1.1 (29/08/2013)
-------------------

* correct Image.add_poisson method
* correct bug in PSF module
* Spectrum/Image/Cube initialization: crval=0 by default and FITS coordinates
  discarded if wave/wcs is not None
* Image: fix bug in gaussian fit
* optimize Image.peak_detection
* correct bug in WCS.isEqual
* correct fscale value in multiprocess functions of Cube
* optimize interactive plots
* update Channel.get_trimmed_image to do bias substraction
* update Image.segment with new parameters
* add warnings according to M Wendt comments
* added method to plot a RawFile object
* added function to reconstruct an image of wavelengths on the detectors from a pixtable
* output of Image.GaussFit updated for rot=None
* correct RawFile to have no crash when a SGS extension is present
* PixTable: multi-extension FITS image format
* add submodule mpdaf_user.fsf (Camille Parisel/DAHLIA)

v1.1.0 (29/01/2013)
-------------------

* mpdaf installation: replace setuptool by distutils
* add structure (mpdaf_user directory) for user library
* mpdaf.drs.RawFile: add output detector option
* mpdaf.drs.CalibFile: add get_image method
* mpdaf.obj.Spectrum: add normalization in polynomial fit
* mpdaf.obj.Cube/Image : correct bug to write/load wcs
* add global parameter CPU for the number of CPUs
* mpdaf.obj.Cube/Image/Spectrum: correct write methods
* mpdaf.obj.Spectrum/Image/Cube : rebin_median method rebins cubes/images/spectra using median values.
* mpdaf.obj.Spectrum : add LSF_convolve method
* mpdaf.MUSE package that contains tools to manipulate MUSE specific data
* mpdaf.obj : correct coordinates rebining
* mpdaf.obj.Image : peaks detection
* mpdaf.MUSE.LSF : simple MUSE LSF model
* mpdaf.obj.Cube : multiprocessing on cube iterator
* mpdaf.obj.Image : update gaussian/moffat fit
* mpdaf.obj.CubeDisk class to open heavy cube fits file with memory mapping

v1.0.2 (19/11/2012)
-------------------

* correct rotation effect in Image.rebin method
* correct bug in spectrum/Image Gaussian fit
* remove prettytable package
* Spectrum/Image/Cube: correct set_item methods
* method to reconstruct image on the sky from pixtable
* ima[:,q] or ima[p,:] return Spectrum objects and not 1D images
* link on new version of HyperFusion
* Image: add iterative methods for Gaussian and Moffat fit
* Image: remove matplotlib clear before ploting
* fusion: update FSF model
* Spectrum/Image/Cube .primary_header and .data_header attributes
* fusion: add copy and clean, continue_fit methods
* pixtable: support new HIERARCH ESO DRS MUSE keywords (MPDAF ticket #23)
  update HIERARCH ESO PRO MUSE PIXTABLE LIMITS keywords when extracting a pixtable (MPDAF ticket #20)
* tools: add a Slicer class to convert slices number between various numbering scheme
* fusion: correct position (cos delta)
* obj package: correct cos(delta) via pywcs
* Spectrum: correct variance computation
* obj package: return np.array in place of list
* Image: correct variance computation
* Cube: correct variance computation
* Cube: add rebin_factor method
* Image: correct Gauss and Moffat fits (cos delta)
* Pixtable: correct cos(delta)
* update documentation

v1.0.1 (27/09/2012)
-------------------

* Creation of mpdaf.obj package:

  - Spectrum class manages spectrum object
  - Image class manages image object
  - Cube class manages cube object

* Creation of mpdaf.obj.coords package:

  - WCS class manages world coordinates in spatial direction (pywcs package is used).
  - WaveCoord class manages world coordinates in spectral direction.
  - deg2sexa and sexa2deg methods transforms coordinates from degree/sexagesimal
    to sexagesimal/degree.

* adding selection and arithmetic methods for Spectrum/Image/Cube objects
  (mpdaf.obj package)
* complete mpdaf.fusion package (python interface for HyperF-1.0.0)
* change mpdaf structure to have "import mpdaf"
* correct bug on memmap file
* new functionalities for Spectrum object (rebining, filtering,
  gaussian/polynomial fitting, plotting)
* documentation
* bug corrections in Spectrum objects
* mpdaf.fusion package: link to HyperF_1.0.1
* add plotting and 2d gaussian fitting for Image objects
* correct bug to read spectrum/image/cube extensions
* correct bug in coords.Wave.pixel() method
* PixTable object:

  - Fix a typo in get_slices output message
  - always read the data from the first exposure
  - use uint32 for origin and dq

* Image: add functionalities (transform, filter, sub-images)
* Spectrum/Image/Cube: correct bug for variance initialization
* Pixtable: optimize and split origin2coords in multiple helpers
* Update WCS object accoriding to the python notation : (dec,ra)
* Image: add methods to mask/unmask the image.
* Udpate the python interface for HyperF v1.1
* Add euro3D package
* Correct error with new version of pywcs (remplace 'UNITLESS' by '' for unit type)
* Compatibility with pyfits 3.0 (The Header.ascardlist() method is deprecated,
  use the .ascard attribute instead)
* Pixtable: rewrite the extract function & keep memory map filenames as private attributes
* Split objs.py in 4 files : spectrum.py, image.py, cube.py, objs.py
* Pixtable: add a reconstruct_det_image method
* New release of Spectrum class
* Create Image from PNG and BMP files
* Use nosetest for unit tests
* Add mpdaf.__info__
* Spectrum/Image/Cube: reorganize copy/in place methods
* Add Cube iterators
* Spectrum/Image/Cube: add clone method
* Add nose and matplotlib as prerequisites
* obj package: correct fscale use
* Cube/Image/Spectrum : add mask_selection method
* Update python interface for HyperFusion v1.2.0
* Spectrum/Image/Cube: bugs corrections
* version 1.0.1

v1.0.0 (02/12/2011)
-------------------

First public release.
