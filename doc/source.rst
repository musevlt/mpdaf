Source object
*************

A lot of tools are currently been developed to detect sources in MUSE data
cubes.  In order to easily exchange information about detected sources, we
needed to define a format for storing source file.  A FITS file format has been
defined and is described in `this document
<http://urania1.univ-lyon1.fr/mpdaf/attachment/wiki/WikiCoreLib/SourceICD.pdf>`_.
The Source class implements input/output for this Source FITS file and methods
to analyse, plot and compare source objects.


Source object format
====================

A Source object O consist of:

+-----------+---------------------------------------------------------------+
| Component | Description                                                   |
+===========+===============================================================+
| O.header  | pyfits header instance                                        |
+-----------+---------------------------------------------------------------+
| O.lines   | astropy table that contains the parameters of spectral lines. |
+-----------+---------------------------------------------------------------+
| O.z       | astropy table that contains redshift values.                  |
+-----------+---------------------------------------------------------------+
| O.mag     | astropy table that contains magnitude values.                 |
+-----------+---------------------------------------------------------------+
| O.spectra | Dictionary that contains spectra.                             |
|           | Keys give description of the spectra.                         |
|           | Values are `mpdaf.obj.Spectrum` objects                       |
+-----------+---------------------------------------------------------------+
| O.images  | Dictionary that contains images.                              |
|           | Keys gives filter names.                                      |
|           | Values are `mpdaf.obj.Image` object                           |
+-----------+---------------------------------------------------------------+
| O.cubes   | Dictionary that contains small data cubes.                    |
|           | Keys give description of the cubes.                           |
|           | Values are `mpdaf.obj.Cube` object                            |
+-----------+---------------------------------------------------------------+
| O.tables  | Dictionary that contains astropy tables.                      |
|           | Keys give description of the tables.                          |
|           | Values are astropy table objects.                             |
+-----------+---------------------------------------------------------------+


Reference
=========

`mpdaf.sdetect.Source.from_data <mpdaf.sdetect.Source.from_data>` constructs a Source object from a list of data.

`mpdaf.sdetect.Source.from_file <mpdaf.sdetect.Source.from_file>` constructs a Source object from a FITS file.

`mpdaf.sdetect.Source.write <mpdaf.sdetect.Source.write>` writes the Source object in a FITS file.

`mpdaf.sdetect.Source.info <mpdaf.sdetect.Source.info>` prints information about the Source object.

`mpdaf.sdetect.Source.add_comment <mpdaf.sdetect.Source.add_comment>` adds a user comment to the FITS header of the Source object.

`mpdaf.sdetect.Source.remove_comment <mpdaf.sdetect.Source.remove_comment>` removes a user comment from the FITS header of the Source object.

`mpdaf.sdetect.Source.add_attr <mpdaf.sdetect.Source.add_attr>` adds a new attribute as a keyword in the primary FITS header.

`mpdaf.sdetect.Source.remove_attr <mpdaf.sdetect.Source.remove_attr>` removes an attribute from the FITS header of the Source object.

`mpdaf.sdetect.Source.add_z <mpdaf.sdetect.Source.add_z>` adds a redshift value to the z table.

`mpdaf.sdetect.Source.add_mag <mpdaf.sdetect.Source.add_mag>` adds a magnitude value to the mag table.

`mpdaf.sdetect.Source.add_line <mpdaf.sdetect.Source.add_line>` adds a line to the lines table.

`mpdaf.sdetect.Source.add_image <mpdaf.sdetect.Source.add_image>` extracts an image centered on the source center and appends it to the images dictionary.

`mpdaf.sdetect.Source.add_cube <mpdaf.sdetect.Source.add_cube>` extracts a cube centered on the source center and appends it to the cubes dictionary.

`mpdaf.sdetect.Source.add_white_image <mpdaf.sdetect.Source.add_white_image>` computes the white images from the MUSE data cube and appends it to the images dictionary.

`mpdaf.sdetect.Source.add_narrow_band_images <mpdaf.sdetect.Source.add_narrow_band_images>` creates narrow band images from a redshift value and a catalog of lines.

`mpdaf.sdetect.Source.add_narrow_band_image_lbdaobs <mpdaf.sdetect.Source.add_narrow_band_image_lbdaobs>` creates a narrow band image around an observed wavelength value.

`mpdaf.sdetect.Source.add_seg_images <mpdaf.sdetect.Source.add_seg_images>` runs SExtractor to create segmentation maps.

`mpdaf.sdetect.Source.find_sky_mask <mpdaf.sdetect.Source.find_sky_mask>` creates a sky mask from a list of segmentation maps.

`mpdaf.sdetect.Source.find_union_mask <mpdaf.sdetect.Source.find_union_mask>` creates an object mask as the union of the segmentation maps.

`mpdaf.sdetect.Source.find_intersection_mask <mpdaf.sdetect.Source.find_intersection_mask>` creates an object mask as the intersection of the segmentation maps.

`mpdaf.sdetect.Source.add_table <mpdaf.sdetect.Source.add_table>` appends an astropy table to the tables dictionary.

`mpdaf.sdetect.Source.extract_spectra <mpdaf.sdetect.Source.extract_spectra>` extracts spectra from the MUSE data cube.

`mpdaf.sdetect.Source.crack_z <mpdaf.sdetect.Source.crack_z>` estimates the best redshift matching the list of emission lines.

`mpdaf.sdetect.Source.sort_lines <mpdaf.sdetect.Source.sort_lines>` sorts the lines by flux in descending order.

`mpdaf.sdetect.Source.show_ima <mpdaf.sdetect.Source.show_ima>` shows image.

`mpdaf.sdetect.Source.show_spec <mpdaf.sdetect.Source.show_spec>` displays a spectra.


Examples
========

Preliminary import::

 >>> from mpdaf.sdetect import Source

For example, we create a source object from spatial coordinates::

 >>> s = Source.from_data(ID=1, ra=150.05654, dec=2.60335, origin=('test','v0.0','DATACUBE-HDFS.fits'))
 >>> s.info()
 [INFO] ID      =                    1 / object ID
 [INFO] RA      =    150.0565338134766 / RA in degrees
 [INFO] DEC     =    2.603349924087524 / DEC in degrees
 [INFO] ORIGIN  = 'test    '           / detection software
 [INFO] ORIGIN_V= 'v0.0    '           / version of the detection software
 [INFO] CUBE    = 'DATACUBE-HDFS.fits' / MUSE data cube

`Source.add_white_image <mpdaf.sdetect.Source.add_white_image>` method computes from the MUSE data cube a white image of 5 arcseconds around the object and appends it to the images dictionary::

 >>> from mpdaf.obj import Cube
 >>> cub = Cube('DATACUBE-HDFS.fits')
 >>> s.add_white_image(cube=cub, size=5)
 >>> s.info()
 [INFO] ID      =                    1 / object ID
 [INFO] RA      =    150.0565338134766 / RA in degrees
 [INFO] DEC     =    2.603349924087524 / DEC in degrees
 [INFO] ORIGIN  = 'test    '           / detection software
 [INFO] ORIGIN_V= 'v0.0    '           / version of the detection software
 [INFO] CUBE    = 'DATACUBE-HDFS.fits' / MUSE data cube

 [INFO] images['MUSE_WHITE'] 25 X 25 .data .var rot=0.0

We can also extract an HST image centered on the source center and append it to the images dictionary::

 >>> from mpdaf.obj import Image
 >>> ima_hst = Image('hst.fits')
 >>> s.add_image(ima_hst, name='HST_F814W')
 >>> s.info()
 [INFO] ID      =                    1 / object ID
 [INFO] RA      =    150.0565338134766 / RA in degrees
 [INFO] DEC     =    2.603349924087524 / DEC in degrees
 [INFO] ORIGIN  = 'test    '           / detection software
 [INFO] ORIGIN_V= 'v0.0    '           / version of the detection software
 [INFO] CUBE    = 'DATACUBE-HDFS.fits' / MUSE data cube

 [INFO] images['HST_F814W']
 [INFO] 168 X 167 image (hst.fits)
 [INFO] .data(168,167) (no unit) fscale=1, no noise
 [INFO] center:(02:36:12.0492,10:00:13.5691) size in arcsec:(5.040,5.010) step in arcsec:(0.030,0.030) rot:0.0

 [INFO] images['MUSE_WHITE']
 [INFO] 25 X 25 image (no name)
 [INFO] .data(25,25) (10**(-20)*erg/s/cm**2/Angstrom) fscale=1, .var(25,25)
 [INFO] center:(02:36:11.9131,10:00:13.5686) size in arcsec:(5.000,5.000) step in arcsec:(0.200,0.200) rot:0.0



SourceList class
================

SourceList is a sub-class of the python list class.  This class contains just
one method `mpdaf.sdetect.SourceList.write
<mpdaf.sdetect.SourceList.write>` that creates a folder and saves all sources
files and the catalog file in it.
