Source object
*************

A lot of tools are currently been developed to detect sources in MUSE data cubes.
In order to easily exchange information about detected sources, we needed to define a format for storing source file.
A FITS file format has been defined and is described in `this document <http://urania1.univ-lyon1.fr/mpdaf/attachment/wiki/WikiCoreLib/SourceICD.pdf>`_. 
The Source class implements input/output for this Source FITS file and methods to analyse, plot and compare source objects. 


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
|           | Values are :class:`mpdaf.obj.Spectrum` objects                |
+-----------+---------------------------------------------------------------+
| O.images  | Dictionary that contains images.                              |
|           | Keys gives filter names.                                      |
|           | Values are :class:`mpdaf.obj.Image` object                    |
+-----------+---------------------------------------------------------------+
| O.cubes   | Dictionary that contains small data cubes.                    |
|           | Keys give description of the cubes.                           |
|           | Values are :class:`mpdaf.obj.Cube` object                     |
+-----------+---------------------------------------------------------------+
| O.tables  | Dictionary that contains astropy tables.                      |
|           | Keys give description of the tables.                          |
|           | Values are astropy table objects.                             |
+-----------+---------------------------------------------------------------+


Reference
=========

:func:`mpdaf.sdetect.Source.from_data <mpdaf.sdetect.Source.from_data>` constructs a Source object from a list of data.

:func:`mpdaf.sdetect.Source.from_file <mpdaf.sdetect.Source.from_file>` constructs a Source object from a FITS file.

:func:`mpdaf.sdetect.Source.write <mpdaf.sdetect.Source.write>` writes the Source object in a FITS file.

:func:`mpdaf.sdetect.Source.info <mpdaf.sdetect.Source.info>` prints information about the Source object.

:func:`mpdaf.sdetect.Source.add_comment <mpdaf.sdetect.Source.add_comment>` adds a user comment to the FITS header of the Source object.

:func:`mpdaf.sdetect.Source.remove_comment <mpdaf.sdetect.Source.remove_comment>` removes a user comment from the FITS header of the Source object.

:func:`mpdaf.sdetect.Source.add_attr <mpdaf.sdetect.Source.add_attr>` adds a new attribute as a keyword in the primary FITS header.

:func:`mpdaf.sdetect.Source.remove_attr <mpdaf.sdetect.Source.remove_attr>` removes an attribute from the FITS header of the Source object.

:func:`mpdaf.sdetect.Source.add_z <mpdaf.sdetect.Source.add_z>` adds a redshift value to the z table.

:func:`mpdaf.sdetect.Source.add_mag <mpdaf.sdetect.Source.add_mag>` adds a magnitude value to the mag table.

:func:`mpdaf.sdetect.Source.add_line <mpdaf.sdetect.Source.add_line>` adds a line to the lines table.

:func:`mpdaf.sdetect.Source.add_image <mpdaf.sdetect.Source.add_image>` extracts an image centered on the source center and appends it to the images dictionary.

:func:`mpdaf.sdetect.Source.add_cube <mpdaf.sdetect.Source.add_cube>` extracts a cube centered on the source center and appends it to the cubes dictionary.

:func:`mpdaf.sdetect.Source.add_white_image <mpdaf.sdetect.Source.add_white_image>` computes the white images from the MUSE data cube and appends it to the images dictionary.

:func:`mpdaf.sdetect.Source.add_narrow_band_images <mpdaf.sdetect.Source.add_narrow_band_images>` creates narrow band images from a redshift value and a catalog of lines.

:func:`mpdaf.sdetect.Source.add_narrow_band_image_lbdaobs <mpdaf.sdetect.Source.add_narrow_band_image_lbdaobs>` creates a narrow band image around an observed wavelength value.

:func:`mpdaf.sdetect.Source.add_seg_images <mpdaf.sdetect.Source.add_seg_images>` runs SExtractor to create segmentation maps.

:func:`mpdaf.sdetect.Source.add_masks <mpdaf.sdetect.Source.add_masks>` runs SExtractor to create masked images.

:func:`mpdaf.sdetect.Source.add_table <mpdaf.sdetect.Source.add_table>` appends an astropy table to the tables dictionary.

:func:`mpdaf.sdetect.Source.extract_spectra <mpdaf.sdetect.Source.extract_spectra>` extracts spectra from the MUSE data cube.

:func:`mpdaf.sdetect.Source.crack_z <mpdaf.sdetect.Source.crack_z>` estimates the best redshift matching the list of emission lines.

:func:`mpdaf.sdetect.Source.sort_lines <mpdaf.sdetect.Source.sort_lines>` sorts the lines by flux in descending order.

:func:`mpdaf.sdetect.Source.show_ima <mpdaf.sdetect.Source.show_ima>` shows image.

:func:`mpdaf.sdetect.Source.show_spec <mpdaf.sdetect.Source.show_spec>` displays a spectra.


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
 
:func:`Source.add_white_image <mpdaf.sdetect.Source.add_white_image>` method computes from the MUSE data cube a white image of 5 arcseconds around the object and appends it to the images dictionary::

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
 
:func:`Source.add_seg_images <mpdaf.sdetect.Source.add_seg_images>` method runs SExtractor on all images present in self.images dictionary to create segmentation maps.
After that, :func:`Source.add_masks <mpdaf.sdetect.Source.add_masks>` uses the list of segmentation maps to compute the union mask and the intersection mask and  the region where no object is detected in any segmentation map is saved in the sky mask.
Union is saved as an image of booleans nammed 'MASK_UNION', intersection is saved as 'MASK_INTER' and sky mask is saved as 'MASK_SKY'::
 
 >>> s.add_seg_images()
 >>> s.add_masks()

 SExtractor  version 2.19.5 (2014-03-21)

 Written by Emmanuel BERTIN <bertin@iap.fr>
 Copyright 2012 IAP/CNRS/UPMC

 visit http://astromatic.net/software/sextractor

 SExtractor comes with ABSOLUTELY NO WARRANTY
 You may redistribute copies of SExtractor
 under the terms of the GNU General Public License.

 ----- SExtractor 2.19.5 started on 2015-06-03 at 14:38:46 with 1 thread

 ----- Measuring from: 0001-HST_F814W.fits [1/1]
       "Unnamed" / no ext. header / 25x25 / 32 bits (floats)
 (M+D) Background: -0.00378665 RMS: 0.121295   / Threshold: 0.0909715  
       Objects: detected 1        / sextracted 1               

 All done (in 0.0 s: 1903.9 lines/s , 76.2 detections/s)
 ----- SExtractor 2.19.5 started on 2015-06-03 at 14:38:46 with 1 thread

 ----- Measuring from: 0001-MUSE_WHITE.fits [1/1]
       "Unnamed" / no ext. header / 25x25 / 32 bits (floats)
 (M+D) Background: 118.647    RMS: 603.26     / Threshold: 452.445    
       Objects: detected 1        / sextracted 1               

 All done (in 0.0 s: 1974.1 lines/s , 79.0 detections/s)
 [INFO] Doing HST_F814W
 [INFO] Doing MUSE_WHITE
 [INFO] Image HST_F814W has one useful object
 [INFO] Image MUSE_WHITE has one useful object

Now, we plot these different images::

 >>> fig = plt.figure()
 >>> ax = fig.add_subplot(1,3,1)
 >>> s.show_ima(ax, 'MUSE_WHITE', showcenter=(0.2, 'r'))
 >>> ax = fig.add_subplot(1,3,2)
 >>> s.show_ima(ax, 'HST_F814W', showcenter=(0.2, 'r'))
 >>> ax = fig.add_subplot(1,3,3)
 >>> s.show_ima(ax, 'MASK_UNION')
 
.. image::  user_manual_sourceselect_images/show_ima.png
 
Now, we extract spectra from the MUSE data cube::
 
 >>> s.extract_spectra(cub)
 >>> fig = plt.figure()
 >>> ax = fig.add_subplot(1,1,1)
 >>> s.show_spec(ax, 'MUSE_WHITE_NOSKY', zero=True, sky=s.spectra['MUSE_SKY'])
 
.. image::  user_manual_sourceselect_images/show_spec.png
 
The source can be saved as a FITS file::
 
 >>> s.write('test.fits')
 
Later we can load it like this::

 >>> s = Source.from_file('test.fits')


SourceList class
================

SourceList is a sub-class of the python list class.  
This class contains just one method :func:`mpdaf.sdetect.SourceList.write <mpdaf.sdetect.SourceList.write>` that creates a folder and saves all sources files and the catalog file in it.
        
