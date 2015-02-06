SourceDetect3D object
*********************

FOCUS: Fast Object CUbe Search

This class contains sources detection methods on cube object.
This software has been developped by Carole Clastres (MUSICOS PhD) under the supervision of David Mary (Lagrange insitute, University of Nice).
Please contact Carole for more info at carole.clastres@univ-lyon1.fr 


SourceDetect3D object format
============================

A SourceDetect3D object O consist of:

+------------+---------------------------------------+
| Component  | Description                           |
+============+=======================================+
| O.cube     | Cube object (:class:`mpdaf.obj.Cube`) |
+------------+---------------------------------------+
| O.expmap   | Exposures map FITS file name.         |
+------------+---------------------------------------+


Examples
========

Preliminary import::

 >>> from mpdaf.sdetect import focus

We will test the source detection on the HDFS cube::
 
 >>> detect = focus.SourceDetect3D(cube='DATACUBE-HDFS.fits', expmap='EXPMAP-HDFS.fits')

First, we compute the corresponding false detection probability cube using the data, variance and nb of exposures.  
The Student cumulative distribution function with expmap-1 degrees of freedom is used.
(note that this first step can take up to 15 mn for a standard size MUSE data cube):

 >>> pval = detect.p_values()
 
:func:`p_values <mpdaf.sdetect.focus.SourceDetect3D.p_values>` method returns a :class:`mpdaf.obj.Cube` object. Each voxel of the cube contain the false detection probability (p-values). We save it as a FITS file::

 >>> pval.write('pval.fits')
 
Then, we use these p_values to detects voxels with P < 1.e-8 and connect them into objects.

 >>> ima, cat = detect.quick_detection(pval, 1.e-8)
 
:func:`quick_detection <mpdaf.sdetect.focus.SourceDetect3D.quick_detection>` return an :class:`mpdaf.obj.Image` object and an :class:`mpdaf.sdetect.focus.SourceCatalog` object.

The image is used to display the location of the detected objects and the wavelength of the peak intensity of the object spectra.::

 >>> ima.plot(vmin=4900, vmax=9300, colorbar='v')
 
.. image::  user_manual_sourceselect_images/detect.png

The catalog (:class:`mpdaf.sdetect.focus.SourceCatalog`) lists the center of gravity of the detected components and the wavelengths of the intensity peaks::

 >>> cat.info()
 [INFO] ra=338.2 dec=-60.6 lbda=[ 6608.75  6610.    6611.25]
 [INFO] ra=338.2 dec=-60.6 lbda=[ 7696.25  7697.5 ]
 [INFO] ra=338.2 dec=-60.6 lbda=[ 4843.75  5055.  ]
 [INFO] ra=338.2 dec=-60.6 lbda=[ 6926.25]
 [INFO] ra=338.2 dec=-60.6 lbda=[ 8220.    8221.25  8222.5 ]
 [INFO] ra=338.2 dec=-60.6 lbda=[ 6821.25  6822.5   6825.    8233.75  8516.25  
 8563.75  8748.75  8803.75]
 ...

 >>> cat.write('cat.fits')
 

Reference
=========

:func:`mpdaf.sdetect.focus.SourceDetect3D <mpdaf.sdetect.focus.SourceDetect3D>` is the classic SourceDetect3D constructor.

:func:`mpdaf.sdetect.focus.SourceDetect3D.p_values <mpdaf.sdetect.focus.SourceDetect3D.p_values>` computes the false detection cube using Student cumulative distribution function.

:func:`mpdaf.sdetect.focus.SourceDetect3D.quick_detection <mpdaf.sdetect.focus.SourceDetect3D.quick_detection>` fast detection of bright voxels and builds a catalog of objects.
