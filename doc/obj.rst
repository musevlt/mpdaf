*******************************************************
Interface for spectra, images and cubes (``mpdaf.obj``)
*******************************************************

The ``mpdaf.obj`` package provides a way to load a MUSE cube created by the MUSE
pipeline (i.e. a 3GB FITS data cube of approximately 300x300x3680 pixels) into a
Python object that manages spatial world-coordinates, wavelength world
coordinates, pixel variances and bad pixel information.


It is then relatively easy to extract smaller cubes, narrow-band images or
spectra from the cube. The world coordinates, associated variances and bad-pixel
masks are propagated into these extracted cubes, images, and spectra.  Many
useful operations like masking, interpolation, re-sampling, smoothing and
profile fitting are also provided.


Contents:

.. toctree::
   :maxdepth: 2

   objformat
   objmask
   objarithmetic
   objtransfo
   objconvol
   objwrite
   spectrum
   image
   cube
   cubelist
   coordinates


Reference/API
=============

.. automodapi:: mpdaf.obj
   :no-main-docstr:
