*******************************************************
Interface for spectra, images and cubes (``mpdaf.obj``)
*******************************************************

The ``mpdaf.obj`` package provides a way to load a MUSE cube created by the MUSE pipeline (i.e. a FITS data cube of 3GB, ~ 300x300x3680 pixels)
into a Python object handling the world coordinates, the variance and the bad pixels information.


It is then relatively easy to extract smaller cubes or narrow-band images from a cube, spectra from an aperture,
and perform common operations like masking, interpolating, re-sampling, smoothing, profile fitting...
The world coordinates, the associated variance and the mask are propagated into the extracted cube, image, or spectra.


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
