DataArray class
===============

The DataArray class is the parent of the :class:`Cube
<mpdaf.obj.Cube>`, :class:`Image <mpdaf.obj.Image>` and
:class:`Spectrum <mpdaf.obj.Spectrum>` classes. Its primary purpose is
to store pixel values in a masked numpy array. For Cube objects this
is a 3D array indexed in the order
[wavelength,declination,right-ascension]. For Image objects it is a 2D
array indexed in the order [declination,right-ascension]. For Spectrum
objects it is a 1D spectrum.

There are a number of optional features. The ``.var`` member
optionally holds an array of variances for each value in the data
array. For cubes and spectra, the wavelengths of the spectral pixels
may be specified in the ``.wave`` member. For cubes and images, the
world-coordinates of the image pixels may be specified in the ``.wcs``
member.

When a DataArray object is constructed from a FITS file, the name of
the file and the file's primary header are recorded. If the data are
read from a FITS extension, the header of this extension is also
recorded.  Primary and data FITS headers can also be passed to the
DataArray constructor. Where FITS headers are not provided, generic
headers are substituted.

Methods are provided for masking and unmasking pixels, and performing
basic arithmetic operations on pixels.

.. autoclass:: mpdaf.obj.DataArray
    :members:
    :special-members:




