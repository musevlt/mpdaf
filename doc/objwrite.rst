*******************
Copy, clone or save
*******************

Copy
----

Spectrum, Image and Cube objects provide a `copy <mpdaf.obj.DataArray.copy>`
method that returns a deep copy of these objects.  For example:

.. ipython::
   :suppress:

   In [4]: import sys

   In [4]: from mpdaf import setup_logging

.. ipython::
   :okwarning:

   @suppress
   In [5]: setup_logging(stream=sys.stdout)

   In [1]: from mpdaf.obj import Spectrum

   In [1]: import numpy as np

   In [2]: specline = Spectrum('../data/obj/Spectrum_lines.fits')

   In [4]: specline.info()

   In [4]: specline.data[42]

   In [3]: spe = specline.copy()

   In [4]: specline.data = np.ones(specline.shape)

   In [4]: spe.info()

   In [4]: spe.data[42]

Clone
-----

The `clone <mpdaf.obj.DataArray.clone>` method returns a new object of the same
shape and coordinates as the original object. However the ``.data`` and ``.var``
attributes of the cloned object are set to None by default.  After cloning an
object, new numpy data and/or variance arrays can be assigned to the ``.data``
and ``.var`` properties of the clone. In the following example, an existing
spectrum is cloned, and then the clone is assigned a zero-filled data array.

.. ipython::

   @suppress
   In [5]: setup_logging(stream=sys.stdout)

   In [5]: spe2 = spe.clone()

   In [1]: spe2.info()

   In [2]: spe2.data = np.zeros(spe.shape)

   In [3]: spe2.info()

   In [3]: spe2.data[42]


Alternatively, the clone method can be passed functions that generate the data
and variances arrays of the cloned object. The clone method passes these
functions the shape of the required data or variance array, and the functions
are expected to return an array of that shape. For example, numpy provides
functions like ``np.zeros`` and ``np.ones`` that can be used for this
purpose. In the following demonstration, an existing spectrum, ``spe``, is
cloned and the cloning method is passed ``np.zeros``, to create a zero-filled
data array.

.. ipython::

   @suppress
   In [5]: setup_logging(stream=sys.stdout)

   In [5]: spe3 = spe.clone(data_init=np.zeros)

   In [1]: spe3.info()

   In [3]: spe3.data[42]

Save
----

The `Spectrum.write <mpdaf.obj.Spectrum.write>`, `Image.write
<mpdaf.obj.Image.write>` and `Cube.write <mpdaf.obj.Cube.write>` methods save
the contents of a Spectrum, Image or Cube to a FITS file.  The data are saved in
a DATA extension, and any variances are saved in a STAT extension.  Depending on
the value of the savemask option, the mask array is either saved as a DQ
extension (the default) or masked data elements are replaced by NaN in the DATA
extension and the mask array is not saved.

The intermediate methods `get_data_hdu <mpdaf.obj.Cube.get_data_hdu>` and
`get_stat_hdu <mpdaf.obj.Cube.get_stat_hdu>` can also be used to generate
astropy.io.fits.ImageHDU objects that correspond to the DATA and STAT
extensions.
