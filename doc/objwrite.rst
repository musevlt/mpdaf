*******************
Copy, clone or save
*******************

Copy
----

`copy <mpdaf.obj.DataArray.copy>` returns a new copy of a Spectrum/Image/Cube object.

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

`clone <mpdaf.obj.DataArray.clone>` returns a new object of the same shape and coordinates with the data and var attributes set to None by default.
We can set directly a Numpy array to the cloned object:

.. ipython::

   @suppress
   In [5]: setup_logging(stream=sys.stdout)
   
   In [5]: spe2 = spe.clone()

   In [1]: spe2.info()
   
   In [2]: spe2.data = np.zeros(spe.shape)
   
   In [3]: spe2.info()
   
   In [3]: spe2.data[42]
   

Or we can fill the data and/or the var arrays by giving an optional function:

.. ipython::

   @suppress
   In [5]: setup_logging(stream=sys.stdout)
   
   In [5]: spe3 = spe.clone(data_init=np.zeros)

   In [1]: spe3.info()
   
   In [3]: spe3.data[42]

Save
----

`Spectrum.write <mpdaf.obj.Spectrum.write>`,  `Image.write <mpdaf.obj.Image.write>` and `Cube.write <mpdaf.obj.Cube.write>` save the object in a FITS file.
``O.data`` is saved in the DATA extension and ``O.var`` is saved in the STAT extension.
According to the savemask option, the mask array is saved in DQ extension (default) or masked data are replaced by nan in DATA extension or masked array is not saved.

The intermediate methods `get_data_hdu <mpdaf.obj.Cube.get_data_hdu>` and `get_stat_hdu <mpdaf.obj.Cube.get_stat_hdu>` could also be used to have the astropy.io.fits.ImageHDU corresponding to the DATA and STAT extensions.