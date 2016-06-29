**************************
Getting Started with MPDAF
**************************

.. ipython::
   :suppress:
   
   In [4]: import sys
   
   In [4]: from mpdaf import setup_logging
   
Importing MPDAF
---------------
   
MPDAF is divided in sub-packages and themselves are composed of classes.
You must import the class of a sub-package with the syntax:

.. ipython::

   In [1]: from mpdaf.obj import Cube
   
   In [2]: from mpdaf.drs import PixTable

   
Loading your first MUSE datacube
--------------------------------
   
The MUSE datacube is now loaded from a FITS file (in which case the flux and variance values are read from specific extensions):

.. ipython::
  :okwarning:

  @suppress
  In [5]: setup_logging(stream=sys.stdout)
  
  # data and variance arrays read from the file (extension DATA and STAT)
  In [2]: cube = Cube('../data/obj/CUBE.fits')
  
  In [10]: cube.info()


The cube format 1595 x 10 x 20 has 10 x 20 spatial pixels and 1595 spectral pixels.
The format follows the indexing used by Python to handle 3D arrays (see :ref:`objformat` for more information).

Let's compute the reconstructed white light image and display it:

.. ipython::

  In [1]: ima = cube.sum(axis=0)
  
  In [1]: type(ima) 
  
  In [2]: plt.figure()
  
  @savefig Cube1.png width=4in
  In [3]: ima.plot(scale='arcsinh', colorbar='v')
  

Let's now compute the total spectrum of the object:

.. ipython::

  In [1]: sp = cube.sum(axis=(1,2))
  
  In [1]: type(sp) 
  
  In [2]: plt.figure()
  
  @savefig Cube2.png width=4in
  In [3]: sp.plot()
   

Helping
-------

Because different subpackages have very different functionality, further suggestions for getting started are in the documentation for the subpackages:
for example :ref:`cube`, :ref:`image`, :ref:`spectrum` for the ``mpdaf.obj`` package.

Or you can either look at docstrings using the ?

.. ipython::

   In [1]: Cube?
   
.. ipython::

   In [7]: Cube.info?
   
.. ipython::
   
   In [2]: ima.plot?
   
.. ipython::
   :suppress:
   
   In [4]: plt.close("all")

   
   