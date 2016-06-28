**************************
Getting Started with MPDAF
**************************

.. ipython::
   :suppress:
   
   In [4]: import sys
   
   In [4]: from mpdaf import setup_logging
   
MPDAF is divided in sub-packages and themselves are composed of classes.
You must import the class of a sub-package with the syntax:

.. ipython::

   In [1]: from mpdaf.obj import Cube
   
Because different subpackages have very different functionality, further suggestions for getting started are in the documentation for the subpackages.
For example you can see :ref:`cube`.

Or you can either look at docstrings for particular a package or object using the ?:

.. ipython::

   In [1]: Cube?

   In [7]: Cube.info?

   
   