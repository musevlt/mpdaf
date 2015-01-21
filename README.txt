 = MPDAF =

== Description ==

MPDAF is the MUSE Python Data Analysis Framework.
Its goal is to develop a python framework in view of the analysis of MUSE data in the context of the GTO. 


== Installation ==

The various software required are:

 * Python (version 2.6 or 2.7)
 * IPython
 * numpy (version 1.6.2 or above)
 * scipy (version 0.10.1 or above)
 * matplotlib (version 1.1.0 or above)
 * astropy (version 0.4 or above)
 * nose
 * PIL
 * numexpr
 * python-development package
 * pkg-config tool
 * C numerics library
 * C CFITSIO library
 * C OpenMP library (optional)
 

To install the mpdaf package:
python setup.py build
python setup.py install


setup.py tries to use pkg-config to find the correct compiler flags and library flags.
Note that on MAC OS, OpenMP is not used by default because clang doesn't support OpenMp.
To force it, the USEOPENMP environment variable can be set to anything except an empty string:
sudo USEOPENMP=0 CC=<local path of gcc> python setup.py build 
 
setup.py informs you that the fusion package is not found.
But it's just a warning, it's not blocking and you can continue to install mpdaf.
To install the fusion submodule: python setup.py fusion
  
  
runs the test:
python setup.py test


== Version History ==

 1.0::
  Initial release
 1.0.1::
  Add fusion module and cube iterators
 1.0.2::
  Bugs correction
 1.1.0::
  Bugs correction + simple MUSE LSF model
 1.1.1::
  Optimization and link to the FSFModel of Camille Parisel (DAHLIA)
 1.1.2::
  Bugs correction
 1.1.3::
  Pixel table visualization
 1.1.4::
  Bugs correction + link to the sky subtraction tool zap (Zurich)
 1.1.5::
  Bugs correction
 1.1.6::
  update of the sky subtraction tool zap
 1.1.7::
  PixTable update
 1.1.8:
  zap v 0.3
 1.1.9:
  zap v 0.4
 1.1.10:
  zap v 0.5
 1.1.11:
   zap v 0.5.1
 1.1.12:
  zap v 0.6
  galpak tool
 1.1.13
  Pixel table setters and transformation methods
  CubeList class
 1.1.14
  New method to correct slice effects on pixel tables
