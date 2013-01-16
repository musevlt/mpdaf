Slicer class
============

.. toctree::
	:maxdepth: 2

Examples::
 
  from mpdaf.MUSE import Slicer
 
  # Convert slice number 4 in CCD numbering to SKY numbering
  print(Slicer.ccd2sky(4))

  # Convert slice number 12 of stack 3 in OPTICAL numbering to CCD numbering
  print(Slicer.optical2sky((2, 12)))

.. autoclass:: mpdaf.MUSE.Slicer
	:members:
	:special-members:
	
	
