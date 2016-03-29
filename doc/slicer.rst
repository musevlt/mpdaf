Python interface for MUSE slicer numbering scheme
*************************************************

The Slicer calls contains a set of static methods to convert a 
slice number between the various numbering schemes. The definition
of the various numbering schemes and the conversion table can be 
found in the “Global Positioning System” document (VLT-TRE-MUSE-14670-0657).

All the methods are static and thus there is no need to instanciate an object
to use this class.


Tutorial
========

  >>> from mpdaf.MUSE import Slicer

  >>> # Convert slice number 4 in CCD numbering to SKY numbering
  >>> print(Slicer.ccd2sky(4))
  10

  >>> # Convert slice number 12 of stack 3 in OPTICAL numbering to CCD numbering
  >>> print(Slicer.optical2sky((2, 12)))
  25


References
==========

:func:`mpdaf.MUSE.Slicer.ccd2optical <mpdaf.MUSE.Slicer.ccd2optical>` converts slicer number from CCD to OPTICAL numbering scheme. 

:func:`mpdaf.MUSE.Slicer.ccd2sky <mpdaf.MUSE.Slicer.ccd2sky>` converts slicer number from CCD to SKY numbering scheme. 

:func:`mpdaf.MUSE.Slicer.optical2ccd <mpdaf.MUSE.Slicer.optical2ccd>` converts slicer number from OPTICAL to CCD numbering scheme. 

:func:`mpdaf.MUSE.Slicer.optical2sky <mpdaf.MUSE.Slicer.optical2sky>` converts slicer number from OPTICAL to SKY numbering scheme. 

:func:`mpdaf.MUSE.Slicer.sky2ccd <mpdaf.MUSE.Slicer.sky2ccd>` converts slicer number from SKY to CCD numbering scheme. 

:func:`mpdaf.MUSE.Slicer.sky2optical <mpdaf.MUSE.Slicer.sky2optical>` converts slicer number from SKY to OPTICAL numbering scheme. 


