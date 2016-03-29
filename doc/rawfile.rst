Python interface for raw FITS files
************************************

`RawFile <mpdaf.drs.RawFile>` python object can handle raw MUSE CCD image, with 24 extensions. 
RawFile object can be read and written to disk as a multi-extension FITS file. To use efficiently the memory, the file is open in memory mapping mode when a RawFile object is created from an input FITS file: i.e. the arrays are not in memory unless they are used by the script.

A few functions can be performed on the RawFile object: recontruct trimmed image of a channel, compute the bias level of channels, mask overscanned pixels ... 
In most cases multiprocessing is used because the same process is applied on each channel.
`Channel <mpdaf.drs.Channel>` python object manages input/output and methods on a channel, i.e. an extension of the raw FITS file.
   
RawFile object
==============

An RawFile object O consists of:

+------------------+-------------------------------------------------------------------+
| Component        | Description                                                       |
+==================+===================================================================+
| O.filename       | raw FITS file name. None if any.                                  |
+------------------+-------------------------------------------------------------------+
| O.channels       | dictionary (extname,Channel) containing the list of extensions    |
+------------------+-------------------------------------------------------------------+
| O.primary_header | FITS primary header                                               |
+------------------+-------------------------------------------------------------------+
| O.nx             |  lengths of data in X                                             |
+------------------+-------------------------------------------------------------------+
| O.ny             |  lengths of data in Y                                             |
+------------------+-------------------------------------------------------------------+
| O.next           | number of extensions                                              |
+------------------+-------------------------------------------------------------------+
| O.progress       | boolean, if True progress of multiprocessing tasks are displayed. |
+------------------+-------------------------------------------------------------------+


Channel object
==============

An Channel object O consists of:

+-----------+--------------------------------------------------------------------+
| Component | Description                                                        |
+===========+====================================================================+
| O.extname | extension name                                                     |
+-----------+--------------------------------------------------------------------+
| O.header  | extension FITS header                                              |
+-----------+--------------------------------------------------------------------+
| O.data    | array containing the pixel values of the image extension           |
+-----------+--------------------------------------------------------------------+
| O.nx      |  lengths of data in X                                              |
+-----------+--------------------------------------------------------------------+
| O.ny      |  lengths of data in Y                                              |
+-----------+--------------------------------------------------------------------+
| O.mask    | boolean arrays (TRUE for overscanned pixels, FALSE for the others) |
+-----------+--------------------------------------------------------------------+


Tutorials
=========

Preliminary imports for all tutorials::

  >>> from mpdaf.drs import RawFile
  >>> import matplotlib.cm as cm


Tutorial 1: RawFile Creation and display a channel image.
---------------------------------------------------------

A RawFile object is created from a raw MUSE CCD image ::

  >>> raw = RawFile('MUSE_IQE_MASK158_0001.fits')
  >>> raw.info()
  MUSE_IQE_MASK158_0001.fits
  Nb extensions:  24 (loaded:24 ['CHAN19', 'CHAN18', 'CHAN15', 'CHAN14', 'CHAN17', 'CHAN16', 'CHAN11', 'CHAN10', 'CHAN13', 'CHAN12', 'CHAN06', 'CHAN02', 'CHAN21', 'CHAN04', 'CHAN23', 'CHAN08', 'CHAN09', 'CHAN20', 'CHAN07', 'CHAN22', 'CHAN05', 'CHAN24', 'CHAN03', 'CHAN01'])
  format: (4224,4240)


Let's extract the channel 12 and display its image::

  >>> chan = raw.get_channel('CHAN12')
  >>> ima = chan.get_image()
  >>> ima.plot(cmap=cm.copper)
  
.. figure::  _static/raw/ima.png
   :align:   center  

Masking overscanned pixels::

  >>> ima = chan.get_image_mask_overscan()
  >>> ima.plot(cmap=cm.copper)
  
.. figure::  _static/raw/ima_mask.png
   :align:   center 

Or displaying only overscan area::

  >>> ima = chan.get_image_just_overscan()
  >>> ima.plot(cmap=cm.copper)
  
.. figure::  _static/raw/ima_overscan.png
   :align:   center 
   
`mpdaf.drs.Channel.get_trimmed_image <mpdaf.drs.Channel.get_trimmed_image>` method returns an Image object without over scanned pixels. If bias option is used, median value of the overscanned pixels is subtracted on each detector::

  >>> ima = chan.get_trimmed_image(bias=True)
  >>> ima.plot(cmap=cm.copper)
  
.. figure::  _static/raw/ima_trimmed.png
   :align:   center 
   
   
Tutorial 2: White image fast reconstruction.
--------------------------------------------

Let's compute the reconstructed white light image and display it::

  >>> ima = raw.reconstruct_white_image()
  >>> ima.info()
      
  >>> ima.plot(cmap=cm.copper)
  
.. figure::  _static/raw/ima_white.png
   :align:   center 

The fast reconstruction used a mask file produced by the drs. By default, the mask constructed during the PAE global test is used.

`mpdaf.drs.RawFile.plot_white_image <mpdaf.drs.RawFile.plot_white_image>` method reconstructs the white image and plots it. It plots also a channel image and provides mouse interaction between the 2 parts in order for the user to be able to click somewhere on one display and exhibit the corresponding data in the other display::

  >>> raw.plot_white_image()
  To select on other channel/slice, click on the images with the right mouse button.
  
.. figure::  _static/raw/visu1.png
   :align:   center 

The selected slice, which corresponds to a single row of pixels on the reconstructed image, is surrounded by a red colored line on the two displays.
Select a slice by clicking with the right mouse button on the right display (channel image), automatically update the slice display on the white image. As a reverse process,
selecting one of the 48 slices on the white image updates the position of the slice on the CCD image. 

.. figure::  _static/raw/visu2.png
   :align:   center 

Select a channel by clicking with the right mouse button on the left display (Reconstructed Image), automatically update the display in the raw exposure image and surround the selected channel by a blue colored line.

.. figure::  _static/raw/visu3.png
   :align:   center 


Reference
=========

`mpdaf.drs.RawFile <mpdaf.drs.RawFile>` is the classic RawFile constructor.

`mpdaf.drs.RawFile.copy <mpdaf.drs.RawFile.copy>` returns a copy of the RawFile object.

`mpdaf.drs.RawFile.info <mpdaf.drs.RawFile.info>` prints information.

`mpdaf.drs.RawFile.write <mpdaf.drs.RawFile.write>` saves the object in a FITS file.


Getters and setters
-------------------

`mpdaf.drs.RawFile.get_keywords <mpdaf.drs.RawFile.get_keywords>` returns a FITS header keyword value.

`mpdaf.drs.RawFile.get_channel <mpdaf.drs.RawFile.get_channel>` returns a Channel object corresponding to an extension name.

`mpdaf.drs.RawFile.get_channels_extname_list <mpdaf.drs.RawFile.get_channels_extname_list>` returns the list of existing channels names.

`mpdaf.drs.RawFile['CHANxx'] <mpdaf.drs.RawFile.__getitem__>` returns a Channel object.

`mpdaf.drs.RawFile['CHANxx'] = mpdaf.drs.Channel <mpdaf.drs.RawFile.__setitem__>` sets channel object in RawFile.channels['CHANxx']


Arithmetic
----------

`\+ <mpdaf.drs.RawFile.__add__>` makes a addition.

`\- <mpdaf.drs.RawFile.__sub__>` makes a subtraction .

`\* <mpdaf.drs.RawFile.__mul__>` makes a multiplication.

`/ <mpdaf.drs.RawFile.__div__>` makes a division.

`\*\* <mpdaf.drs.RawFile.__pow__>`  computes the power exponent of data extensions.

`mpdaf.drs.RawFile.sqrt <mpdaf.drs.RawFile.sqrt>` computes the square root of each channel.


Plotting
--------

`mpdaf.drs.RawFile.plot <mpdaf.drs.RawFile.plot>` plots the raw images.

`mpdaf.drs.RawFile.plot_white_image <mpdaf.drs.RawFile.plot_white_image>` reconstructs the white image of the FOV using a mask file and plots this image.



Transformation
--------------

`mpdaf.drs.RawFile.overscan <mpdaf.drs.RawFile.overscan>` returns a RawFile object containing only overscanned pixels.

`mpdaf.drs.RawFile.trimmed <mpdaf.drs.RawFile.trimmed>` returns a RawFile object containing only reference to the valid pixels.

`mpdaf.drs.RawFile.reconstruct_white_image <mpdaf.drs.RawFile.reconstruct_white_image>` reconstructs the white image using a mask file.


Function on Channel object
--------------------------

`mpdaf.drs.Channel <mpdaf.drs.Channel>` object corresponds to an extension of a raw FITS file.

`mpdaf.drs.Channel.get_bias_level <mpdaf.drs.Channel.get_bias_level>` computes median value of the overscanned pixels for a given detector.

`mpdaf.drs.Channel.get_image <mpdaf.drs.Channel.get_image>` returns an Image object.

`mpdaf.drs.Channel.get_image_just_overscan <mpdaf.drs.Channel.get_image_just_overscan>` returns an Image object in which only overscanned pixels are not masked.

`mpdaf.drs.Channel.get_image_mask_overscan <mpdaf.drs.Channel.get_image_mask_overscan>` returns an Image object in which overscanned pixels are masked.

`mpdaf.drs.Channel.get_trimmed_image <mpdaf.drs.Channel.get_trimmed_image>` returns an Image object without over scanned pixels (bias could be subtracted).
