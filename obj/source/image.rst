Image class
===========

.. toctree::
	:maxdepth: 2
	
Examples::
 
  import numpy as np
  from mpdaf.obj import Image
  from mpdaf.obj import WCS
  
  wcs1 = WCS(crval=0,cdelt=0.2)
  wcs2 = WCS(crval=0,cdelt=0.2,shape=400)
  MyData = np.ones((300,300))
  
  ima = Image(filename="image.fits",ext=1) # image from file without variance (extension number is 1)  
  ima = Image(filename="image.fits",ext=(1,2)) # image from file with variance (extension numbers are 1 and 2)
  ima = Image(shape=300, wcs=wcs1) # image 300x300 filled with zeros
  ima = Image(wcs=wcs1, data=MyData) # image 300x300 filled with MyData
  ima = Image(shape=300, wcs=wcs2) # warning: world coordinates and image have not the same dimensions
				   # ima.wcs = None
  ima = Image(wcs=wcs2, data=MyData) # warning: world coordinates and data have not the same dimensions
				       # ima.wcs = None

.. autoclass:: mpdaf.obj.Image
	:members:
	
	
	
Functions to create a new image
===============================

.. autofunction:: mpdaf.obj.gauss_image

Examples::
 
    import numpy as np
    from mpdaf.obj import gauss_image
    from mpdaf.obj import WCS
    wcs = WCS (cdelt=(0.2,0.3), crval=(8.5,12),shape=(40,30))
    ima = gauss_image(wcs=wcs,width=(1,2),factor=2, rot = 60)
    ima.plot()
    gauss = ima.gauss_fit(pos_min=(4, 7), pos_max=(13,17), cont=0, plot=True)
    gauss.print_param()

.. autofunction:: mpdaf.obj.moffat_image

.. autofunction:: mpdaf.obj.make_image

.. autofunction:: mpdaf.obj.composite_image

Examples::
 
  import numpy as np
  from mpdaf.obj import Image
  from mpdaf.obj import composite_image
  
  stars = Image(filename="stars.fits")
  lowz = Image(filename="lowz.fits")
  highz = Image(filename="highz.fits")
  imalist = [stars, lowz, highz]
  tab = zip(imalist,linspace(250,0,3),ones(3)*100)
  p1 = composite_image(tab,cuts=(0,99.5),mode='sqrt')
  p1.show()
  p1.save('test_composite.jpg')

