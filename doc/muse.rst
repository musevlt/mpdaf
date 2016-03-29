MUSE specific tools (``mpdaf.MUSE``)
====================================

The ``mpdaf.MUSE`` package contains tools to manipulate MUSE specific data.

Python interface for MUSE slicer numbering scheme
-------------------------------------------------

The `mpdaf.MUSE.Slicer` class contains a set of static methods to convert
a slice number between the various numbering schemes. The definition of the
various numbering schemes and the conversion table can be found in the *“Global
Positioning System”* document (VLT-TRE-MUSE-14670-0657).

All the methods are static and thus there is no need to instanciate an object
to use this class.


Example::

    >>> from mpdaf.MUSE import Slicer

    >>> # Convert slice number 4 in CCD numbering to SKY numbering
    >>> print(Slicer.ccd2sky(4))
    10

    >>> # Convert slice number 12 of stack 3 in OPTICAL numbering to CCD numbering
    >>> print(Slicer.optical2sky((2, 12)))
    25

MUSE PSF models
---------------

'qsim_v1' LSF model
^^^^^^^^^^^^^^^^^^^

This is a simple model where the LSF is supposed to be constant over the filed
of view. It uses a simple parametric model of variation with wavelength.
    
The model is a convolution of a step function with a gaussian. The resulting
function is then sample by the pixel size::

    LSF = T(y2+dy/2) - T(y2-dy/2) - T(y1+dy/2) + T(y1-dy/2)

    T(x) = exp(-x**2/2) + sqrt(2*pi)*x*erf(x/sqrt(2))/2

    y1 = (y-h/2) / sigma

    y2 = (y+h/2) / sigma

The slit width is assumed to be constant (h = 2.09 pixels).  The gaussian sigma
parameter is a polynomial approximation of order 3 with wavelength::

    c = [-0.09876662, 0.44410609, -0.03166038, 0.46285363]

    sigma(x) = c[3] + c[2]*x + c[1]*x**2 + c[0]*x**3


Tutorial::

    >>> from mpdaf.MUSE import LSF

    >>> lsf = LSF(type='qsim_v1')

    >>> lsf.get_LSF(lbda=6000,step=1.25,size=11)
    array([  1.35563937e-15,   1.29241981e-09,   2.87088720e-05,
            1.45978758e-02,   2.55903993e-01,   4.58938842e-01,
            2.55903993e-01,   1.45978758e-02,   2.87088720e-05,
            1.29241998e-09,   1.69454922e-15])

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> plt.plot(np.arange(-5,6),lsf.get_LSF(lbda=6000,step=1.25,size=11),drawstyle='steps-mid')
    >>> plt.show()


.. image:: psf_images/simple_LSF.png



Reference/API
-------------

.. automodapi:: mpdaf.MUSE
