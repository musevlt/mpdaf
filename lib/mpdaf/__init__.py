__version__  = '1.0.1'
__date__     = '2012/09/27 17:35'

import tools
import obj
import drs
try:
    import fusion
except:
    pass

import scipy, numpy, pyfits, pywcs
__info__   = 'numpy %s - scipy %s - pyfits %s - pywcs %s' %(numpy.__version__,scipy.__version__,pyfits.__version__,pywcs.__version__)
