__version__  = '1.0.2'
__date__     = '2012/11/19 16:47'

import tools
import obj
import drs
import MUSE
try:
    import fusion
except:
    pass

import scipy, numpy, pyfits, pywcs
__info__   = 'numpy %s - scipy %s - pyfits %s - pywcs %s' %(numpy.__version__,scipy.__version__,pyfits.__version__,pywcs.__version__)

CPU = 0
