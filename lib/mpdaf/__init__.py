__version__  = '1.0.1-dev'
__date__     = '2012/08/27 17:00'

import tools
import obj
import drs
try:
    import fusion
except:
    pass

import scipy, numpy, pyfits, pywcs, prettytable
__info__   = 'numpy %s - scipy %s - pyfits %s - pywcs %s - prettytable %s' %(numpy.__version__,scipy.__version__,pyfits.__version__,pywcs.__version__,prettytable.__version__)
