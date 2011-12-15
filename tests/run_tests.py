import unittest

import os
import sys
sys.path.append('tests/drs')
sys.path.append('tests/obj')

from test_calibobj import TestCalibObj
from test_rawobj import TestRawObj
from test_coords import TestWCS
from test_coords import TestWaveCoord
from test_objs import TestObj

if __name__=='__main__':

    if not os.path.exists("data/drs/masterflat/MASTER_FLAT_01.fits"):
        print 'IOError: file data/drs/masterflat/*.fits not found.'
        print 'Test files are not stored on the git repository to limit its memory size.'
        print 'Please download it from http://urania1.univ-lyon1.fr/mpdaf/login'
        print ''
    else:
        loader = unittest.TestLoader()

        suite = loader.loadTestsFromTestCase(TestCalibObj)
        suite.addTests(loader.loadTestsFromTestCase(TestRawObj))
        suite.addTests(loader.loadTestsFromTestCase(TestWCS))
        suite.addTests(loader.loadTestsFromTestCase(TestWaveCoord))
        suite.addTests(loader.loadTestsFromTestCase(TestObj))

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
