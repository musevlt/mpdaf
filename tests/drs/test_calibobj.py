"""Test on CalibFile objects to be used with py.test."""

import nose.tools
import numpy
import os
import unittest

from nose.plugins.attrib import attr
from mpdaf.drs import CalibDir

DATA_PATH = "data/drs/masterflat"
DATA_MISSING = not os.path.exists(DATA_PATH)


class TestCalibObj(object):

    def setUp(self):
        self.flat = CalibDir("MASTER_FLAT", DATA_PATH)
        self.flat.progress = False

    def tearDown(self):
        del self.flat

    @unittest.skipIf(DATA_MISSING, "Missing test data (data/drs/masterflat)")
    @attr(speed='slow')
    def test_init(self):
        """Calib objects: tests initialization"""
        data = self.flat[1].get_data()
        shape = numpy.shape(data)
        nose.tools.assert_equal(shape, (self.flat[1].ny, self.flat[1].nx))

    @unittest.skipIf(DATA_MISSING, "Missing test data (data/drs/masterflat)")
    @attr(speed='slow')
    def test_copy(self):
        """Calib objects: tests copy"""
        flat2 = self.flat.copy()
        value_data1 = self.flat[2].get_data()[32, 28]
        value_dq1 = self.flat[2].get_dq()[32, 28]
        value_stat1 = self.flat[2].get_stat()[32, 28]
        value_data2 = flat2[2].get_data()[32, 28]
        value_dq2 = flat2[2].get_dq()[32, 28]
        value_stat2 = flat2[2].get_stat()[32, 28]
        nose.tools.assert_equal(value_data1, value_data2)
        nose.tools.assert_equal(value_dq1, value_dq2)
        nose.tools.assert_equal(value_stat1, value_stat2)
        del flat2

    # def test_write(self):
    #    """tests write of CalibFile objects"""
    #    self.flat[1].write("tmp.fits")
    #    value_data1 = self.flat[1].get_data()[32,28]
    #    value_dq1 = self.flat[1].get_dq()[32,28]
    #    value_stat1 = self.flat[1].get_stat()[32,28]
    #    flat2 = CalibFile("tmp.fits")
    #    value_data2 = flat2.get_data()[32,28]
    #    value_dq2 = flat2.get_dq()[32,28]
    #    value_stat2 = flat2.get_stat()[32,28]
    #    nose.tools.assert_equal(value_data1,value_data2)
    #    nose.tools.assert_equal(value_dq1,value_dq2)
    #    nose.tools.assert_equal(value_stat1,value_stat2)
    #    del flat2
    #    cmd_rm= "rm -f tmp.fits"
    #    os.system(cmd_rm)

    @unittest.skipIf(DATA_MISSING, "Missing test data (data/drs/masterflat)")
    @attr(speed='slow')
    def test_operator(self):
        # Error using memmap objects shared among processes created by the multprocessing module.
        # This error only happen with new version of numpy.
        # See http://mail.scipy.org/pipermail/numpy-discussion/2011-April/056134.html
        # and http://projects.scipy.org/numpy/ticket/1809
        """Calib objects: tests arithmetic functions"""
        flat2 = self.flat.copy()
        sum = self.flat + flat2
        value_data1 = self.flat[2].get_data()[32, 28]
        value_dq1 = self.flat[2].get_dq()[32, 28]
        value_stat1 = self.flat[2].get_stat()[32, 28]
        value_data2 = sum[2].get_data()[32, 28]
        value_dq2 = sum[2].get_dq()[32, 28]
        value_stat2 = sum[2].get_stat()[32, 28]
        nose.tools.assert_equal(value_data1 * 2, value_data2)
        nose.tools.assert_equal(value_dq1, value_dq2)
        nose.tools.assert_equal(value_stat1 * 2, value_stat2)
        del sum
        sub = self.flat - flat2
        value_data2 = sub[2].get_data()[32, 28]
        value_dq2 = sub[2].get_dq()[32, 28]
        value_stat2 = sub[2].get_stat()[32, 28]
        nose.tools.assert_equal(0, value_data2)
        nose.tools.assert_equal(value_dq1, value_dq2)
        nose.tools.assert_equal(value_stat1 * 2, value_stat2)
        del sub
        mul = self.flat * 2
        value_data2 = mul[2].get_data()[32, 28]
        value_dq2 = mul[2].get_dq()[32, 28]
        value_stat2 = mul[2].get_stat()[32, 28]
        nose.tools.assert_equal(value_data1 * 2, value_data2)
        nose.tools.assert_equal(value_dq1, value_dq2)
        nose.tools.assert_equal(value_stat1 * 4, value_stat2)
        del mul
        del flat2

# if __name__=='__main__':
#    if not os.path.exists("data/drs/masterflat/MASTER_FLAT_01.fits"):
#        print 'IOError: file data/drs/masterflat/*.fits not found.'
#        print 'Test files are not stored on the git repository to limit its memory size.'
#        print 'Please download it from http://urania1.univ-lyon1.fr/mpdaf/login'
#        print ''
#    else:
#        unittest.main()
