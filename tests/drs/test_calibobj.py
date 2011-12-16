"""Test on CalibFile objects to be used with py.test."""

import os
import sys
import numpy
import unittest

from drs import CalibDir
from drs import CalibFile

class TestCalibObj(unittest.TestCase):
    
    def setUp(self):
        self.flat = CalibDir("MASTER_FLAT","data/drs/masterflat")
        self.flat.progress = False
        
    def tearDown(self):
        del self.flat

    def test_init(self):
        """tests CalibDir initialization"""
        data = self.flat[1].get_data()
        shape = numpy.shape(data)
        self.assertEqual(shape, (self.flat[1].ny, self.flat[1].nx))

    def test_copy(self):
        """tests copy of CalibDir objects"""
        flat2 = self.flat.copy()
        value_data1 = self.flat[2].get_data()[32,28]
        value_dq1 = self.flat[2].get_dq()[32,28]
        value_stat1 = self.flat[2].get_stat()[32,28]
        value_data2 = flat2[2].get_data()[32,28]
        value_dq2 = flat2[2].get_dq()[32,28]
        value_stat2 = flat2[2].get_stat()[32,28]
        self.assertEqual(value_data1,value_data2)
        self.assertEqual(value_dq1,value_dq2)
        self.assertEqual(value_stat1,value_stat2)
        del flat2
    
    #def test_write(self):
    #    """tests write of CalibFile objects"""
    #    self.flat[1].write("tmp.fits")
    #    value_data1 = self.flat[1].get_data()[32,28]
    #    value_dq1 = self.flat[1].get_dq()[32,28]
    #    value_stat1 = self.flat[1].get_stat()[32,28]
    #    flat2 = CalibFile("tmp.fits")
    #    value_data2 = flat2.get_data()[32,28]
    #    value_dq2 = flat2.get_dq()[32,28]
    #    value_stat2 = flat2.get_stat()[32,28]
    #    self.assertEqual(value_data1,value_data2)
    #    self.assertEqual(value_dq1,value_dq2)
    #    self.assertEqual(value_stat1,value_stat2)
    #    del flat2
    #    cmd_rm= "rm -f tmp.fits"
    #    os.system(cmd_rm)
        
    def test_operator(self):
        """tests arithmetic functions on CalibDir and CalibFile objects"""
        flat2 = self.flat.copy()
        sum = self.flat + flat2
        value_data1 = self.flat[2].get_data()[32,28]
        value_dq1 = self.flat[2].get_dq()[32,28]
        value_stat1 = self.flat[2].get_stat()[32,28]
        value_data2 = sum[2].get_data()[32,28]
        value_dq2 = sum[2].get_dq()[32,28]
        value_stat2 = sum[2].get_stat()[32,28]
        self.assertEqual(value_data1*2,value_data2)
        self.assertEqual(value_dq1,value_dq2)
        self.assertEqual(value_stat1*2,value_stat2)
        del sum
        sub = self.flat - flat2
        value_data2 = sub[2].get_data()[32,28]
        value_dq2 = sub[2].get_dq()[32,28]
        value_stat2 = sub[2].get_stat()[32,28]
        self.assertEqual(0,value_data2)
        self.assertEqual(value_dq1,value_dq2)
        self.assertEqual(value_stat1*2,value_stat2)
        del sub
        mul = self.flat * 2
        value_data2 = mul[2].get_data()[32,28]
        value_dq2 = mul[2].get_dq()[32,28]
        value_stat2 = mul[2].get_stat()[32,28]
        self.assertEqual(value_data1*2,value_data2)
        self.assertEqual(value_dq1,value_dq2)
        self.assertEqual(value_stat1*4,value_stat2)
        del mul
        del flat2
        
if __name__=='__main__':
    if not os.path.exists("data/drs/masterflat/MASTER_FLAT_01.fits"):
        print 'IOError: file data/drs/masterflat/*.fits not found.'
        print 'Test files are not stored on the git repository to limit its memory size.'
        print 'Please download it from http://urania1.univ-lyon1.fr/mpdaf/login'
        print ''
    else:
        unittest.main()