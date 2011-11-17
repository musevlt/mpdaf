"""Test on CalibFile objects to be used with py.test."""

import os
import sys
import numpy

if not os.path.exists("data/MASTER_FLAT.fits"):
    print 'test files are not stored on the git repository to limit its memory size.'
    print 'please download it from http://urania1.univ-lyon1.fr/mpdaf/login'
    print ''

import drs.calibobj as calibobj

def test_init():
    """tests CalibFile initialization"""
    if not os.path.exists("data/MASTER_FLAT.fits"):
        print 'test files are not stored on the git repository to limit its memory size.'
        print 'please download it from http://urania1.univ-lyon1.fr/mpdaf/login'
        print ''
    flat = calibobj.CalibFile("data/MASTER_FLAT.fits")
    data = flat.get_data()
    shape = numpy.shape(data)
    assert shape==(flat.ny, flat.nx)


def test_copy():
    """test copy of CalibFile object"""
    flat = calibobj.CalibFile("data/MASTER_FLAT.fits")
    flat2 = flat.copy()
    value_data1 = flat.get_data()[32,28]
    value_dq1 = flat.get_dq()[32,28]
    value_stat1 = flat.get_stat()[32,28]
    del flat
    value_data2 = flat2.get_data()[32,28]
    value_dq2 = flat2.get_dq()[32,28]
    value_stat2 = flat2.get_stat()[32,28]
    assert value_data1 == value_data2
    assert value_dq1 == value_dq2
    assert value_stat1 == value_stat2

def test_write():
    """test write of CalibFile object"""
    flat = calibobj.CalibFile("data/MASTER_FLAT.fits")
    flat.write("tests/tmp.fits")
    value_data1 = flat.get_data()[32,28]
    value_dq1 = flat.get_dq()[32,28]
    value_stat1 = flat.get_stat()[32,28]
    del flat
    flat = calibobj.CalibFile("tests/tmp.fits")
    value_data2 = flat.get_data()[32,28]
    value_dq2 = flat.get_dq()[32,28]
    value_stat2 = flat.get_stat()[32,28]
    cmd_rm= "rm -f tests/tmp.fits"
    os.system(cmd_rm)
    assert value_data1 == value_data2
    assert value_dq1 == value_dq2
    assert value_stat1 == value_stat2