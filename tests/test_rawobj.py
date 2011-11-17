"""Test on RawFile objects to be used with py.test."""

import os
import sys
import numpy

import drs.rawobj as rawobj

def test_init():
    """tests RawFile initialization"""
    if not os.path.exists("data/raw.fits"):
        print 'test files are not stored on the git repository to limit its memory size.'
        print 'please download it from http://urania1.univ-lyon1.fr/mpdaf/login'
        print ''
    raw = rawobj.RawFile("data/raw.fits")
    chan1 = raw.get_channel("CHAN01")
    shape = numpy.shape(chan1.data)
    assert shape==(4240, 4224)

def test_operator():
    """tests arithmetic function on RawFile objects"""
    raw = rawobj.RawFile("data/raw.fits")
    chan1 = raw.get_channel("CHAN01")
    value1 = chan1.data[32,28]

    out = raw - raw
    chan2 = out.get_channel("CHAN01")
    value2 = chan2.data[32,28]
    assert value2 == 0

    out = raw + raw
    chan2 = out.get_channel("CHAN01")
    value2 = chan2.data[32,28]
    assert value2 == 2*value1

    out = raw * raw
    chan2 = out.get_channel("CHAN01")
    value2 = chan2.data[32,28]
    assert value2 == value1 * value1

    out = raw.sqrt()
    chan2 = out.get_channel("CHAN01")
    value2 = chan2.data[32,28]
    assert value2 == numpy.sqrt(value1)

def test_copy():
    """test copy of RawFile object"""
    raw = rawobj.RawFile("data/raw.fits")
    raw2 = raw.copy()
    out = raw - raw2
    out2 = out.copy()
    del out
    chan2 = out2.get_channel("CHAN02")
    value2 = chan2.data[24,12]
    assert value2 == 0

def test_mask():
    """test strimmed and overscan functionalities"""
    raw = rawobj.RawFile("data/raw.fits")
    overscan = raw[1].data[24,12]
    pixel = raw[1].data[240,120]
    out = raw[1].trimmed() * 10
    assert out.data[24,12] == overscan
    assert out.data[240,120] == 10*pixel

    out = raw[1].overscan() * 2
    assert out.data[24,12] == 2*overscan
    assert out.data[240,120] == pixel
