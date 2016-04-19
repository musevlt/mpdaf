"""Test on SEA algorythm."""

from __future__ import absolute_import, division

import nose.tools
import os
from nose.plugins.attrib import attr

from mpdaf.obj import Cube, Image
from mpdaf.sdetect import SEA

from astropy.table import Table

DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       '..', '..', 'data', 'sdetect')


@attr(speed='slow')
def test_SEA():
    """test SEA"""
    cube = Cube(os.path.join(DATADIR, 'minicube.fits'))
    ima = Image(os.path.join(DATADIR, 'a478hst.fits'))
    cat = Table.read(os.path.join(DATADIR, 'cat.txt'), format='ascii')
    sources = SEA(cat, cube, {'hst': ima})

    nose.tools.assert_equal(len(sources), 8)
