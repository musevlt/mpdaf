"""Test on SEA algorythm."""
import nose.tools
from nose.plugins.attrib import attr

from mpdaf.obj import Cube, Image
from mpdaf.sdetect import SEA

from astropy.table import Table


class TestSEA():

    @attr(speed='slow')
    def test_SEA(self):
        """test SEA"""
        cube = Cube('data/sdetect/minicube.fits')
        ima = Image('data/sdetect/a478hst.fits')
        cat = Table.read('../mpdaf.git/data/sdetect/cat.txt', format='ascii')
        sources = SEA(cat, cube, {'hst': ima})

        nose.tools.assert_equal(len(sources), 8)
