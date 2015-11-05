"""Test on Source objects."""
import nose.tools
from nose.plugins.attrib import attr

from mpdaf.sdetect import Catalog
from mpdaf.sdetect import FOCUS


class TestFOCUS():

    def setUp(self):
        self.cube = 'data/sdetect/minicube.fits'
        self.expmap = 'data/sdetect/miniexpmap.fits'

    def tearDown(self):
        del self.cube
        del self.expmap

    @attr(speed='fast')
    def test_quick(self):
        """FOCUS: tests quick detection"""
        foc = FOCUS(self.cube, self.expmap)
        p_values = foc.p_values()
        imaobj, sources = foc.quick_detection(p_values, p0=1e-3)
        nose.tools.assert_equal(len(sources), 1)
