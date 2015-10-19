"""Test on muselet script."""
import nose.tools
from nose.plugins.attrib import attr

from mpdaf.sdetect import muselet

class TestMuselet():
        
    @attr(speed='slow')
    def test_muselet(self):
        """test MUSELET"""
        continuum, single, raw = muselet('data/sdetect/minicube.fits', nbcube=False, del_sex=True)
        nose.tools.assert_equal(len(continuum), 1)
        nose.tools.assert_equal(len(single), 8)
        nose.tools.assert_equal(len(raw), 39)
