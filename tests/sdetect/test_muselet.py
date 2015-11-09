"""Test on muselet script."""
import nose.tools
import os
from nose.plugins.attrib import attr

from mpdaf.sdetect import muselet

DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       '..', '..', 'data', 'sdetect')


@attr(speed='slow')
def test_muselet():
    """test MUSELET"""
    continuum, single, raw = muselet(os.path.join(DATADIR, 'minicube.fits'),
                                     nbcube=False, del_sex=True)
    nose.tools.assert_equal(len(continuum), 1)
    nose.tools.assert_equal(len(single), 8)
    nose.tools.assert_equal(len(raw), 39)
