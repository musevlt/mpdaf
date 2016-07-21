# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import pytest
from mpdaf.sdetect import SourceList


def test_sourcelist(tmpdir, source1, source2):
    with pytest.raises(ValueError):
        SourceList.from_path('not/a/real/path')

    source1.CUBE_V = '0.1'
    source2.CUBE_V = '0.2'
    source1.write(str(tmpdir.join('source1.fits')))
    source2.write(str(tmpdir.join('source2.fits')))
    slist = SourceList.from_path(str(tmpdir))
    assert len(slist) == 2

    with pytest.raises(ValueError):
        slist.write('cat', path='not/a/real/path')

    slist.write('out', path=str(tmpdir))
    slist.write('out', path=str(tmpdir), overwrite=True)
    assert tmpdir.join('out.fits').isfile()
    assert tmpdir.join('out').join('out-0001.fits').isfile()
    assert tmpdir.join('out').join('out-0032.fits').isfile()
