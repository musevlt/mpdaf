# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import pytest
import sys
from mpdaf.sdetect import SourceList


@pytest.mark.xfail(sys.version_info >= (3, 3),
                   reason="not compatible with python 3")
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
