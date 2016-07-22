# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from mpdaf.sdetect import Catalog


def test_catalog(source1, source2):
    source1.CUBE_V = '0.1'
    source2.CUBE_V = '0.2'
    cat = Catalog.from_sources([source1, source2], fmt='working')
    print(cat)
    assert len(cat) == 2
    assert len(cat.colnames) == 29
    assert list(cat['ID']) == [1, 32]
    assert list(cat['CUBE_V']) == ['0.1', '0.2']
