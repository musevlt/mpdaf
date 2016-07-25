# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from mpdaf.sdetect import Catalog

from numpy.testing import assert_array_equal

@pytest.mark.xfail(sys.version_info >= (3, 3),
                   reason="not compatible with python 3")
def test_catalog(source1, source2):
    source1.CUBE_V = '0.1'
    source2.CUBE_V = '0.2'
    lines1 = source1.lines['LINE'].data.copy()
    lines2 = source2.lines['LINE'].data.copy()
    cat = Catalog.from_sources([source1, source2], fmt='working')
    print(cat)
    assert len(cat) == 2
    assert len(cat.colnames) == 29
    assert list(cat['ID']) == [1, 32]
    assert list(cat['CUBE_V']) == ['0.1', '0.2']
    assert_array_equal(source1.lines['LINE'].data, lines1)
    assert_array_equal(source2.lines['LINE'].data, lines2)
    cat2 = Catalog.from_sources([source1, source2])
    assert len(cat2) == 2
    assert len(cat2.colnames) == 31
    assert list(cat['ID']) == list(cat2['ID'])
    assert_array_equal(source1.lines['LINE'].data, lines1)
    assert_array_equal(source2.lines['LINE'].data, lines2)
    
    
    
    

