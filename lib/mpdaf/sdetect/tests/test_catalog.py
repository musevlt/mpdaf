# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import pytest
from mpdaf.sdetect import Catalog
from numpy.testing import assert_array_equal


@pytest.mark.parametrize('fmt,ncols', (('default', 44),
                                       ('working', 42)))
def test_catalog(source1, source2, fmt, ncols):
    source1.CUBE_V = '0.1'
    source2.CUBE_V = '0.2'
    lines1 = source1.lines['LINE'].data.copy()
    lines2 = source2.lines['LINE'].data.copy()
    cat = Catalog.from_sources([source1, source2], fmt=fmt)
    print(cat)
    assert len(cat) == 2
    assert len(cat.colnames) == ncols
    assert list(cat['ID']) == [1, 32]
    assert list(cat['CUBE_V']) == ['0.1', '0.2']
    assert_array_equal(source1.lines['LINE'].data, lines1)
    assert_array_equal(source2.lines['LINE'].data, lines2)
