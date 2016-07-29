# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
from mpdaf.sdetect import Catalog
from numpy.testing import assert_array_equal


def test_catalog():
    cat = Catalog(rows=[[1, 50., 10., 2., -9999],
                        [2, 40., 20., np.nan, 2]],
                  names=('ID', 'ra', 'dec', 'z', 'flag'), masked=True)
    print(cat)
    assert len(cat) == 2
    assert cat.masked
    assert cat.colnames == ['ID', 'RA', 'DEC', 'Z', 'flag']
    assert cat['flag'][0] is np.ma.masked
    assert cat['Z'][1] is np.ma.masked


@pytest.mark.parametrize('fmt,ncols', (('default', 44),
                                       ('working', 42)))
def test_from_sources(source1, source2, fmt, ncols):
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


def test_from_path(source1, source2, tmpdir):
    source1.write(str(tmpdir.join('source1.fits')))
    source2.write(str(tmpdir.join('source2.fits')))
    cat = Catalog.from_path(str(tmpdir))
    assert len(cat) == 2
    # 2 additional columns vs from_sources: FILENAME is added by from_path, and
    # SOURCE_V which was added in the Source.write
    assert len(cat.colnames) == 46

    filename = str(tmpdir.join('cat.fits'))
    cat.write(filename)

    c = Catalog.read(filename)
    assert c.colnames == cat.colnames
    assert len(cat) == 2
    assert isinstance(c, Catalog)
