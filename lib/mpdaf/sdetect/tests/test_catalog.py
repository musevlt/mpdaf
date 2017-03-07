# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2016 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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


@pytest.mark.parametrize('fmt,ncols', (('default', 45),
                                       ('working', 43)))
def test_from_sources(source1, source2, fmt, ncols):
    source1.CUBE_V = '0.1'
    source2.CUBE_V = '0.2'
    lines1 = source1.lines['LINE'].data.copy()
    lines2 = source2.lines['LINE'].data.copy()
    cat = Catalog.from_sources([source1, source2], fmt=fmt)
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
    assert len(cat.colnames) == 47

    filename = str(tmpdir.join('cat.fits'))
    cat.write(filename)

    c = Catalog.read(filename)
    assert c.colnames == cat.colnames
    assert len(cat) == 2
    assert isinstance(c, Catalog)
