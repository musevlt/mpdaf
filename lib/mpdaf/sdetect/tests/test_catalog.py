# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2017 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2017 Simon Conseil <simon.conseil@univ-lyon1.fr>
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
import six
from astropy.coordinates import SkyCoord
from mpdaf.sdetect import Catalog
from numpy.testing import assert_array_equal, assert_almost_equal

from ...tests.utils import get_data_file


def test_catalog():
    cat = Catalog(rows=[[1, 50., 10., 2., -9999],
                        [2, 40., 20., np.nan, 2]],
                  names=('ID', 'ra', 'dec', 'z', 'flag'), masked=True)
    print(cat)
    assert len(cat) == 2
    assert cat.masked
    assert cat.colnames == ['ID', 'ra', 'dec', 'z', 'flag']
    assert cat['flag'][0] is np.ma.masked
    assert cat['z'][1] is np.ma.masked


@pytest.mark.parametrize('fmt,ncols', (('default', 48),
                                       ('working', 46)))
def test_from_sources(source1, source2, fmt, ncols):
    source1.CUBE_V = '0.1'
    source2.CUBE_V = '0.2'
    source1.UCUSTOM = (1000, 'some custom keyword u.Angstrom')
    source2.UCUSTOM = (2000, 'some custom keyword u.Angstrom')
    source1.FCUSTOM = (1000, 'some custom keyword %.2f')
    source2.FCUSTOM = (2000, 'some custom keyword %.2f')
    source1.UFCUSTOM = (1000.1234, 'some custom keyword u.Angstrom %.2f')
    source2.UFCUSTOM = (2000.1234, 'some custom keyword u.Angstrom %.2f')
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
    with pytest.raises(IOError):
        cat = Catalog.from_path('/not/a/valid/path')

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


@pytest.mark.xfail(six.PY2, reason="issue with astropy coordinates and numpy")
def test_match():
    c1 = Catalog()
    c1['RA'] = np.arange(10, dtype=float)
    c1['DEC'] = np.arange(10, dtype=float)

    c2 = Catalog()
    c2['ra'] = np.arange(20, dtype=float) + 0.5 / 3600
    c2['dec'] = np.arange(20, dtype=float) - 0.5 / 3600

    match = c1.match(c2, colc2=('ra', 'dec'), full_output=False)
    assert len(match) == 10
    assert_almost_equal(match['Distance'], 0.705, decimal=2)

    # create a duplicate match
    c1['RA'][4] = c1['RA'][3] - 0.1 / 3600
    c1['DEC'][4] = c1['DEC'][3] - 0.1 / 3600

    c2['ra'][:5] = np.arange(5, dtype=float) + 0.1 / 3600
    c2['dec'][:5] = np.arange(5, dtype=float) + 0.1 / 3600

    match, nomatch1, nomatch2 = c1.match(c2, colc2=('ra', 'dec'), radius=0.5,
                                         full_output=True)
    assert len(match) == 4
    assert len(nomatch1) == 6
    assert len(nomatch2) == 16


@pytest.mark.xfail(six.PY2, reason="issue with astropy coordinates and numpy")
def test_nearest():
    c1 = Catalog()
    c1['RA'] = np.arange(10, dtype=float)
    c1['DEC'] = np.arange(10, dtype=float)

    res = c1.nearest((5 + 1 / 3600, 5 + 1 / 3600))
    assert_almost_equal(list(res[0]), (5.0, 5.0, 1.41), decimal=2)

    res = c1.nearest((5, 5), ksel=2)
    assert len(res) == 2

    pos = SkyCoord(5, 5, unit='deg', frame='fk5')
    res = c1.nearest(pos, ksel=2)
    assert len(res) == 2

    res = c1.nearest(pos.to_string('hmsdms').split(' '),
                     ksel=10, maxdist=6000)
    assert len(res) == 3


@pytest.mark.xfail(six.PY2, reason="issue with astropy coordinates and numpy")
def test_select(minicube):
    cat = Catalog.read(get_data_file('sdetect', 'cat.txt'), format='ascii')
    im = minicube.mean(axis=0)

    # Note im.shape is (40, 40) and cat has 8 rows all inside the image
    assert len(cat) == 8

    # all sources are in the image
    assert len(cat.select(im.wcs, margin=0)) == 8

    # using a margin removing sources on the edges
    assert len(cat.select(im.wcs, margin=5)) == 4

    # Create a mask with the bottom and left edges masked
    mask = np.ones(im.shape, dtype=bool)
    mask[5:, 5:] = False

    # using a margin removing sources on the edges
    assert len(cat.select(im.wcs, mask=mask)) == 4
    assert len(cat.select(im.wcs, margin=1, mask=mask)) == 4
