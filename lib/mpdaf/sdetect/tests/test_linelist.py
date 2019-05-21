# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2018-2019 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c) 2018-2019 Yannick Roehlly <yannick.roehlly@univ-lyon1.fr>

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

import pytest
from numpy.testing import assert_allclose
from mpdaf.sdetect import get_emlines, z_from_linepos


def test_linelist():
    assert len(get_emlines()) == 61
    assert len(get_emlines(family=1)) == 9
    assert len(get_emlines(doublet=True, z=3.0, vac=False)) == 12
    assert get_emlines(iden='FOO') is None

    em = get_emlines(z=0, vac=False, lbrange=(4750, 9350), margin=20, sel=0,
                     ltype='is')
    assert len(em) == 2
    assert em['id'].tolist() == ['MGB', 'NAD']


def test_restframe():
    em = get_emlines(iden='LYALPHA', z=3.0, restframe=True)[0]
    assert_allclose(em[1], 1215.67)
    em = get_emlines(iden='LYALPHA', z=3.0, restframe=False)[0]
    assert_allclose(em[1], 4862.68)

    em = get_emlines(z=1, vac=False, lbrange=(4750, 9350), margin=20, sel=0,
                     ltype='is', restframe=True)
    assert len(em) == 6
    assert em['id'].tolist() == ['FEII2587', 'FEII2600', 'MGI2853', 'CAK',
                                 'CAH', 'CAG']


def test_table():
    em = get_emlines(table=True, iden=('LYALPHA', 'HALPHA'))
    assert_allclose(em[0]['LBDA_OBS'], 1215.67)
    assert_allclose(em[1]['LBDA_OBS'], 6564.61)

    em = get_emlines(table=True, iden=('LYALPHA', 'HALPHA'), vac=False)
    assert_allclose(em[0]['LBDA_OBS'], 1215.67)
    assert_allclose(em[1]['LBDA_OBS'], 6562.794)


def test_z_linepos():
    assert_allclose(z_from_linepos(iden='LYALPHA', wavelength=4862.68), 3)

    with pytest.raises(ValueError):
        z_from_linepos(iden='FOO', wavelength=4862.68)


def test_z_linepos_air():
    assert_allclose(
        z_from_linepos(iden='LYALPHA', wavelength=4861.32, vac=False),
        3, atol=0.1)
