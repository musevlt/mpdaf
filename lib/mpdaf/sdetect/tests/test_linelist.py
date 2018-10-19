# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2018 Yannick Roehlly <yannick.roehlly@univ-lyon1.fr>

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

from numpy.testing import assert_allclose
from ..linelist import get_emlines


def test_linelist():
    em = get_emlines()
    assert len(em) == 54

    em = get_emlines(doublet=True, z=3.0, vac=False)
    assert len(em) == 6

    em = get_emlines(z=0, vac=False, lbrange=(4750, 9350), margin=20, sel=0,
                     ltype='is')
    assert len(em) == 2


def test_restframe():
    em = get_emlines(iden='LYALPHA', z=3.0, restframe=True)[0]
    assert_allclose(em[1], 1215.67)
    em = get_emlines(iden='LYALPHA', z=3.0, restframe=False)[0]
    assert_allclose(em[1], 4862.68)


def test_table():
    em = get_emlines(table=True, iden=('LYALPHA', 'HALPHA'))
    assert_allclose(em[0]['LBDA_OBS'], 1215.67)
    assert_allclose(em[1]['LBDA_OBS'], 6564.61)

    em = get_emlines(table=True, iden=('LYALPHA', 'HALPHA'), vac=False)
    assert_allclose(em[0]['LBDA_OBS'], 1215.67)
    assert_allclose(em[1]['LBDA_OBS'], 6562.794)
