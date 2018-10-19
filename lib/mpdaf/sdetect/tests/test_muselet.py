"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
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

import os
import pytest
import subprocess
from mpdaf.sdetect import muselet, Catalog


try:
    subprocess.check_call(['sex', '-v'])
    HAS_SEX = True
except OSError:
    try:
        subprocess.check_call(['sextractor', '-v'])
        HAS_SEX = True
    except OSError:
        HAS_SEX = False


@pytest.mark.skipif(not HAS_SEX, reason="requires sextractor")
def test_muselet_fast(tmpdir, minicube):
    """test MUSELET"""
    outdir = str(tmpdir)
    filename = str(tmpdir.join('cube.fits'))
    cube = minicube[1800:2000, :, :]
    cube.write(filename, savemask='nan')
    print('Working directory:', outdir)
    cont, single, raw = muselet(filename, nbcube=True, del_sex=True,
                                workdir=outdir)
    assert len(cont) == 1
    assert len(single) == 7
    assert len(raw) == 22
    assert os.path.isfile(str(tmpdir.join('NB_cube.fits')))

    cont.write('cont', path=str(tmpdir), fmt='working')
    single.write('sing', path=str(tmpdir), fmt='working')
    raw.write('raw', path=str(tmpdir), fmt='working')
    cat_cont = Catalog.read(str(tmpdir.join('cont.fits')))
    cat_sing = Catalog.read(str(tmpdir.join('sing.fits')))
    cat_raw = Catalog.read(str(tmpdir.join('raw.fits')))
    assert len(cont) == len(cat_cont)
    assert len(single) == len(cat_sing)
    assert len(raw) == len(cat_raw)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SEX, reason="requires sextractor")
def test_muselet_full(tmpdir, minicube):
    """test MUSELET"""
    outdir = str(tmpdir)
    print('Working directory:', outdir)
    cont, single, raw = muselet(minicube.filename, nbcube=True, del_sex=True,
                                workdir=outdir)
    assert len(cont) == 1
    assert len(single) == 8
    assert len(raw) == 39
    assert os.path.isfile(str(tmpdir.join(
        'NB_' + os.path.basename(minicube.filename))))

    cont.write('cont', path=str(tmpdir), fmt='working')
    single.write('sing', path=str(tmpdir), fmt='working')
    raw.write('raw', path=str(tmpdir), fmt='working')
    cat_cont = Catalog.read(str(tmpdir.join('cont.fits')))
    cat_sing = Catalog.read(str(tmpdir.join('sing.fits')))
    cat_raw = Catalog.read(str(tmpdir.join('raw.fits')))
    assert len(cont) == len(cat_cont)
    assert len(single) == len(cat_sing)
    assert len(raw) == len(cat_raw)
