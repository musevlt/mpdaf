"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c)      2019 David Carton <cartondj@gmail.com>

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

from glob import glob
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
    muselet(filename, write_nbcube=True, cleanup=True, workdir=outdir)

    #NB cube produced?
    assert os.path.isfile(str(tmpdir.join('NB_cube.fits')))

    #get catalogs
    cat_lines = Catalog.read(str(tmpdir.join('lines.fit')))
    cat_objects = Catalog.read(str(tmpdir.join('objects.fit')))
    
    files_lines = glob(str(tmpdir.join('lines/*')))
    files_objects = glob(str(tmpdir.join('objects/*')))

    #check same length as number of sources
    assert len(cat_lines) == len(files_lines)
    assert len(cat_objects) ==  len(files_objects)

    assert len(cat_lines) == 34
    assert len(cat_objects) == 12


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SEX, reason="requires sextractor")
def test_muselet_full(tmpdir, minicube):
    """test MUSELET"""
    outdir = str(tmpdir)
    print('Working directory:', outdir)

    muselet(minicube.filename, write_nbcube=True, cleanup=True,
            workdir=outdir, n_cpu=2)

    #NB cube produced?
    assert os.path.isfile(str(tmpdir.join(
        'NB_' + os.path.basename(minicube.filename))))

    #get catalogs
    cat_lines = Catalog.read(str(tmpdir.join('lines.fit')))
    cat_objects = Catalog.read(str(tmpdir.join('objects.fit')))
    
    files_lines = glob(str(tmpdir.join('lines/*')))
    files_objects = glob(str(tmpdir.join('objects/*')))

    #check same length as number of sources
    assert len(cat_lines) == len(files_lines)
    assert len(cat_objects) ==  len(files_objects)

    assert len(cat_lines) == 61
    assert len(cat_objects) == 15

