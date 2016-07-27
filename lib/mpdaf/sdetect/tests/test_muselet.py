"""Test on muselet script."""

from __future__ import absolute_import, division, print_function

import pytest
import sys
from mpdaf.sdetect import muselet
from mpdaf.sdetect import Catalog


@pytest.mark.xfail(sys.version_info >= (3, 3),
                   reason="not compatible with python 3")
def test_muselet_fast(tmpdir, minicube):
    """test MUSELET"""
    outdir = str(tmpdir)
    filename = str(tmpdir.join('cube.fits'))
    cube = minicube[1800:2000, :, :]
    cube.write(filename, savemask='nan')
    print('Working directory:', outdir)
    cont, single, raw = muselet(filename, nbcube=False, del_sex=True,
                                workdir=outdir)
    assert len(cont) == 1
    assert len(single) == 7
    assert len(raw) == 22
    
    cont.write('cont', fmt='working')
    single.write('sing', fmt='working')
    raw.write('raw', fmt='working')
    cat_cont = Catalog.read('cont.fits')
    cat_sing = Catalog.read('sing.fits')
    cat_raw = Catalog.read('raw.fits')
    assert len(cont) == len(cat_cont)
    assert len(single) == len(cat_sing)
    assert len(raw) == len(cat_raw)


@pytest.mark.xfail(sys.version_info >= (3, 3),
                   reason="not compatible with python 3")
@pytest.mark.slow
def test_muselet_full(tmpdir, minicube):
    """test MUSELET"""
    outdir = str(tmpdir)
    print('Working directory:', outdir)
    cont, single, raw = muselet(minicube.filename, nbcube=False, del_sex=True,
                                workdir=outdir)
    assert len(cont) == 1
    assert len(single) == 8
    assert len(raw) == 39
    
    cont.write('cont', fmt='working')
    single.write('sing', fmt='working')
    raw.write('raw', fmt='working')
    cat_cont = Catalog.read('cont.fits')
    cat_sing = Catalog.read('sing.fits')
    cat_raw = Catalog.read('raw.fits')
    assert len(cont) == len(cat_cont)
    assert len(single) == len(cat_sing)
    assert len(raw) == len(cat_raw)
