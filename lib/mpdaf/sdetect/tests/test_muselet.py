"""Test on muselet script."""

from __future__ import absolute_import, division, print_function

import pytest
import sys
from mpdaf.sdetect.muselet import muselet


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
