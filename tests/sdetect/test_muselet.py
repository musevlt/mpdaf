"""Test on muselet script."""

from __future__ import absolute_import, division, print_function

import os
import shutil
import tempfile
import pytest

from mpdaf.sdetect.muselet import muselet

DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       '..', '..', 'data', 'sdetect')


@pytest.mark.veryslow
def test_muselet():
    """test MUSELET"""
    try:
        outdir = tempfile.mkdtemp(prefix='muselet.')
        print('Working directory:', outdir)
        cont, single, raw = muselet(os.path.join(DATADIR, 'minicube.fits'),
                                    nbcube=False, del_sex=True, workdir=outdir)
        assert len(cont) == 1
        assert len(single) == 8
        assert len(raw) == 39
    finally:
        shutil.rmtree(outdir)
