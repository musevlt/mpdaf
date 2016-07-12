"""Test on muselet script."""

from __future__ import absolute_import, division, print_function

import shutil
import tempfile
import pytest

from mpdaf.sdetect.muselet import muselet

from ..utils import get_data_file


@pytest.mark.slow
def test_muselet():
    """test MUSELET"""
    try:
        outdir = tempfile.mkdtemp(prefix='muselet.')
        print('Working directory:', outdir)
        cont, single, raw = muselet(get_data_file('sdetect', 'minicube.fits'),
                                    nbcube=False, del_sex=True, workdir=outdir)
        assert len(cont) == 1
        assert len(single) == 8
        assert len(raw) == 39
    finally:
        shutil.rmtree(outdir)
