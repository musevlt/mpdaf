"""Test on muselet script."""

from __future__ import absolute_import, division, print_function

import nose.tools
import os
import shutil
import tempfile
from nose.plugins.attrib import attr

from mpdaf.sdetect.muselet import muselet

DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       '..', '..', 'data', 'sdetect')


@attr(speed='veryslow')
def test_muselet():
    """test MUSELET"""
    try:
        outdir = tempfile.mkdtemp(prefix='muselet.')
        print('Working directory:', outdir)
        cont, single, raw = muselet(os.path.join(DATADIR, 'minicube.fits'),
                                    nbcube=False, del_sex=True, workdir=outdir)
        nose.tools.assert_equal(len(cont), 1)
        nose.tools.assert_equal(len(single), 8)
        nose.tools.assert_equal(len(raw), 39)
    finally:
        shutil.rmtree(outdir)
