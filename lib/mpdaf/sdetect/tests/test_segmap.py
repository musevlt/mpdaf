"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2019 Simon Conseil <simon.conseil@univ-lyon1.fr>

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
import numpy as np
import pytest

from astropy.io import fits
from glob import glob
from mpdaf.obj import Image
from mpdaf.sdetect import Segmap, create_masks_from_segmap
from mpdaf.tests.utils import get_data_file
from numpy.testing import assert_array_equal

try:
    import joblib  # noqa
except ImportError:
    HAS_JOBLIB = False
else:
    HAS_JOBLIB = True


def test_segmap():
    segfile = get_data_file('segmap', 'segmap.fits')
    img = Image(segfile)
    refdata = np.arange(14)

    for arg in (segfile, img, img.data):
        segmap = Segmap(arg)
        assert segmap.img.shape == (90, 90)
        assert str(segmap.img.data.dtype) == '>i8'
        assert np.max(segmap.img._data) == 13
        assert_array_equal(np.unique(segmap.img._data), refdata)

    assert_array_equal(segmap.copy().img.data, segmap.img.data)

    cmap = segmap.cmap()
    assert cmap.N == 14  # nb of values in the segmap


def test_align_segmap():
    segmap = Segmap(get_data_file('segmap', 'segmap.fits'))
    ref = Image(get_data_file('segmap', 'image.fits'))
    aligned = segmap.align_with_image(ref, truncate=True)
    assert aligned.img.shape == ref.shape
    assert (aligned.img.wcs.get_rot() - ref.wcs.get_rot()) < 1e-3


def test_cut_header():
    segmap = Segmap(get_data_file('segmap', 'segmap.fits'),
                    cut_header_after='NAXIS2')
    assert 'RADESYS' not in segmap.img.primary_header
    assert 'RADESYS' not in segmap.img.data_header


@pytest.mark.skipif(not HAS_JOBLIB, reason="requires joblib")
def test_create_masks(tmpdir):
    segfile = get_data_file('segmap', 'segmap.fits')
    reffile = get_data_file('segmap', 'image.fits')
    catalog = get_data_file('segmap', 'catalog.fits')

    create_masks_from_segmap(
        segfile, catalog, reffile, n_jobs=1,
        masksky_name=str(tmpdir.join('mask-sky.fits')),
        maskobj_name=str(tmpdir.join('mask-source-%05d.fits')),
        idname='id', raname='ra', decname='dec', margin=5, mask_size=(10, 10))

    assert len(glob(str(tmpdir.join('mask-source*')))) == 13
    assert len(glob(str(tmpdir.join('mask-sky*')))) == 1

    mask = fits.getdata(str(tmpdir.join('mask-source-00001.fits')))
    assert mask.shape == (50, 50)
    assert mask.sum() == 56

    # test skip_existing
    create_masks_from_segmap(
        segfile, catalog, reffile, n_jobs=1, skip_existing=True,
        masksky_name=str(tmpdir.join('mask-sky.fits')),
        maskobj_name=str(tmpdir.join('mask-source-%05d.fits')),
        idname='id', raname='ra', decname='dec', margin=5, mask_size=(10, 10),
        convolve_fwhm=0)

    # test convolve_fwhm and callables for mask filenames
    masksky_func = lambda: str(tmpdir.join('mask2-sky.fits'))
    maskobj_func = lambda x: str(tmpdir.join('mask2-source-%05d.fits' % x))
    create_masks_from_segmap(
        segfile, catalog, reffile, n_jobs=1, skip_existing=True,
        masksky_name=masksky_func, maskobj_name=maskobj_func,
        idname='id', raname='ra', decname='dec', margin=5, mask_size=(10, 10),
        convolve_fwhm=1, psf_threshold=0.5)

    mask = fits.getdata(str(tmpdir.join('mask2-source-00001.fits')))
    assert mask.shape == (50, 50)
    assert mask.sum() == 106
