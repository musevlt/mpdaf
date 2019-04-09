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

    # test convolve_fwhm
    create_masks_from_segmap(
        segfile, catalog, reffile, n_jobs=1, skip_existing=True,
        masksky_name=str(tmpdir.join('mask2-sky.fits')),
        maskobj_name=str(tmpdir.join('mask2-source-%05d.fits')),
        idname='id', raname='ra', decname='dec', margin=5, mask_size=(10, 10),
        convolve_fwhm=1, psf_threshold=0.5)

    mask = fits.getdata(str(tmpdir.join('mask2-source-00001.fits')))
    assert mask.shape == (50, 50)
    assert mask.sum() == 106
