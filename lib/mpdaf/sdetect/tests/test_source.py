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

import astropy.units as u
import numpy as np
import os
import pickle
import pytest
import subprocess
import warnings

from astropy.io import fits
from astropy.table import Table
from mpdaf.obj import Cube, Image
from mpdaf.sdetect import Source
from mpdaf.tools import MpdafWarning
from numpy.testing import assert_array_equal, assert_almost_equal

from ...tests.utils import get_data_file

try:
    subprocess.check_call(['sex', '-v'])
    HAS_SEX = True
except OSError:
    try:
        subprocess.check_call(['sextractor', '-v'])
        HAS_SEX = True
    except OSError:
        HAS_SEX = False


def test_init():
    with pytest.raises(ValueError):
        Source({})


def test_from_data():
    s1 = Source.from_data(ID=1, ra=63.35592651367188, dec=10.46536922454834,
                          origin=('test', 'v0', 'minicube.fits', 'v0'),
                          proba=1.0, confid=2, extras={'FOO': 'BAR'},
                          default_size=10)
    s2 = pickle.loads(pickle.dumps(s1))

    assert 'ID' in dir(s1)
    assert 'FOO' in dir(s1)

    for src in (s1, s2):
        assert src.DPROBA == 1.0
        assert src.CONFID == 2
        assert src.FOO == 'BAR'

        assert src.default_size == 10
        src.default_size = 24.12
        assert src.default_size == 24.12

        src.test = 24.12
        assert src.test == 24.12

        src.add_attr('test', 'toto')
        assert src.test == 'toto'

        src.add_attr('test', 1.2345, desc='my keyword', unit=u.deg, fmt='.2f')
        assert src.header.comments['TEST'] == 'my keyword u.deg %.2f'

        src.remove_attr('test')
        with pytest.raises(AttributeError):
            src.test


def test_from_file():
    filename = get_data_file('sdetect', 'sing-0032.fits')
    src = Source.from_file(filename, ext='NB*')
    assert 'NB7317' in src.images

    src = Source.from_file(filename, ext=['NB*'])
    assert 'NB7317' in src.images

    src = Source.from_file(filename, ext='FOO')
    assert 'NB7317' not in src.images


@pytest.mark.parametrize('filename', ('sing-0032.fits', 'origin-00026.fits'))
def test_pickle(filename):
    filename = get_data_file('sdetect', filename)
    src = Source.from_file(filename)

    s1 = pickle.loads(pickle.dumps(src))

    # Force loading all extensions
    for objtype in ('images', 'cubes', 'spectra', 'tables'):
        for name, obj in getattr(src, objtype).items():
            print(name, obj)

    s2 = pickle.loads(pickle.dumps(src))

    for s in (s1, s2):
        assert src.header.tostring() == s.header.tostring()
        for objtype in ('images', 'cubes', 'spectra', 'tables'):
            attr_ref = getattr(src, objtype)
            attr_new = getattr(s, objtype)
            if attr_ref is None:
                assert attr_new is None
            else:
                assert list(attr_ref.keys()) == list(attr_new.keys())


def test_loc(source1):
    assert source1.z.primary_key == ('Z_DESC', )
    assert source1.mag.primary_key == ('BAND', )

    assert source1.z.loc['z_test']['Z'] == 0.07
    assert source1.mag.loc['TEST2']['MAG'] == 24.5


def test_write(tmpdir, source1):
    filename = str(tmpdir.join('source.fits'))

    with pytest.raises(ValueError):
        source1.add_z('z_error', 2.0, errz=[0.001])

    sel = np.where(source1.z['Z_DESC'] == 'z_test2')[0][0]
    assert source1.z['Z'][sel] == 1.0
    assert source1.z['Z_MIN'][sel] is np.ma.masked
    assert source1.z['Z_MAX'][sel] is np.ma.masked

    source1.add_z('z_test2', -9999)
    assert 'z_test2' not in source1.z['Z_DESC']

    table = Table(rows=[[1, 2.34, 'Hello']], names=('ID', 'value', 'name'))
    source1.tables['TEST'] = table

    source1.write(filename)

    with pytest.raises(OSError):
        source1.write(filename, overwrite=False)

    source1.info()
    source1 = None

    src = Source.from_file(filename)

    sel = np.where(src.mag['BAND'] == 'TEST2')[0][0]
    assert src.mag['MAG'][sel] == 24.5
    assert src.mag['MAG_ERR'][sel] == 0.01

    assert 'z_test2' not in src.z['Z_DESC']

    sel = np.where(src.z['Z_DESC'] == 'z_test')[0][0]
    assert src.z['Z'][sel] == 0.07
    assert src.z['Z_MIN'][sel] == 0.07 - 0.007 / 2
    assert src.z['Z_MAX'][sel] == 0.07 + 0.007 / 2

    sel = np.where(src.z['Z_DESC'] == 'z_test3')[0][0]
    assert src.z['Z'][sel] == 2.0
    assert src.z['Z_MIN'][sel] == 1.8
    assert src.z['Z_MAX'][sel] == 2.5

    assert src.tables['TEST'].colnames == table.colnames
    assert src.tables['TEST'][0].as_void() == table[0].as_void()


def test_delete_extension(tmpdir, source2):
    filename = str(tmpdir.join('source.fits'))

    table = Table(rows=[[1, 2.34, 'Hello']], names=('ID', 'value', 'name'))
    source2.tables['TEST'] = table
    table = Table(rows=[[2, 0.34, 'Foo']], names=('ID2', 'value2', 'name2'))
    source2.tables['TEST2'] = table

    source2.images['NB7317'].var = np.ones(source2.images['NB7317'].shape)
    source2.images['MUSE_WHITE'] = source2.images['NB7317'].copy()
    source2.add_image(source2.images['NB7317'], 'TESTIM')
    del source2.images['NB7317']

    # Write file from scratch
    source2._filename = None
    source2.write(filename)
    source2 = None

    src = Source.from_file(filename)
    assert 'NB7317' not in src.images
    assert len(src.tables['TEST2']) == 1
    del src.tables['TEST']
    del src.images['TESTIM']
    src.write(filename)

    src = Source.from_file(filename)
    assert list(src.tables.keys()) == ['TEST2']
    assert list(src.images.keys()) == ['MUSE_WHITE']

    assert [h.name for h in fits.open(filename)] == [
        'PRIMARY', 'LINES', 'IMA_MUSE_WHITE_DATA', 'IMA_MUSE_WHITE_STAT',
        'TAB_TEST2']


def test_comments(source1):
    source1.add_comment('This is a test', 'mpdaf', '2016-09-02')
    assert source1.comment[0] == '[mpdaf 2016-09-02] This is a test'
    source1.add_comment('an other', 'mpdaf', '2016-09-02')
    assert source1.comment[1] == '[mpdaf 2016-09-02] an other'


def test_history(source1):
    source1.add_history('test_arg unitary test', 'mpdaf')
    assert source1.history[0].find('test_arg unitary test') != -1
    source1.add_history('an other', 'mpdaf')
    assert source1.history[1].find('an other') != -1
    source1.add_history('yet an other', 'mpdaf', '2016-09-02')
    assert 'yet an other (mpdaf 2016-09-02)' in source1.history[2]


def test_line():
    """Source class: testing add_line methods"""
    src = Source.from_data(ID=1, ra=63.35, dec=10.46,
                           origin=('test', 'v0', 'minicube.fits', 'v0'))

    src.add_line(['LBDA_OBS', 'LBDA_OBS_ERR', 'LINE'], [4810.123, 3.0, 'TEST'],
                 units=[u.angstrom, u.angstrom, None],
                 desc=['wavelength', 'error', 'name'],
                 fmt=['.2f', '.3f', None])
    lines = src.lines
    assert lines['LBDA_OBS'].unit == u.angstrom
    assert lines['LBDA_OBS'][lines['LINE'] == 'TEST'][0] == 4810.123

    src.add_line(['LBDA_OBS', 'NEWCOL', 'NEWCOL2', 'NEWCOL3', 'LINE'],
                 [4807.0, 2, 5.55, 'my new col', 'TEST2'])
    src.add_line(['LBDA_OBS'], [4807.0], match=('LINE', 'TEST'))
    src.add_line(['LBDA_OBS'], [6000.0], match=('LINE', 'TESTMISS', False))

    assert 'NEWCOL' in lines.colnames
    assert 'TESTMISS' not in lines['LINE']
    assert lines['LBDA_OBS'][lines['LINE'] == 'TEST'][0] == 4807.


def test_add_cube(source2, minicube, tmpdir):
    """Source class: testing add_cube method"""
    with pytest.raises(ValueError):
        source2.add_cube(minicube, 'TEST')

    lbda = (5000, 5500)
    source2.add_white_image(minicube, size=minicube.shape[1:], unit_size=None)
    source2.add_cube(minicube, 'TEST1', lbda=lbda)
    lmin, lmax = minicube.wave.pixel(lbda, unit=u.angstrom, nearest=True)
    assert (source2.cubes['TEST1'].shape ==
            (lmax - lmin + 1,) + source2.images['MUSE_WHITE'].shape)

    filename = str(tmpdir.join('source.fits'))
    source2.write(filename)

    src = Source.from_file(filename)
    assert 'MUSE_WHITE' in src.images
    assert 'TEST1' in src.cubes

    # Add image again to be sure that the extension is correctly updated
    src.add_white_image(minicube, size=(20, 20), unit_size=None)
    src.write(filename)
    src = Source.from_file(filename)
    assert src.images['MUSE_WHITE'].shape == (20, 20)

    source3 = Source.from_data(source2.ID, source2.RA, source2.DEC,
                               (source2.FROM, source2.FROM_V, '', ''),
                               default_size=source2.default_size)
    source3.add_cube(minicube, 'TEST1', lbda=lbda, add_white=True)
    assert_array_equal(source3.images['MUSE_WHITE'].data,
                       source2.images['MUSE_WHITE'].data)


def test_add_image(source2, a478hst, a370II):
    """Source class: testing add_image method"""
    minicube = Cube(get_data_file('sdetect', 'minicube.fits'), dtype=float)
    source2.add_white_image(minicube)
    ima = minicube.mean(axis=0)

    # The position source2.dec, source2.ra corresponds
    # to pixel index 18.817,32.432 in the cube. The default 5
    # arcsecond requested size of the white-light image corresponds
    # to 25 pixels. There will thus be 12 pixels on either side of
    # a central pixel. The nearest pixel to the center is 19,32, so
    # we expect add_white_image() to have selected the following
    # range of pixels cube[19-12:19+12+1, 32-12:32+12+1], which
    # is cube[7:32, 20:45]. However the cube only has 40 pixels
    # along the X-axis, so the white-light image should correspond
    # to:
    #
    #  white[:,:] = cube[7:32, 20:40]
    #  white.shape=(25,20)
    #
    # So: cube[15,25] = white[15-7, 25-20] = white[8, 5]
    assert ima[15, 25] == source2.images['MUSE_WHITE'][8, 5]

    # Add a square patch of an HST image equal in width and height
    # to the height of the white-light image, which has a height
    # of 25 white-light pixels.
    source2.add_image(a478hst, 'HST1')

    # Add the same HST image, but this time set the width and height
    # equal to the height of the above HST patch (ie. 50 pixels). This
    # should have the same result as giving it the same size as the
    # white-light image.
    size = source2.images['HST1'].shape[0]
    source2.add_image(a478hst, 'HST2', size=size, minsize=size, unit_size=None)
    assert source2.images['HST1'][10, 10] == source2.images['HST2'][10, 10]

    # Add the HST image again, but this time rotate it to the same
    # orientation as the white-light image, then check that they end
    # up with the same rotation angles.
    source2.add_image(a478hst, 'HST3', rotate=True)
    assert_almost_equal(source2.images['HST3'].get_rot(),
                        source2.images['MUSE_WHITE'].get_rot(), 3)

    # Trying to add image not overlapping with Source
    assert source2.add_image(a370II, 'ERROR') is None


@pytest.mark.skipif(not HAS_SEX, reason="requires sextractor")
def test_add_narrow_band_image(minicube, tmpdir):
    """Source class: testing methods on narrow bands images"""
    src = Source.from_data(ID=1, ra=63.35592651367188, dec=10.46536922454834,
                           origin=('test', 'v0', 'minicube.fits', 'v0'),
                           proba=1.0, confid=2, extras={'FOO': 'BAR'})
    src.add_z('EMI', 0.086, 0.0001)
    src.add_white_image(minicube)
    src.add_narrow_band_images(minicube, 'EMI')
    assert 'NB_OIII5007' in src.images
    assert 'NB_HALPHA' in src.images
    assert 'NB_HBETA' in src.images
    src.add_narrow_band_image_lbdaobs(minicube, 'OBS7128', 7128)
    assert 'OBS7128' in src.images
    src.add_seg_images()
    assert 'SEG_MUSE_WHITE' in src.images
    assert 'SEG_NB_OIII5007' in src.images
    assert 'SEG_NB_HALPHA' in src.images
    assert 'SEG_NB_HBETA' in src.images
    assert 'SEG_OBS7128' in src.images

    seg_tags = ['SEG_MUSE_WHITE', 'SEG_NB_OIII5007', 'SEG_OBS7128',
                'SEG_NB_HBETA', 'SEG_NB_HALPHA']
    src.find_sky_mask(seg_tags=seg_tags)
    src.find_union_mask(union_mask='MASK_OBJ', seg_tags=seg_tags)
    src.find_intersection_mask(seg_tags=seg_tags)
    assert_array_equal(src.images['MASK_OBJ'].data.data.astype(bool),
                       ~(src.images['MASK_SKY'].data.data.astype(bool)))
    assert_array_equal(src.images['MASK_INTER'].data.data,
                       np.zeros(src.images['MASK_INTER'].shape))
    src.extract_spectra(minicube, obj_mask='MASK_OBJ', skysub=True, psf=None)
    src.extract_spectra(minicube, obj_mask='MASK_OBJ', skysub=False,
                        psf=0.2 * np.ones(minicube.shape[0]))

    filename = str(tmpdir.join('source.fits'))
    src.write(filename)
    src = Source.from_file(filename)

    for name in ('MASK_OBJ', 'MASK_INTER', 'MASK_SKY'):
        assert name in src.images

    for name in ('MUSE_SKY', 'MUSE_TOT_SKYSUB', 'MUSE_WHITE_SKYSUB',
                 'NB_HALPHA_SKYSUB', 'MUSE_PSF', 'MUSE_TOT', 'MUSE_WHITE',
                 'NB_HALPHA'):
        assert name in src.spectra

    Ny = np.array([ima.shape[0] for ima in src.images.values()])
    assert len(np.unique(Ny)) == 1
    Nx = np.array([ima.shape[1] for ima in src.images.values()])
    assert len(np.unique(Nx)) == 1


def test_sort_lines(source1):
    """Source class: testing sort_lines method"""
    source1.sort_lines()
    assert source1.lines['LINE'][0] == '[OIII]2'


@pytest.mark.skipif(not HAS_SEX, reason="requires sextractor")
def test_SEA(minicube, a478hst):
    """test SEA"""
    cat = Table.read(get_data_file('sdetect', 'cat.txt'), format='ascii')
    size = 10
    width = 8
    margin = 10.
    fband = 3.
    origin = ('sea', '0.0', os.path.basename(minicube.filename), 'v0')

    for obj in cat[0:3]:
        source = Source.from_data(obj['ID'], obj['RA'], obj['DEC'], origin)
        z = float(obj['Z'])
        try:
            errz = (float(obj['Z_MAX']) - float(obj['Z_MIN'])) / 2.0
        except:
            errz = np.nan
        source.add_z('CAT', z, errz)
        # create white image
        source.add_white_image(minicube, size, unit_size=u.arcsec)

        # create narrow band images
        source.add_narrow_band_images(cube=minicube, z_desc='CAT',
                                      size=None, unit_size=u.arcsec,
                                      width=width, margin=margin,
                                      fband=fband, is_sum=False)

        # extract images stamps
        source.add_image(a478hst, 'HST_')

        # segmentation maps
        source.add_seg_images(DIR=None)
        tags = [tag for tag in source.images.keys() if tag[0:4] == 'SEG_']
        source.find_sky_mask(tags)
        source.find_union_mask(tags)
        source.find_intersection_mask(tags)

        # extract spectra
        source.extract_spectra(minicube, skysub=True, psf=None)
        source.extract_spectra(minicube, skysub=False, psf=None)

        Nz = np.array([sp.shape[0] for sp in source.spectra.values()])
        assert len(np.unique(Nz)) == 1
        tags = [tag for tag in source.images.keys() if tag[0:4] != 'HST_']
        Ny = np.array([source.images[tag].shape[0] for tag in tags])
        assert len(np.unique(Ny)) == 1
        Nx = np.array([source.images[tag].shape[1] for tag in tags])
        assert len(np.unique(Nx)) == 1


def test_SEA2(minicube):
    size = 9
    shape = (5, size, size)
    center = size // 2
    np.random.seed(22)
    data = np.random.choice([-0.01, 0.02],
                            np.prod(shape[1:])).reshape(shape[1:])

    # Put fake data at the center of the image
    sl = slice(2, -2)
    for i in range(sl.start, center + 1):
        data[i:-i, i:-i] = i - 1

    data = np.repeat([data], 5, axis=0)
    # Mask some values in the background
    data[1:4, 0, 0] = np.nan
    # Ideally we should test with NaNs in the data, but the algorithm used
    # makes it diffcult to compute expected values without reimplementing the
    # same code here
    # data[:, 3, 3] = np.nan

    cube = Cube(data=data, var=np.ones(shape), wcs=minicube.wcs,
                wave=minicube.wave, copy=False)
    cube_novar = Cube(data=data, wcs=minicube.wcs, wave=minicube.wave,
                      copy=False)
    dec, ra = cube.wcs.pix2sky([4, 4])[0]

    origin = ('sea', '0.0', 'test', 'v0')
    s = Source.from_data('1', ra, dec, origin)
    s.add_white_image(cube, size, unit_size=None)
    white = s.images['MUSE_WHITE'].data
    assert_array_equal(white, data[0])

    mask = np.zeros(shape[1:], dtype=bool)
    mask[sl, sl] = True
    s.images['MASK_OBJ'] = Image(data=mask, wcs=cube.wcs)
    s.images['MASK_SKY'] = Image(data=~mask, wcs=cube.wcs)
    sky_value = np.nanmean(data[:, ~mask], axis=1)

    # extract spectra errors
    with pytest.raises(ValueError):
        s.extract_spectra(cube, skysub=False)

    with pytest.raises(ValueError):
        s.extract_spectra(cube, skysub=True, obj_mask='MASK_OBJ',
                          sky_mask='NONE')

    dist = np.sum((np.mgrid[:5, :5] - 2) ** 2, axis=0)
    psf = np.zeros(shape)
    psf[:, sl, sl] = np.max(dist) - dist

    # extract spectra without var
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        s.extract_spectra(cube_novar, skysub=False, obj_mask='MASK_OBJ')
        assert len(w) == 1
        assert issubclass(w[-1].category, MpdafWarning)

    # MUSE_TOT
    sptot = cube[:, sl, sl].sum(axis=(1, 2))
    assert_almost_equal(s.spectra['MUSE_TOT'].data, sptot.data)
    assert s.spectra['MUSE_TOT'].var is None

    # extract spectra
    s.extract_spectra(cube, skysub=False, psf=psf, obj_mask='MASK_OBJ',
                      apertures=(0.3, 1.0))
    s.extract_spectra(cube, skysub=True, psf=psf, obj_mask='MASK_OBJ',
                      apertures=(0.3, 1.0))

    # Check that extraction does not modified the data
    assert_almost_equal(cube.data.data, data)
    assert_almost_equal(s.images['MUSE_WHITE'].data, data[0])

    # MUSE_TOT
    npix_sky = np.sum((~cube.mask) & mask, axis=(1, 2))
    assert_almost_equal(s.spectra['MUSE_TOT'].data, sptot.data)
    assert_almost_equal(s.spectra['MUSE_TOT'].var, sptot.var)
    assert_almost_equal(s.spectra['MUSE_TOT_SKYSUB'].data,
                        sptot.data - sky_value * npix_sky)

    # MUSE_APER
    sl03 = slice(3, -3)
    sp03 = cube[:, sl03, sl03].sum(axis=(1, 2))
    assert_almost_equal(s.spectra['MUSE_APER_0.3'].data, sp03.data)
    assert_almost_equal(s.spectra['MUSE_APER_0.3'].var, sp03.var)
    # 1" aperture is wider than the object mask, so we get the same flux
    assert_almost_equal(s.spectra['MUSE_APER_1.0'].data, sptot.data)
    assert_almost_equal(s.spectra['MUSE_APER_1.0'].var, sptot.var)
    assert_almost_equal(s.spectra['MUSE_APER_1.0_SKYSUB'].data,
                        sptot.data - sky_value * npix_sky)

    # MUSE_PSF
    # mask = (psf > 0) & s.images['MASK_OBJ']._data
    # sp = np.nansum(cube.data * mask, axis=(1, 2))
    # npix_sky = np.sum((~cube.mask) & mask, axis=(1, 2))
    assert_almost_equal(31.11, s.spectra['MUSE_PSF'].data, decimal=2)
    assert_almost_equal(18.51, s.spectra['MUSE_PSF'].var, decimal=2)
    assert_almost_equal(30.98, s.spectra['MUSE_PSF_SKYSUB'].data, decimal=2)

    # MUSE_WHITE
    # here compute_spectrum subtracts the min value of the weight image, to
    # ensure that weights are positive, but this means that on our fake data
    # the weights of the boundary are put to 0. So we need to compare on the
    # inner part only.
    # mask = ((white - white[sl, sl].min()) > 0) & s.images['MASK_OBJ'].data
    # mask = broadcast_to(mask, cube.shape)
    # sp = np.nansum(cube.data * mask, axis=(1, 2))
    # npix_sky = np.sum((~cube.mask) & mask, axis=(1, 2))
    assert_almost_equal(18.33, s.spectra['MUSE_WHITE'].data, decimal=2)
    assert_almost_equal(8.33, s.spectra['MUSE_WHITE'].var, decimal=2)
    assert_almost_equal(18.27, s.spectra['MUSE_WHITE_SKYSUB'].data, decimal=2)


def test_add_FSF(minicube):
    """Source class: testing add_FSF method"""
    src = Source.from_file(get_data_file('sdetect', 'origin-00026.fits'))
    assert src.get_FSF() is None

    with pytest.raises(ValueError):
        src.add_FSF(minicube)

    cube = Cube(get_data_file('sdetect', 'subcub_mosaic.fits'))
    src.add_FSF(cube)
    assert src.FSF99BET == 2.8
    assert src.FSF99FWA == 0.855
    assert src.FSF99FWB == -3.551e-05
    assert src.get_FSF() == (0.855, -3.551e-05, 2.8, 99)
