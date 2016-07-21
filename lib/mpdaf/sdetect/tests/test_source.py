"""Test on Source objects."""

from __future__ import absolute_import, division

import astropy.units as u
import numpy as np
import os
import pytest
import six

from astropy.table import Table
from mpdaf.obj import Cube
from mpdaf.sdetect import Source
from numpy.testing import assert_array_equal, assert_almost_equal

from ...tests.utils import get_data_file


def test_light():
    """Source class; testing initialisation"""
    src = Source._light_from_file(get_data_file('sdetect', 'sing-0032.fits'))
    assert len(src.lines) == 1


def test_init():
    with pytest.raises(ValueError):
        Source({})


def test_from_data():
    src = Source.from_data(ID=1, ra=63.35592651367188, dec=10.46536922454834,
                           origin=('test', 'v0', 'minicube.fits'),
                           proba=1.0, confi=2, extras={'FOO': 'BAR'})

    assert src.DPROBA == 1.0
    assert src.CONFI == 2
    assert src.FOO == 'BAR'

    src.test = 24.12
    assert src.test == 24.12

    src.add_attr('test', 'toto')
    assert src.test == 'toto'

    src.add_attr('test', 1.2345, desc='my keyword', unit=u.deg, fmt='.2f')
    assert src.header.comments['TEST'] == 'my keyword u.deg %.2f'

    src.remove_attr('test')
    with pytest.raises(AttributeError):
        src.test


def test_from_file(tmpdir, source2):
    filename = get_data_file('sdetect', 'sing-0032.fits')
    src = Source.from_file(filename, ext='NB*')
    assert 'NB7317' in src.images

    src = Source.from_file(filename, ext=['NB*'])
    assert 'NB7317' in src.images

    src = Source.from_file(filename, ext='FOO')
    assert 'NB7317' not in src.images


def test_write(tmpdir, source2):
    filename = str(tmpdir.join('source.fits'))

    source2.add_mag('TEST', 2380, 46)
    source2.add_mag('TEST2', 23.5, 0.1)
    source2.add_mag('TEST2', 24.5, 0.01)

    source2.add_z('z_test', 0.07, errz=0.007)
    source2.add_z('z_test2', 1.0, errz=-9999)
    source2.add_z('z_test3', 2.0, errz=(1.5, 2.2))
    source2.add_z('z_test3', 2.0, errz=(1.8, 2.5))

    with pytest.raises(ValueError):
        source2.add_z('z_error', 2.0, errz=[0.001])

    sel = np.where(source2.z['Z_DESC'] == six.b('z_test2'))[0][0]
    assert source2.z['Z'][sel] == 1.0
    assert source2.z['Z_MIN'][sel] is np.ma.masked
    assert source2.z['Z_MAX'][sel] is np.ma.masked

    source2.add_z('z_test2', -9999)
    assert six.b('z_test2') not in source2.z['Z_DESC']

    # cube = Cube(data=np.zeros((2, 2, 2)))
    # source2.add_cube(cube, 'MUSE_WHITE', size=2, unit_size=None)
    # source2.add_white_image(cube)
    # source2.extract_spectra(cube)
    source2.write(filename)
    source2.info()
    source2 = None

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


def test_comments(source1):
    source1.add_comment('This is a test', 'mpdaf')
    assert source1.com001 == 'This is a test'
    source1.add_comment('an other', 'mpdaf')
    assert source1.com002 == 'an other'
    source1.remove_comment(2)
    source1.add_comment('an other', 'mpdaf')
    assert source1.com002 == 'an other'
    source1.remove_comment(1)
    source1.remove_comment(2)


def test_history(source1):
    source1.add_history('test_arg unitary test', 'mpdaf')
    assert source1.hist001 == 'test_arg unitary test'
    source1.add_history('an other', 'mpdaf')
    assert source1.hist002 == 'an other'
    source1.remove_history(2)
    source1.add_history('an other', 'mpdaf')
    assert source1.hist002 == 'an other'
    source1.remove_history(1)
    source1.remove_history(2)


def test_line():
    """Source class: testing add_line methods"""
    src = Source.from_data(ID=1, ra=63.35, dec=10.46,
                           origin=('test', 'v0', 'minicube.fits'))

    src.add_line(['LBDA_OBS', 'LBDA_OBS_ERR', 'LINE'], [4810.123, 3.0, 'TEST'],
                 units=[u.angstrom, u.angstrom, None],
                 desc=['wavelength', 'error', 'name'],
                 fmt=['.2f', '.3f', None])
    lines = src.lines
    assert lines['LBDA_OBS'].unit == u.angstrom
    assert lines['LBDA_OBS'][lines['LINE'] == six.b('TEST')][0] == 4810.123

    src.add_line(['LBDA_OBS', 'NEWCOL', 'NEWCOL2', 'NEWCOL3', 'LINE'],
                 [4807.0, 2, 5.55, 'my new col', 'TEST2'])
    src.add_line(['LBDA_OBS'], [4807.0], match=('LINE', 'TEST'))
    src.add_line(['LBDA_OBS'], [6000.0], match=('LINE', 'TESTMISS', False))

    assert 'NEWCOL' in lines.colnames
    assert six.b('TESTMISS') not in lines['LINE']
    assert lines['LBDA_OBS'][lines['LINE'] == six.b('TEST')][0] == 4807.


def test_add_image(source2, a478hst):
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


def test_add_narrow_band_image(minicube):
    """Source class: testing methods on narrow bands images"""
    src = Source.from_data(ID=1, ra=63.35592651367188, dec=10.46536922454834,
                           origin=('test', 'v0', 'minicube.fits', 'v0'),
                           proba=1.0, confi=2, extras={'FOO': 'BAR'})
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
    assert 'MASK_OBJ' in src.images
    assert 'MASK_INTER' in src.images
    assert 'MASK_SKY' in src.images
    src.extract_spectra(minicube, obj_mask='MASK_OBJ', skysub=True, psf=None)
    assert 'MUSE_SKY' in src.spectra
    assert 'MUSE_TOT_SKYSUB' in src.spectra
    assert 'MUSE_WHITE_SKYSUB' in src.spectra
    assert 'NB_HALPHA_SKYSUB' in src.spectra
    src.extract_spectra(minicube, obj_mask='MASK_OBJ', skysub=False,
                        psf=0.2 * np.ones(minicube.shape[0]))
    assert 'MUSE_PSF' in src.spectra
    assert 'MUSE_TOT' in src.spectra
    assert 'MUSE_WHITE' in src.spectra
    assert 'NB_HALPHA' in src.spectra

    Ny = np.array([ima.shape[0] for ima in src.images.values()])
    assert len(np.unique(Ny)) == 1
    Nx = np.array([ima.shape[1] for ima in src.images.values()])
    assert len(np.unique(Nx)) == 1


def test_sort_lines(source1):
    """Source class: testing sort_lines method"""
    source1.sort_lines()
    assert source1.lines['LINE'][0] == six.b('[OIII]2')


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


def test_add_FSF():
    """Source class: testing add_FSF method"""
    src = Source.from_file(get_data_file('sdetect', 'origin-00026.fits'))
    cube = Cube(get_data_file('sdetect', 'subcub_mosaic.fits'))
    src.add_FSF(cube)
    assert src.FSF99BET == 2.8
    assert src.FSF99FWA == 0.855
    assert src.FSF99FWB == -3.551e-05
