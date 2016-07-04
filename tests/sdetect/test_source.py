"""Test on Source objects."""

from __future__ import absolute_import, division

from nose.plugins.attrib import attr
from nose.tools import assert_equal, assert_true, assert_almost_equal

import astropy.units as u
import numpy as np
import os
import six

from astropy.table import Table
from mpdaf.obj import Image, Cube
from mpdaf.sdetect import Source
from numpy.testing import assert_array_equal
from os.path import join

DATADIR = join(os.path.abspath(os.path.dirname(__file__)),
               '..', '..', 'data', 'sdetect')


class TestSource(object):

    def setUp(self):
        col_lines = ['LBDA_OBS', 'LBDA_OBS_ERR',
                     'FWHM_OBS', 'FWHM_OBS_ERR',
                     'LBDA_REST', 'LBDA_REST_ERR',
                     'FWHM_REST', 'FWHM_REST_ERR',
                     'FLUX', 'FLUX_ERR', 'LINE']
        line1 = [5550, 10, 2.3, 0.2, 5600.0, 11.0, 2.5, 0.4, 28.0, 3.1,
                 six.b('[OIII]')]
        line2 = [5550, 10, 2.3, 0.2, 5600.0, 11.0, 2.5, 0.4, 28.0879, 3.1,
                 six.b('[OIII]2')]
        lines = Table(names=col_lines, rows=[line1, line2])
        self.source1 = Source.from_data(ID=1, ra=-65.1349958, dec=140.3057987,
                                        origin=('test', 'v0', 'cube.fits'),
                                        lines=lines)
        self.source2 = Source.from_file(join(DATADIR, 'sing-0032.fits'))

    def tearDown(self):
        del self.source1
        del self.source2

    @attr(speed='fast')
    def test_init(self):
        """Source class; testing initialisation"""
        src = Source._light_from_file(join(DATADIR, 'sing-0032.fits'))
        assert_equal(len(src.lines), len(self.source2.lines))

    @attr(speed='fast')
    def test_arg(self):
        """Source class: testing argument setter/getter"""
        self.source1.add_comment('This is a test', 'mpdaf')
        assert_equal(self.source1.com001, 'This is a test')
        self.source1.add_comment('an other', 'mpdaf')
        assert_equal(self.source1.com002, 'an other')
        self.source1.remove_comment(2)
        self.source1.add_comment('an other', 'mpdaf')
        assert_equal(self.source1.com002, 'an other')
        self.source1.remove_comment(1)
        self.source1.remove_comment(2)
        self.source1.test = 24.12
        assert_equal(self.source1.test, 24.12)
        self.source1.add_attr('test', 'toto')
        assert_equal(self.source1.test, 'toto')
        self.source1.remove_attr('test')
        self.source1.add_history('test_arg unitary test', 'mpdaf')
        assert_equal(self.source1.hist001, 'test_arg unitary test')
        self.source1.add_history('an other', 'mpdaf')
        assert_equal(self.source1.hist002, 'an other')
        self.source1.remove_history(2)
        self.source1.add_history('an other', 'mpdaf')
        assert_equal(self.source1.hist002, 'an other')
        self.source1.remove_history(1)
        self.source1.remove_history(2)

    @attr(speed='fast')
    def test_z(self):
        """Source class: testing add_z method"""
        self.source1.add_z('z_test', 0.07, 0.007)
        z = self.source1.z
        key = six.b('z_test')
        assert_equal(z['Z'][z['Z_DESC'] == key], 0.07)
        assert_equal(z['Z_MIN'][z['Z_DESC'] == key], 0.07 - 0.007 / 2)
        assert_equal(z['Z_MAX'][z['Z_DESC'] == key], 0.07 + 0.007 / 2)

    @attr(speed='fast')
    def test_mag(self):
        """Source class: testing add_mag method"""
        self.source1.add_mag('TEST', 2380, 46)
        mag = self.source1.mag
        assert_equal(mag['MAG'][mag['BAND'] == six.b('TEST')], 2380)
        assert_equal(mag['MAG_ERR'][mag['BAND'] == six.b('TEST')], 46)

    @attr(speed='fast')
    def test_line(self):
        """Source class: testing add_line methods"""
        cols = ['LBDA_OBS', 'LBDA_OBS_ERR', 'LINE']
        values = [4810.0, 3.0, 'TEST']
        self.source1.add_line(cols, values)
        lines = self.source1.lines
        assert_equal(lines['LBDA_OBS'][lines['LINE'] == six.b('TEST')], 4810.)
        cols = ['LBDA_OBS']
        values = [4807.0]
        self.source1.add_line(cols, values, match=('LINE', 'TEST'))
        assert_equal(lines['LBDA_OBS'][lines['LINE'] == six.b('TEST')], 4807.)

    @attr(speed='fast')
    def test_add_image(self):
        """Source class: testing add_image method"""
        cube = Cube(join(DATADIR, 'minicube.fits'), dtype=np.float64)
        self.source2.add_white_image(cube)
        ima = cube.mean(axis=0)

        # The position self.source2.dec, self.source2.ra corresponds
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
        assert_equal(ima[15, 25], self.source2.images['MUSE_WHITE'][8, 5])

        # Add a square patch of an HST image equal in width and height
        # to the height of the white-light image, which has a height
        # of 25 white-light pixels.
        hst = Image(join(DATADIR, 'a478hst-cutout.fits'))
        self.source2.add_image(hst, 'HST1')

        # Add the same HST image, but this time set the width and height
        # equal to the height of the above HST patch (ie. 50 pixels). This
        # should have the same result as giving it the same size as the
        # white-light image.
        size = self.source2.images['HST1'].shape[0]
        self.source2.add_image(hst, 'HST2', size=size, minsize=size,
                               unit_size=None)
        assert_equal(self.source2.images['HST1'][10, 10],
                     self.source2.images['HST2'][10, 10])

        # Add the HST image again, but this time rotate it to the same
        # orientation as the white-light image, then check that they end
        # up with the same rotation angles.
        self.source2.add_image(hst, 'HST3', rotate=True)
        assert_almost_equal(self.source2.images['HST3'].get_rot(),
                            self.source2.images['MUSE_WHITE'].get_rot(), 3)

    @attr(speed='fast')
    def test_add_narrow_band_image(self):
        """Source class: testing methods on narrow bands images"""
        cube = Cube(join(DATADIR, 'minicube.fits'))
        src = Source.from_data(ID=1,
                               ra=63.35592651367188, dec=10.46536922454834,
                               origin=('test', 'v0', 'minicube.fits'))
        src.add_z('EMI', 0.086, 0.0001)
        src.add_white_image(cube)
        src.add_narrow_band_images(cube, 'EMI')
        assert_true('NB_OIII5007' in src.images)
        assert_true('NB_HALPHA' in src.images)
        assert_true('NB_HBETA' in src.images)
        src.add_narrow_band_image_lbdaobs(cube, 'OBS7128', 7128)
        assert_true('OBS7128' in src.images)
        src.add_seg_images()
        assert_true('SEG_MUSE_WHITE' in src.images)
        assert_true('SEG_NB_OIII5007' in src.images)
        assert_true('SEG_NB_HALPHA' in src.images)
        assert_true('SEG_NB_HBETA' in src.images)
        assert_true('SEG_OBS7128' in src.images)

        seg_tags = ['SEG_MUSE_WHITE', 'SEG_NB_OIII5007', 'SEG_OBS7128',
                    'SEG_NB_HBETA', 'SEG_NB_HALPHA']
        src.find_sky_mask(seg_tags=seg_tags)
        src.find_union_mask(union_mask='MASK_OBJ', seg_tags=seg_tags)
        src.find_intersection_mask(seg_tags=seg_tags)
        assert_array_equal(src.images['MASK_OBJ'].data.data.astype(bool),
                           ~(src.images['MASK_SKY'].data.data.astype(bool)))
        assert_array_equal(src.images['MASK_INTER'].data.data,
                           np.zeros(src.images['MASK_INTER'].shape))
        assert_true('MASK_OBJ' in src.images)
        assert_true('MASK_INTER' in src.images)
        assert_true('MASK_SKY' in src.images)
        src.extract_spectra(cube, obj_mask='MASK_OBJ', skysub=True, psf=None)
        assert_true('MUSE_SKY' in src.spectra)
        assert_true('MUSE_TOT_SKYSUB' in src.spectra)
        assert_true('MUSE_WHITE_SKYSUB' in src.spectra)
        assert_true('NB_HALPHA_SKYSUB' in src.spectra)
        src.extract_spectra(cube, obj_mask='MASK_OBJ', skysub=False,
                            psf=0.2 * np.ones(cube.shape[0]))
        assert_true('MUSE_PSF' in src.spectra)
        assert_true('MUSE_TOT' in src.spectra)
        assert_true('MUSE_WHITE' in src.spectra)
        assert_true('NB_HALPHA' in src.spectra)

        Ny = np.array([ima.shape[0] for ima in src.images.values()])
        assert_equal(len(np.unique(Ny)), 1)
        Nx = np.array([ima.shape[1] for ima in src.images.values()])
        assert_equal(len(np.unique(Nx)), 1)

    @attr(speed='fast')
    def test_sort_lines(self):
        """Source class: testing sort_lines method"""
        self.source1.sort_lines()
        assert_equal(self.source1.lines['LINE'][0], six.b('[OIII]2'))

    @attr(speed='slow')
    def test_SEA(self):
        """test SEA"""
        cube = Cube(join(DATADIR, 'minicube.fits'))
        ima = Image(join(DATADIR, 'a478hst-cutout.fits'))
        cat = Table.read(join(DATADIR, 'cat.txt'), format='ascii')
        size = 10
        width = 8
        margin = 10.
        fband = 3.
        origin = ('sea', '0.0', os.path.basename(cube.filename))

        for obj in cat[0:6]:
            source = Source.from_data(obj['ID'], obj['RA'], obj['DEC'], origin)
            z = float(obj['Z'])
            try:
                errz = (float(obj['Z_MAX']) - float(obj['Z_MIN'])) / 2.0
            except:
                errz = np.nan
            source.add_z('CAT', z, errz)
            # create white image
            source.add_white_image(cube, size, unit_size=u.arcsec)

            # create narrow band images
            source.add_narrow_band_images(cube=cube, z_desc='CAT',
                                          size=None, unit_size=u.arcsec,
                                          width=width, margin=margin,
                                          fband=fband, is_sum=False)

            # extract images stamps
            source.add_image(ima, 'HST_')

            # segmentation maps
            source.add_seg_images(DIR=None)
            tags = [tag for tag in source.images.keys() if tag[0:4] == 'SEG_']
            source.find_sky_mask(tags)
            source.find_union_mask(tags)
            source.find_intersection_mask(tags)

            # extract spectra
            source.extract_spectra(cube, skysub=True, psf=None)
            source.extract_spectra(cube, skysub=False, psf=None)

            Nz = np.array([sp.shape[0] for sp in source.spectra.values()])
            assert_equal(len(np.unique(Nz)), 1)
            tags = [tag for tag in source.images.keys() if tag[0:4] != 'HST_']
            Ny = np.array([source.images[tag].shape[0] for tag in tags])
            assert_equal(len(np.unique(Ny)), 1)
            Nx = np.array([source.images[tag].shape[1] for tag in tags])
            assert_equal(len(np.unique(Nx)), 1)

    @attr(speed='fast')
    def test_add_FSF(self):
        """Source class: testing add_FSF method"""
        src = Source.from_file(join(DATADIR, 'origin-00026.fits'))
        cube = Cube(join(DATADIR, 'subcub_mosaic.fits'))
        src.add_FSF(cube)
        assert_equal(src.FSF99BET, 2.8)
        assert_equal(src.FSF99FWA, 0.855)
        assert_equal(src.FSF99FWB, -3.551e-05)

#
#     @attr(speed='fast')
#     def test_catalog(self):
#         """Source class: tests catalog creation"""
#         cat = Catalog.from_sources([self.source1, self.source2])
#         print cat
