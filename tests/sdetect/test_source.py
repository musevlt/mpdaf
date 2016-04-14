"""Test on Source objects."""
from __future__ import absolute_import
import nose.tools
from nose.plugins.attrib import attr

from mpdaf.sdetect import Source
from mpdaf.obj import Image, Cube
from astropy.table import Table
import numpy as np
from numpy.testing import assert_array_equal


class TestSource():

    def setUp(self):
        col_lines = ['LBDA_OBS', 'LBDA_OBS_ERR',
                     'FWHM_OBS', 'FWHM_OBS_ERR',
                     'LBDA_REST', 'LBDA_REST_ERR',
                     'FWHM_REST', 'FWHM_REST_ERR',
                     'FLUX', 'FLUX_ERR', 'LINE']
        line1 = [5550, 10, 2.3, 0.2, 5600.0, 11.0, 2.5, 0.4, 28.0, 3.1, '[OIII]']
        line2 = [5550, 10, 2.3, 0.2, 5600.0, 11.0, 2.5, 0.4, 28.0879, 3.1, '[OIII]2']
        lines = Table(names=col_lines, rows=[line1, line2])
        self.source1 = Source.from_data(ID=1, ra=-65.1349958, dec=140.3057987, origin=('test', 'v0', 'cube.fits'),
                                        lines=lines)
        self.source2 = Source.from_file('data/sdetect/sing-0032.fits')

    def tearDown(self):
        del self.source1
        del self.source2

    @attr(speed='fast')
    def test_init(self):
        """Source class; testing initialisation"""
        src = Source._light_from_file('data/sdetect/sing-0032.fits')
        nose.tools.assert_equal(len(src.lines), len(self.source2.lines))

    @attr(speed='fast')
    def test_arg(self):
        """Source class: testing argument setter/getter"""
        self.source1.add_comment('This is a test', 'mpdaf')
        nose.tools.assert_equal(self.source1.com001, 'This is a test')
        self.source1.add_comment('an other', 'mpdaf')
        nose.tools.assert_equal(self.source1.com002, 'an other')
        self.source1.remove_comment(2)
        self.source1.add_comment('an other', 'mpdaf')
        nose.tools.assert_equal(self.source1.com002, 'an other')
        self.source1.remove_comment(1)
        self.source1.remove_comment(2)
        self.source1.test = 24.12
        nose.tools.assert_equal(self.source1.test, 24.12)
        self.source1.add_attr('test', 'toto')
        nose.tools.assert_equal(self.source1.test, 'toto')
        self.source1.remove_attr('test')
        self.source1.add_history('test_arg unitary test', 'mpdaf')
        nose.tools.assert_equal(self.source1.hist001, 'test_arg unitary test')
        self.source1.add_history('an other', 'mpdaf')
        nose.tools.assert_equal(self.source1.hist002, 'an other')
        self.source1.remove_history(2)
        self.source1.add_history('an other', 'mpdaf')
        nose.tools.assert_equal(self.source1.hist002, 'an other')
        self.source1.remove_history(1)
        self.source1.remove_history(2)

    @attr(speed='fast')
    def test_z(self):
        """Source class: testing add_z method"""
        self.source1.add_z('z_test', 0.07, 0.007)
        nose.tools.assert_equal(self.source1.z['Z'][self.source1.z['Z_DESC'] == 'z_test'], 0.07)
        nose.tools.assert_equal(self.source1.z['Z_MIN'][self.source1.z['Z_DESC'] == 'z_test'], 0.07 - 0.007 / 2)
        nose.tools.assert_equal(self.source1.z['Z_MAX'][self.source1.z['Z_DESC'] == 'z_test'], 0.07 + 0.007 / 2)

    @attr(speed='fast')
    def test_mag(self):
        """Source class: testing add_mag method"""
        self.source1.add_mag('TEST', 2380, 46)
        nose.tools.assert_equal(self.source1.mag['MAG'][self.source1.mag['BAND'] == 'TEST'], 2380)
        nose.tools.assert_equal(self.source1.mag['MAG_ERR'][self.source1.mag['BAND'] == 'TEST'], 46)

    @attr(speed='fast')
    def test_line(self):
        """Source class: testing add_line methods"""
        cols = ['LBDA_OBS', 'LBDA_OBS_ERR', 'LINE']
        values = [4810.0, 3.0, 'TEST']
        self.source1.add_line(cols, values)
        nose.tools.assert_equal(self.source1.lines['LBDA_OBS'][self.source1.lines['LINE'] == 'TEST'], 4810.)
        cols = ['LBDA_OBS']
        values = [4807.0]
        self.source1.add_line(cols, values, match=('LINE', 'TEST'))
        nose.tools.assert_equal(self.source1.lines['LBDA_OBS'][self.source1.lines['LINE'] == 'TEST'], 4807.)

    def test_add_image(self):
        """Source class: testing add_image method"""
        cube = Cube('data/sdetect/minicube.fits', dtype=np.float64)
        self.source2.add_white_image(cube)
        ima = cube.mean(axis=0)
        nose.tools.assert_equal(ima[15, 25], self.source2.images['MUSE_WHITE'][9, 5])
        hst = Image('data/sdetect/a478hst.fits')
        self.source2.add_image(hst, 'HST1')
        size = self.source2.images['HST1'].shape[0]
        self.source2.add_image(hst, 'HST2', size=size, minsize=size, unit_size=None)
        nose.tools.assert_equal(self.source2.images['HST1'][10, 10],
                                self.source2.images['HST2'][10, 10])
        self.source2.add_image(hst, 'HST3', rotate=True)
        nose.tools.assert_almost_equal(self.source2.images['HST3'].get_rot(),
                                       self.source2.images['MUSE_WHITE'].get_rot(), 3)

    def test_add_narrow_band_image(self):
        """Source class: testing methods on narrow bands images"""
        cube = Cube('data/sdetect/minicube.fits')
        src = Source.from_data(ID=1, ra=63.35592651367188, dec=10.46536922454834,
                               origin=('test', 'v0', 'minicube.fits'))
        src.add_z('EMI', 0.086, 0.0001)
        src.add_white_image(cube)
        src.add_narrow_band_images(cube, 'EMI')
        nose.tools.assert_true('NB_OIII5007' in src.images)
        nose.tools.assert_true('NB_HALPHA' in src.images)
        nose.tools.assert_true('NB_HBETA' in src.images)
        src.add_narrow_band_image_lbdaobs(cube, 'OBS7128', 7128)
        nose.tools.assert_true('OBS7128' in src.images)
        src.add_seg_images()
        nose.tools.assert_true('SEG_MUSE_WHITE' in src.images)
        nose.tools.assert_true('SEG_NB_OIII5007' in src.images)
        nose.tools.assert_true('SEG_NB_HALPHA' in src.images)
        nose.tools.assert_true('SEG_NB_HBETA' in src.images)
        nose.tools.assert_true('SEG_OBS7128' in src.images)
        src.find_sky_mask(seg_tags=['SEG_MUSE_WHITE', 'SEG_NB_OIII5007', 'SEG_OBS7128', 'SEG_NB_HBETA', 'SEG_NB_HALPHA'])
        src.find_union_mask(union_mask='MASK_OBJ', seg_tags=['SEG_MUSE_WHITE', 'SEG_NB_OIII5007', 'SEG_OBS7128', 'SEG_NB_HBETA', 'SEG_NB_HALPHA'])
        src.find_intersection_mask(seg_tags=['SEG_MUSE_WHITE', 'SEG_NB_OIII5007', 'SEG_OBS7128', 'SEG_NB_HBETA', 'SEG_NB_HALPHA'])
        assert_array_equal(np.array(src.images['MASK_OBJ'].data.data, dtype=bool),
                           ~(np.array(src.images['MASK_SKY'].data.data, dtype=bool)))
        assert_array_equal(src.images['MASK_INTER'].data.data, np.zeros(src.images['MASK_INTER'].shape))
        nose.tools.assert_true('MASK_OBJ' in src.images)
        nose.tools.assert_true('MASK_INTER' in src.images)
        nose.tools.assert_true('MASK_SKY' in src.images)
        src.extract_spectra(cube, obj_mask='MASK_OBJ', skysub=True, psf=None)
        nose.tools.assert_true('MUSE_SKY' in src.spectra)
        nose.tools.assert_true('MUSE_TOT_SKYSUB' in src.spectra)
        nose.tools.assert_true('MUSE_WHITE_SKYSUB' in src.spectra)
        nose.tools.assert_true('NB_HALPHA_SKYSUB' in src.spectra)
        src.extract_spectra(cube, obj_mask='MASK_OBJ', skysub=False, psf=0.2 * np.ones(cube.shape[0]))
        nose.tools.assert_true('MUSE_PSF' in src.spectra)
        nose.tools.assert_true('MUSE_TOT' in src.spectra)
        nose.tools.assert_true('MUSE_WHITE' in src.spectra)
        nose.tools.assert_true('NB_HALPHA' in src.spectra)

    def test_sort_lines(self):
        """Source class: testing sort_lines method"""
        self.source1.sort_lines()
        nose.tools.assert_equal(self.source1.lines['LINE'][0], '[OIII]2')

#
#     @attr(speed='fast')
#     def test_catalog(self):
#         """Source class: tests catalog creation"""
#         cat = Catalog.from_sources([self.source1, self.source2])
#         print cat
