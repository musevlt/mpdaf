"""Test on Image objects."""

from __future__ import absolute_import, print_function

import numpy as np
import os
import pytest
import shutil
import tempfile
import unittest

from mpdaf.obj import CubeList, CubeMosaic
from numpy.testing import assert_array_equal
from ...tests.utils import generate_cube

try:
    import fitsio  # NOQA
except ImportError:
    HAS_FITSIO = False
else:
    HAS_FITSIO = True


class TestCubeList(unittest.TestCase):

    shape = (5, 4, 3)
    ncubes = 3
    cubevals = [0, 1, 5]

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        print('\n>>> Create cubes in', cls.tmpdir)
        cls.cubenames = []
        for i in cls.cubevals:
            cube = generate_cube(data=i, shape=cls.shape)
            cube.primary_header['CUBEIDX'] = i
            cube.primary_header['OBJECT'] = 'OBJECT %d' % i
            cube.primary_header['EXPTIME'] = 100
            filename = os.path.join(cls.tmpdir, 'cube-%d.fits' % i)
            cube.write(filename, savemask='nan')
            cls.cubenames.append(filename)
        cls.expmap = np.full(cls.shape, cls.ncubes, dtype=int)

    @classmethod
    def tearDownClass(cls):
        print('>>> Remove test dir')
        shutil.rmtree(cls.tmpdir)

    def assert_header(self, cube):
        assert cube.primary_header['FOO'] == 'BAR'
        assert 'CUBEIDX' not in cube.primary_header
        assert cube.primary_header['OBJECT'] == 'OBJECT 0'
        assert cube.data_header['OBJECT'] == 'OBJECT 0'
        assert cube.primary_header['EXPTIME'] == 100 * self.ncubes

    def test_get_item(self):
        clist = CubeList(self.cubenames)
        assert_array_equal(clist[0, 2, 2], self.cubevals)
        assert_array_equal(np.array([a.data for a in clist[0, :, :]])[:, 0, 0],
                           self.cubevals)
        assert_array_equal(np.array([a.data for a in clist[0]])[:, 0, 0],
                           self.cubevals)

    def test_checks(self):
        cube = generate_cube(shape=(3, 2, 1))
        cube.write(os.path.join(self.tmpdir, 'cube-tmp.fits'), savemask='nan')
        clist = CubeList(self.cubenames[:1] + [cube.filename])
        assert clist.check_dim() is False
        assert clist.check_wcs() is False

        cube = generate_cube(shape=self.shape, crval=12.)
        cube.write(os.path.join(self.tmpdir, 'cube-tmp.fits'), savemask='nan')
        clist = CubeList(self.cubenames[:1] + [cube.filename])
        assert clist.check_dim() is True
        assert clist.check_wcs() is False

    @pytest.mark.skipif(not HAS_FITSIO, reason="requires fitsio")
    def test_median(self):
        clist = CubeList(self.cubenames)
        combined_cube = np.ones(self.shape)

        for method in (clist.median, clist.pymedian):
            cube, expmap, stat_pix = method(header={'FOO': 'BAR'})
            self.assert_header(cube)
            assert_array_equal(cube.data, combined_cube)
            assert_array_equal(expmap.data, self.expmap)

    @pytest.mark.skipif(not HAS_FITSIO, reason="requires fitsio")
    def test_combine(self):
        clist = CubeList(self.cubenames)
        combined_cube = np.full(self.shape, 2, dtype=float)

        for method in (clist.combine, clist.pycombine):
            out = method(header={'FOO': 'BAR'})
            if method == clist.combine:
                cube, expmap, stat_pix = out
            else:
                cube, expmap, stat_pix, rejmap = out

            self.assert_header(cube)
            assert_array_equal(cube.data, combined_cube)
            assert_array_equal(expmap.data, self.expmap)

        for method in (clist.combine, clist.pycombine):
            cube = method(nclip=(5., 5.), var='stat_mean')[0]
            assert_array_equal(cube.data, combined_cube)

    @pytest.mark.skipif(not HAS_FITSIO, reason="requires fitsio")
    def test_combine_scale(self):
        clist = CubeList(self.cubenames, scalelist=[2.]*self.ncubes)
        combined_cube = np.full(self.shape, 2*2, dtype=float)
        cube, expmap, stat_pix = clist.combine(header={'FOO': 'BAR'})
        assert_array_equal(cube.data, combined_cube)

    @pytest.mark.skipif(not HAS_FITSIO, reason="requires fitsio")
    def test_mosaic_combine(self):
        clist = CubeMosaic(self.cubenames, self.cubenames[0])
        combined_cube = np.full(self.shape, 2, dtype=float)

        cube, expmap, stat_pix, rejmap = clist.pycombine(header={'FOO': 'BAR'})

        self.assert_header(cube)
        assert_array_equal(cube.data, combined_cube)
        assert_array_equal(expmap.data, self.expmap)
