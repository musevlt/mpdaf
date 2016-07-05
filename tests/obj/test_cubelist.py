"""Test on Image objects."""

from __future__ import absolute_import, print_function
from nose.plugins.attrib import attr

import numpy as np
import os
import shutil
import tempfile
import unittest
from mpdaf.obj import CubeList, CubeMosaic
from numpy.testing import assert_array_equal
from ..utils import generate_cube


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

    def test_header(self, cube):
        self.assertEqual(cube.primary_header['FOO'], 'BAR')
        self.assertNotIn('CUBEIDX', cube.primary_header)
        self.assertEqual(cube.primary_header['OBJECT'], 'OBJECT 0')
        self.assertEqual(cube.data_header['OBJECT'], 'OBJECT 0')
        self.assertEqual(cube.primary_header['EXPTIME'], 100 * self.ncubes)

    @attr(speed='fast')
    def test_median(self):
        clist = CubeList(self.cubenames)
        combined_cube = np.ones(self.shape)

        for method in (clist.median, clist.pymedian):
            cube, expmap, stat_pix = method(header={'FOO': 'BAR'})
            self.test_header(cube)
            assert_array_equal(cube.data, combined_cube)
            assert_array_equal(expmap.data, self.expmap)

    @attr(speed='fast')
    def test_combine(self):
        clist = CubeList(self.cubenames)
        combined_cube = np.full(self.shape, 2, dtype=float)

        for method in (clist.combine, clist.pycombine):
            out = method(header={'FOO': 'BAR'})
            if method == clist.combine:
                cube, expmap, stat_pix = out
            else:
                cube, expmap, stat_pix, rejmap = out

            self.test_header(cube)
            assert_array_equal(cube.data, combined_cube)
            assert_array_equal(expmap.data, self.expmap)

    @attr(speed='fast')
    def test_mosaic_combine(self):
        clist = CubeMosaic(self.cubenames, self.cubenames[0])
        combined_cube = np.full(self.shape, 2, dtype=float)

        cube, expmap, stat_pix, rejmap = clist.pycombine(header={'FOO': 'BAR'})

        self.test_header(cube)
        assert_array_equal(cube.data, combined_cube)
        assert_array_equal(expmap.data, self.expmap)
