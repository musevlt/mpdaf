"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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
import os
import pytest
import shutil
import tempfile
import unittest

from mpdaf.obj import CubeList, CubeMosaic
from numpy.testing import assert_array_equal
from ...tests.utils import generate_cube

try:
    import fitsio  # noqa
except ImportError:
    HAS_FITSIO = False
else:
    HAS_FITSIO = True

try:
    import mpdaf.tools.ctools  # noqa
except OSError:
    HAS_CFITSIO = False
else:
    HAS_CFITSIO = True


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

    @pytest.mark.skipif(not HAS_CFITSIO, reason="requires cfitsio")
    def test_median(self):
        clist = CubeList(self.cubenames)
        combined_cube = np.ones(self.shape)
        cube, expmap, stat_pix = clist.median(header={'FOO': 'BAR'})
        self.assert_header(cube)
        assert_array_equal(cube.data, combined_cube)
        assert_array_equal(expmap.data, self.expmap)

    @pytest.mark.skipif(not HAS_FITSIO, reason="requires fitsio")
    def test_pymedian(self):
        clist = CubeList(self.cubenames)
        combined_cube = np.ones(self.shape)
        cube, expmap, stat_pix = clist.pymedian(header={'FOO': 'BAR'})
        self.assert_header(cube)
        assert_array_equal(cube.data, combined_cube)
        assert_array_equal(expmap.data, self.expmap)

    @pytest.mark.skipif(not HAS_CFITSIO, reason="requires cfitsio")
    def test_combine(self):
        clist = CubeList(self.cubenames)
        combined_cube = np.full(self.shape, 2, dtype=float)

        cube, expmap, stat_pix = clist.combine(header={'FOO': 'BAR'})
        cube2, expmap2, stat_pix2 = clist.combine(header={'FOO': 'BAR'},
                                                  mad=True)
        assert_array_equal(cube.data, cube2.data)
        assert_array_equal(expmap.data, expmap2.data)

        self.assert_header(cube)
        assert_array_equal(cube.data, combined_cube)
        assert_array_equal(expmap.data, self.expmap)

        cube = clist.combine(nclip=(5., 5.), var='stat_mean')[0]
        assert_array_equal(cube.data, combined_cube)

    @pytest.mark.skipif(not HAS_FITSIO, reason="requires fitsio")
    def test_pycombine(self):
        clist = CubeList(self.cubenames)
        combined_cube = np.full(self.shape, 2, dtype=float)

        cube, expmap, stat_pix, rejmap = clist.pycombine(header={'FOO': 'BAR'})
        cube2, expmap2, stat_pix2, rejmap2 = clist.pycombine(
            header={'FOO': 'BAR'}, mad=True)
        assert_array_equal(cube.data, cube2.data)
        assert_array_equal(expmap.data, expmap2.data)

        self.assert_header(cube)
        assert_array_equal(cube.data, combined_cube)
        assert_array_equal(expmap.data, self.expmap)

        cube = clist.pycombine(nclip=(5., 5.), var='stat_mean')[0]
        assert_array_equal(cube.data, combined_cube)

    @pytest.mark.skipif(not HAS_CFITSIO, reason="requires cfitsio")
    def test_combine_scale(self):
        clist = CubeList(self.cubenames, scalelist=[2.] * self.ncubes)
        combined_cube = np.full(self.shape, 2 * 2, dtype=float)
        cube, expmap, stat_pix = clist.combine(header={'FOO': 'BAR'})
        assert_array_equal(cube.data, combined_cube)

    @pytest.mark.skipif(not HAS_FITSIO, reason="requires fitsio")
    def test_pycombine_scale(self):
        clist = CubeList(self.cubenames, scalelist=[2.] * self.ncubes)
        combined_cube = np.full(self.shape, 2 * 2, dtype=float)

        cube2, expmap2, _, _ = clist.pycombine(header={'FOO': 'BAR'})
        assert_array_equal(cube2.data, combined_cube)

        clist = CubeList(self.cubenames, scalelist=[2.] * self.ncubes,
                         offsetlist=[0.5] * self.ncubes)
        combined_cube = np.full(self.shape, 5, dtype=float)

        cube2, expmap2, _, _ = clist.pycombine(header={'FOO': 'BAR'})
        assert_array_equal(cube2.data, combined_cube)

    @pytest.mark.skipif(not HAS_FITSIO, reason="requires fitsio")
    def test_mosaic_combine(self):
        clist = CubeMosaic(self.cubenames, self.cubenames[0])
        combined_cube = np.full(self.shape, 2, dtype=float)

        cube, expmap, stat_pix, rejmap = clist.pycombine(header={'FOO': 'BAR'})

        self.assert_header(cube)
        assert_array_equal(cube.data, combined_cube)
        assert_array_equal(expmap.data, self.expmap)

        cube2, expmap2, _, _ = clist.pycombine(header={'FOO': 'BAR'}, mad=True)
        assert_array_equal(cube.data, cube2.data)
        assert_array_equal(expmap.data, expmap2.data)
