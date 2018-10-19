# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2017 Simon Conseil <simon.conseil@univ-lyon1.fr>
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
from numpy.testing import (assert_array_equal, assert_allclose,
                           assert_almost_equal)

from mpdaf.obj import (is_float, is_int, bounding_box, flux2mag,
                       mag2flux, UnitArray, UnitMaskedArray)


def test_is_float():
    assert is_float(1.2)
    assert is_float(1)


def test_is_int():
    assert is_int(1)
    assert not is_int(1.2)


def test_mag_flux():
    assert_almost_equal(flux2mag(mag2flux(20, 7000), 0, 7000)[0], 20)


def test_unit_array():
    arr = np.arange(5)
    assert UnitArray(arr, u.m, u.m) is arr
    assert_array_equal(UnitArray(arr, u.m, u.mm), arr * 1e3)


def test_unit_masked_array():
    arr = np.ma.arange(5)
    assert UnitMaskedArray(arr, u.m, u.m) is arr
    assert_array_equal(UnitMaskedArray(arr, u.m, u.mm), arr * 1e3)


def test_bounding_box():
    shape = (4, 5)

    # Check that specifying just one radius, or two identical radii
    # produce the same expected result. The requested half-width of 1
    # pixel should result in a region of 2x2 pixels being masked,
    # regardless of the chosen center.  The center of a 2x2 pixel
    # region is at the corner of the 4 selected pixels. The closest
    # corner to the requested center of pixel index 2.2,1.8 is
    # 2.5, 1.5, so the center should be at that position, and the
    # selected pixels should be 2 and 3 along the Y axis, and
    # 1 and 2 along the X axis.

    for radius in (1, (1, 1)):
        [sy, sx], [uy, yx], c = bounding_box("rectangle", (2.2, 1.8), radius,
                                             posangle=0.0, shape=shape,
                                             step=[1.0, 1.0])
        assert sy == slice(2, 4)
        assert sx == slice(1, 3)
        assert_allclose(c, np.array([2.5, 1.5]))

    # Ask for a 6x6 region centered at 2.2, 1.8.
    # If the image were large enough, this would select pixel indexes
    # 0-5 (inclusive) along the Y axis, with a center of 2.5, and
    # pixel indexes -1-4 (inclusive) along the X axis, with a center of
    # 1.5. After clipping the indexes to acommodate the shape (4,5),
    # we expect the center to still be 2.5,1.5, but the selected pixels
    # should change to 0-3 (inclusive) along the Y axis, and 0-4 along
    # the Y axis.
    [sy, sx], [uy, ux], c = bounding_box("rectangle", (2.2, 1.8), (3, 3),
                                         posangle=0.0, shape=shape,
                                         step=[1.0, 1.0])
    assert sy == slice(0, 4)
    assert sx == slice(0, 5)
    assert_allclose(c, np.array([2.5, 1.5]))

    # Place the center at 0.1,-0.1 and the radius to 1, to check the
    # behavior for centers close to zero. A radius of 1 requests a
    # 2x2 region of pixels, which can only be centered half way between
    # pixels. The closest corner to 0.1,-0.1 is 0.5,-0.5, so before
    # clipping the selected pixels would be 0,1 along the Y-axis and
    # -1,0 along the X axis. After clipping only pixel 0 should remain
    # selected along the X-axis.
    [sy, sx], [uy, ux], c = bounding_box("rectangle", (0.1, -0.1), 1,
                                         posangle=0.0, shape=shape,
                                         step=[1.0, 1.0])
    assert sy == slice(0, 2)
    assert sx == slice(0, 1)
    assert_allclose(c, np.array([0.5, -0.5]))

    # Request a region that is an odd number of pixels in width and height
    # (3x3), centered at (2.1,2.8). A 3x3 region of pixels can only be
    # centered at the center of a pixel, and the nearest pixel center is
    # 2.0,3.0, so we expect it to select pixels 1,2,3 along the Y axis,
    # and 2,3,4 along the X axis.
    [sy, sx], [uy, ux], c = bounding_box("rectangle", (2.1, 2.8), 1.5,
                                         posangle=0.0,
                                         shape=shape, step=[1.0, 1.0])
    assert sy == slice(1, 4)
    assert sx == slice(2, 5)
    assert_allclose(c, np.array([2.0, 3.0]))

    # Request a region that is entirely outside the array.
    # The requested size is 3x3. This would be centered at the requested
    # center of -5,10 if it fitted in the image. The selected range along
    # the Y axis is off the lower edge of the array, so its slice should
    # be the zero pixel range slice(0,0). The selected range along the
    # X axis is above the maximum pixel index of shape[1], so a zero
    # pixel-range slice of slice(shape[1]-1,shape[1]-1) should be returned.
    [sy, sx], [uy, ux], c = bounding_box("rectangle", (-5, 10), 1.5,
                                         posangle=0.0,
                                         shape=shape, step=[1.0, 1.0])
    assert sy == slice(0, 0)
    assert sx == slice(shape[1] - 1, shape[1] - 1)
