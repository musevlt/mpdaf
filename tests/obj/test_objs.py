# -*- coding: utf-8 -*-

import nose.tools
from nose.plugins.attrib import attr

from mpdaf.obj.objs import is_float, is_int, circular_bounding_box


@attr(speed='fast')
def test_is_float():
    nose.tools.assert_true(is_float(1.2))
    nose.tools.assert_true(is_float(1))


@attr(speed='fast')
def test_is_int():
    nose.tools.assert_true(is_int(1))
    nose.tools.assert_false(is_int(1.2))


@attr(speed='fast')
def test_circular_bounding_box():
    shape = (4, 5)
    center = (2, 2)

    for radius in (1, (1, 1)):
        sy, sx = circular_bounding_box(center, radius, shape)
        nose.tools.assert_equal(sx, slice(1, 4))
        nose.tools.assert_equal(sy, slice(1, 4))

    sy, sx = circular_bounding_box((2, 2), (3, 3), shape)
    nose.tools.assert_equal(sy, slice(0, 4))
    nose.tools.assert_equal(sx, slice(0, 5))

    sy, sx = circular_bounding_box((0, 0), 1, shape)
    nose.tools.assert_equal(sy, slice(0, 2))
    nose.tools.assert_equal(sx, slice(0, 2))

    sy, sx = circular_bounding_box((-1, -1), 1, shape)
    nose.tools.assert_equal(sy, slice(0, 1))
    nose.tools.assert_equal(sx, slice(0, 1))

    sy, sx = circular_bounding_box((3, 4), 1, shape)
    nose.tools.assert_equal(sy, slice(2, 4))
    nose.tools.assert_equal(sx, slice(3, 5))
