# -*- coding: utf-8 -*-

import nose.tools
from nose.plugins.attrib import attr

from mpdaf.obj import objs


@attr(speed='fast')
def test_is_float():
    nose.tools.assert_true(objs.is_float(1.2))
    nose.tools.assert_false(objs.is_float(1))


@attr(speed='fast')
def test_is_int():
    nose.tools.assert_true(objs.is_int(1))
    nose.tools.assert_false(objs.is_int(1.2))


@attr(speed='fast')
def test_circular_bounding_box():
    sy, sx = objs.circular_bounding_box((2, 2), 1, (5, 5))
    nose.tools.assert_tuple_equal((sx.start, sx.stop), (1, 4))
    sy, sx = objs.circular_bounding_box((2, 2), (1, 1), (5, 5))
    nose.tools.assert_tuple_equal((sx.start, sx.stop), (1, 4))
    sy, sx = objs.circular_bounding_box((2, 2), (3, 3), (5, 5))
    nose.tools.assert_tuple_equal((sx.start, sx.stop), (0, 5))
