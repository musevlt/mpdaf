# -*- coding: utf-8 -*-

from numpy.testing import assert_array_equal


def assert_image_equal(ima, shape=None, start=None, end=None, step=None):
    if shape is not None:
        assert_array_equal(ima.shape, shape)
    if start is not None:
        assert_array_equal(ima.get_start(), start)
    if end is not None:
        assert_array_equal(ima.get_end(), end)
    if step is not None:
        assert_array_equal(ima.get_step(), step)
