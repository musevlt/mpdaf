# -*- coding: utf-8 -*-

import numpy as np
from nose.tools import assert_true


def assert_array_equal(arr1, arr2):
    assert_true(np.array_equal(arr1, arr2))


def assert_image_equal(ima, shape=None, start=None, end=None, step=None):
    if shape is not None:
        assert_array_equal(ima.shape, shape)
    if start is not None:
        assert_array_equal(ima.get_start(), start)
    if end is not None:
        assert_array_equal(ima.get_end(), end)
    if step is not None:
        assert_array_equal(ima.get_step(), step)
