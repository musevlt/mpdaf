# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import os
import pytest
import warnings

from mpdaf.tools.util import chdir, deprecated, broadcast_to_cube


def test_chdir(tmpdir):
    cwd = os.getcwd()
    tmp = str(tmpdir)
    with chdir(tmp):
        assert tmp == os.getcwd()

    assert cwd == os.getcwd()


def test_deprecated():
    msg = 'This function is deprecated'

    @deprecated(msg)
    def func():
        pass

    with warnings.catch_warnings(record=True) as w:
        func()
        assert (w[0].message.args[0] ==
                'Call to deprecated function `func`. ' + msg)


def test_broadcast_to_cube():
    shape = (5, 4, 3)

    with pytest.raises(AssertionError):
        broadcast_to_cube(np.zeros(5), (2, 2))

    for s in (5, (4, 3), shape):
        assert broadcast_to_cube(np.zeros(s), shape).shape == shape

    for s in (4, (5, 3), (4, 4, 3)):
        with pytest.raises(ValueError):
            broadcast_to_cube(np.zeros(s), shape)
