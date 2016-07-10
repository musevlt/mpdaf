# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os
import tempfile
import warnings

from mpdaf.tools import util


def test_chdir():
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    with util.chdir(tmp):
        assert tmp == os.getcwd()

    assert cwd == os.getcwd()


def test_deprecated():
    msg = 'This function is deprecated'

    @util.deprecated(msg)
    def func():
        pass

    with warnings.catch_warnings(record=True) as w:
        func()
        assert (w[0].message.args[0] ==
                'Call to deprecated function `func`. ' + msg)
