# -*- coding: utf-8 -*-

from __future__ import absolute_import
import nose.tools
import os
import tempfile
import warnings
from nose.plugins.attrib import attr

from mpdaf.tools import util


@attr(speed='fast')
def test_chdir():
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    with util.chdir(tmp):
        nose.tools.assert_equal(tmp, os.getcwd())

    nose.tools.assert_equal(cwd, os.getcwd())


@attr(speed='fast')
def test_deprecated():
    msg = 'This function is deprecated'

    @util.deprecated(msg)
    def func():
        pass

    with warnings.catch_warnings(record=True) as w:
        func()
        nose.tools.assert_equal(w[0].message.args[0],
                                'Call to deprecated function `func`. ' + msg)
