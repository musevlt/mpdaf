# -*- coding: utf-8 -*-
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
import re
import time
import warnings

from astropy.utils import minversion
from mpdaf.tools.util import (chdir, deprecated, broadcast_to_cube, timeit,
                              timer)

PYTEST_LT_3_3 = not minversion('pytest', '3.3')


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


@pytest.mark.skipif(PYTEST_LT_3_3, reason="caplog requires Pytest 3.3+")
def test_timeit(caplog):

    @timeit
    def func(foo, bar=0):
        time.sleep(bar)
        return foo

    assert func('a', bar=0.1) == 'a'
    assert re.search(r"INFO     'func' \(\('a',\), {'bar': 0.1}\) 0.1\d sec",
                     caplog.text) is not None


@pytest.mark.skipif(PYTEST_LT_3_3, reason="caplog requires Pytest 3.3+")
def test_timer(caplog):

    def func(foo, bar=0):
        time.sleep(bar)
        return foo

    with timer():
        out = func('a', bar=0.1)

    assert out == 'a'
    assert re.search(r'Request took 0.1\d\d sec.', caplog.text) is not None
