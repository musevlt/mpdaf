# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c)      2018 Simon Conseil <simon.conseil@univ-lyon1.fr>

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

import re
from astropy.io import fits

from ...tests.utils import get_data_file

from mpdaf.tools.fits import (copy_header, add_mpdaf_method_keywords,
                              add_mpdaf_keywords_to_file)
from mpdaf.version import __version__

REFHDR = """\
HIERARCH MPDAF METH1 VERSION = '{0}\\s*' / MPDAF version
HIERARCH MPDAF METH1 ID = 'func1   ' / MPDAF method identifier
HIERARCH MPDAF METH1 PARAM1 NAME = 'a       ' / test
HIERARCH MPDAF METH1 PARAM1 VALUE = 1
HIERARCH MPDAF METH1 PARAM2 NAME = 'b       ' / test2
HIERARCH MPDAF METH1 PARAM2 VALUE = 'hello   '
HIERARCH MPDAF METH2 VERSION = '{0}\\s*' / MPDAF version
HIERARCH MPDAF METH2 ID = 'func2   ' / MPDAF method identifier
HIERARCH MPDAF METH2 PARAM1 NAME = 'c       ' / the comment
HIERARCH MPDAF METH2 PARAM1 VALUE = 'with a very long parameter that i'
END
""".format(__version__)


def test_add_mpdaf_method_keywords():
    hdr = fits.Header()
    add_mpdaf_method_keywords(hdr, 'func1', ['a', 'b'], [1, 'hello'],
                              ['test', 'test2'])
    add_mpdaf_method_keywords(hdr, 'func2', ['c'],
                              ['with a very long parameter that is too long'],
                              ['the comment'])

    hdrtxt = hdr.tostring(sep='\n', padding=False)
    hdrlines = hdrtxt.splitlines()
    for ref, line in zip(REFHDR.splitlines(), [l.strip() for l in hdrlines]):
        assert re.match(ref, line) is not None


def test_add_mpdaf_keywords_to_file(tmpdir):
    testf = str(tmpdir.join('test.fits'))
    fits.PrimaryHDU().writeto(testf)
    add_mpdaf_keywords_to_file(testf, 'func1', ['a', 'b'], [1, 'hello'],
                               ['test', 'test2'])
    add_mpdaf_keywords_to_file(testf, 'func2', ['c'],
                               ['with a very long parameter that is too long'],
                               ['the comment'])

    hdr = fits.getheader(testf)
    hdrtxt = hdr[4:].tostring(sep='\n', padding=False)
    hdrlines = hdrtxt.splitlines()
    for ref, line in zip(REFHDR.splitlines(), [l.strip() for l in hdrlines]):
        assert re.match(ref, line) is not None


def test_copy_header():
    hdr = fits.getheader(get_data_file('sdetect', 'a478hst-cutout.fits'))[:50]
    hdr2 = copy_header(hdr)

    for c, c2 in zip(hdr.cards, hdr2.cards):
        assert ((c.keyword, c.value, c.comment) ==
                (c2.keyword, c2.value, c2.comment))
