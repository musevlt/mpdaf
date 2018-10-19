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

import os
import sys
from collections import OrderedDict
from subprocess import check_output

LICENSE = """\
\"\"\"
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon

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
\"\"\"
"""

GIT_CMD = ("git log --date=format:%Y --format='%ad %aN <%aE>' --date-order "
           "--reverse {} | cut -f1 --complement")

EXCLUDES = (
    'lib/mpdaf/_githash.py',
    'lib/mpdaf/obj/wavelet1D.py',
)

ADDITIONAL_COPYRIGHTS = {
    'lib/mpdaf/sdetect/sea.py': [
        'Copyright (c) 2015-2016 Jarle Brinchman <jarle@strw.leidenuniv.nl>\n'
    ],
}


def modify(lines, license, start=0):
    print('Modify license text ... ', end='')
    for i, l in enumerate(lines[start + 1:], start + 1):
        if l == '"""\n':
            end = i
            break

    newlines = license + lines[end + 1:]
    if start == 1:
        newlines.insert(0, lines[0])
    return newlines


def insert(lines, license, pos=0):
    print('Insert license text ... ', end='')
    if pos == 0:
        return license + lines
    else:
        return lines[:pos] + license + lines[pos:]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: {} FILES...'.format(os.path.basename(sys.argv[0])))
        print('Insert or update copyright in docstrings')
        sys.exit()

    files = sys.argv[1:]

    for filename in files:
        print('- {} : '.format(filename), end='')

        if filename in EXCLUDES:
            print('SKIP')
            continue

        with open(filename, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            print('Empty file')
            continue

        authors = OrderedDict()
        for l in check_output(GIT_CMD.format(filename),
                              shell=True).splitlines():
            l = l.decode()
            year, author = l.split(' ', 1)
            if author in authors:
                authors[author].append(year)
            else:
                authors[author] = [year]

        authorlist = []
        if filename in ADDITIONAL_COPYRIGHTS:
            authorlist.extend(ADDITIONAL_COPYRIGHTS[filename])

        for author, years in authors.items():
            years = sorted(set(years))
            if len(years) == 1:
                year = '     ' + years[0]
            else:
                year = '{}-{}'.format(years[0], years[-1])
            authorlist.append('Copyright (c) {} {}\n'.format(year, author))

        license = LICENSE.splitlines(True)
        license = license[:2] + authorlist + license[2:]

        start = 1 if lines[0].startswith('# -*- coding') else 0

        if lines[start].startswith('"""Copyright'):
            lines = modify(lines, license, start=start)
        elif lines[start] == '"""\n' and \
                lines[start + 1].startswith('Copyright '):
            lines = modify(lines, license, start=start)
        else:
            lines = insert(lines, license, pos=start)

        with open(filename, 'w') as f:
            f.write(''.join(lines))

        print('OK')
