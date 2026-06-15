# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2011-2017 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2015-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2015 Aurelien Jarno <aurelien.jarno@univ-lyon1.fr>

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

# See README.rst for details on how to install MPDAF.

import numpy
import os
import sys

from Cython.Build import cythonize
from setuptools import setup, Extension
from extension_helpers import add_openmp_flags_if_available, pkg_config


def use_openmp():
    """Find if OpenMP must be used or not. Disabled by default on MacOS,
    enabled otherwise. Usage can be forced with the USEOPENMP env var.
    """
    openmp_env = os.environ.get('USEOPENMP')
    if openmp_env == '1':
        print('OPENMP enabled from USEOPENMP env var')
        return True
    elif openmp_env == '0':
        print('OPENMP disabled from USEOPENMP env var')
        return False

    if openmp_env is not None:
        print('USEOPENMP env var must be set to 0 or 1')

    if sys.platform.startswith('darwin'):
        print('OPENMP disabled by default on MacOS')
        return False
    else:
        print('OPENMP enabled by default')
        return True


ext_modules = [
    Extension(
        'mpdaf.obj.merging',
        ['src/tools.c', './lib/mpdaf/obj/merging.pyx'],
        include_dirs=[numpy.get_include()],
    ),
]

try:
    options = pkg_config(['cfitsio'], [])
    if options["libraries"]:
        ext = Extension('mpdaf.tools._ctools',
                        ['src/tools.c', 'src/merging.c'],
                        **options)
        if use_openmp():
            add_openmp_flags_if_available(ext)
        ext_modules.append(ext)
    else:
        print("cfitsio is missing, the cube combination extension will not be built")
except Exception as e:
    print("problem while building the cube merging extension:")
    print(e)

ext_modules = cythonize(ext_modules, compiler_directives={'language_level': 3})

print('Configuration done, now running setup() ...\n')

setup(ext_modules=ext_modules)
