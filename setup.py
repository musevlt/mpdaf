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
import subprocess
import sys

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand

# Check if Cython is available
try:
    from Cython.Build import cythonize
except ImportError:
    HAVE_CYTHON = False
else:
    HAVE_CYTHON = True
    print('Cython detected, building from sources.')

# Check if pkg-config is available
try:
    out = subprocess.check_output(['pkg-config', '--version'])
except subprocess.CalledProcessError as e:
    sys.exit(e.output)
except OSError:
    print('pkg-config is required to build C extensions for some MPDAF '
          'features (cube combination). Continuing the installation without '
          'building these extensions. Please check if pkg-config is installed '
          'and in your $PATH and rebuild MPDAF if you need them.')
    HAVE_PKG_CONFIG = False
else:
    out = out.decode(encoding='utf-8', errors='replace')
    print('Found pkg-config {}'.format(out.strip('\n')))
    del out
    HAVE_PKG_CONFIG = True


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


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


def options(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}

    for package in packages:
        try:
            out = subprocess.check_output(['pkg-config', '--modversion',
                                           package])
        except subprocess.CalledProcessError:
            msg = "package '{}' not found.".format(package)
            print(msg)
            raise Exception(msg)
        else:
            out = out.decode('utf8')
            print('Found {} {}'.format(package, out.strip('\n')))

    for token in subprocess.check_output(["pkg-config", "--libs", "--cflags",
                                          ' '.join(packages)]).split():
        token = token.decode('utf8')
        if token[:2] in flag_map:
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
        else:  # throw others to extra_link_args
            kw.setdefault('extra_link_args', []).append(token)

    kw.setdefault('libraries', []).append('m')

    if use_openmp():
        kw.setdefault('extra_link_args', []).append('-lgomp')
        kw.setdefault('extra_compile_args', []).append('-fopenmp')

    for k, v in kw.items():  # remove duplicated
        kw[k] = list(set(v))
    return kw


with open('README.rst') as f:
    README = f.read()

ext = '.pyx' if HAVE_CYTHON else '.c'
ext_modules = [
    Extension('obj.merging',
              ['src/tools.c', './lib/mpdaf/obj/merging' + ext],
              include_dirs=[numpy.get_include()]),
]

if HAVE_PKG_CONFIG:
    try:
        ext_modules.append(
            Extension('tools._ctools',
                      ['src/tools.c', 'src/merging.c'],
                      **options('cfitsio')),
        )
    except Exception:
        pass

if HAVE_CYTHON:
    ext_modules = cythonize(ext_modules,
                            compiler_directives={'language_level': 3})

print('Configuration done, now running setup() ...\n')

setup(
    name='mpdaf',
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'numpy>=1.10.0'],
    maintainer='Laure Piqueras',
    maintainer_email='laure.piqueras@univ-lyon1.fr',
    description='MUSE Python Data Analysis Framework is a python framework '
    'in view of the analysis of MUSE data in the context of the GTO.',
    long_description=README,
    license='BSD',
    url='https://git-cral.univ-lyon1.fr/MUSE/mpdaf',
    python_requires='>=3.9',
    install_requires=['numpy>=1.10.0', 'scipy>=1.11.0', 'matplotlib', 'astropy>=1.0','specutils'],
    extras_require={
        'all': ['numexpr', 'fitsio', 'adjustText', 'joblib', 'tqdm',
                'specutils'],
        'docs': ['sphinx', 'sphinx_rtd_theme', 'sphinx_automodapi',
                 'numpydoc', 'ipython', 'psutil','specutils'],
    },
    tests_require=['pytest'],
    package_dir={'': 'lib'},
    packages=find_packages('lib'),
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    cmdclass={'test': PyTest, 'build_ext': build_ext},
    entry_points={
        'console_scripts': [
            'make_white_image = mpdaf.scripts.make_white_image:main',
        ],
    },
    scripts=['lib/mpdaf/scripts/topcat_show_ds9'],
    ext_package='mpdaf',
    ext_modules=ext_modules,
    keywords=['astronomy', 'astrophysics', 'science', 'muse', 'vlt', 'cube',
              'image', 'spectrum'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics'
    ],
)
