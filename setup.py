# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2016 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2011-2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>
Copyright (c) 2015-2016 Simon Conseil <simon.conseil@univ-lyon1.fr>
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

from __future__ import print_function

import os
import subprocess
import sys

# Bootstrap setuptools if not available
import ez_setup
ez_setup.use_setuptools(version='18.0')  # NOQA

from setuptools import setup, find_packages, Command, Extension

# os.environ['DISTUTILS_DEBUG'] = '1'

PY2 = sys.version_info[0] == 2

# Check if Numpy is available
try:
    import numpy
except ImportError:
    sys.exit('You must install Numpy before MPDAF, as it is required to '
             'build C extensions.')

# Check if Cython is available
try:
    from Cython.Distutils import build_ext
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
    print('Found pkg-config {}'.format(out))
    del out
    HAVE_PKG_CONFIG = True

# Generate version.py
__version__ = None
with open('lib/mpdaf/version.py') as f:
    exec(f.read())

# If the version is not stable, we can add a git hash to the __version__
if '.dev' in __version__:
    # Find hash for __githash__ and dev number for __version__ (can't use hash
    # as per PEP440)
    command_hash = 'git rev-list --max-count=1 --abbrev-commit HEAD'
    command_number = 'git rev-list --count HEAD'

    try:
        commit_hash = subprocess.check_output(command_hash, shell=True)\
            .decode('ascii').strip()
        commit_number = subprocess.check_output(command_number, shell=True)\
            .decode('ascii').strip()
    except Exception:
        pass
    else:
        # We write the git hash and value so that they gets frozen if installed
        with open(os.path.join('lib', 'mpdaf', '_githash.py'), 'w') as f:
            f.write("__githash__ = \"{}\"\n".format(commit_hash))
            f.write("__dev_value__ = \"{}\"\n".format(commit_number))

        # We modify __version__ here too for commands such as egg_info
        __version__ += commit_number


class UnitTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = subprocess.call(['nosetests', '-a speed=fast',
                                 'lib/mpdaf', 'tests'])
        raise SystemExit(errno)


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
            if not PY2:
                out = out.decode('utf8')
            print('Found {} {}'.format(package, out))

    for token in subprocess.check_output(["pkg-config", "--libs", "--cflags",
                                          ' '.join(packages)]).split():
        if not PY2:
            token = token.decode('utf8')
        if token[:2] in flag_map:
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
        else:  # throw others to extra_link_args
            kw.setdefault('extra_link_args', []).append(token)

    kw.setdefault('libraries', []).append('m')

    # Use OpenMP if directed or not on a Mac
    if os.environ.get('USEOPENMP') or not sys.platform.startswith('darwin'):
        kw.setdefault('extra_link_args', []).append('-lgomp')
        kw.setdefault('extra_compile_args', []).append('-fopenmp')
    else:
        print("Unable to find OPENMP")

    for k, v in kw.items():  # remove duplicated
        kw[k] = list(set(v))
    return kw


with open('README.rst') as f:
    README = f.read()

with open('CHANGELOG') as f:
    CHANGELOG = f.read()

cmdclass = {'test': UnitTest}

ext = '.pyx' if HAVE_CYTHON else '.c'
ext_modules = [
    Extension('merging', ['src/tools.c', './lib/mpdaf/obj/merging' + ext],
              include_dirs=[numpy.get_include()]),
]
if HAVE_PKG_CONFIG:
    try:
        ext_modules.append(
            Extension('tools._ctools', [
                'src/tools.c', 'src/subtract_slice_median.c', 'src/merging.c'],
                **options('cfitsio')),
        )
    except Exception:
        pass
if HAVE_CYTHON:
    cmdclass.update({'build_ext': build_ext})
    ext_modules = cythonize(ext_modules)

setup(
    name='mpdaf',
    version=__version__,
    maintainer='Laure Piqueras',
    maintainer_email='laure.piqueras@univ-lyon1.fr',
    description='MUSE Python Data Analysis Framework is a python framework '
    'in view of the analysis of MUSE data in the context of the GTO.',
    long_description=README + '\n' + CHANGELOG,
    license='BSD',
    url='https://git-cral.univ-lyon1.fr/MUSE/mpdaf',
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy>=1.0', 'six'],
    extras_require={
        'all':  ['numexpr', 'fitsio'],
    },
    tests_require=['nose'],
    package_dir={'': 'lib'},
    packages=find_packages('lib'),
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    cmdclass=cmdclass,
    entry_points={
        'console_scripts': [
            'make_white_image = mpdaf.scripts.make_white_image:main'
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics'
    ],
)
