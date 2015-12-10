# -*- coding: utf-8 -*-

# Copyright (C) 2011  Centre de Recherche Astronomique de Lyon (CRAL)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     3. The name of AURA and its representatives may not be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY CRAL ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
#

# See README.rst for details on how to install MPDAF.

from __future__ import print_function

import os
import subprocess
import sys
import shutil

# Bootstrap setuptools if not available
import ez_setup
ez_setup.use_setuptools(version='18.0')  # NOQA

from setuptools import setup, find_packages, Command, Extension

# os.environ['DISTUTILS_DEBUG'] = '1'

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
    sys.exit('pkg-config is required to install MPDAF. Please check if it '
             'is installed and in your $PATH.')
else:
    print('Found pkg-config {}'.format(out))
    del out

# rm old focus directory
try:
    import mpdaf
    d = mpdaf.__path__[0]+'/sdetect/focus'
    if os.path.exists(d):
        shutil.rmtree('build')
        shutil.rmtree(mpdaf.__path__[0]+'/sdetect/focus')
except:
    pass

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
        errno = subprocess.call(['nosetests', '-v', '-a speed=fast',
                                 '--logging-clear-handlers', 'tests/'])
        raise SystemExit(errno)


def options(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}

    for package in packages:
        try:
            out = subprocess.check_output(['pkg-config', '--modversion',
                                           package])
        except subprocess.CalledProcessError:
            sys.exit("package '{}' not found.".format(package))
        else:
            print('Found {} {}'.format(package, out))

    for token in subprocess.check_output(["pkg-config", "--libs", "--cflags",
                                          ' '.join(packages)]).split():
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

    for k, v in kw.iteritems():  # remove duplicated
        kw[k] = list(set(v))
    return kw


with open('README.rst') as f:
    README = f.read()

with open('CHANGES.rst') as f:
    CHANGELOG = f.read()

cmdclass = {'test': UnitTest}

ext = '.pyx' if HAVE_CYTHON else '.c'
ext_modules = [
    Extension('libCmethods', [
        'src/tools.c', 'src/subtract_slice_median.c', 'src/merging.c'],
        **options('cfitsio')),
    Extension('merging', ['src/tools.c', './lib/mpdaf/obj/merging' + ext],
              include_dirs=[numpy.get_include()]),
]
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
    url='http://urania1.univ-lyon1.fr/mpdaf/login',
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'numexpr'],
    tests_require=['nose'],
    extras_require={'Image':  ['Pillow']},
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
)
