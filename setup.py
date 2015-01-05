#!/usr/bin/python

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

import os
import commands
import subprocess
# import setuptools

from distutils.core import setup, Command, Extension


class UnitTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = subprocess.call(['nosetests', '-v', '-a speed=fast'])
        raise SystemExit(errno)


class MakeFusion(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import shutil
        import mpdaf.fusion
        errno = subprocess.call(['make', '-C', 'lib/mpdaf/fusion/'])
        shutil.copy('lib/mpdaf/fusion/fusion_fit', '/usr/local/bin/fusion_fit')
        shutil.copy('lib/mpdaf/fusion/fusion_FSF', '/usr/local/bin/fusion_FSF')
        shutil.copy('lib/mpdaf/fusion/fusion_LSF', '/usr/local/bin/fusion_LSF')
        shutil.copy('lib/mpdaf/fusion/fusion_residual', '/usr/local/bin/fusion_residual')
        shutil.copy('lib/mpdaf/fusion/fusion_resampling', '/usr/local/bin/fusion_resampling')
        shutil.copy('lib/mpdaf/fusion/fusion_variance', '/usr/local/bin/fusion_variance')
        errno = subprocess.call(['make', 'cleanall', '-C', 'lib/mpdaf/fusion/'])
        path = os.path.abspath(os.path.dirname(mpdaf.fusion.__file__))
        shutil.copy('lib/mpdaf/fusion/examples/LSF_V1.fits', path + '/LSF_V1.fits')

package_dir = {'mpdaf': 'lib/mpdaf/', 'mpdaf_user': 'mpdaf_user/'}
if os.path.isfile('lib/mpdaf/fusion/__init__.py'):
    packages = ['mpdaf', 'mpdaf.tools', 'mpdaf.obj', 'mpdaf.fusion', 'mpdaf.drs', 'mpdaf.MUSE', 'mpdaf_user']
else:
    packages = ['mpdaf', 'mpdaf.tools', 'mpdaf.obj', 'mpdaf.drs', 'mpdaf.MUSE', 'mpdaf_user']
for path in os.listdir('mpdaf_user'):
    if os.path.isdir('mpdaf_user/' + path + '/lib/' + path):
        package_dir['mpdaf_user.' + path] = 'mpdaf_user/' + path + '/lib/' + path
        packages.append('mpdaf_user.' + path)
    if os.path.isfile('mpdaf_user/' + path + '/__init__.py'):
        package_dir['mpdaf_user.' + path] = 'mpdaf_user/' + path + '/lib/'
        packages.append('mpdaf_user.' + path)
        #package_dir[path] = 'mpdaf_user/'+path
        # packages.append(path)
    if os.path.isfile('mpdaf_user/' + path + '/' + path + '.py'):
        package_dir['mpdaf_user.' + path] = 'mpdaf_user/' + path
        packages.append('mpdaf_user.' + path)


def options(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}

    err = commands.getstatusoutput("pkg-config --version")
    if err[0] != 0:
        raise ImportError(err[1])

    for package in packages:
        err = commands.getstatusoutput("pkg-config %s" % package)
        if err[0] != 0:
            raise ImportError(err[1])

    for token in commands.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
        if flag_map.has_key(token[:2]):
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
        else:  # throw others to extra_link_args
            kw.setdefault('extra_link_args', []).append(token)

    kw.setdefault('libraries', []).append('m')
    kw.setdefault('extra_link_args', []).append('-fopenmp')
    kw.setdefault('extra_compile_args', []).append('-fopenmp')

    for k, v in kw.iteritems():  # remove duplicated
        kw[k] = list(set(v))
    return kw


setup(name='mpdaf',
      version='1.1',
      description='MUSE Python Data Analysis Framework is a python framework '
      'in view of the analysis of MUSE data in the context of the GTO.',
      url='http://urania1.univ-lyon1.fr/mpdaf/login',
      requires=['numpy (>= 1.0)', 'scipy (>= 0.10)', 'matplotlib', 'astropy',
                'nose', 'PIL'],
      package_dir=package_dir,
      packages=packages,
      package_data={'mpdaf.drs': ['mumdatMask_1x1/*.fits']},
      maintainer='Laure Piqueras',
      maintainer_email='laure.piqueras@univ-lyon1.fr',
      platforms='any',
      cmdclass={'test': UnitTest, 'fusion': MakeFusion},
      ext_package='mpdaf',
      ext_modules=[Extension('libCmethods', ['src/subtract_slice_median.c', 'src/merging.c'], **options('cfitsio'))],
      )
