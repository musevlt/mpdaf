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

from distutils.core import setup, Command
import setuptools

class UnitTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        #errno = subprocess.call([sys.executable, 'python', 'tests/run_tests.py'])
        errno = subprocess.call(['python', 'tests/run_tests.py'])
        raise SystemExit(errno)

setup(name = 'mpdaf',
      version = '0.1.0',
      description = 'MUSE Python Data Analysis Framework is a python framework in view of '
                    'the analysis of MUSE data in the context of the GTO.',
      url = 'http://urania1.univ-lyon1.fr/mpdaf/login',
      requires = ['numpy (>= 1.0)', 'pyfits (>= 2.3)'],
      install_requires = ['pywcs','astropysics'],
      provides = ['mpdaf'],
      package_dir = {'': 'lib/'},
      packages = ['drs','obj'],
      maintainer = 'Laure Piqueras',
      maintainer_email = 'laure.piqueras@univ-lyon1.fr',
      platforms = 'any', 
      cmdclass = {'test': UnitTest},
     )
