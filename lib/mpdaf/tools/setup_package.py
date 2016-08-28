from __future__ import print_function

import os
import sys
from astropy_helpers.setup_helpers import pkg_config
from distutils.extension import Extension


def get_extensions():
    res = pkg_config(['cfitsio'], None)
    if res['libraries'] is not None:
        kw = {'libraries': ['m'] + res['libraries']}

        # Use OpenMP if directed or not on a Mac
        if os.environ.get('USEOPENMP') or not sys.platform.startswith('darwin'):
            kw.setdefault('extra_link_args', []).append('-lgomp')
            kw.setdefault('extra_compile_args', []).append('-fopenmp')
        else:
            print("Unable to find OPENMP")

        return [Extension('tools._ctools', [
            'src/tools.c', 'src/subtract_slice_median.c', 'src/merging.c'], **kw)]
    else:
        print('WARNING: mpdaf.tools.ctools extension was not build.\n'
              'cfitsio and pkg-config are required to build this module.')
