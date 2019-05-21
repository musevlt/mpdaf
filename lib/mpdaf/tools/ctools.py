# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2016-2019 Simon Conseil <simon.conseil@univ-lyon1.fr>

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

# Load functions from the C extension.
#
# Note: closing the shared library issue with OpenMP
#
# old close libray code:
# import _ctypes
# _ctypes.dlclose(libCmethods._handle)
# libCmethods._handle = None
# libCmethods._name = None
# libCmethods._FuncPtr = None
# del libCmethods


import ctypes
import logging
import os
from numpy.ctypeslib import ndpointer, load_library


LIBRARY_PATH = os.path.dirname(__file__)

try:
    # load the library, using numpy mechanisms
    ctools = load_library("_ctools", LIBRARY_PATH)
except OSError:  # pragma: no cover
    logging.getLogger(__name__).error(
        "MPDAF's C extension is missing, probably it was not compiled because "
        "of missing dependencies.\n Try rebuilding MPDAF.")
else:
    # define argument types
    charptr = ctypes.POINTER(ctypes.c_char)
    array_1d_double = ndpointer(dtype=ctypes.c_double, ndim=1,
                                flags='C_CONTIGUOUS')
    array_1d_int = ndpointer(dtype=ctypes.c_int, ndim=1, flags='C_CONTIGUOUS')

    # mpdaf_merging_median
    ctools.mpdaf_merging_median.argtypes = [
        charptr,          # char* input
        array_1d_double,  # double* data
        array_1d_int,     # int* expmap
        array_1d_int      # int* valid_pix
    ]

    # mpdaf_merging_sigma_clipping
    ctools.mpdaf_merging_sigma_clipping.argtypes = [
        charptr,          # char* input
        array_1d_double,  # double* data
        array_1d_double,  # double* var
        array_1d_int,     # int* expmap
        array_1d_double,  # double* scale
        array_1d_double,  # double* offset
        array_1d_int,     # int* selected_pix
        array_1d_int,     # int* valid_pix
        ctypes.c_int,     # int nmax
        ctypes.c_double,  # double nclip_low
        ctypes.c_double,  # double nclip_up
        ctypes.c_int,     # int nstop
        ctypes.c_int,     # int typ_var
        ctypes.c_int      # int mad
    ]
