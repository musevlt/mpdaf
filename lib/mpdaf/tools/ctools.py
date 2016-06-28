# -*- coding: utf-8 -*-

"""
Load functions from the C extension.

Note: closing the shared library issue with OpenMP

    # close libray
    # import _ctypes
    # _ctypes.dlclose(libCmethods._handle)
    # libCmethods._handle = None
    # libCmethods._name = None
    # libCmethods._FuncPtr = None
    # del libCmethods

"""

import ctypes
import logging
import numpy as np
import os


LIBRARY_PATH = os.path.dirname(__file__)

try:
    # load the library, using numpy mechanisms
    ctools = np.ctypeslib.load_library("_ctools", LIBRARY_PATH)
except OSError:
    _logger = logging.getLogger(__name__)
    _logger.error(
        "MPDAF's C extension is missing, probably it was not compiled because "
        "of missing dependencies.\n Try rebuilding MPDAF.")
    raise
else:
    # define argument types
    charptr = ctypes.POINTER(ctypes.c_char)
    array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                             flags='CONTIGUOUS')
    array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                                          flags='CONTIGUOUS')

    # mpdaf_merging_median
    ctools.mpdaf_merging_median.argtypes = [
        charptr, array_1d_double, array_1d_int, array_1d_int
    ]

    # mpdaf_merging_sigma_clipping
    ctools.mpdaf_merging_sigma_clipping.argtypes = [
        charptr, array_1d_double, array_1d_double, array_1d_int,
        array_1d_double, array_1d_int, array_1d_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
        ctypes.c_int
    ]

    # mpdaf_sky_ref
    ctools.mpdaf_sky_ref.restype = None
    ctools.mpdaf_sky_ref.argtypes = [
        array_1d_double, array_1d_double, array_1d_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
        array_1d_double
    ]

    # mpdaf_slice_median
    ctools.mpdaf_slice_median.restype = None
    ctools.mpdaf_slice_median.argtypes = [
        array_1d_double, array_1d_double, array_1d_double, array_1d_int,
        array_1d_int, array_1d_int, array_1d_double, array_1d_double,
        ctypes.c_int, array_1d_int, array_1d_double, array_1d_double,
        ctypes.c_int, array_1d_int, array_1d_int, ctypes.c_int
    ]

    # mpdaf_slice_median
    ctools.mpdaf_slice_median.restype = None
    ctools.mpdaf_slice_median.argtypes = [
        array_1d_double, array_1d_double, array_1d_double, array_1d_int,
        array_1d_int, array_1d_int, array_1d_double, array_1d_double,
        ctypes.c_int, array_1d_int, array_1d_double, array_1d_double,
        ctypes.c_int, array_1d_int, array_1d_int, ctypes.c_int
    ]
