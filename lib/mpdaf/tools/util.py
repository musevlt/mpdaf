"""
Copyright (c) 2010-2017 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2017 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Laure Piqueras <laure.piqueras@univ-lyon1.fr>

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
from __future__ import absolute_import, print_function

import functools
import inspect
import logging
import numpy as np
import os
import warnings

from astropy.units import UnitsWarning
from contextlib import contextmanager
from functools import wraps
from time import time

from .numpycompat import broadcast_to

__all__ = ('MpdafWarning', 'MpdafUnitsWarning', 'deprecated', 'chdir',
           'timeit', 'timer', 'broadcast_to_cube')


# NOTE(kgriffs): We don't want our deprecations to be ignored by default,
# so create our own type.
class MpdafWarning(UserWarning):
    pass


class MpdafUnitsWarning(UnitsWarning):
    pass


def deprecated(instructions):
    """Flags a method as deprecated.

    Args:
        instructions: A human-friendly string of instructions, such
            as: 'Please migrate to add_proxy() ASAP.'
    """
    def decorator(func):
        """This is a decorator which can be used to mark functions as
        deprecated.

        It will result in a warning being emitted when the function is
        used.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = 'Call to deprecated function `{}`. {}'.format(
                func.__name__,
                instructions)

            frame = inspect.currentframe().f_back
            warnings.warn_explicit(message,
                                   category=MpdafWarning,
                                   filename=inspect.getfile(frame.f_code),
                                   lineno=frame.f_lineno)
            return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def chdir(dirname):
    """Context manager to change the current working directory."""
    curdir = os.getcwd()
    try:
        os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)


def timeit(f):
    """Decorator which prints the execution time of a function."""
    @wraps(f)
    def timed(*args, **kw):
        logger = logging.getLogger(__name__)
        t0 = time()
        result = f(*args, **kw)
        logger.info('%r (%r, %r) %2.2f sec', f.__name__, args, kw, time() - t0)
        return result
    return timed


@contextmanager
def timer():
    """Context manager which prints the execution time."""
    logger = logging.getLogger(__name__)
    start = time()
    yield
    logger.info('Request took %.03f sec.', time() - start)


def broadcast_to_cube(arr, shape):
    """Broadcast an array (image or spectrum to a cube."""
    assert len(shape) == 3
    excmsg = 'Incorrect dimensions for the weights (%s) (it must be (%s))'
    if arr.ndim == 3 and arr.shape != shape:
        raise ValueError(excmsg % (arr.shape, shape))
    elif arr.ndim == 2 and arr.shape != shape[1:]:
        raise ValueError(excmsg % (arr.shape, shape[1:]))
    elif arr.ndim == 1:
        if arr.shape[0] != shape[0]:
            raise ValueError(excmsg % (arr.shape[0], shape[0]))
        arr = arr[:, np.newaxis, np.newaxis]

    return broadcast_to(arr, shape)
