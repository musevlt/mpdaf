from __future__ import print_function

from __future__ import absolute_import
import functools
import inspect
import logging
import os
import warnings

from astropy.units import UnitsWarning
from contextlib import contextmanager
from functools import wraps
from time import time

__all__ = ('MpdafWarning', 'MpdafUnitsWarning', 'deprecated', 'chdir',
           'timeit', 'timer')


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
