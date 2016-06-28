# -*- coding: utf-8 -*-

from __future__ import absolute_import
from . import drs, MUSE, obj, sdetect, tools
from .log import setup_logging, setup_logfile, clear_loggers
from .version import __version__, __date__

"""The maximum number of processes that should be started by
multiprocessing MPDAF functions. By default this is zero, which
requests that MPDAF multiprocessing functions use one process per
physical CPU core."""
CPU = 0

setup_logging()
