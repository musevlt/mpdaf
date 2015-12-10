# -*- coding: utf-8 -*-

import logging

from . import drs, MUSE, obj, sdetect, tools
from .version import __version__, __date__

# cpu numbers
CPU = 0


def setup_logging(level):
    """Setup logging to stdout."""
    logger = logging.getLogger(__name__)
    logger.handlers = []
    logger.setLevel(level)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(level)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    steam_handler.setFormatter(formatter)
    logger.addHandler(steam_handler)


def setup_logfile(level, logfile):
    """Setup logging to file."""
    from logging.handlers import RotatingFileHandler
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] {%(name)s:%(lineno)d} %(message)s')
    file_handler = RotatingFileHandler(logfile, 'a', 1000000, 1)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

setup_logging(logging.DEBUG)
