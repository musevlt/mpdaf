# -*- coding: utf-8 -*-

__version__ = '1.2b1'
__date__ = '2015/11/05'

import tools
import obj
import drs
import sdetect
import MUSE
import logging
from logging.handlers import RotatingFileHandler

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
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] {%(name)s:%(lineno)d} %(message)s')
    file_handler = RotatingFileHandler(logfile, 'a', 1000000, 1)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

setup_logging(logging.DEBUG)
