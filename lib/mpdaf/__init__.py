# -*- coding: utf-8 -*-

__version__ = '1.1.14'
__date__ = '2015/01/21'

import tools
import obj
import drs
import sdetect
import MUSE
try:
    import fusion
except:
    pass
import logging
from logging.handlers import RotatingFileHandler

# cpu numbers
CPU = 0

# logging


def setup_logging(level, logfile):
    logger = logging.getLogger('mpdaf corelib')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] {mpdaf corelib %(class)s.%(method)s} %(message)s')
    file_handler = RotatingFileHandler(logfile, 'a', 1000000, 1)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(level)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    steam_handler.setFormatter(formatter)
    logger.addHandler(steam_handler)

setup_logging(logging.DEBUG, 'mpdaf.log')
