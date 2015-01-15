__version__ = '1.1.13'
__date__ = '2014/12/17'

import tools
import obj
import drs
import MUSE
try:
    import fusion
except:
    pass
import logging
from logging.handlers import RotatingFileHandler

# FORMAT = "WARNING mpdaf corelib %(class)s.%(method)s: %(message)s"
# logging.basicConfig(format=FORMAT)
# logger = logging.getLogger('mpdaf corelib')

CPU = 0

# 
logger = logging.getLogger('mpdaf corelib')
logger.setLevel(logging.DEBUG)
 
formatter = logging.Formatter('%(asctime)s [%(levelname)s] {mpdaf corelib %(class)s.%(method)s} %(message)s')
file_handler = RotatingFileHandler('mpdaf.log', 'a', 1000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
 
steam_handler = logging.StreamHandler()
steam_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] %(message)s')
steam_handler.setFormatter(formatter)
logger.addHandler(steam_handler)