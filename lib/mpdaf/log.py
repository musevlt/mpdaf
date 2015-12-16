# -*- coding: utf-8 -*-

import logging
import os
import sys

# The background is set with 40 plus the number of the color, and the
# foreground with 30
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = [30 + i
                                                         for i in range(8)]
COLORS = {
    'DEBUG': BLUE,
    'INFO': GREEN,
    'WARNING': YELLOW,
    'ERROR': RED,
    'CRITICAL': MAGENTA,
}
# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;{0}m{1}{2}"
BOLD_SEQ = "\033[1m"


def colored(text, color):
    return COLOR_SEQ.format(color, text, RESET_SEQ)


def setup_logging(level=logging.DEBUG, color=False,
                  fmt='[%(levelname)s] %(message)s'):
    """Setup logging to stdout."""
    logger = logging.getLogger('mpdaf')
    logger.handlers = []
    logger.setLevel(level)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(level)
    if (color and os.isatty(sys.stdout.fileno()) and
            not sys.platform.startswith('win')):
        formatter = ColoredFormatter(fmt)
    else:
        formatter = logging.Formatter(fmt)
    steam_handler.setFormatter(formatter)
    logger.addHandler(steam_handler)


def setup_logfile(level=logging.DEBUG, logfile='mpdaf.log'):
    """Setup logging to file."""
    from logging.handlers import RotatingFileHandler
    logger = logging.getLogger('mpdaf')
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] {%(name)s:%(lineno)d} %(message)s')
    file_handler = RotatingFileHandler(logfile, 'a', 1000000, 1)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        level = record.levelname
        process = record.process
        record.levelname = colored(level, COLORS[level])
        record.process = colored(process, 30 + process % 8)
        s = logging.Formatter.format(self, record)
        record.levelname = level
        record.process = process
        return s
