# -*- coding: utf-8 -*-
"""
Copyright (c) 2010-2018 CNRS / Centre de Recherche Astrophysique de Lyon
Copyright (c) 2015-2018 Simon Conseil <simon.conseil@univ-lyon1.fr>
Copyright (c)      2016 Roland Bacon <roland.bacon@univ-lyon1.fr>
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


def setup_logging(name='mpdaf', level='DEBUG', color=False, stream=None,
                  fmt='[%(levelname)s] %(message)s', clear_handlers=True):
    """Setup stream handler for a given logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : str
        Level for the stream handler (default: DEBUG).
    color : bool
        Use colored output (default: True). Used only if a compatible
        stream or tty is used.
    stream : {sys.stdout, sys.stderr}
        Used stream (default: stderr)
    fmt : str
        Messages format.
    clear_handlers : bool
        Tell if other handlers must be removed (default: True).

    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if clear_handlers:
        logger.handlers.clear()

    steam_handler = logging.StreamHandler(stream)
    steam_handler.setLevel(level)
    formatter = logging.Formatter(fmt)
    if color:
        # Jupyter
        try:
            import ipykernel
            # check if iostream attribute is available
            ipykernel.iostream  # noqa
        except (ImportError, AttributeError):
            pass
        else:
            if isinstance(sys.stdout, ipykernel.iostream.OutStream):
                formatter = ColoredFormatter(fmt)

        # Try to detect if stdout is a tty
        try:
            if (os.isatty(sys.stdout.fileno()) and
                    not sys.platform.startswith('win')):
                formatter = ColoredFormatter(fmt)
        except IOError:
            pass
    steam_handler.setFormatter(formatter)
    logger.addHandler(steam_handler)


def setup_logfile(name='mpdaf', level=logging.DEBUG, logfile='mpdaf.log',
                  fmt='%(asctime)s [%(levelname)s] {%(name)s:%(lineno)d} '
                      '%(message)s', rotating=True):
    """Setup logging to file."""

    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt)
    if rotating:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(logfile, 'a', 1000000, 1)
    else:
        file_handler = logging.FileHandler(logfile, 'a')

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def clear_loggers(name='mpdaf'):
    """Remove all handlers for a given logger."""
    logger = logging.getLogger(name)
    logger.handlers = []


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
