***********************
Logging (``mpdaf.log``)
***********************

When imported, MPDAF initialize a logger by default. This logger uses the
`logging` module, and log messages to stderr, for instance for the ``.info()``
methods.

It is possible to remove this logger with `~mpdaf.log.clear_loggers`, and to
use `~mpdaf.log.setup_logging` to setup a logger with a different level or
format. `~mpdaf.log.setup_logfile` can also be used to setup a logger with
a file.

Functions
=========

.. autofunction:: mpdaf.log.clear_loggers

.. autofunction:: mpdaf.log.setup_logging

.. autofunction:: mpdaf.log.setup_logfile
