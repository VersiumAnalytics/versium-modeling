import logging
import os
import sys
from typing import Optional

import numpy as np

LOG_FORMATS = {'CRITICAL': '%(asctime)s - %(message)s',
                   'ERROR': '%(asctime)s - %(message)s',
                   'WARNING': '%(asctime)s - %(message)s',
                   'INFO': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   'DEBUG': '[%(asctime)s - %(name)30s - %(filename)15s:%(lineno)5s - %(funcName)20s() - %(levelname)8s] - %(message)s',
                   'NOTSET': '%(asctime)s - %(name)s - %(levelname)s - %(message)s', }


class MultiFilter(logging.Filter):
    """Filter LogRecords.

    Allows multiple criteria for allowing events through the filter.

    For each criterion, the filter will admit records which are below a certain point in the logger hierarchy. A record's name must start with
    the criterion string to be admitted.
    """

    def __init__(self, name: Optional[str | list[str]] = None):
        """
        Initialize a filter.

        Initialize with the name of the logger which, together with its
        children, will have its events allowed through the filter. If no
        name is specified, allow every event.
        """
        if name is None:
            name = ""

        if isinstance(name, str):
            name = [name]

        self.name = name

    def filter(self, record):
        return any(filter(lambda x: record.name.startswith(x), self.name))


def setup_logging(*, logger=None, level='INFO', log_file=None, packages=['pyversium'], file_mode='a+'):
    logger = logging.getLogger() if logger is None else logger

    if log_file is None:
        log_file = []
    elif isinstance(log_file, str):
        log_file = [log_file]

    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper()))

    # Set warnings to never be printed if we are already logging them.
    if logger.level >= 20:
        import warnings

        warnings.filterwarnings("ignore")

    # Assign the exception hook to catch exceptions.
    sys.excepthook = excepthook

    # Configure formatting and filters
    filt = MultiFilter(packages)
    formatter = logging.Formatter(LOG_FORMATS[level])

    # Set up logging to console.
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.addFilter(filt)

    logger.addHandler(sh)

    fh = None
    for lf in log_file:
        if os.path.dirname(lf):
            os.makedirs(os.path.dirname(lf), exist_ok=True)
        fh = logging.FileHandler(lf, mode=file_mode)
        fh.setFormatter(formatter)
        fh.addFilter(filt)

        logger.addHandler(fh)

    # ignore invalid value runtime warnings.
    np.seterr(invalid='ignore')

    this_logger = logging.getLogger(__name__)
    msg = "Logger setup complete. Logging to stderr"
    msg = msg + ("." if fh is None else " and {}".format(", ".join(log_file)))
    this_logger.info(msg)


def excepthook(type, value, traceback):
    """Replaces sys.excepthook to log uncaught exceptions.

    See Also
    --------
    sys.excepthook

    """

    # Don't log KeyboardInterrupt.
    if issubclass(type, KeyboardInterrupt):
        sys.__excepthook__(type, value, traceback)
        return

    # Get the current logger.
    logger = logging.getLogger("pyversium")

    # An uncaught exception is critical.
    logger.critical("Uncaught exception", exc_info=(type, value, traceback))

    return



class Tee:
    """Tee catches anything written to sys.stdout and copies to given logger with given level.

    Parameters
    ----------
    logger: logging.logger
        The logger object to use to.

    level: str
        The level of logging to be used.
    """
    levels = {'CRITICAL': 50,
              'ERROR':    40,
              'WARNING':  30,
              'INFO':     20,
              'DEBUG':    10,
              'NOTSET':   0, }

    def __init__(self, logger, keep_stdout=True, level="INFO"):

        # Get the sys.stdout and copy over.
        self.stdout = sys.stdout

        # Set the logger and its level.
        self.logger = logger
        self.level = self.levels[level]
        self.keep_stdout = keep_stdout

        # Become sys.stdout, so that we can catch anything written to sys.stdout.
        sys.stdout = self

        return

    def __del__(self):
        self.close()
        return

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return

    def write(self, message):
        """Write method for sys.stdout."""

        if self.keep_stdout:
            # Output as usual.
            self.stdout.write(message)

        # If it is all whitespace, do not log it.
        if message.strip():
            self.logger.log(self.level, message)
        return

    def flush(self):
        """Flush to the sys.stdout."""
        self.stdout.flush()
        return

    def close(self):
        """Flush and close method for assigning sys.stdout back."""
        if self.stdout is not None:
            self.flush()
            sys.stdout = self.stdout
            self.stdout = None

        return
