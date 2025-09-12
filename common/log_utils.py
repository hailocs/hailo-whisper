from functools import partial, partialmethod
import logging
import os


def add_log_level(name, value):
    """
    Adds a custom log level

    Args:
        - name (str): new log level name
        - value (int): value for the log level
    """
    # Add a custom log level to all loggers
    setattr(logging, name.upper(), value)
    logging.addLevelName(value, name.upper())
    setattr(logging.Logger, name.lower(),
            partialmethod(logging.Logger.log, value))
    setattr(logging, name.lower(), partial(logging.log, value))


# Add a level for the titles
add_log_level('title', logging.CRITICAL + 5)


class LogFormatter(logging.Formatter):
    """Custom formatter for output to console"""
    # define color for log messages
    grey = "\x1b[38;0m"
    bold_yellow = "\x1b[33;1m"
    bold_red = "\x1b[31;1m"
    white_bold_fg_red_bg = "\x1b[37;41;1m"
    bold_green = "\x1b[32;1m"
    reset = "\x1b[0m"

    # define log format for console
    format = "[%(levelname)s] - %(message)s"
    format_title = "%(message)s"

    # custom formats
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: bold_yellow + format + reset,
        logging.ERROR: bold_red + format + reset,
        logging.CRITICAL: white_bold_fg_red_bg + format + reset,
        logging.TITLE: bold_green + format_title + reset
    }

    """Sets the custom formats"""
    def format(self, record):
        record.levelname = record.levelname.lower()
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def add_console_handler(level=logging.INFO):

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(LogFormatter())

    # add console handler to the logger
    logger.addHandler(console_handler)



def setup_logger(log_level="info"):
    """
    Configures the logger and adds the handlers

    Args:
        - level (int): the log level for console handler
    """
    numeric_log_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        handler_log_level = logging.INFO
    else:
        handler_log_level = numeric_log_level
    add_console_handler(handler_log_level)


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
setup_logger()
