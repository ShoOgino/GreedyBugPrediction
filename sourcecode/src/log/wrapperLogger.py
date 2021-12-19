from logging import getLogger, FileHandler, DEBUG, Formatter, log
from src.config.config import config
import logging
import os
class wrapperLogger:
    def setup_logger(name, logfile):
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        open(logfile, 'a').close()
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # create file handler which logs even DEBUG messages
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter(
            fmt = "%(asctime)s %(name)s\t:%(lineno)s\t%(funcName)s\t[%(levelname)s]:\t%(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(fh_formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.propagate = False
        return logger