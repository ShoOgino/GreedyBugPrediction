from logging import getLogger, FileHandler, DEBUG, Formatter
from src.config.config import config
import logging
import os
class wrapperLogger:
    def setup_logger(name, logfile):
        os.makedirs(os.path.dirname(config.pathLog), exist_ok=True)
        open(config.pathLog, 'a').close()
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # create file handler which logs even DEBUG messages
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
        fh.setFormatter(fh_formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.propagate = False
        return logger