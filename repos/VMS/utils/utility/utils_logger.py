import logging.config
import os


def init_logger():
    logging.getLogger('cassandra').setLevel(logging.DEBUG)
    logging.config.fileConfig('./utils/utility/logging.conf',disable_existing_loggers=False)
    # This allow changing log level in production via docker environment
    logging.getLogger().setLevel(get_logging_level())
    return logging.getLogger()


def get_logging_level() -> str:
    return os.environ.get('VMS_LOG_LEVEL', 'DEBUG').upper()


logger = init_logger()
