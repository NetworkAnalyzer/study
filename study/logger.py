import logging

logger = logging.getLogger()


def set_level(level):
    logger.setLevel(getattr(logging, level.upper()))

