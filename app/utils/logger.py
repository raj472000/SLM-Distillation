import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        logger.addHandler(ch)

    return logger