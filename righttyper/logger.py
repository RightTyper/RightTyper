import logging

logger = logging.getLogger(__name__.split('.')[0])
logger.setLevel(logging.INFO)
logger.propagate = False    # excludes other packages' messages

_handler = logging.FileHandler(logger.name + '.log')
_handler.setFormatter(logging.Formatter("[%(filename)s:%(lineno)s] %(message)s"))

logger.addHandler(_handler)
