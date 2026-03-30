import logging

from functools import partial, wraps
from flask import request


class _DeclaredPathFilter(logging.Filter):
    def __init__(self, path, name=''):
        self.path = path
        super().__init__(name)

    def filter(self, record):
        return f"{self.path} " not in record.getMessage()


def disable_request_logging(func=None, *args, **kwargs):
    """Place this annotation on a flask's method will make the http request to it won't be logged"""

    if not func:
        return partial(disable_request_logging, *args, **kwargs)

    already_filtered = set()

    @wraps(func)
    def wrapper(*args, **kwargs):
        path = request.environ['PATH_INFO']
        if path not in already_filtered:
            already_filtered.add(path)
            logging.getLogger('werkzeug').addFilter(_DeclaredPathFilter(path))
            logging.getLogger("gunicorn.access").addFilter(_DeclaredPathFilter(path))
            logging.getLogger("geventwebsocket.handler").addFilter(_DeclaredPathFilter(path))

        return func(*args, **kwargs)
    return wrapper
