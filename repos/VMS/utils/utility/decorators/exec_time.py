import logging
import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import NoReturn, Callable

from dynaconf import settings
from flask import request

from utility.utils import Utils

thread_local_storage = threading.local()


def measure_exec_time(func):
    @wraps(func)
    def wrapper_measure_exec_time(*args, **kwargs):
        if isinstance(func, classmethod):
            raise RuntimeError('Place @measure_exec_time UNDER @classmethod')
        callee = func.__qualname__
        request_uri = ''
        if request and request.path.startswith('/api/'):
            request_uri = f'{request.method} {request.path}'

        if request:
            if getattr(thread_local_storage, 'stopwatch_latch_counter', None) is None:
                thread_local_storage.stopwatch_latch_counter = 0
                thread_local_storage.stopwatch_measurements = OrderedDict()
            thread_local_storage.stopwatch_latch_counter = thread_local_storage.stopwatch_latch_counter + 1

        start = time.time()
        ret = func(*args, **kwargs)
        stop = time.time()

        # accumulation measurement is only supported in flask context for now.
        if not request:
            log_exec_time(stop - start, lambda duration: f'Exec time : {duration} {request_uri} func: {callee}')
        else:
            thread_local_storage.stopwatch_latch_counter = thread_local_storage.stopwatch_latch_counter - 1
            if thread_local_storage.stopwatch_measurements.get(callee) is None:
                thread_local_storage.stopwatch_measurements[callee] = {
                    'exec_time': stop - start,
                    'invocations': 1
                }
            else:
                measurement = thread_local_storage.stopwatch_measurements[callee]
                measurement['exec_time'] = measurement['exec_time'] + (stop - start)
                measurement['invocations'] = measurement['invocations'] + 1

            if thread_local_storage.stopwatch_latch_counter == 0:
                # dump measurement
                for callee_name in thread_local_storage.stopwatch_measurements.keys():
                    measurement = thread_local_storage.stopwatch_measurements[callee_name]

                    log_exec_time(measurement["exec_time"],
                                  lambda duration: f'Exec time : {duration} '
                                                   f'({measurement["invocations"]} times) '
                                                   f'{request_uri}'
                                                   f' func: {callee_name}')

        return ret

    return wrapper_measure_exec_time


def log_exec_time(duration: float, log_message_callback: Callable[[str], str]) -> NoReturn:
    log_all_exec_time = Utils.to_boolean(settings.get('LOG_ALL_EXEC_TIME', 'false'))

    if log_all_exec_time or duration > 2.0:  # seconds
        logging.getLogger('stopwatch').info(log_message_callback(f'{"{:8.3f}".format(duration)}s'))
