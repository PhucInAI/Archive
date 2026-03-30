import threading
import time

from threading import Event
from utility.utils_logger import logger as log
# from ai_core.utility.ai_logger import aiLogger as log


def handle_exception(ex):
    log.exception(ex)


def finish_execution():
    pass


class InfinityThread(threading.Thread):
    def __init__(self, target, sleep_seconds=0.01, name=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        super(InfinityThread, self).__init__(target=target, name=name, args=args, kwargs=kwargs, daemon=True)
        self.sleep_seconds = sleep_seconds
        self.event = Event()
        self.is_stop = False
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def run(self):
        while not self.is_stop:
            try:
                if self.target:
                    self.target(*self.args, **self.kwargs)
            except BaseException as ex:
                handle_exception(ex)
            finally:
                finish_execution()

            if self.sleep_seconds > 0:
                self.event.clear()
                self.event.wait(timeout=self.sleep_seconds)
            else:
                time.sleep(0.001)

    def stop(self):
        self.is_stop = True
        if self.sleep_seconds > 0:
            self.event.set()
