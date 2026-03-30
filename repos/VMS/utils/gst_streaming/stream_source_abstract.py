from abc import abstractmethod
from utils.enums.stream_info import StreamInfo
from utils.enums.stream_status import StreamStatus

import time


class BaseStreamSource:
    def __init__(self, stream_info):
        self.stream_info = StreamInfo(stream_info)
        self.stream_status = StreamStatus.INIT
        self.stream_message = None
        self.connection_retry = 0

    def _on_update_status(self, status: StreamStatus, message: str):
        self.stream_status = status
        self.stream_message = message
        self.latest_update_status = int(time.time())

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass
