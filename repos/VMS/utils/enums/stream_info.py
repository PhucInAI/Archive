import uuid
from utils.enums.stream_format import StreamFormat, StreamType

class StreamInfo:
    def __init__(self, stream_info):
        # inpurt type: json
        # parse type of stream
        if 'stream_id' in stream_info.keys():
            self.stream_id = stream_info['stream_id']
        else:
            self.stream_id = str(uuid.uuid1())
        assert stream_info['connection_uri']
        self.uri = stream_info['connection_uri']
        assert stream_info['name']
        self.stream_name = stream_info['name']
        self.type = StreamType.NA
        # self.height = stream_info["height"]
        # self.width = stream_info["width"]
        if self.uri.startswith('rtsp'):
            self.type = StreamType.RTSP
            # assert stream_info['username']
            # self.username = stream_info['username']
            # assert stream_info['password']
            # self.password = stream_info['password']

            self.fmt = StreamFormat.DEFAULT
            # assert stream_info['fmt']
            # if stream_info['fmt'] == StreamFormat.H264:
            #     self.fmt = StreamFormat.H264
            # elif stream_info['fmt'] == StreamFormat.H265:
            #     self.fmt = StreamFormat.H265
            # else:
            #     self.fmt = StreamFormat.DEFAULT
        elif self.uri.startswith('/dev/video'):
            self.type = StreamType.WEBCAM
