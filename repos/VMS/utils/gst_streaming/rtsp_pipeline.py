import gi
import sys
from utils.enums.stream_format import StreamFormat, StreamType

if 'gi' in sys.modules:
    gi.require_version('Gst', '1.0')
    gi.require_version('GstBase', '1.0')
    gi.require_version('GstVideo', '1.0')
    gi.require_version('GstApp', '1.0')
    from gi.repository import Gst, GLib, GObject, GstApp, GstVideo  # noqa:F401,F402
from .base_gst_pipeline import BaseGstPipeline
from utils.enums.stream_info import StreamInfo
from utils.enums.stream_status import StreamStatus
RTSP_PIPELINE1 = "rtspsrc location={location} user-id={username} user-pw={password} protocols=4 ! rtpjitterbuffer ! " \
                 "rtph264depay ! nvdec ! gldownload ! videoconvert ! video/x-raw, format=BGR !   appsink name=appsink sync=false emit-signals=true max-buffers=1 drop=true"

RTSP_PIPELINE = "rtspsrc location={location} user-id={username} user-pw={password} protocols=4 ! rtpjitterbuffer ! " \
                "rtph264depay ! h264parse ! avdec_h264 !  videoconvert ! video/x-raw, format=BGR !   appsink name=appsink sync=false emit-signals=true max-buffers=3 drop=false"

RTSP_PIPELINE3 = "rtspsrc location={location} user-id={username} user-pw={password} protocols=4 ! rtpjitterbuffer ! " \
                "rtph264depay ! h264parse ! avdec_h264 ! videoscale !  videoconvert !  video/x-raw,width=640,height=480,format=BGR  ! appsink name=appsink sync=false emit-signals=true max-buffers=10 drop=true"

RTSP_PIPELINE4 = "rtspsrc location={location} user-id={username} user-pw={password} protocols=4 ! rtpjitterbuffer ! " \
                "rtph264depay ! h264parse ! avdec_h264 ! videoscale ! videoconvert ! video/x-raw,format=BGR  ! appsink name=appsink sync=false emit-signals=true max-buffers=5 drop=true"

RTSP_PIPELINE5 = "rtspsrc location={location} user-id={username} user-pw={password} protocols=4 ! rtpjitterbuffer ! " \
                "rtph265depay ! h265parse ! avdec_h265 ! videoscale ! videoconvert ! video/x-raw,format=BGR ! appsink name=appsink sync=false emit-signals=true max-buffers=5 drop=true"


F_RTSP_PIPELINE = "filesrc location=../../data/input/Nova_cam/Nova_Street1_02.mp4 ! qtdemux name=demux ! " \
                " avdec_h264 ! videoconvert ! video/x-raw,format=BGR !  appsink name=appsink sync=true emit-signals=true max-buffers=5 drop=false"

F_QA_RTSP_PIPELINE = "filesrc location=./di_cham_nhin_thang_5attempts_1passed_4failed.mp4 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink name=appsink sync=true emit-signals=true max-buffers=5 drop=false"

F_RTSP_PIPELINE2 = "filesrc location=../../data/input/Di_nguoc_chieu/118.h264 ! matroskademux ! avdec_h265 ! videoconvert ! video/x-raw,format=BGR ! appsink name=appsink sync=true emit-signals=true max-buffers=5 drop=false "

class RTSPPipelineSource(BaseGstPipeline):
    def __init__(self, stream_info: StreamInfo, on_new_buffer, on_update_status):
        self._stream_info = stream_info
        self._pipeline_cmd = self._build_pipeline_cmd()
        # self.log.debug("Gstreamer %s:" , self._pipeline_cmd)
        print("aaaaaaaaaaaaaaaaaaaaa",self._pipeline_cmd)
        super(RTSPPipelineSource, self).__init__(self._pipeline_cmd, on_update_status)
        self.on_new_buffer = on_new_buffer
        self.appsink = None

    def center_crop(img, dim):
        """Returns center cropped image
        Args:
        img: image to be center cropped
        dim: dimensions (width, height) to be cropped
        """
        width, height = img.shape[1], img.shape[0]

        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img

    def _build_pipeline_cmd(self):
        # print(self._stream_info.fmt)
        if (self._stream_info.fmt == StreamFormat.H264):
            pipeline_cmd = RTSP_PIPELINE4.format(#change PIPELINE3 to F_RTSP_PIPELINE for video testing, also update video path in location=
                location=self._stream_info.uri,
                username=self._stream_info.username,
                password=self._stream_info.password
            )
        elif (self._stream_info.fmt == StreamFormat.H265):
            pipeline_cmd = RTSP_PIPELINE5.format(#change PIPELINE3 to F_RTSP_PIPELINE for video testing, also update video path in location=
                location=self._stream_info.uri,
                # username=self._stream_info.username,
                # password=self._stream_info.password
                username='viact',
                password='W%23znXWL6Dz'
            )

        return pipeline_cmd.replace('\n', '')

    def _on_pipeline_init(self) -> None:
        # init data
        self.appsink = self._pipeline.get_by_name('appsink')
        assert self.appsink
        self.appsink.connect('new-sample', self._on_new_sample, None)

    def _on_new_sample(self, sink, data) -> Gst.FlowReturn.OK:
        sample = sink.emit('pull-sample')
        self.on_new_buffer(sample)
        return Gst.FlowReturn.OK

    def on_error(self, bus: Gst.Bus, message: Gst.Message):
        err, debug = message.parse_error()
        self.log.error("Gstreamer.%s: Error %s: %s. ", self, err, debug)
        self._on_update_status(StreamStatus.ERROR, err)
        self.shutdown()
        self._on_update_status(StreamStatus.SHUTDOWN, "Shutdown because Got ERROR Message")

    def on_eos(self, bus: Gst.Bus, message: Gst.Message):
        self.log.debug("Gstreamer.%s: Received stream EOS event", self)
        self._on_update_status(StreamStatus.EOS, "Got EOS Message")
        self.shutdown()
        self._on_update_status(StreamStatus.SHUTDOWN, "Shutdown because Got EOS Message")

    def on_warning(self, bus: Gst.Bus, message: Gst.Message):
        warn, debug = message.parse_warning()
        self.log.warning("Gstreamer.%s: %s. %s", self, warn, debug)
        self._on_update_status(StreamStatus.WARNING, warn)

    def update_stream_info(self, stream_info):
        self._stream_info = stream_info
        self._pipeline_cmd = self._build_pipeline_cmd()
        self.update_pipeline_cmd(self._pipeline_cmd)
