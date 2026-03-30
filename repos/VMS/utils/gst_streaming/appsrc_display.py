# Send Numpy image to UDP sink
# gst-launch-1.0 udpsrc port="5000" caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)RAW, sampling=(string)BGR, width=(string)320, height=(string)240" ! rtpvrawdepay ! videoconvert ! queue ! xvimagesink sync=false
# gst-launch-1.0 -v videotestsrc ! video/x-raw, format="(string)BGR, width=(int)320, height=(int)240, framerate=(fraction)30/1" ! rtpvrawpay ! udpsink host="127.0.0.1" port="5000"
# gst-launch-1.0 rtspsrc location=rtsp://170.93.143.139:1935/rtplive/0b01b57900060075004d823633235daa protocols=4 ! decodebin ! videoconvert ! capsfilter ! video/x-raw, format="(string)BGR, width=(int)320, height=(int)240" ! rtpvrawpay ! udpsink host="224.1.1.1" port="5000" auto-multicast=true
from logging import exception
import gi
import sys

if 'gi' in sys.modules:
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib, GObject
from abc import ABC
from utils.utility.utils_logger import logger as log
# from ai_core.utility.ai_logger import aiLogger as log

from .base_gst_pipeline import BaseGstPipeline
from utils.enums.stream_format import StreamFormat
from utils.enums.stream_status import StreamStatus
from utils.enums.pipeline_status import PipelineStatus
from .frame_queue import FrameBufferQueue
from utils.enums.stream_info import StreamInfo
from ai_functions.utility.config import DisplayProcessedFrameConfig
from ai_functions.utility.config import AiProcessorConfig
import cv2
import time

from utils.utility import utils_image
from utils.utility.utils_image import colors
import random

CAPS = "video/x-raw, width=(int){WIDTH}, height=(int){HEIGHT}, framerate=(fraction){FPS}/10, format=(string)BGR"

classes = {'person':0,'car':1,'bird':2,'truck':3,'bench':4,'motorbike':5,'bicycle':6,'bus':7,'dog':8,'cat':9,'face':10,'plate':11, 'clothes':12, 'tree':13 , 'excavator':14, 'dump_truck':15, 'concrete_mixer_truck':16, 'van':17, 'container_truck':18, 'bulldozer':19, 'vehicle':20, 'boat':21, 'roller':22}
labels = {0:'person',1:'car',2:'bird',3:'truck',4:'bench',5:'motorbike',6:'bicycle',7:'bus',8:'dog',9:'cat',10:'face',11:'plate', 12:'clothes', 13:'tree' , 14:'excavator', 15:'dump_truck', 16:'concrete_mixer_truck', 17:'van', 18:'container_truck', 19:'bulldozer', 20:'vehicle', 21:'boat', 22:'roller'}

class AppSrcDisplay(BaseGstPipeline, ABC):
    def __init__(self, 
                 stream_info: StreamInfo,
                 codec_encode: StreamFormat = StreamFormat.RAW_RGB, 
                 fps: int = 15, 
                 pipeline_cmd = None,
                 buffer = None):
        config = DisplayProcessedFrameConfig()
        self.codec = codec_encode                   # Contain format of stream video
        self.display_buffer = None

        self.stream_info = stream_info              # Information of stream that consists of name, pass,... 
        self.pipeline_cmd = self._build_pipeline_cmd(pipeline_cmd)  #-------------------
        self.message = None                                         #-------------------
        super(AppSrcDisplay, self).__init__(self.pipeline_cmd, self._on_update_status) #-------------------
        self.fps = fps                                              #-------------------
        self.duration = 1 / self.fps * Gst.SECOND                   #-------------------
        self.pts = 0                                                #-------------------
        self.latest_frame = None                                    #-------------------
        self.factor = 2
        self.stream_info.width = 640*self.factor
        self.stream_info.height = 360*self.factor

        camConfig = AiProcessorConfig()
        self.fps = camConfig.processRate
        self.vmsDisplay = not camConfig.using_internal_display

        self.info_out = {}
        self.yolo_info = []
        self.face_info = []
        self.fire_info = []

    def _build_pipeline_cmd(self, cmd):
        # if self.codec == StreamFormat.RAW_RGB:
        # RAW format
        if cmd is not None:
            return cmd
        pipeline_cmd = '''  appsrc emit-signals=True is-live=True name=appsrc format=GST_FORMAT_TIME !  
                            queue max-size-buffers=4 !                       
                            autovideoconvert !
                            ximagesink 
                            '''
        return pipeline_cmd.replace('\n', ' ')

    def _on_pipeline_init(self) -> None:
        self.appsrc = self._pipeline.get_by_name('appsrc')
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("block", True)
        appsrc_caps = CAPS.format(
            WIDTH=self.stream_info.width,
            HEIGHT=self.stream_info.height,
            FPS=self.fps
        )
        self.appsrc.set_caps(Gst.Caps.from_string(appsrc_caps))
        assert self.appsrc
        self.appsrc.connect('need-data', self._need_data, None)

    def _need_data(self, appsrc, data, dd):
        frame = None
        # print('need-data', self.display_buffer.qsize())
        try:
            # if self.buffer_queue.qsize() > 0:
            # print(self.buffer_queue.qsize())
            self.info_out = self.display_buffer.get()

            # print(self.latest_frame)
            # print(self.latest_frame['frame'], self.latest_frame['camera_id'])
            # print('info_out',self.info_out)
            # self.yolo_info = self.info_out['yolo5']['bounding_box']
            # [('plate', 0.705078125, (0.19270833333333334, 0.3074074074074074, 0.2109375, 0.37592592592592594), 'ABCD78410'), \
            # ('motorbike', 0.6826171875, (0.19375, 0.3212962962962963, 0.2140625, 0.38333333333333336), '')]
            # print('yolo_info', self.yolo_info['plate'])

            # self.statistic_info = self.info_out['yolo5']['statistic']
            # {'object_counter': array([[[8.000, 45.000],  # out in 2 banh'
            #                         [5.000, 12.000],      # out in 4 banh'
            #                         [3.000, 45.000]]])}   # out in person
            # print('statistic_info', self.statistic_info)

            # self.face_info = self.info_out['retinaface']
        # print('face_info', self.face_info)

            # self.fire_info = self.info_out['fire_classification']

            self.meta = self.info_out['meta']
            # print('eeeeeeeeeeeeeeeee')
            for m in self.meta:
                log.debug("Output data {}".format(m))
            #     if m['feature_id'] == 'face':
            #         print('3',m)

            self.latest_frame = self.info_out['frame']

            self.frame_time = self.info_out['ts']

            self.uuid = self.info_out['frame_id']

            # print('meta', self.meta)

            # [{'feature_id': 'statistic', 'meta': {'start_time': 1649247777.4361053,
            #                                      'end_time': 1649247807.4773195, 
            #                                      'object_counter': array([[[1.000, 0.000],[0.000, 0.000],[0.000, 0.000]]]),  # out in  of 3 class clusters (2 banh - 4 banh - person)
            #                                     'velocity_histogram': [{58: 1, 37: 1, 2: 1}, {14: 1, 16: 1, 2: 1}, {57: 1, 33: 1, 0: 1}], #histogram of velocity V, Vx,Vy pixels/sec
            #                                     'heatmap': np.array(16, 16, 3), 
            #                                     'heatmap_direction': np.array(16, 16, 12, 3), 
            #                                     'frame_size': (720, 1280, 3)}}]

            # [{'feature_id': 'face', 'meta': [{'appeared_at': datetime.datetime(2022, 4, 2, 18, 48, 27, 655682), 'image_paths': './data/face_camera0/face/1648900107.6557055.jpg', 'full_image_paths': './data/face_camera0/full/1648900107.6552663.jpg', 'face_id': 'Duc', 'is_vip': None, 'age': None, 'gender': -1, 'appearance_metadata': {'wear_glass': None, 'face_mask': 'NoMaskFront', 'face_expression': None, 'hat_meta': {'hat_type': None, 'hat_color': None}}, 'conf': 0.74}]}]
            # [{'feature_id': 'fire', 'meta': [{'appeared_at': datetime.datetime(2022, 4, 2, 18, 51, 2, 122842), 'image_paths': [], 'full_image_paths': './data/face_camera0/fire/1648900262.1228466.jpg'}]}]
            # [{'feature_id': 'crowd', 'meta': [{'appeared_at': datetime.datetime(2022, 4, 4, 19, 52, 39, 605828), 'image_paths': [], 'full_image_paths': ['./data/cam21/crowd/1649076759.6060867.jpg'], 'appearance_metadata': {'count': 8, 'zone': [(0, 0.7), (0.3, 0), (1, 0.7), (0, 1), (0.3, 1)]}}, ]}]
        except BaseException as ex:
            log.exception('fsd')

        try:
            if self.latest_frame is not None:

                if (self.vmsDisplay):
                    # print('1')
                    self.draw_object_detection(self.latest_frame, self.yolo_info)
                    # print('2')
                    self.draw_face(self.latest_frame, self.face_info)
                    # print('3')
                    self.draw_fire(self.latest_frame, self.fire_info)
                # print('4')
                cv2.putText(self.latest_frame, self.stream_info.stream_id + " " + str(self.latest_frame.shape), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 3)
                # print('5')
                self.latest_frame = cv2.resize(self.latest_frame,(self.stream_info.width,self.stream_info.height))
                # print('6')
                # cv2.imwrite('./test/' + str(time.time()) + '.png',self.latest_frame)

                delivered_buffer = Gst.Buffer.new_wrapped(bytes(self.latest_frame))
                # print('7')
                # set pts and duration to be able to record video, calculate fps
                # self.pts += self.duration
                # delivered_buffer.pts = self.pts
                # delivered_buffer.duration = self.duration
                appsrc.emit("push-buffer", delivered_buffer)
        except BaseException as ex:
            log.exception(ex)

    def draw_fire(self,image, results):
        for i in range(len(results)):
            for dict_e in results[i]:
                if dict_e is None:
                    continue
                cv2.putText(image, dict_e['text'], dict_e['position'], dict_e['font'], 
                    dict_e['fontScale'], dict_e['color'], dict_e['thickness'], cv2.LINE_AA)

    def draw_face(self, image, results):
        # print(len(image), len(results), results)
        # for i in range(len(results)):
            # print('draw_face---------------- ',i, results, len(results), image.shape)
        for predict in results:
            if predict is None:
                continue
            text = f'{predict.name} - {round(predict.score, 2)} - {predict.state_of_the_face}'
            det = list(map(int, predict.xlylxryr[:]))
            # print('textttttttttttttttttttttt',text, det)
            cv2.rectangle(image, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 2)
            cx = det[0]
            cy = det[1] - 12

            # if ('NoMask' in predict.state_of_the_face):
            #     cv2.putText(image, text, (cx, cy),
            #                 cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))
            # else:
            cv2.putText(image, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

        return image

    def draw_object_detection(self, img, results):
        ih, iw, _ = img.shape
        for r in results:
            if (r is not None):
                label, config, c1, c2, text = self.extract_box(r, iw, ih)
                color=colors(classes[label], True)
                cv2.rectangle(img, c1, c2, color, 1)
                if (label in ['dog', 'cat', 'bird', 'bench', 'clothes']):
                    continue
                # print(label, c1, c2)
                if (label == 'plate'):
                    # print('texxxtttt', text)
                    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                    cv2.rectangle(img, c1, c2, color, -1)
                    cv2.putText(img, text, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
                else:
                    l_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    c2 = c1[0] + l_size[0] + 3, c1[1] + l_size[1] + 4
                    cv2.rectangle(img, c1, c2, color, -1)
                    cv2.putText(img, label, (c1[0], c1[1] + l_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

        # print("==================================")
    def extract_box(self,detection_result, iw, ih):
        # print('detection_result',detection_result)
        label, config, (x1, y1, x2, y2),text = detection_result
        x1 = int(x1 * iw)
        y1 = int(y1 * ih)
        x2 = int(x2 * iw)
        y2 = int(y2 * ih)
        return labels[label], config, (x1, y1), (x2, y2), text

    def _on_update_status(self, pipeline_status: PipelineStatus, message: str):
        self.message = message
        log.info(message)

    def on_error(self, bus: Gst.Bus, message: Gst.Message):
        err, debug = message.parse_error()
        self.log.error("Gstreamer.%s: Error %s: %s. ", self, err, debug)
        self._on_update_status(StreamStatus.ERROR, err)

    def on_eos(self, bus: Gst.Bus, message: Gst.Message):
        self.log.debug("Gstreamer.%s: Received stream EOS event", self)
        self._on_update_status(StreamStatus.EOS, "Got EOS Message")

    def on_warning(self, bus: Gst.Bus, message: Gst.Message):
        warn, debug = message.parse_warning()
        self.log.warning("Gstreamer.%s: %s. %s", self, warn, debug)
        self._on_update_status(StreamStatus.WARNING, warn)
