# import threading
import time
from typing import Counter
import gi
import sys
from numpy.core.numeric import count_nonzero
import cv2
if 'gi' in sys.modules:
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
import os 

from abc import ABC
from .stream_source_abstract import BaseStreamSource
from utils.enums.stream_format import StreamType
from .rtsp_pipeline import RTSPPipelineSource
from .frame_queue import FrameBufferQueue
from utils.enums.stream_status import StreamStatus
from utils.enums.stream_info import StreamInfo
from utils.utility.utils_logger import logger as log
# from ai_core.utility.ai_logger import aiLogger as log

import numpy as np
import queue
import uuid

#=======================================
# This class is used to pull data from 
# stream and save into buffer
#=======================================

class GstStreamSource(BaseStreamSource, ABC):
    def __init__(self,  camera,buffer,input_condition):
        stream_info=camera
        super().__init__(stream_info) 
        
        self.camera  = camera
        #--------------------
        # Condition object and
        # buffer connected with
        # ai_thread
        #--------------------
        self.c_streaming    = input_condition                  
        self.stream_buffer  = buffer              # contain frame read from stream
        # self.processRate = config.processRate
        #--------------------
        # Using for stream
        #--------------------
        self._stream = None 
        if self.stream_info.type == StreamType.RTSP:
            self._stream = RTSPPipelineSource(self.stream_info, self._on_new_buffer, self._on_update_status)
        assert self._stream
        
    #==================================
    # This function is ultilized 
    # for enabling to check fps
    #----------------------------------
    # def check_fps_func(self):
    #     print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    #     # print(sc)
    #     self.check_fps = True
    #     self.get_system_fps_again.enter(2, 1, self.check_fps_func, ())
        
    #==================================
    # This function starts the 
    # Gstreamer and timer thread
    #----------------------------------
    def start(self):
        # self.get_system_fps_again.run()
        try:
            self.connection_retry += 1
            self._stream.play()
        except BaseException as ex:
            self._on_update_status(StreamStatus.ERROR, 'cant not start Streaming because ' + str(ex))
            
    def center_crop(self,img, dim0, dim1):
        """Returns center cropped image
        Args:
        img: image to be center cropped
        dim: dimensions (width, height) to be cropped
        """
        width, height = img.shape[1], img.shape[0]

        # process crop width and height for max available dimension
        crop_width = dim0 if dim0<img.shape[1] else img.shape[1]
        crop_height = dim1 if dim1<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img


    #==================================
    # This function get frames from 
    # cameras through new_data signal
    # and store in stream buffer
    #----------------------------------
    def _on_new_buffer(self, sample) -> Gst.FlowReturn.OK:
        buf = sample.get_buffer()
        caps_format = sample.get_caps().get_structure(0)
        app_width, app_height = caps_format.get_value('width'), caps_format.get_value('height')
        result, mapinfo = buf.map(Gst.MapFlags.READ)

        if result:
            numpy_frame = np.ndarray(shape=(app_height, app_width, 3), dtype=np.uint8, buffer=mapinfo.data)
            # print('aaaaaaaaaaa')

            # numpy_frame = cv2.imread('./speed.jpg')
            self.stream_buffer.put({
                    "frame": numpy_frame,
                    "ts": time.time(),
                    "frame_id": str(uuid.uuid4())
                })
            
            if self.stream_buffer.full():
                self.stream_buffer.get()
            
            
           #--------------------------
           # Notify for ai_thread
           #--------------------------
            with self.c_streaming:
                if self.stream_buffer.qsize()>=2:
                    self.c_streaming.notifyAll()
                    
        buf.unmap(mapinfo)
        return Gst.FlowReturn.OK
    
    #==================================
    # This function refreshes 
    # Gstreamer
    #----------------------------------

    def refresh(self, stream_info):
        self.stream_info = StreamInfo(stream_info)
        if self.stream_status != StreamStatus.PLAYING:
            self._stream.update_stream_info(self.stream_info)
            self.stop()
            self.start()
        log.info("#stream_id: {}, status: {}, message: {}".format(self.stream_info.stream_id,
                                                                      self.stream_status,
                                                                      self.stream_message))

    #==================================
    # This function stops 
    # Gstreamer
    #----------------------------------
    def stop(self):
        try:
            self._stream.stop()
        except BaseException as ex:
            self._on_update_status(StreamStatus.ERROR, 'cant not stop Streaming because ' + str(ex))
