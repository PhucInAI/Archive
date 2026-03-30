# from common.enums.stream_info                   import StreamInfo
from utils.gst_streaming.gst_stream_source     import GstStreamSource

#=========================================
# Name_class: Get_camera_frames
# Purpose   : Get camera from camera and save
# into specific buffers
#=========================================
class GetCameraFrames:
    def __init__(self,
                 camera, 
                 buffer,
                 input_condition
                 ):   
        self.camera = camera
        self.frame_reader = GstStreamSource(camera,buffer,input_condition)
                                            
        
    def start(self):
        self.frame_reader.start()                           # Get data from default pipeline buffer and feed to Ai_buffer through new_sample signal
    
    def refresh(self, camera):   
        self.camera = camera
        self.frame_reader.refresh(camera.to_ai_stream_info())
    def stop(self):
        self.frame_reader.stop() 
        