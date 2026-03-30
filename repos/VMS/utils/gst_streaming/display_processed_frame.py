#=========================================
# Name_class: DisplayProcessedFrame
# Purpose: This class is ultilized for display 
# purpose from frames contained in display_buffer
# through need_data signal
#=========================================
from utils.gst_streaming.appsrc_display    import AppSrcDisplay
from utils.enums.stream_info               import StreamInfo

class DisplayProcessedFrame:
    def __init__(self, camera):
        self.camera = camera
        self.display = AppSrcDisplay(StreamInfo(camera))
                                         
    def start(self):
        self.display.play()
        
    def stop(self,):
        self.display.stop()

        
        
        # self.try:
        #     self._stream.stop()
        # except BaseException as ex:
        #     self._on_update_status(StreamStatus.ERROR, 'cant not stop Streaming because ' + str(ex))
