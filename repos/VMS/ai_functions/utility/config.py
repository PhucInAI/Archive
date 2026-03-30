import queue
# from ai_core.utils   import get_GPU_available

#=========================================
# Name_class: BufferConfig
# Purpose   : Config the size of buffer
#=========================================    
buffer_size = 25

class BufferConfig():
    def __init__(self):
        self.buffer_size            = 25
        self.rawBuffer              = queue.Queue(maxsize=self.buffer_size) 
        self.processedBuffer        = queue.Queue(maxsize=self.buffer_size) 

        

#=========================================
# Name_class: AiProcessorConfig
# Purpose   : Config for AiProcessorConfig
#=========================================      
class AiProcessorConfig():
    def __init__(self):
        
        # --------------------------------
        # General config
        # --------------------------------
        
        self.using_internal_display     = False 
        self.sync_thread_output         = True
        self.batch_size                 = 1
        self.gpu_for_model              = True
        self.use_cuda_empty_cache       = False

        self.force_fps                  = True     # Set true to force the fps the system to use the prossessRate value 
        self.processRate                = 30       # Control the input fps of the system
        
        self.runMotionDetection         = False
        self.motionDetectionStableTime  = 5  #min


        # --------------------------------
        # Fps Log config
        # --------------------------------
        self.time_to_print              = 20
        self.measuringYoloApp           = True



        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # TRACKING CONFIG
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        self.numGetTrackPackage         = 2


        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # YOLO CONFIG
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        self.yoloMetaType   = [ 'human_counting',
                                'vehicle_counting',
                                'human_multiple_camera_tracking',
                                'heat_map',
                                'crowd_detection',
                                'virtual_fences',
                                'traffic_jam_detection',
                                'parking_violation_detection',
                                'way_driving_violation_detection',
                                'vehicle_speed_violation'
                                ]

        # -----------------------
        # Config for Object counter
        # -----------------------
        self.drawObjectCounter          = True
        self.drawCrowdDetection         = True
        self.drawRedTrafficLight        = True
        self.drawDirection              = True
        self.drawWrongDirection         = True
        self.drawVelocity               = True


        



        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # FACE CONFIG
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        self.faceMetaType   = []
        self.fireMetaType   = []


        self.systemMetaType = self.yoloMetaType + self.faceMetaType + self.fireMetaType 
                        
 
    


#=========================================
# Name_class: DisplayProcessedFrameConfig
# Purpose   : Config for DisplayProcessedFrame
#=========================================
class DisplayProcessedFrameConfig():
    def __init__(self):
        self.display_buffer = queue.Queue(maxsize=buffer_size)           # Contain display frame after assgined what pipeline to display 
        
        
