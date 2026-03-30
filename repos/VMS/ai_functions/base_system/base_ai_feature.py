import threading
import queue
import time
import os
from ai_core.utility.config     import BufferConfig
from ai_core.utility.ai_logger  import aiLogger

#=========================================
# This class create threads for ai feature 
# 
#=========================================
class BaseAiFeature:
    def __init__(self, fps_log):
        
        config = BufferConfig()
        self.__fps_log = fps_log
        # self._folder_path = folder_path
        # os.makedirs(self._folder_path, exist_ok=True)
        #----------------------------
        # Declare running flag and 
        # tracking thread
        #----------------------------
        self._running = False                                                                                   # Using this flag to running or stop thread
        self._ai_feature_thread         = threading.Thread(target=self.__ai_feature_processing,daemon=False)    # Declare backbone_thread with __ai_process function
        self.__get_buffer_per_once      = 0                                                                     # The number of element wanting to get from buffer each time
                         
        #----------------------------
        # Buffer
        #----------------------------
        self.processed_feature_buffer  = config.processedBuffer
        self.raw_feature_buffer        = config.rawBuffer 
        
        
        #----------------------------
        # Condition variable
        #----------------------------
        self.c_raw_neckhead         = None                          # Condition object to get frame from other threads
        self.c_processed_neckhead   = threading.Condition()         # Condition object to notify other threads that this stage was completed    
        
        #----------------------------
        # Sending variable
        #----------------------------
        self.base_info      = {}
        self.utility_info   = {}
        
        
    #==========================================
    # This function brings frame from stream 
    # buffer and let into AI block
    #------------------------------------------
    def __ai_feature_processing(self):
        while True:
            try:
                with self.c_raw_neckhead:
                    self.c_raw_neckhead.wait()
                
                self.__get_buffer_per_once    = 0
                while (self.__get_buffer_per_once < 2 and self.raw_feature_buffer.qsize()):
                    curTime = time.time()
                    #-------------------------------
                    # get frames from raw buffer
                    #-------------------------------
                    base_info, utility_info = self.raw_feature_buffer.get()
                    
                    #-------------------------------
                    # Extract info
                    #------------------------------- 
                    raw_frame   = base_info['frame']
                    frame_id    = base_info['uuid']
                    camera      = base_info['camera']
                    batch_size  = base_info['batchsize']
                    layer       = utility_info['layer']
                    motion      = utility_info['motion']
                    
                    #-------------------------------
                    # Process Ai features
                    #-------------------------------
                    if motion is True:
                        frame,feature_info = self._ai_feature(layer,raw_frame, frame_id,camera)
                    else:
                        feature_info       = self.feature_default(batch_size)
                        frame              = raw_frame
                    
                    #-------------------------------
                    # Compress info
                    #-------------------------------
                    base_info['frame']                          = frame
                    utility_info['meta_data']                   = {}
                    utility_info['meta_data'][self.feature]     = feature_info['meta_data']
                    utility_info[self.result_feature_name]      = feature_info[self.result_feature_name]
          
                    


                    #-------------------------------
                    # Put frames into processed buffer
                    #-------------------------------
                    self.processed_feature_buffer.put((base_info,utility_info.copy()))

                    if self.processed_feature_buffer.full():
                        self.processed_feature_buffer.get()

                    self.__get_buffer_per_once += 1
                    #-------------------------------
                    # Notify to other threads
                    #-------------------------------
                    with self.c_processed_neckhead:
                
                        if self.processed_feature_buffer.qsize() >=0:
                            self.c_processed_neckhead.notifyAll()

                    #-------------------------------
                    # Get process time and frame count to calculate FPS
                    #-------------------------------
                    if camera.id in self.__fps_log.measure_neckhead.keys():
                        self.__fps_log.measure_neckhead[camera.id][self.feature]['time']+=(time.time()- curTime)
                        self.__fps_log.measure_neckhead[camera.id][self.feature]['fps'] +=1
                
                if not self._running:
                    break
            except Exception as e:
                aiLogger.exception("__ai_feature_processing Exception" + str(e))
                
               
               
    #==========================================
    # This function is used to return a default 
    # feature in the case if we use motion detection
    #------------------------------------------   
    def feature_default(self,batch_size):
        feature = {}
    
        feature['meta_data']                =  [[] for _ in range(batch_size)]
        feature[self.result_feature_name]   =  [[None] for _ in range(batch_size)]
        return feature
            
    #==========================================
    #
    #------------------------------------------
    def start(self):
        raise RuntimeError(' start() method need to be override by child class.')
    
    #==========================================
    #
    #------------------------------------------
    def stop(self):
        raise RuntimeError('stop() method need to be override by child class.')
    
    #==========================================
    #
    #------------------------------------------                                    
    def _ai_feature(self):
    #    raise RuntimeError('_ai_feature method need to be override by child class.')
        return True
    
