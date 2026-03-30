import threading
import queue
from ai_core.utility.config import BufferConfig
import time
# from utility.utils_logger import logger
from ai_core.utility.ai_logger import aiLogger
import os
#=========================================
# This class create threads for ai feature 
# 
#=========================================
class BaseAiAppLicensePlate:
    def __init__(self,fps_log):
        
        config = BufferConfig()
        #----------------------------
        # Declare running flag and 
        # tracking thread
        #----------------------------
        self.__fps_log  = fps_log
        self._running   = False                                                                                    # Using this flag to running or stop thread
        self._ai_application_thread = threading.Thread(target=self.__ai_application_processing,daemon=False)       # Declare backbone_thread with __ai_process function
    
                         
        #----------------------------
        # Buffer
        #----------------------------
        self.processedBuffer  = config.processedBuffer
        self.rawBuffer        = None
        
        
        #----------------------------
        # Condition variable
        #----------------------------
        self.rawCondition = None                                         # Condition object to get frame from other threads
        self.processedCondition = threading.Condition()                  # Condition object to notify other threads that this stage was completed    
        
    
    # def update_setting(camera):
    #     self.camera = camera
    #==========================================
    # This function brings frame from stream 
    # buffer and let into AI block
    #------------------------------------------
    def __ai_application_processing(self):
        while True:
            try:
                with self.rawCondition:
                    self.rawCondition.wait()
                
                if not self.rawBuffer.empty():
                    curTime = time.time()

                    #--------------------------------
                    # Get frame from rawBuffer 
                    #--------------------------------
                    base_info,utility_info = self.rawBuffer.get()
                    
                    
                    #-------------------------------
                    # Extract info
                    #------------------------------- 
                    frames      = base_info['frame']
                    # frame_times = base_info['frame_time']
                    # frame_ids   = base_info['uuid']
                    camera      = base_info['camera']
                    batch_size  = base_info['batchsize']
                    motion      = utility_info['motion']
                    
                    # aiLogger.debug(camera)
                    #--------------------------------
                    # Process _ai_application 
                    #--------------------------------
                    if motion is True:
                        frames,application_info = self._ai_application(base_info,utility_info)
                        
                    else:
                        application_info        = self.application_default(batch_size)
                        frames                  = frames

                       
                    #--------------------------------
                    # Compress info
                    #--------------------------------
                    base_info['frame']                               = frames
                    utility_info['meta_data'][self.application_name] = application_info['meta_data']
                    utility_info[self.result_application_name]       = application_info[self.result_application_name]

                    
                    #--------------------------------
                    # Put info into the system
                    #--------------------------------
                
                    self.processedBuffer.put((base_info,utility_info))
                    if self.processedBuffer.full():
                        self.processedBuffer.get()
                        
                
                    #--------------------------------
                    # Notify for later threads
                    #--------------------------------
                    with self.processedCondition:
                        if self.processedBuffer.qsize() >=1:
                            self.processedCondition.notifyAll()
                    # if self.application_name    == 'object_counter':
                    #     tm = time.time()-curTime
                    #     if tm>0.04:
                    #         aiLogger.error("Time too large: {}".format(tm))

                        # aiLogger.debug("Current time of object counter:{}".format(tm))
                    #-------------------------------
                    # Get process time and frame count to calculate FPS
                    #-------------------------------
                    # camera = base_info['camera']
                    if camera.id in self.__fps_log.measure_license_plate.keys():
                        self.__fps_log.measure_license_plate[camera.id]['time']+= (time.time()-curTime)
                        self.__fps_log.measure_license_plate[camera.id]['fps'] +=1  
                        
                if not self._running:
                    break
                # print("the entier time of neckhead in _ai_feature", time.time()- curTime)
            except Exception as e:
                aiLogger.exception("__ai_application_processing {} Exception {}".format(self.application_name,e))

                # print("------- __ai_application_processing Exception " + str(e))
                
                
    #==========================================
    # This function is used to return a default 
    # application in the case if we use 
    # motion detection
    #------------------------------------------   
    def application_default(self,batch_size):
        returned_application_rst = {}
        
            
        returned_application_rst['meta_data']                  = [[] for _ in range(batch_size)]
        returned_application_rst[self.result_application_name] = [[None] for _ in range(batch_size)]
        return returned_application_rst
    #==========================================
    #
    #------------------------------------------
    def start(self):
        raise RuntimeError(' start() method need to be overrided by child class.')
    
    #==========================================
    #
    #------------------------------------------
    def stop(self):
        raise RuntimeError('stop() method need to be overrided by child class.')
    
    #==========================================
    #
    #------------------------------------------                                    
    def _ai_application(self):
    #    raise RuntimeError('_ai_feature method need to be override by child class.')
        return True
    