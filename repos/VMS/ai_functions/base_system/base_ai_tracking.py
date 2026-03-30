import threading
import queue
from ai_core.utility.config import BufferConfig
import time
from ai_core.utility.ai_logger import aiLogger

#=========================================
# This class create threads for ai feature 
# 
#=========================================
class BaseAiTracking:
    def __init__(self,fps_log):
        
        config = BufferConfig()
        self.__fps_log = fps_log
        
        #----------------------------
        # Declare running flag and 
        # tracking thread
        #----------------------------
        self._running = False             # Using this flag to running or stop thread
        self._ai_tracking_thread = threading.Thread(target=self.__ai_tracking_processing,daemon=False)   
        
        #----------------------------
        # Buffer
        #----------------------------
        self.rawBuffer        = None                                # From neck head of Yolo
        self.processedBuffer  = config.processedBuffer              # processed tracking frame will be stored in this buffer
        
        #----------------------------
        # Condition object
        #----------------------------
        self.processedCondition = threading.Condition()             # Notify for other places
        self.rawCondition = None
        
        #----------------------------
        # Sending variable
        #----------------------------
        self.base_info = {}
        self.utility_info = {}
    #==========================================
    # This function is the base function of
    # tracking block to transfer the data through
    # pipeline
    #------------------------------------------
    def __ai_tracking_processing(self):
        while True:
            try:
                with self.rawCondition:
                    self.rawCondition.wait()
                    
                if not self.rawBuffer.empty():
                    curTime = time.time()
                    base_info,utility_info = self.rawBuffer.get()
                    frame   = base_info['frame']
                    camera  = base_info['camera']

                    if camera.run_track_in_yolo5:
                        #--------------------------------
                        # Get frame from Yolo neck head 
                        # buffer
                        #--------------------------------
                        frame,tracking_info             = self._ai_tracking(frame,utility_info['bounding_box'],camera)
                        base_info['frame']              = frame
                        utility_info['track']           = tracking_info
                        
                    else:
                        utility_info['track']  = None

                    #--------------------------------
                    # push processed tracking frame
                    # into buffer to transfer to next
                    # steps
                    #--------------------------------
                    # if self.processedBuffer.qsize()>0:
                    #     aiLogger.warning("Tracking buffer {}".format(self.processedBuffer.qsize()))
                    self.processedBuffer.put((base_info,utility_info))
                    if self.processedBuffer.full():
                        self.processedBuffer.get()
                        aiLogger.warning("The tracking thread lost frame, please check again!!!")

                    #--------------------------------
                    # Notify for next steps that this
                    # stage was completed
                    #--------------------------------
                    with self.processedCondition:
                        if self.processedBuffer.qsize() >=1:
                            # aiLogger.debug("TRACKING PUSH--------------------------------------")
                            self.processedCondition.notifyAll()
                            # aiLogger.debug("Tracking-------------------------------- {}".format(self.processedBuffer.qsize()))
                            # print('notifyAll', base_info['camera'].id)
                    # print("the entier time of neckhead in _ai_tracking", time.time()- curTime)
                    
                    
                
                    #-------------------------------
                    # Get process time and frame count 
                    # to calculate FPS
                    #-------------------------------
                    if camera.id in self.__fps_log.measure_tracking.keys():
                        self.__fps_log.measure_tracking[camera.id]['time']+=time.time()-curTime
                        self.__fps_log.measure_tracking[camera.id]['fps']+=1  
                    
                    
                if not self._running:
                    break
            except Exception as e:
                aiLogger.debug("__ai_tracking_processing Exception" + str(e))

                # print("------- __ai_tracking_processing Exception " + str(e))
                    
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
    def _ai_tracking(self):
    #    raise RuntimeError('_ai_feature method need to be override by child class.')
        return True
    