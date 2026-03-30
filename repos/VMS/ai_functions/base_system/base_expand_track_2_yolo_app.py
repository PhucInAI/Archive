from logging import raiseExceptions
import queue
import threading
import time
from ai_core.utility.ai_logger import aiLogger
from ai_core.utility.config import BufferConfig
from ai_core.utility.config import AiProcessorConfig


class BaseExpandTrack2YoloApp():
    def __init__(self,fps_log,availableYoloApp):
        
        # config              = AiProcessorConfig()
        
        self.__fps_log      = fps_log
        #-----------------------------  
        # Create thread and others
        #-----------------------------
        self.track2YoloApp  = threading.Thread(target=self.__expandTrack2YoloApp, daemon=False) 
        

        self.__available_application = availableYoloApp
        self.aiConfig       = AiProcessorConfig()
        #----------------------------
        # Condition object
        #----------------------------
        self.rawCondition        = None
        self.processedCondition  = threading.Condition()
        
        
        #----------------------------
        # Buffer
        #----------------------------
        self.rawBuffer          = None
        self.processedBuffer    = {}
        config                  = {}
        for app in self.__available_application:
            config[app]               =   BufferConfig()
            self.processedBuffer[app] =   config[app].processedBuffer


    #==========================================
    # This function expands tracking info to 
    # multiple application buffers 
    #------------------------------------------     
    def __expandTrack2YoloApp(self):
        
        while True:
            try:
                #-------------------------------
                # wait till get new frame from 
                # previous thread
                #-------------------------------
                #startTime = time.time()
                with self.rawCondition:
                    self.rawCondition.wait()

                # if self.processedBuffer.qsize()>0:
                # aiLogger.warning("Tracking buffer {}".format(self.rawBuffer.qsize()))

                for _ in range(self.aiConfig.numGetTrackPackage):
                    if not self.rawBuffer.empty():
                        curTime = time.time()
                        # aiLogger.debug("---------------------------------------")
                        # -------------------------------
                        # get the frame from buffers
                        # ------------------------------- 
                        base_info, utility_info = self.rawBuffer.get()

                        # -------------------------------
                        # Extract info
                        # ------------------------------- 
                        camera      = base_info['camera']
                        # motion      = utility_info['motion']


                        # -------------------------------
                        # If tracking algorithm is turned off, 
                        # the memory and other apps will not 
                        # handle anything.
                        # -------------------------------
                        if camera.run_track_in_yolo5:
                            frames,memory_info = self._track2app(base_info,utility_info)

                            # -------------------------------
                            # Compress the info
                            # -------------------------------
                            base_info['frame']                      = frames 
                            utility_info['general_yolo_app_info']   = memory_info

                        
                        # -------------------------------
                        # Extend the info to yolo application
                        # -------------------------------
                        for app in self.__available_application:
                            if camera.run[app] and app in self.processedBuffer:
                                self.processedBuffer[app].put((base_info, utility_info.copy()))
                                    
                                #-------------------------------
                                # get the frame if the buffer is 
                                # full
                                #-------------------------------
                                # aiLogger.debug("The size of memory buffer: {}".format(self.processedBuffer[app].qsize()))
                                if self.processedBuffer[app].full():
                                    self.processedBuffer[app].get()
                                    aiLogger.warning("The yolo application info is lost, please increase your buffer or check the system again!!!")

                                #-------------------------------
                                # Notify for a specific application
                                #-------------------------------
                                with self.processedCondition:
                                    self.processedCondition.notifyAll()
                            
                        # aiLogger.debug("The time in update mem: {}".format(time.time()-curTime))
                        #-------------------------------
                        # Get process time and frame count to calculate FPS
                        #-------------------------------
                        if camera.id in self.__fps_log.measure_updating_memory.keys():
                            self.__fps_log.measure_updating_memory[camera.id]['time'] += time.time()-curTime
                            self.__fps_log.measure_updating_memory[camera.id]['fps']  += 1       
                    
                if not self._running:
                    break  
            except Exception as e:
                aiLogger.exception(str(e))
                
            
    #==========================================
    # This function is used to start the thread
    #------------------------------------------         
    def start(self):
        raise RuntimeError(' start() method need to be override by child class.')
        
        
    #==========================================
    # This function is used to stop the thread
    #------------------------------------------   
    def stop(self):
        raise RuntimeError(' stop() method need to be override by child class.')
                
        
            
    
        
         
            
     