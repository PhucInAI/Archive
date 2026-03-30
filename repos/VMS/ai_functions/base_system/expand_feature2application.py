import queue
import threading
import time
from ai_functions.utility.ai_logger  import aiLogger
from ai_functions.utility.config     import BufferConfig


class ExpandFeature2Application():
    def __init__(self,):
        
        self.config = BufferConfig()
        #-----------------------------  
        # Create thread and others
        #-----------------------------
        self.feature2application = threading.Thread(target=self.__expand_feature2application, daemon=False) 
        self.__running = False
        self.cameras   = []
        #----------------------------
        # Condition object
        #----------------------------
        self.c_raw_feature2application        = None
        self.processedCondition  = {}
        
        
        #----------------------------
        # Buffer
        #----------------------------
        self.raw_feature2application_buffer = None
        self.processedBuffer = {}
            
        
    #==========================================
    # This function is used to start the thread
    #------------------------------------------         
            
    def start(self):
        self.__running = True
        self.feature2application.start()
        
        
    #==========================================
    # This function is used to stop the thread
    #------------------------------------------   
    def stop(self):
        self.__running = False 
        
    #==========================================
    # This function expands layers buffer to 
    # multiple buffers connected to Ai Features
    #------------------------------------------     
    def __expand_feature2application(self):
        
        while True:
            try:
                #-------------------------------
                # wait till get new frame from 
                # backbone thread
                #-------------------------------
                with self.c_raw_feature2application:
                    self.c_raw_feature2application.wait()
                    
                if not self.raw_feature2application_buffer.empty():
                    #-------------------------------
                    # get the frame from buffers
                    #------------------------------- 
                    base_info, utility_info = self.raw_feature2application_buffer.get()

                    #-------------------------------
                    # Put frames into specific buffer
                    #------------------------------- 
                    camera_id = base_info['camera'].id
                
                    if (camera_id in self.processedBuffer):
                        # print('1 aaaaaaaaaaaa',camera_id, base_info['uuid'])
                        self.processedBuffer[camera_id].put((base_info, utility_info))
                    else: 
                        continue
                    #-------------------------------
                    # get the frame if the buffer is 
                    # full
                    #-------------------------------
                    if self.processedBuffer[camera_id].full():
                        self.processedBuffer[camera_id].get()
                                
                    #-------------------------------
                    # Notify for a specific Ai feature 
                    # threads
                    #-------------------------------
                    with self.processedCondition[camera_id]:
                        self.processedCondition[camera_id].notifyAll()
                        
                    
                # #-------------------------------
                # # Get process time and frame count to calculate FPS
                # #-------------------------------
                # self.__fps_log.measure_bottleneck[camera.id]['time']+=time.time()-curTime
                # self.__fps_log.measure_bottleneck[camera.id]['fps']+=1      
                    
                if not self.__running:
                    break  
            except Exception as e:
                aiLogger.exception("__expand_feature2application Exception" + str(e))
                
            
    def add_camera(self,added_cameras):
        for camera in added_cameras:
            self.processedCondition[camera.id]      = threading.Condition()
            self.processedBuffer[camera.id]         = self.config.processedBuffer
        self.cameras += added_cameras    
            
    def remove_camera(self,rm_camera_id):
        for cur_cam in self.cameras[:]:
            if cur_cam.id in rm_camera_id:
                self.cameras.remove(cur_cam) 
                time.sleep(0.25)
                del self.processedCondition[cur_cam.id]   
                del self.processedBuffer[cur_cam.id]
                
        
            
    
        
         
            
     