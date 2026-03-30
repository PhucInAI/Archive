import threading
import queue
from ai_functions.utility.ai_logger import aiLogger
from copy import deepcopy
import copy
import time

#=========================================
# Name_class: ExpandBackbone2Features
# Purpose   : This class is used to dupplicate
# the results after processing backbone to 
# avoid the bottleneck at backbone
#=========================================

class ExpandBackbone2Features():
    def __init__(self,running_features,
                    fps_log,
                    sync_thread_output):
        
        # self.curFrameId = -1
        self.__fps_log=fps_log
        #-----------------------------  
        # Create thread and others
        #-----------------------------
        self.release_bottleneck = threading.Thread(target=self.__release_feature_bottleneck, daemon=False)
        self.__running_features = running_features
        self.__running = False
        self.__sync_thread_output = sync_thread_output
        #----------------------------
        # Condition object
        #----------------------------
        self.c_raw_rl_bottleneck = None
        self.c_processed_rl_bottleneck = {}
        for feature in self.__running_features:
            self.c_processed_rl_bottleneck[feature] = threading.Condition()
        
        
        #----------------------------
        # Buffer
        #----------------------------
        self.raw_rl_bottleneck_buffer = None
        self.processed_rl_bottleneck_buffer = {}
        for feature in self.__running_features:
            self.processed_rl_bottleneck_buffer[feature] = None      
        
        
        
    #==========================================
    # This function is used to start the thread
    #------------------------------------------         
            
    def start(self):
        self.__running = True
        self.release_bottleneck.start()
        
        
    #==========================================
    # This function is used to stop the thread
    #------------------------------------------   
    def stop(self):
        self.__running = False 
        
    #==========================================
    # This function expands layers buffer to 
    # multiple buffers connected to Ai Features
    #------------------------------------------     
    def __release_feature_bottleneck(self):
        while True:
            try:
                #-------------------------------
                # wait till get new frame from 
                # backbone thread
                #-------------------------------
                with self.c_raw_rl_bottleneck:
                    self.c_raw_rl_bottleneck.wait()
                    
                #-------------------------------
                # get the frame from buffers
                #------------------------------- 
                if not self.raw_rl_bottleneck_buffer.empty():
                    curTime = time.time() 
                    base_info, utility_info = self.raw_rl_bottleneck_buffer.get()
                    
                    #------------------------
                    # Extract info
                    #------------------------
                    frames      = base_info['frame']
                    camera      = base_info['camera']
                    layer       = utility_info['layer']
                    
                
                    #-------------------------------
                    # if run ai and any one of features
                    # is turned on, put frames into 
                    # assigned ai thread
                    #-------------------------------  
                    
                    if camera.run_ai:        
                        for feature in self.__running_features:
                            if camera.run[feature]:
                                #-------------------------------
                                # If sync, use the same underline
                                # data of numpy for frame
                                #-------------------------------
                                if not self.__sync_thread_output:
                                    sending_frames = []
                                    for frame in frames:
                                        sending_frames.append(frame.copy())
                                    base_info['frame']    = sending_frames
                                    utility_info['layer'] = layer.copy()
                                else:
                                    utility_info['layer'] = layer.copy()
                                                                    
                                
                                self.processed_rl_bottleneck_buffer[feature].put((base_info, utility_info.copy()))
                                
                                #-------------------------------
                                # get the frame if the buffer is 
                                # full
                                #-------------------------------
                                if self.processed_rl_bottleneck_buffer[feature].full():
                                    self.processed_rl_bottleneck_buffer[feature].get()
                                    
                                #-------------------------------
                                # Notify for a specific Ai feature 
                                # threads
                                #-------------------------------
                                with self.c_processed_rl_bottleneck[feature]:
                                    self.c_processed_rl_bottleneck[feature].notifyAll()
                                
                            else:
                                continue   
                            
                    else:
                        self.processed_rl_bottleneck_buffer["without_ai"].put((base_info, utility_info.copy()))
                        #-------------------------------
                        # get the frame if the buffer is 
                        # full
                        #-------------------------------
                        if self.processed_rl_bottleneck_buffer['without_ai'].full():
                            self.processed_rl_bottleneck_buffer['without_ai'].get()
                            
                        #-------------------------------
                        # Notify for a specific Ai feature 
                        # threads
                        #-------------------------------
                        with self.c_processed_rl_bottleneck['without_ai']:
                            self.c_processed_rl_bottleneck['without_ai'].notifyAll()
                        
                        
                                
                        
                    #-------------------------------
                    # Get process time and frame count to calculate FPS
                    #-------------------------------
                    if camera.id in self.__fps_log.measure_bottleneck.keys():
                        self.__fps_log.measure_bottleneck[camera.id]['time']+=time.time()-curTime
                        self.__fps_log.measure_bottleneck[camera.id]['fps']+=1      
                    
                if not self.__running:
                    break    
                

            except Exception as e:
                aiLogger.exception("__release_feature_bottleneck Exception" + str(e))
                # print("------- __release_feature_bottleneck Exception " + str(e))
                
                
                    
                    
                    
                    
                
                
            
        
        
        
        
        