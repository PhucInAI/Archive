import threading
import queue
from ai_core.utility.ai_logger import aiLogger
from copy import deepcopy
import copy
import time

#=========================================
# Name_class: ReleaseFeatureBottleneck
# Purpose   : This class is used to dupplicate
# the results after processing backbone to 
# avoid the bottleneck at backbone
#=========================================

class ReleaseFeatureBottleneck():
    def __init__(self,running_features,
                    fps_calculation,
                    sync_thread_output):
        
        # self.curFrameId = -1
        self.__fps_calculation=fps_calculation
        #-----------------------------  
        # Create thread and others
        #-----------------------------
        self.__release_bottleneck = threading.Thread(target=self.__release_feature_bottleneck, daemon=False)
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
        self.__release_bottleneck.start()
        
        
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
            #-------------------------------
            # wait till get new frame from 
            # backbone thread
            #-------------------------------
            with self.c_raw_rl_bottleneck:
                self.c_raw_rl_bottleneck.wait()
                
            #-------------------------------
            # get the frame from buffers
            #------------------------------- 
            curTime = time.time() 
            raw_frame, layer, camera = self.raw_rl_bottleneck_buffer.get()
           
            #-------------------------------
            # if run ai and any one of features
            # is turned on, put frames into 
            # assigned ai thread
            #-------------------------------  
            for feature in self.__running_features:
                if camera.run[feature] == True:
                    
                    #-------------------------------
                    # If sync, use the same underline
                    # data of numpy for frame
                    #-------------------------------
                    if not self.__sync_thread_output:
                        sending_frames = []
                        for frame in raw_frame:
                            sending_frames.append(copy.deepcopy(frame))
                            # put_layer = layer[:]
                    else:
                        sending_frames = raw_frame

                    self.processed_rl_bottleneck_buffer[feature].put((sending_frames,layer,camera)) 
                    
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
                
            #-------------------------------
            # Get process time and frame count to calculate FPS
            #-------------------------------
            self.__fps_calculation.measure_bottleneck[camera.id]['time']+=time.time()-curTime
            self.__fps_calculation.measure_bottleneck[camera.id]['fps']+=1      
                
            if not self.__running:
                break    