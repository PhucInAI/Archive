
from distutils.command.config import config
import threading
from threading import Lock
import queue
import time
from typing import List
# from utility.utils_logger import logger
from ai_functions.utility.ai_logger import aiLogger
from ai_functions.base_system.fps_control import FpsControl
from ai_functions.utility.config import AiProcessorConfig,BufferConfig

# from ai_functions.motion_detection import MotionDetection


class InputFrame2Ai():
    def __init__(self,
                 fps_log,
                 batch_size,
                 processRate
                ): 
                        
        self.cameras            = []
        self.__fps_log          = fps_log
        self.config             = AiProcessorConfig()
        self.__processRate      = processRate
        
        #----------------------------- 
        # Create thread and others
        #-----------------------------
        self.input_input_frame2ai   = threading.Thread(target=self.__input_frame2ai, daemon=False)
        self.mutex                  = Lock()
        
        self.__running              = False
        self.control_fps            = {}
        
        
        #----------------------------
        # For fps log
        #----------------------------
        self.begin_fps_time = {}
        
        for camera in self.cameras:
            self.begin_fps_time[camera.id]    = None
        
        #----------------------------
        # Other variables
        #----------------------------
        self.batch_size         = batch_size
        
        self.base_info          = {}
        self.utility_info       = {}

        # self.runMotionDetection   = self.config.runMotionDetection
        #----------------------------
        # Condition object
        #----------------------------
        self.c_raw_fps_control       = None
        self.c_processed_fps_control = threading.Condition()
        
        
        #----------------------------
        # Buffer
        #----------------------------
        buffConfig                          = BufferConfig()
        self.processed_fps_control_buffer   = buffConfig.processedBuffer
        
        self.raw_fps_control_buffer         = {}
        for camera in self.cameras:
            self.raw_fps_control_buffer[camera.id] = None
            
        #----------------------------
        # Motion detection variable
        #----------------------------
        self.motion_detection = {}
        
        
    #==========================================
    # This function is used to start the thread
    #------------------------------------------         
    def start(self):
        self.__running = True
        self.input_input_frame2ai.start()
        
        
    #==========================================
    # This function is used to stop the thread
    #------------------------------------------   
    def stop(self):
        self.__running = False 
        
    #==========================================
    # This function expands layers buffer to 
    # multiple buffers connected to Ai Features
    #------------------------------------------     
    def __input_frame2ai(self):
        while True:
            try:
                #-------------------------------
                # wait till get new frame 
                #-------------------------------
                if self.c_raw_fps_control is not None:
                    with self.c_raw_fps_control:
                        self.c_raw_fps_control.wait()
                
                #-------------------------------
                # get the frame from buffers
                #------------------------------- 
                for camera in self.cameras:
                    if not self.raw_fps_control_buffer[camera.id].empty():
                        self.begin_fps_time[camera.id] = time.time()
                        
                        input_data   = self.raw_fps_control_buffer[camera.id].get()
                        # print(input_data)
                        stream_frame = input_data["frame"]
                        uuid         = input_data["frame_id"]
                        ts           = input_data["ts"]
                        # print(stream_frame)
                        #-------------------------------
                        # get frames base on processRate
                        #------------------------------- 
                        
                        if self.control_fps[camera.id].get_frame_now():
                            #----------------------
                            # Compress info
                            #----------------------
                            self.base_info[camera.id]['batchsize']  = self.batch_size 
                            self.base_info[camera.id]['camera']     = camera
                            self.base_info[camera.id]['frame'].append(stream_frame)
                            self.base_info[camera.id]['frame_time'].append(ts)
                            self.base_info[camera.id]['uuid'].append(uuid)
                            
                            curLen = len(self.base_info[camera.id]['frame'])
                                
                            if curLen == self.batch_size:
                                
                                #-------------------------------
                                # Motion detection 
                                #-------------------------------
                                # if self.runMotionDetection:
                                #     motion = self.motion_detection[camera.id].batch_detect(self.base_info[camera.id]['frame'])
                                #     self.utility_info[camera.id]['motion'] = motion
                                #     # aiLogger.debug("motion detection {}".format(motion))
                                # else:
                                self.utility_info[camera.id]['motion'] = True
                                
                                # if camera.id == 'cam6':
                                #     print("motion detection",motion, camera.id)
                                
                                
                                self.processed_fps_control_buffer.put((self.base_info[camera.id].copy(),self.utility_info[camera.id].copy()))
                                self.base_info[camera.id]['frame']      = []
                                self.base_info[camera.id]['frame_time'] = []
                                self.base_info[camera.id]['uuid']       = []
                                # self.utility_info[camera.id]['motion']  = False
                                
                            
                                #-------------------------------
                                # get frame from full buffer and 
                                # notify
                                #-------------------------------
                                if self.processed_fps_control_buffer.full():
                                    self.processed_fps_control_buffer.get()  
                                    
                                with self.c_processed_fps_control:
                                    if (self.processed_fps_control_buffer.qsize() >= 1):
                                        self.c_processed_fps_control.notifyAll()  
    
                            #-------------------------------
                            # Get process time and frame count
                            # to calculate FPS of AI input
                            #-------------------------------
                            self.__fps_log.measure_input_frame2ai[camera.id]['time']    += time.time()- self.begin_fps_time[camera.id]
                            self.__fps_log.measure_input_frame2ai[camera.id]['fps']     +=  1
                            self.__fps_log.measure_input_frame2ai[camera.id]['nb_skip_frame'] = self.control_fps[camera.id].skip_frames
                    
                        #-------------------------------
                        # Get process time and frame count
                        # to calculate FPS from VMS input
                        #-------------------------------
                        self.__fps_log.measure_vms_fps[camera.id]['time']   += time.time()- self.begin_fps_time[camera.id]
                        self.__fps_log.measure_vms_fps[camera.id]['fps']    += 1
                            
                        
                    if not self.__running:
                        break   


            except Exception as e:
                aiLogger.exception("__input_frame2ai Exception" + str(e))
                # print("------- __input_frame2ai Exception " + str(e))    
    
    #==========================================
    # This function is used to add cameras into
    # input_frame2ai
    #------------------------------------------         
    def add_camera(self,
                   added_cameras,
                   in_buffer,
                   in_condition):

        
        # self.cameras    += added_cameras
        self.c_raw_fps_control  = in_condition
        for camera in added_cameras:
            #---------------------------
            # Update buffer and other 
            # variables for InputFrame2Ai
            #---------------------------
            self.control_fps[camera.id]               = FpsControl(self.__processRate)
            curProcessRate = self.update_process_rate(camera)                           #Update the processRate for fpsControl
            
            
            self.raw_fps_control_buffer[camera.id]    = in_buffer[camera.id]
            self.begin_fps_time[camera.id]            = None 
            
            
            self.base_info[camera.id]                 = {}
            self.base_info[camera.id]['frame']        = []
            self.base_info[camera.id]['frame_time']   = []
            self.base_info[camera.id]['uuid']         = []
            
            self.utility_info[camera.id]              = {}
            self.utility_info[camera.id]['motion']    = False
            
            self.motion_detection[camera.id]          = MotionDetection(motion_thr=40,
                                                                        updated_background = 5,
                                                                        updated_false_motion= self.config.motionDetectionStableTime*60*curProcessRate)  #5min x 60second x processRate
        self.cameras    += added_cameras
    
    # 

    #==========================================
    # This function is used to remove cameras 
    # in input_frame2ai
    #------------------------------------------   
    def remove_camera(self,rm_camera_id):
        for cur_cam in self.cameras[:]:
            if cur_cam.id in rm_camera_id:
                self.cameras.remove(cur_cam)
                time.sleep(0.25)
                del self.raw_fps_control_buffer[cur_cam.id]
                del self.begin_fps_time[cur_cam.id]   
                del self.control_fps[cur_cam.id]  
                del self.base_info[cur_cam.id]
                del self.utility_info[cur_cam.id]
                
    #==========================================
    # This function is used to update the 
    # process rate for each camera
    #------------------------------------------
    def update_process_rate(self,cameras,newProcessRate=None):

        if self.config.force_fps:
            newProcessRate = self.__processRate if newProcessRate is None else newProcessRate
    
        else:
            if cameras.run['yolo5'] and cameras.run['vehicle_speed_violation']:
                newProcessRate = 25
            elif cameras.run['yolo5']:
                newProcessRate = 10
            elif not cameras.run['yolo5']: 
                newProcessRate = 5
            else:
                newProcessRate = self.__processRate if newProcessRate is None else newProcessRate

        self.control_fps[cameras.id].processRate = newProcessRate
        aiLogger.warning("The processRate was updated successfully!!!" + str(newProcessRate))
        return newProcessRate

    #==========================================
    # This function is used to update the camera
    # feature in Input block
    #------------------------------------------
    def update_utility(self,updated_camera):

        newProcessRate = self.update_process_rate(updated_camera)

        # Update processRate for motion
        self.motion_detection[updated_camera.id].update_motion_parameter(newProcessRate)
        
        return newProcessRate
                
        


        
        
        
            

                
                
                
                    
                    
                    
                    
                
                
            
        
        
        
        
        