import queue
import threading
import camera.listcam as listcam
from camera.camera_manager                              import Camera
from utils.gst_streaming.get_camera_frames              import GetCameraFrames
from utils.gst_streaming.display_processed_frame        import DisplayProcessedFrame
import time

deencam3            =   listcam.deen_cam3

class VmsSimulation():
    def __init__(self):
        self.update_flag = True
        #--------------------------------------
        # Initialize camera, buffer, and 
        # condition
        #--------------------------------------
        self.init_cameras = [deencam3]
        
        self.init_in_buffer     = {}
        self.init_out_buffer    = {}

        self.init_in_condition = threading.Condition()
        self.init_out_condition = {}
        for camera in self.init_cameras:
            self.init_in_buffer[camera['name']]      = queue.Queue(5)
            self.init_out_buffer[camera['name']]     = queue.Queue(5)
            self.init_out_condition[camera['name']]  = threading.Condition()

        #--------------------------------------
        # Init some variables
        #--------------------------------------
        self.get_camera_frame = {}
        self.display_process_frame = {}


        #----------------------------------
        # Init the get_frame_camera
        #----------------------------------
        for camera in self.init_cameras:
            self.get_camera_frame[camera['name']]        = GetCameraFrames(    camera,
                                                                                    self.init_in_buffer[camera['name']],
                                                                                    self.init_in_condition) 
            self.display_process_frame[camera['name']]   = DisplayProcessedFrame(camera,
                                                                                    )
            self.display_process_frame[camera['name']].display.display_buffer = self.init_out_buffer[camera['name']]
    
    #==========================================
    # This function is used to start to
    # intialize the pipeline 
    #------------------------------------------
    def start(self,):
        for camera in self.init_cameras:
            self.get_camera_frame[camera['name']].start()
            self.display_process_frame[camera['name']].start()
            
    def remove_camera(self,):
        self.removed_cameras = [deencam3]
        
    def add_new_camera(self,):
        #--------------------------------------
        # Update camera, buffer, and 
        # condition
        #--------------------------------------
        self.added_cameras       = [deencam3]
        self.added_in_buffer     = {}
        self.added_out_buffer    = {}

        # self.added_in_condition = threading.Condition()
        self.added_out_condition = {}
        for camera in self.added_cameras:
            self.added_in_buffer[camera['name']]       = self.init_in_buffer[camera['name']] 
            self.added_out_buffer[camera['name']]      = self.init_out_buffer[camera['name']]
            self.added_out_condition[camera['name']]   = self.init_out_condition[camera['name']]
    
    def update_camera(self,):
        self.update_flag                            = not self.update_flag
        # nova_cam1['feature']['face']['enabled']     = self.update_flag   
        deencam3['run_ai']   = self.update_flag 
        
        # camera1['run_ai']   = self.update_flag 
        self.updated_camera  = [deencam3]
        
        
        


            


    


        

    
