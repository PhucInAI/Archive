import threading
import cv2
#=========================================
# Name_class: BaseExtractCamera2Display
# Purpose   : Extract camera from batchs
# and transfer to display
# Created by: An Nguyen
# Created on: 
# Last modified by:
# Modified on: 
#=========================================

class OutputAiInfoUnsync():
    def __init__(self,cameras):
        
        #----------------------------
        # Create thread and others
        #----------------------------
        self.cameras = cameras
        self.output_ai_info_unsync = threading.Thread(target=self.__output_ai_info_unsync, daemon=False)       
        self.running = False
        
        #----------------------------
        # Buffer
        #----------------------------
        self.raw_extract_camera_buffer = None
        self.processed_extract_camera_buffer ={}
        for camera in cameras:
            self.processed_extract_camera_buffer[camera.id] = None
        
        
        #----------------------------
        # Condition object
        #----------------------------
        self.c_raw_output_ai_info_unsync = None
        
    def drawInOutLine(self, frame, camera):
        h,w,d = frame.shape
        for l in camera.in_out_lines:
            if (len(l) == 2 and len(l[0]) == 2 and len(l[1]) == 2):
                l0 = (int(l[0][0]*w), int(l[0][1]*h))
                l1 = (int(l[1][0]*w), int(l[1][1]*h))
                cv2.line(frame, l0, l1, (255,0,0), 5)
        return frame
        
    #==========================================
    # This function is used to extract frames 
    # from particular camera and display 
    #------------------------------------------    
    def __output_ai_info_unsync(self):
        while True:
            try:
                if not self.running:
                    break
                #----------------------------
                # Wait until get new frame
                #----------------------------
                with self.c_raw_output_ai_info_unsync:
                    self.c_raw_output_ai_info_unsync.wait()
                       
                       
                #----------------------------
                # Get frames from raw buffer
                #----------------------------   
                frames,previous_info,camera = self.raw_extract_camera_buffer.get() 
                
                
                #----------------------------
                # Put frames into processed 
                # buffer
                #---------------------------- 
                if (self.processed_extract_camera_buffer[camera.id] is not None):
                    for frame in frames:
                        self.processed_extract_camera_buffer[camera.id].put(frame)
                        
                        if self.processed_extract_camera_buffer[camera.id].full():
                            self.processed_extract_camera_buffer[camera.id].get()
                    # print("Display buffer  ",self.raw_extract_camera_buffer.qsize())
            except Exception as e:
                print("------- __output_ai_info_unsync Exception " + str(e))
    #==========================================
    # This function is used to start
    # the thread
    #------------------------------------------   
    def start(self,):
        self.running = True
        self.output_ai_info_unsync.start()
        
    #==========================================
    # This function is used to stop
    # the thread
    #------------------------------------------  
    def stop(self,):
        self.running= False


            
            
                
                
    
