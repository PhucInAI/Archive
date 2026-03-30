import threading
import queue
import time
from ai_functions.utility.ai_logger import aiLogger
from ai_functions.utility.config    import BufferConfig
from ai_functions.utility.config    import AiProcessorConfig

class MergeYoloApp2Output():
    def __init__(self,availableYoloApp):
        
        #-----------------------------  
        # Create thread and others
        #-----------------------------
        self.application2output = threading.Thread(target=self.__merge_application2output, daemon=False) 
        self.__running          = False
        self.cameras            = []
        self.__availableYoloApp = availableYoloApp
        buffConfig              = BufferConfig()
        sysConfig               = AiProcessorConfig()

        

        #----------------------------
        # Contain the stable meta before
        # sending
        #----------------------------
        # self.yoloMeta                       = {}
        # self.yoloMeta['heat_map']           = []
        # self.yoloMeta['yolo_app_center']    = []
        # self.yoloMeta['object_counter']     = []




        #----------------------------
        # Contain the meta data
        #----------------------------
        
        self.yoloMetaType   = sysConfig.yoloMetaType

        self.returnedMeta   = {}
        for metaType in self.yoloMetaType:
            self.returnedMeta[metaType] = []

        #----------------------------
        # Condition object
        #----------------------------
        self.rawCondition        = threading.Condition()
        self.c_processed_merge_application2output  = threading.Condition()
    
        
        #----------------------------
        # Buffer
        #----------------------------
        self.rawBuffer       = {}
        self.processed_app2out_buffer = buffConfig.processedBuffer
        
        
        
    #==========================================
    # This function is used to start the thread
    #------------------------------------------         
            
    def start(self):
        self.__running = True
        self.application2output.start()
        
        
    #==========================================
    # This function is used to stop the thread
    #------------------------------------------   
    def stop(self):
        self.__running = False 
        
    #==========================================
    # This function is used to merge information
    # from cameras into one 
    #------------------------------------------     
    def __merge_application2output(self):
        while True:
            try:
                #-------------------------------
                # wait till get new frame from 
                # backbone thread
                #-------------------------------
                with self.rawCondition:
                    self.rawCondition.wait()
                
                self.reset_meta()
                #-------------------------------
                # get the frame from buffers
                #------------------------------- 
                for camera in self.cameras:
                    for app in self.__availableYoloApp:

                        # -------------------------
                        # Get all meta info from buffer
                        # before assigning to avoid
                        # losing data
                        # -------------------------
                        while not self.rawBuffer[camera.id][app].empty():
                            temp_base_info, utility_info = self.rawBuffer[camera.id][app].get()
                            # aiLogger.debug("The utility in {}  {} \n Meta: {}".format(app,utility_info.keys(),utility_info['meta_data']))
                            
                            appBatchMeta = utility_info[app]
                            # --------------------
                            # Get meta Object counter
                            if app == 'object_counter':
                                # ----------------
                                # Merge meta from each
                                # element in a batch into
                                # one  
                                # ----------------
                                for singleMeta in appBatchMeta:
                                    for object in singleMeta:
                                        if (object['metadata']['type'] == 'person'):
                                            self.returnedMeta['human_counting'].append(object)

                                        elif (object['metadata']['type'] == 'vehicle'):
                                            self.returnedMeta['vehicle_counting'].append(object)

                                        elif object['metadata']['type']  == 'tracking':
                                            self.returnedMeta['human_multiple_camera_tracking'].append(object)

                            else:
                                for singleMeta in appBatchMeta:
                                    # aiLogger.debug(singleMeta,app)
                                    if len(singleMeta):
                                        self.returnedMeta[app].append(singleMeta)

                            # # --------------------
                            # # Get meta from heat map
                            # elif app == 'heat_map':
                            #     # ----------------
                            #     # Get meta from batch
                            #     for singleMeta in appBatchMeta:
                            #         if len(singleMeta):
                            #             self.returnedMeta['heat_map'].append(singleMeta)
                            
                            # # --------------------
                            # # Get meta from crowd detection
                            # elif app == 'crowd_detection':
                            #     for singleMeta in appBatchMeta:
                            #         if len(singleMeta):
                            #             self.returnedMeta['crowd_detection'].append(singleMeta)

                            # # --------------------
                            # # Get meta from virtual fence
                            # elif app == 'virtual_fences':
                            #     for singleMeta in appBatchMeta:
                            #         if len(singleMeta):
                            #             self.returnedMeta['virtual_fences'].append(singleMeta)




                        
                        # -------------------------
                        # Get frames from yolo app
                        # center
                        # -------------------------
                        if app == 'yolo_app_center':
                            base_info = temp_base_info
                            
            
                    # aiLogger.debug(self.returnedMeta)
                    #-------------------------------
                    # Put frames into processed buff
                    #------------------------------- 
                    self.processed_app2out_buffer.put((base_info,self.returnedMeta))

                    #-------------------------------
                    # get the frame if the buffer is 
                    # full
                    #-------------------------------
                    if self.processed_app2out_buffer.full():
                        self.processed_app2out_buffer.get()
                        
                    #-------------------------------
                    # Not using notifyAll here because
                    # the notifyAll notified at backbone
                    #-------------------------------
                    with self.c_processed_merge_application2output:
                        self.c_processed_merge_application2output.notifyAll()
                        
                    
                # #-------------------------------
                # # Get process time and frame count to calculate FPS
                # #-------------------------------
                # self.__fps_log.measure_bottleneck[camera.id]['time']+=time.time()-curTime
                # self.__fps_log.measure_bottleneck[camera.id]['fps']+=1      
                    
                if not self.__running:
                    break   
            except Exception as e:
                aiLogger.exception("__merge_application2output Exception" + str(e))
                # print("------- __merge_application2output Exception " + str(e))
          

    #==========================================
    # Reset the meta dictionary
    #------------------------------------------
    def reset_meta(self):
        self.returnedMeta   = {}
        for metaType in self.yoloMetaType:
            self.returnedMeta[metaType] = []


    #==========================================
    # This function is used to add camera in to
    # merge_app function
    #------------------------------------------         
    def add_cameras(self,added_cameras):
        for camera in added_cameras:
            self.rawBuffer[camera.id] = {}
            for app in self.__availableYoloApp:
                self.rawBuffer[camera.id][app] = None

        self.cameras += added_cameras    
            
    def remove_cameras(self,rm_camera_id):
        for cur_cam in self.cameras[:]:
            if cur_cam.id in rm_camera_id:
                self.cameras.remove(cur_cam)  
                time.sleep(0.5)
                del self.rawBuffer[cur_cam.id]
                
                
        
            
            
        
        
                