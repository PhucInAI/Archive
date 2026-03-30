#=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=--=
#                       PLEASE DECLARE YOUR LIBRARY BELOW
#=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=--=
from ai_functions.utility.ai_logger import aiLogger
import time
import torch
import os
import yaml

# --------------------------
# IMPORT INTERFACE THREADS
# --------------------------
from camera.camera_manager                                      import Camera
from ai_functions.utility.config                                import AiProcessorConfig

# --------------------------
# IMPORT BASE SYSTEM
# --------------------------
from ai_functions.base_system.input_frame2ai                     import InputFrame2Ai
from ai_functions.base_system.output_ai_info_unsync              import OutputAiInfoUnsync
from ai_functions.base_system.expand_backbone_2_features         import ExpandBackbone2Features
from ai_functions.base_system.expand_feature2application         import ExpandFeature2Application
from ai_functions.base_system.output_ai_info_sync                import OutputAiInfoSync
from ai_functions.base_system.merge_yolo_app2_output             import MergeYoloApp2Output
from ai_functions.base_system.fps_control                        import FpsControl
# from ai_functions.reid.tracking_engine                           import TrackingEngine
from ai_functions.utility.utils                                  import get_GPU_available
from ai_functions.utility.fps_log                                import FpsLog

#=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=--=

class AiProcessor:
    def __init__(self, data_folder_path, model_path, isTensorrt=False):

        print("Init AI Processor")

        config = AiProcessorConfig()
        
        #=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        #           DECLARE SOME VARIABLES FOR AI 
        #
        #=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        cfg_hydranet= model_path + 'configs/Hydranet_engine.yml'
        with open(cfg_hydranet) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        torch.backends.cudnn.benchmark      = cfg['cudnn_params']['benchmark']
        torch.backends.cudnn.enabled        = cfg['cudnn_params']['enabled']
        torch.backends.cudnn.deterministic  = cfg['cudnn_params']['deterministic']
        #----------------------------------
        # Init general variable
        #----------------------------------
        
        self.__processRate                  = config.processRate
        self.__batch_size                   = config.batch_size
        self.__gpu_for_model                = config.gpu_for_model
        self.__fps_log                      = None
        gpu_idx,self.is_server              = get_GPU_available()
        
        self.gpu_is_available               = True if gpu_idx >=0 else False
        device                              = "cuda:" + str(gpu_idx) if self.gpu_is_available  else "cpu"
        self.__device                       = torch.device(device)
        self.__tensorrt                     = isTensorrt

        # self.__device                       = torch.device("cuda:2")
        #----------------------------------
        # path
        #----------------------------------
        if (not model_path.endswith('/')):
            model_path = model_path + '/'
        
        if (not data_folder_path.endswith('/')):
            data_folder_path = data_folder_path + '/'
        os.makedirs(data_folder_path, exist_ok=True)
        
        self.__model_path   = model_path
        self.__data_path    = data_folder_path
        #-----------CAMERA LIST

        # self.__cameras = []

        #----------------------------------
        # General ai variable
        #----------------------------------
        self.__available_features           = ['yolo5','fire_classification','retinaface','without_ai'] # 'fire_classification'

        self.__availableYoloApp             = [ 'yolo_app_center',
                                                'object_counter',
                                                'heat_map',
                                                'way_driving_violation_detection',
                                                'vehicle_speed_violation',
                                                # 'heat_map_direction',
                                                'crowd_detection',
                                                'virtual_fences',
                                                'traffic_jam_detection',
                                                'parking_violation_detection',
                                                'red_traffic_light']
                                                    


        self.__backboneThread               = None
        self.__backbone2FeaturesThread      = None
        self.__inputFrame2AiThread          = None
        #----------------------------------
        # Init for Yolo5
        #----------------------------------
        self.__yolo5NeckHeadThread          = None

        self.__licensePlateThread           = None
        self.__expandYoloNeckHead2ApplicationThread   = None 
        self.__tracking                     = {}
        self.__statistic                    = {}
        self.__track2YoloApp                = {}
        self.__yoloApp                      = {}

        #----------------------------------
        # Neck head of retina, deeplab,
        # without ai
        #----------------------------------
        self.__retinaFaceNeckHeadThread         = None
        self.__deeplabV3pNeckHeadThread         = None
        self.__withoutAiNeckHeadThread          = None
        self.__fireClassificationNeckHeadThread = None
        
        #----------------------------------
        # Display and synchronize 
        #----------------------------------
        self.__use_internal_display         = config.using_internal_display
        self.__sync_thread_output           = config.sync_thread_output 
        
        
        #----------------------------------
        # Output to connect with VMS
        #----------------------------------
        self.__outputAiInfoSync          = None
        self.__output_ai_info_unsync     = {}
        
        
        #=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        #           DECLARE SOME CLASSES FOR AI 
        #
        #=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

        nn_config_path = self.__model_path + '/reid/' + f"configs/person_search_engine.yml"
        with open(nn_config_path) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)

        # self.__person_reid = TrackingEngine(cfg, self.__device, False, self.is_server, None, self.__model_path, self.__data_path)
        self.reid_time = 0
        #----------------------------------
        # Using this one to measure the 
        # time and fps of each block
        #----------------------------------
        self.__fps_log = FpsLog(self.__available_features,
                                self.__availableYoloApp,
                                self.__batch_size,
                                self.__device)
                                            
        #----------------------------------
        # Initialize backbone thread 
        # and ExpandBackbone2Features thread
        # and FpsControl
        #----------------------------------
        self.__inputFrame2AiThread              = InputFrame2Ai(self.__fps_log,
                                                                self.__batch_size,
                                                                self.__processRate)        
                                                                                                                                     
        # self.__backboneThread                   = BackboneThread(self.__fps_log,
        #                                                          self.__device,
        #                                                          self.__tensorrt,
        #                                                          self.is_server,
        #                                                          self.__model_path)
                                                                                
        # self.__backbone2FeaturesThread          = ExpandBackbone2Features(  self.__available_features,
        #                                                                     self.__fps_log,
        #                                                                     self.__sync_thread_output)           # To avoid bottleneck after backbone thread    

        # #----------------------------------
        # # Threads for yolo5
        # #---------------------------------- 
        # if 'yolo5' in self.__available_features:
        #     self.__yolo5NeckHeadThread                      =   NeckHeadYolo5Thread(self.__fps_log,
        #                                                                             self.__use_internal_display,
        #                                                                             self.__device,
        #                                                                             self.__tensorrt,
        #                                                                             self.is_server,
        #                                                                             self.__data_path,
        #                                                                             self.__model_path)         # Contain neck and head of Yolo5     

        #     self.__licensePlateThread                       =   LicensePlateThread( self.__fps_log,
        #                                                                             self.__use_internal_display,
        #                                                                             self.__device,
        #                                                                             self.__tensorrt,
        #                                                                             self.is_server,
        #                                                                             self.__data_path,
        #                                                                             self.__model_path,
        #                                                                             )
        #     self.__expandYoloNeckHead2ApplicationThread     =   ExpandFeature2Application()
        #     self.__yoloApp2OutThread                        =   MergeYoloApp2Output(self.__availableYoloApp)


        # #----------------------------------
        # # Threads for retinaface
        # #----------------------------------
        # if 'retinaface' in self.__available_features:
        #     self.__retinaFaceNeckHeadThread     =  NeckHeadRetinaFaceThread(self.__fps_log,
        #                                                                     self.__use_internal_display,
        #                                                                     self.__device,
        #                                                                     self.__tensorrt,
        #                                                                     self.is_server,
        #                                                                     self.__data_path,
        #                                                                     self.__model_path)         # Contain neck and head of retinaface
            
        # #----------------------------------
        # # Threads for deeplabv3p
        # #----------------------------------
        # if 'deeplabv3p' in self.__available_features:
        #     self.__deeplabV3pNeckHeadThread     =  NeckHeadDeepLabV3P(self.__fps_log,
        #                                                               self.__device,
        #                                                               self.__model_path)               # Contain neck and head of 
            
        # #----------------------------------
        # # Threads without Ai
        # #----------------------------------
        # if 'without_ai' in self.__available_features:
        #     self.__withoutAiNeckHeadThread      = NeckHeadWithoutAiThread(self.__fps_log)
            

        # #----------------------------------
        # # Threads for fire
        # #----------------------------------
        # if 'fire_classification' in self.__available_features:
        #     self.__fireClassificationNeckHeadThread =  NeckHeadFireClassificationThread(self.__fps_log,
        #                                                                                 self.__batch_size,
        #                                                                                 self.__use_internal_display,
        #                                                                                 self.__device,
        #                                                                                 self.__data_path,
        #                                                                                 self.__model_path)  # Contain neck and head of 
            
        #----------------------------------
        # Output of the system, there are
        # 2 options consisting of sync and 
        # not sync. The sync will return
        # the result with results of the 
        # same frame, meanwhile the 
        # opposite for not sync option
        #----------------------------------    
        if self.__sync_thread_output:
            self.__outputAiInfoSync  = OutputAiInfoSync(self.__fps_log,
                                                        self.__available_features,
                                                        self.__batch_size,
                                                        self.__device)
        else:
            for feature in self.__available_features:
                self.__output_ai_info_unsync[feature]   = OutputAiInfoUnsync(self.__available_features)
            

        #----------------------------------
        # Connect all block in the system
        # Notice: Each block is isolated and 
        # need this one to connect
        #----------------------------------
        self.__all_connection(  self.__sync_thread_output,
                                self.__available_features,
                                
                                self.__inputFrame2AiThread,
                                self.__backboneThread,
                                self.__backbone2FeaturesThread,
                                self.__expandYoloNeckHead2ApplicationThread,
                                self.__yoloApp2OutThread,

                                self.__yolo5NeckHeadThread,
                                self.__retinaFaceNeckHeadThread,
                                self.__deeplabV3pNeckHeadThread,
                                self.__withoutAiNeckHeadThread,
                                self.__fireClassificationNeckHeadThread,

                                self.__licensePlateThread,
                                
                                self.__output_ai_info_unsync,    
                                self.__outputAiInfoSync 
                                )  

        aiLogger.info("The ai model at the process {} was loaded successfully!!!".format(os.getpid()))   
    #==========================================
    # This function used to connect 4 blocks 
    # consisting of GetCameraFrames,
    # DisplayProcessedFrame, AiThread, 
    # ExtractCamera2Display together
    #------------------------------------------
    def __all_connection(   self,
                            sync_thread_output,
                            available_features,
                            
                            inputFrame2AiThread,
                            backboneThread,
                            backbone2FeaturesThread,
                            expandYoloNeckHead2ApplicationThread,
                            yoloApplication2OutputThread,
                        
                            yolo5NeckHeadThread,
                            retinaFaceNeckHeadThread,
                            deeplabV3pNeckHeadThread,
                            withoutAiNeckHeadThread,
                            fireClassificationNeckHeadThread,

                            licensePlateThread,
                            
                            output_ai_info_unsync,
                            outputAiInfoSync,
                            ):
        #--------------------------------
        # Connect FpsControlThread and 
        # BackboneThread 
        #--------------------------------
        backboneThread.c_raw_backbone      = inputFrame2AiThread.c_processed_fps_control
        backboneThread.raw_backbone_buffer = inputFrame2AiThread.processed_fps_control_buffer
        
        #--------------------------------
        # Connect BackboneThread and 
        # ExpandBackbone2Features 
        #--------------------------------
        backbone2FeaturesThread.raw_rl_bottleneck_buffer     = backboneThread.processed_backbone_buffer 
        backbone2FeaturesThread.c_raw_rl_bottleneck          = backboneThread.c_processed_backbone
        
        #--------------------------------
        # Connect BackboneThread and 
        # outputAiInfoSync to syschonize
        #--------------------------------
        if sync_thread_output:
            outputAiInfoSync.c_raw_ai_info_output = backboneThread.c_processed_backbone

        #--------------------------------
        # Connect Yolo5 into the system
        #-------------------------------- 
        if 'yolo5' in available_features:   
            #---------------------------- 
            # Connect ExpandBackbone2Features
            # and Yolo5 neck head
            #----------------------------
            yolo5NeckHeadThread.c_raw_neckhead                              = backbone2FeaturesThread.c_processed_rl_bottleneck['yolo5']
            backbone2FeaturesThread.processed_rl_bottleneck_buffer['yolo5'] = yolo5NeckHeadThread.raw_feature_buffer   
            
            #--------------------------------
            # Connect NeckHeadYolo5Thread and 
            # LicensePlateThread
            #--------------------------------
            licensePlateThread.rawCondition     =  yolo5NeckHeadThread.c_processed_neckhead
            licensePlateThread.rawBuffer        =  yolo5NeckHeadThread.processed_feature_buffer

            #--------------------------------
            # Connect LicensePlateThread and 
            # ExpandFeature2Application
            #--------------------------------
            expandYoloNeckHead2ApplicationThread.c_raw_feature2application          = licensePlateThread.processedCondition
            expandYoloNeckHead2ApplicationThread.raw_feature2application_buffer     = licensePlateThread.processedBuffer
                
            #--------------------------------
            # If using_internal_display,
            # connect with internal display
            # 
            #--------------------------------
            if sync_thread_output:
                outputAiInfoSync.raw_ai_info_buffer['yolo5'] = yoloApplication2OutputThread.processed_app2out_buffer

        #--------------------------------
        # Connect Retinaface into the system
        #-------------------------------- 
        if 'retinaface' in available_features:   
            #--------------------------------
            # Connect ExpandBackbone2Features
            # and NeckHeadRetinaFaceThread
            #-------------------------------- 
            retinaFaceNeckHeadThread.c_raw_neckhead      = backbone2FeaturesThread.c_processed_rl_bottleneck['retinaface']
            backbone2FeaturesThread.processed_rl_bottleneck_buffer['retinaface'] = retinaFaceNeckHeadThread.raw_feature_buffer
              
            if not sync_thread_output:
                #--------------------------------
                # Connect NeckHeadRetinaFaceThread
                #  and integration
                #--------------------------------
                output_ai_info_unsync['retinaface'].raw_extract_camera_buffer       = retinaFaceNeckHeadThread.processed_feature_buffer      
                output_ai_info_unsync['retinaface'].c_raw_output_ai_info_unsync     = retinaFaceNeckHeadThread.c_processed_neckhead    
            else:
                outputAiInfoSync.raw_ai_info_buffer['retinaface'] = retinaFaceNeckHeadThread.processed_feature_buffer   
            
        #--------------------------------
        # Connect 'deeplabv3p' into the system
        #-------------------------------- 
        if 'deeplabv3p' in available_features:  
            #--------------------------------
            # Connect ExpandBackbone2Features
            # and NeckHeadYoloPhucThread
            #--------------------------------  
            deeplabV3pNeckHeadThread.c_raw_neckhead                               = backbone2FeaturesThread.c_processed_rl_bottleneck['deeplabv3p']
            backbone2FeaturesThread.processed_rl_bottleneck_buffer['deeplabv3p']  = deeplabV3pNeckHeadThread.raw_feature_buffer 
            
            if not sync_thread_output:
                #--------------------------------
                # Connect NeckHeadYoloPhucThread
                #  and output_ai_info_unsync
                #--------------------------------
                output_ai_info_unsync['deeplabv3p'].raw_extract_camera_buffer        = deeplabV3pNeckHeadThread.processed_feature_buffer      
                output_ai_info_unsync['deeplabv3p'].c_raw_output_ai_info_unsync     = deeplabV3pNeckHeadThread.c_processed_neckhead      

            else:
                pass    
                
        #--------------------------------
        # Connect 'without_ai' into the system
        #-------------------------------- 
        if 'without_ai' in available_features:  
            #--------------------------------
            # Connect ExpandBackbone2Features
            # and NeckHeadYoloPhucThread
            #--------------------------------  
            withoutAiNeckHeadThread.c_raw_neckhead                               = backbone2FeaturesThread.c_processed_rl_bottleneck['without_ai']
            backbone2FeaturesThread.processed_rl_bottleneck_buffer['without_ai'] = withoutAiNeckHeadThread.raw_feature_buffer 
            
            if not sync_thread_output:
                #--------------------------------
                # Connect NeckHeadYoloPhucThread
                #  and output_ai_info_unsync
                #--------------------------------
                output_ai_info_unsync['without_ai'].raw_extract_camera_buffer       = withoutAiNeckHeadThread.processed_feature_buffer      
                output_ai_info_unsync['without_ai'].c_raw_output_ai_info_unsync     = withoutAiNeckHeadThread.c_processed_neckhead      
  
            else:
                outputAiInfoSync.raw_ai_info_buffer['without_ai'] = withoutAiNeckHeadThread.processed_feature_buffer       
                        
        
        #--------------------------------
        # Connect Fire into the system
        #-------------------------------- 
        if 'fire_classification' in available_features:   
            #--------------------------------
            # Connect ReleaseFeatureBottleneck
            # and NeckHeadRetinaFaceThread
            #-------------------------------- 
            fireClassificationNeckHeadThread.c_raw_neckhead                               = backbone2FeaturesThread.c_processed_rl_bottleneck['fire_classification']
            backbone2FeaturesThread.processed_rl_bottleneck_buffer['fire_classification'] = fireClassificationNeckHeadThread.raw_feature_buffer
              
            if not sync_thread_output:
                #--------------------------------
                # Connect NeckHeadRetinaFaceThread
                #  and integration
                #--------------------------------
                output_ai_info_unsync['fire_classification'].raw_extract_camera_buffer       = fireClassificationNeckHeadThread.processed_feature_buffer      
                output_ai_info_unsync['fire_classification'].c_raw_output_ai_info_unsync     = fireClassificationNeckHeadThread.c_processed_neckhead    
            else:
                outputAiInfoSync.raw_ai_info_buffer['fire_classification'] = fireClassificationNeckHeadThread.processed_feature_buffer     
                 
                    
    #==========================================
    # This function is used to start the Ai 
    # system
    #------------------------------------------                
    def start(self,):
        try:
            #----------------------------------
            # Start Yolo5
            #----------------------------------
            if 'yolo5' in self.__available_features:
                self.__yolo5NeckHeadThread.start()
                self.__licensePlateThread.start()
                self.__expandYoloNeckHead2ApplicationThread.start()
                self.__yoloApp2OutThread.start()

            #----------------------------------
            # Start retinaface
            #----------------------------------
            if 'retinaface' in self.__available_features:
                self.__retinaFaceNeckHeadThread.start()

            #----------------------------------
            # Start deeplabv3p
            #----------------------------------
            if 'deeplabv3p' in self.__available_features:
                self.__deeplabV3pNeckHeadThread.start()
                
            #----------------------------------
            # Start without_ai thread
            #----------------------------------
            if 'without_ai' in self.__available_features:
                self.__withoutAiNeckHeadThread.start()
                
            if 'fire_classification' in self.__available_features:
                self.__fireClassificationNeckHeadThread.start()

            #----------------------------------
            # Start this one if display option
            # was turned on
            #----------------------------------
            if self.__sync_thread_output:
                self.__outputAiInfoSync.start()
            
            else:
                for feature in self.__available_features: 
                    self.__output_ai_info_unsync[feature].start()
            
            #----------------------------------
            # Start backboneThread,  
            # backbone2FeaturesThread
            # and fps_log
            #----------------------------------
            self.__inputFrame2AiThread.start()
            self.__backboneThread.start()  
            self.__backbone2FeaturesThread.start()
            self.__fps_log.start()
            
            aiLogger.warning("Ai core was started successfully!!!")
            
        except Exception as e:
            aiLogger.exception("------- ai_processor start() attribute Exception " + str(e))  

    #==========================================
    # This function is used to stop the Ai 
    # system
    #------------------------------------------
    def stop(self,):
        aiLogger.warning("The Ai core was stopped!!!!")
        
        
        
    #==========================================
    # This function is used to release the Ai 
    # system
    #------------------------------------------
    def release(self,):
        try:
            aiLogger.warning("The Ai core will be released soon !!! Please wait...")

            #----------------------------------
            # Stop backboneThread,  
            # backbone2FeaturesThread
            # and __inputFrame2AiThread
            #----------------------------------
            self.__inputFrame2AiThread.stop()
            time.sleep(1.5)                       # Wait to other threads complete their processing
            self.__backboneThread.stop() 
            self.__backbone2FeaturesThread.stop()

            #----------------------------------
            # Stop Yolo5
            #----------------------------------
            if 'yolo5' in self.__available_features:
                self.__yolo5NeckHeadThread.stop()
                self.__licensePlateThread.stop()
                self.__expandYoloNeckHead2ApplicationThread.stop()
                self.__yoloApp2OutThread.stop()
                
                for camera in self.__inputFrame2AiThread.cameras:
                    self.__tracking[camera.id].stop() 
                    self.__statistic[camera.id].stop()

            #----------------------------------
            # Stop retinaface
            #----------------------------------
            if 'retinaface' in self.__available_features:
                self.__retinaFaceNeckHeadThread.stop()

            #----------------------------------
            # Stop yolo_phuc
            #----------------------------------
            if 'deeplabv3p' in self.__available_features:
                self.__deeplabV3pNeckHeadThread.stop()
                
            #----------------------------------
            # Stop without_ai thread
            #----------------------------------
            if 'without_ai' in self.__available_features:
                self.__withoutAiNeckHeadThread.stop()
                
            
            if 'fire_classification' in self.__available_features:
                self.__fireClassificationNeckHeadThread.stop()

            #----------------------------------
            # Stop this one if display option
            # was turned on
            #----------------------------------
            self.__outputAiInfoSync.stop()
                
            #----------------------------------
            # Stop timer to measure threads, and
            # notify all threads that still in the
            # wait command
            #----------------------------------
            self.__fps_log.calculation_thread.cancel()
            self.__notify_to_stop()                 
            time.sleep(1.5)
            
            #----------------------------------
            # Check alive thread
            #----------------------------------
            # print(self.__backboneThread._ai_thread.is_alive())
            # print(self.__backbone2FeaturesThread.release_bottleneck.is_alive())
            # print(self.__yolo5NeckHeadThread._ai_feature_thread.is_alive())
            # print(self.__expandYoloNeckHead2ApplicationThread.feature2application.is_alive())
            # print(self.__yoloApp2OutThread.application2output.is_alive())
            # print(self.__retinaFaceNeckHeadThread._ai_feature_thread.is_alive())
            # print(self.__withoutAiNeckHeadThread._ai_feature_thread.is_alive())
            # print(self.__fireClassificationNeckHeadThread._ai_feature_thread.is_alive())
            # print(self.__outputAiInfoSync.ai_information.is_alive())
            # for camera in self.__inputFrame2AiThread.cameras:
            #     print("tracking:",self.__tracking[camera.id]._ai_tracking_thread.is_alive())
            #     print(self.__statistic[camera.id]._ai_application_thread.is_alive())
                
            #----------------------------------
            # Delete thread
            #----------------------------------
            del self.__inputFrame2AiThread
            del self.__backboneThread
            del self.__yolo5NeckHeadThread
            del self.__licensePlateThread
            del self.__expandYoloNeckHead2ApplicationThread
            del self.__yoloApp2OutThread
            del self.__retinaFaceNeckHeadThread
            del self.__deeplabV3pNeckHeadThread
            del self.__withoutAiNeckHeadThread
            del self.__fireClassificationNeckHeadThread
            del self.__outputAiInfoSync
            del self.__tracking
            del self.__statistic
            # del self.__fps_log
        
            aiLogger.warning("The Ai core at the process {} was released successfully!!!".format(os.getpid()))
        except Exception as e:
            aiLogger.exception("------- ai_processor release() attribute Exception " + str(e))    

    #==========================================
    # This function is used to notify to break 
    # the while loop before stop completely
    #------------------------------------------
    def __notify_to_stop(self,):
        #----------------------------------
        # InputFrame2Ai
        #----------------------------------
        if self.__inputFrame2AiThread.input_input_frame2ai.is_alive():
            with self.__inputFrame2AiThread.c_raw_fps_control:
                self.__inputFrame2AiThread.c_raw_fps_control.notifyAll()
                
        #----------------------------------
        # BackboneThread
        #----------------------------------
        if self.__backboneThread._ai_thread.is_alive():
            with self.__backboneThread.c_raw_backbone:
                self.__backboneThread.c_raw_backbone.notifyAll()
                
        #----------------------------------
        # ExpandBackbone2Features
        #----------------------------------
        if self.__backbone2FeaturesThread.release_bottleneck.is_alive():
            with self.__backbone2FeaturesThread.c_raw_rl_bottleneck:
                self.__backbone2FeaturesThread.c_raw_rl_bottleneck.notifyAll()
            
            
        #----------------------------------
        # YOLO
        #----------------------------------        
        if 'yolo5' in self.__available_features:
            #----------------------------------
            # NeckHeadYolo5Thread
            #----------------------------------
            if self.__yolo5NeckHeadThread._ai_feature_thread.is_alive():
                with self.__yolo5NeckHeadThread.c_raw_neckhead:
                    self.__yolo5NeckHeadThread.c_raw_neckhead.notifyAll()

            #----------------------------------
            # LicensePlateThread
            #----------------------------------      
            if self.__licensePlateThread._ai_application_thread.is_alive():
                with self.__licensePlateThread.rawCondition:
                    self.__licensePlateThread.rawCondition.notifyAll()

            #----------------------------------
            # ExpandFeature2Application
            #----------------------------------  
            if self.__expandYoloNeckHead2ApplicationThread.feature2application.is_alive():
                with self.__expandYoloNeckHead2ApplicationThread.c_raw_feature2application:
                    self.__expandYoloNeckHead2ApplicationThread.c_raw_feature2application.notifyAll()
                    
            #----------------------------------
            # MergeYoloApp2Output
            #----------------------------------  
            if self.__yoloApp2OutThread.application2output.is_alive():
                with self.__yoloApp2OutThread.rawCondition:
                    self.__yoloApp2OutThread.rawCondition.notifyAll()
                    
                    
            #----------------------------------
            # TrackingThread and StatisticThread
            #----------------------------------
            for camera in self.__inputFrame2AiThread.cameras:
                if self.__tracking[camera.id]._ai_tracking_thread.is_alive():
                    with self.__tracking[camera.id].rawCondition:
                        self.__tracking[camera.id].rawCondition.notifyAll()
                        
                if self.__statistic[camera.id]._ai_application_thread.is_alive():
                    with self.__statistic[camera.id].rawCondition:
                        self.__statistic[camera.id].rawCondition.notifyAll()
                        
        #----------------------------------
        # RETINAFACE
        #---------------------------------- 
        if 'retinaface' in self.__available_features: 
            #----------------------------------
            # NeckHeadRetinaFaceThread
            #----------------------------------  
            if self.__retinaFaceNeckHeadThread._ai_feature_thread.is_alive():
                with self.__retinaFaceNeckHeadThread.c_raw_neckhead:
                    self.__retinaFaceNeckHeadThread.c_raw_neckhead.notifyAll()
        
        #----------------------------------
        # WITHOUT AI
        #----------------------------------        
        if 'without_ai' in self.__available_features:  
            #----------------------------------
            # NeckHeadWithoutAiThread
            #----------------------------------  
            if self.__withoutAiNeckHeadThread._ai_feature_thread.is_alive():
                with self.__withoutAiNeckHeadThread.c_raw_neckhead:
                    self.__withoutAiNeckHeadThread.c_raw_neckhead.notifyAll()
            
        #----------------------------------
        # FIRE CLASSIFICATION
        #----------------------------------         
        if 'fire_classification' in self.__available_features:  
            #----------------------------------
            # NeckHeadFireClassificationThread
            #----------------------------------  
            if self.__fireClassificationNeckHeadThread._ai_feature_thread.is_alive():
                with self.__fireClassificationNeckHeadThread.c_raw_neckhead:
                    self.__fireClassificationNeckHeadThread.c_raw_neckhead.notifyAll()
        
        #----------------------------------
        # OutputAiInfoSync
        #----------------------------------
        if self.__outputAiInfoSync.ai_information.is_alive():
            with self.__outputAiInfoSync.c_raw_ai_info_output:
                self.__outputAiInfoSync.c_raw_ai_info_output.notifyAll()
                
        
                
               
    #==========================================
    # This function is used to add camera 
    #------------------------------------------
    def add_camera( self,
                    raw_cameras,
                    in_buffer,
                    in_condition,
                    out_buffer,
                    out_condition):
        added_ai_cam = self.__convert_cam2_ai_option(raw_cameras)

        # self.__cameras.extend(added_ai_cam)
        curCamId = [camera.id for camera in self.__inputFrame2AiThread.cameras]

        print("\n\n")
        aiLogger.warning("======= ADDING CAMERA LOG ================")
        for camera in added_ai_cam.copy():
            if camera.id in curCamId:
                aiLogger.warning("The camera {} was existing in the system, please add another one".format(camera.name))
                added_ai_cam.remove(camera) 
            else:
                aiLogger.warning("Adding the camera {} into the AI system".format(camera.name))
        # aiLogger.info("======================================")
        try:
            #------------------------
            # Add cam for log
            #------------------------
            self.__fps_log.add_cameras(added_ai_cam)
            #------------------------
            # Add cam for output 
            #------------------------
            self.__outputAiInfoSync.add_cameras(added_ai_cam,out_buffer,out_condition)
            #------------------------
            # Add cam for Yolo branch
            #------------------------
            if 'yolo5' in self.__available_features:
                self.__yoloApp2OutThread.add_cameras(added_ai_cam)
                self.__add_cam_to_yolo_application(added_ai_cam)
                self.__expandYoloNeckHead2ApplicationThread.add_camera(added_ai_cam)
            #------------------------
            # Add cam for Fire branch
            #------------------------
            if 'fire_classification' in self.__available_features:
                self.__fireClassificationNeckHeadThread.add_camera(added_ai_cam)
            #------------------------
            # Update connection and
            # start new threads
            #------------------------
            self.__update_connection(added_ai_cam)
            self.__start_updated_threads(added_ai_cam)
            
            #------------------------
            # Add cam for input
            #------------------------
            self.__inputFrame2AiThread.add_camera(added_ai_cam,in_buffer,in_condition)   
            
            # print("Cameras was added successfully!!!")
            aiLogger.warning("Completed adding cameras !!!")
            aiLogger.warning("================================== \n\n")

        except Exception as e:
            aiLogger.exception("ai_processor add_camera() attribute Exception " + str(e))  
    
    
    #==========================================
    # This function is used to add yolo 
    # application thread into the system
    #------------------------------------------
    def __add_cam_to_yolo_application(self,added_ai_cam):
        for camera in added_ai_cam:
            if 'yolo5' in self.__available_features:

                self.__tracking[camera.id]                     = TrackingThread(    self.__fps_log,
                                                                                    self.__use_internal_display, 
                                                                                    self.__processRate, 
                                                                                    None, 
                                                                                    self.__data_path)   

                self.__track2YoloApp[camera.id]                = GeneralYoloAppInfoThread(  self.__fps_log,
                                                                                            self.__availableYoloApp)

                self.__yoloApp[camera.id]                      = {}
                self.__yoloApp[camera.id]['yolo_app_center']   = YoloAppCenterThread(   self.__fps_log,
                                                                                        camera,
                                                                                        self.__data_path,)
                self.__yoloApp[camera.id]['object_counter']    = ObjectCounterThread(   self.__fps_log,
                                                                                        self.__use_internal_display,
                                                                                        camera,
                                                                                        self.__data_path,
                                                                                        self.__person_reid
                                                                                        )
                self.__yoloApp[camera.id]['heat_map']          = HeatMapThread( self.__fps_log,
                                                                                camera,
                                                                                self.__data_path,
                                                                                )

                self.__yoloApp[camera.id]['crowd_detection']            = CrowdDetectionThread( self.__fps_log,
                                                                                                camera,
                                                                                                self.__data_path
                                                                                                )
                self.__yoloApp[camera.id]['virtual_fences']             = VirtualFenceThread(   self.__fps_log,
                                                                                                camera,
                                                                                                self.__data_path)

                self.__yoloApp[camera.id]['traffic_jam_detection']      =   TrafficJamThread(   self.__fps_log,
                                                                                                camera,
                                                                                                self.__data_path)
                
                self.__yoloApp[camera.id]['parking_violation_detection'] =   ParkingViolationThread(  self.__fps_log,
                                                                                                    camera,
                                                                                                    self.__data_path)

                self.__yoloApp[camera.id]['red_traffic_light']          = RedTrafficLightThread(self.__fps_log,
                                                                                                camera,
                                                                                                self.__data_path
                                                                                                )

                self.__yoloApp[camera.id]['way_driving_violation_detection'] = WayDrivingViolationThread(   self.__fps_log,
                                                                                                            camera,
                                                                                                            self.__data_path)
                
                self.__yoloApp[camera.id]['vehicle_speed_violation']        = VehicleSpeedThread(self.__fps_log,
                                                                                                            camera,
                                                                                                            self.__data_path)

                
    def __start_updated_threads(self,cameras):
        for camera in cameras:
            self.__tracking[camera.id].start()
            self.__track2YoloApp[camera.id].start()
            
            for app in self.__availableYoloApp:
                if app in self.__yoloApp[camera.id]:
                    self.__yoloApp[camera.id][app].start()


    # =========================================
    # This function is used to connect block 
    # together when adding the new camera
    # -----------------------------------------
    def __update_connection(self,cameras):
        for camera in cameras:
            #--------------------------------
            # Connect tracking and yolo neckhead
            #--------------------------------
            self.__tracking[camera.id].rawCondition             = self.__expandYoloNeckHead2ApplicationThread.processedCondition[camera.id]
            self.__tracking[camera.id].rawBuffer                = self.__expandYoloNeckHead2ApplicationThread.processedBuffer[camera.id]

            #--------------------------------
            # Connect tracking and statistic
            #--------------------------------
            # self.__statistic[camera.id].rawCondition      = self.__tracking[camera.id].processedCondition
            # self.__statistic[camera.id].rawBuffer         = self.__tracking[camera.id].processedBuffer
            
            #--------------------------------
            # Connect tracking and 
            # Memory
            #--------------------------------
            self.__track2YoloApp[camera.id].rawCondition        = self.__tracking[camera.id].processedCondition
            self.__track2YoloApp[camera.id].rawBuffer           = self.__tracking[camera.id].processedBuffer


            #--------------------------------
            # Connect Memory
            # and Yolo Applications
            #--------------------------------
            for app in self.__availableYoloApp:
                if app in self.__yoloApp[camera.id]:
                    self.__yoloApp[camera.id][app].rawCondition     = self.__track2YoloApp[camera.id].processedCondition
                    self.__yoloApp[camera.id][app].rawBuffer        = self.__track2YoloApp[camera.id].processedBuffer[app]

            #--------------------------------
            # Connect Yolo App and 
            # merge yolo2ouput 
            #--------------------------------
            for app in self.__availableYoloApp:
                if app in self.__yoloApp[camera.id]:
                    self.__yoloApp2OutThread.rawBuffer[camera.id][app]  = self.__yoloApp[camera.id][app].processedBuffer
            self.__yoloApp[camera.id]['yolo_app_center'].processedCondition   = self.__yoloApp2OutThread.rawCondition
                    




            #--------------------------------
            # Connect statistic and ouput sync
            #--------------------------------
            # self.__statistic[camera.id].processedCondition     = self.__yoloApp2OutThread.rawCondition
            # self.__yoloApp2OutThread.rawBuffer[camera.id] = self.__statistic[camera.id].processedBuffer 
            
            

    #==========================================
    # This function is used to remove camera out
    # of the system 
    #------------------------------------------    
    def remove_camera(self,cameras):
        print("\n\n")
        aiLogger.warning("======= REMOVING CAMERA LOG ================")
        
        removed_cameras = self.__convert_cam2_ai_option(cameras)
        for camera in removed_cameras:
            aiLogger.warning("Removing the camera {} out of the AI system".format(camera.name))

        try:
            rm_cam_id = [camera.id for camera in removed_cameras]

            #------------------------
            # Remove cam at input
            #------------------------
            self.__inputFrame2AiThread.remove_camera(rm_cam_id)
            time.sleep(1.5)                       # Wait to other threads complete their processing
            
            #------------------------
            # Remove cam at Yolo branch
            #-----------------------
            self.__expandYoloNeckHead2ApplicationThread.remove_camera(rm_cam_id)
            self.__remove_yolo_application_threads(rm_cam_id)
            self.__yoloApp2OutThread.remove_cameras(rm_cam_id)
            
            #------------------------
            # Remove cam at Fire branch
            #-----------------------
            self.__fireClassificationNeckHeadThread.remove_camera(removed_cameras)
            
            #------------------------
            # Remove cam at output
            #------------------------
            self.__outputAiInfoSync.remove_cameras(rm_cam_id)
            
            #------------------------
            # Remove cam at fpslog
            #------------------------
            self.__fps_log.remove_cameras(rm_cam_id)
            
            
            aiLogger.warning("Removed cameras out of AI system successfully!!!")
            aiLogger.warning("======================================\n\n")    
            
        except Exception as e:
            aiLogger.exception("------- ai_processor remove_camera() attribute Exception " + str(e))  

    def __remove_yolo_application_threads(self,rm_cam_id):
        for cur_cam in self.__inputFrame2AiThread.cameras[:]:
            if cur_cam.id in rm_cam_id:
                #-------------------------------
                # Remove application at each camera
                #-------------------------------
                if 'yolo5' in self.__available_features:
                    self.__tracking[cur_cam.id].stop()
                    self.__statistic[cur_cam.id].stop()

                    del self.__tracking[cur_cam.id]
                    del self.__statistic[cur_cam.id]
                
    #==========================================
    # This function is used to add or remove 
    # features from a particular camera
    #------------------------------------------
    def update_camera_features(self,updated_cameras):
        print("\n\n")
        aiLogger.warning("======= UPDATING CAMERA LOG ============")
        aiLogger.warning("Updating cameras...!!! Please wait...!!!")
        
        # ------------------
        # Convert original cam
        # to AI cam
        # ------------------
        updated_cameras = self.__convert_cam2_ai_option(updated_cameras)

        # ------------------
        # Update camera's feature
        # ------------------
        updateInCurCam = False
        for idx,cur_camera in enumerate(self.__inputFrame2AiThread.cameras):
            for updated_camera in updated_cameras:
                if updated_camera.id == cur_camera.id:
                    updateInCurCam = True
                    
                    self.__inputFrame2AiThread.cameras[idx] = updated_camera
                    newProcessRate = self.__inputFrame2AiThread.update_utility(updated_camera)


                    self.__outputAiInfoSync.cameras[idx]    = updated_camera
                    self.__fps_log.cameras[idx]             = updated_camera
                    self.__statistic[cur_camera.id].update_setting(updated_camera)

                    self.__retinaFaceNeckHeadThread.update_process_rate(newProcessRate, updated_camera)

                    aiLogger.warning(" The camera {} was updated into AI system successfully".format(cur_camera.name))

        if not updateInCurCam:
            aiLogger.warning("The updated camera did not appear in the AI camera collection, please check again!!!")

        aiLogger.warning("====================================== \n\n")
                
    
    
    #==========================================
    # This function is used to convert raw camera
    # info to ai option
    #------------------------------------------
    def __convert_cam2_ai_option(self,cameras):
        ai_cameras = []
        for camera in cameras:
            ai_cameras.append(Camera(camera))
            
        return ai_cameras
                    
            
    def update_face_bank(self,face_bank_list=[],config="add"):
                           #[{ 
                           #     "id" "duc",
                           #     "images": ["ai_core/face_recognition/ArcFace/resources/registers/images/Duc/IMG_3982.jpg",
                           #                 "ai_core/face_recognition/ArcFace/resources/registers/images/Duc/IMG_3983.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Duc/IMG_3984.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Duc/IMG_3985.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Duc/IMG_3986.jpg"   
                           #                 ],
                           #     "features": ["ai_core/face_recognition/ArcFace/resources/registers/features/Duc/IMG_3982.npy",
                           #                 "ai_core/face_recognition/ArcFace/resources/registers/features/Duc/IMG_3983.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Duc/IMG_3984.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Duc/IMG_3985.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Duc/IMG_3986.npy"   
                           #                 ]
                           #  },
                           #  {"id" "Bang",
                           #     "images": ["ai_core/face_recognition/ArcFace/resources/registers/images/Bang/1.jpg",
                           #                 "ai_core/face_recognition/ArcFace/resources/registers/images/Bang/2.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Bang/3.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Bang/4.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Bang/5.jpg"   
                           #                 ],
                           #     "features": ["ai_core/face_recognition/ArcFace/resources/registers/features/Bang/1.npy",
                           #                 "ai_core/face_recognition/ArcFace/resources/registers/features/Bang/2.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Bang/3.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Bang/4.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Bang/5.npy" 
                           #   }]*/
        if config=="add":
            self.__retinaFaceNeckHeadThread.face_recognition._face_recognition.face_en.add_face(face_bank_list)
        elif (config=="delete"):
            self.__retinaFaceNeckHeadThread.face_recognition._face_recognition.face_en.delete_face(face_bank_list)

        
    def update_human_bank(self,human_bank_list=[],config="add"):
        if config=="add":
            self.__person_reid.add_human(human_bank_list)
        elif (config=="delete"):
            self.__person_reid.delete_human(human_bank_list)


    

