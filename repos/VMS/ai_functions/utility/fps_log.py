
from ai_functions.utility.config import AiProcessorConfig
# from utility.utils_logger import logger
from ai_functions.utility.ai_logger import aiLogger
from threading import Timer
import os

class FpsLog():
    def __init__(self,
                    available_features,
                    availabelYoloApp,
                    batch_size,
                    device):

        self.device                     = device
        #----------------------------------
        # Init for calculation_system_time
        #----------------------------------
        self.__available_application    = availabelYoloApp
        
        self.running                    = False
        self.cameras                    = []
        self.measure_backbone           = {}         # Contain the total time of backbone before getting average  
        self.measure_neckhead           = {}         # Contain the total time of features's neckhead before getting average  
        self.measure_bottleneck         = {}         # Contain the total time of releasing bottleneck before getting average  
        self.measure_input_frame2ai     = {}         # Contain the total time of measure_input_frame2ai before getting average 
        self.measure_tracking           = {}
        self.measure_updating_memory    = {}
        self.measure_application        = {}
        self.measure_vms_fps            = {}
        self.measure_license_plate      = {}
        self.measure_vms_output         = {"time":0,"fps":0}
        self.features                   = available_features      # Contain features of the system
        
        if self.cameras is not None:
            for camera in self.cameras:
                self.measure_neckhead[camera.id]        = {}
                self.measure_application[camera.id]     = {}
                self.measure_backbone[camera.id]        = {"time":0,"fps":0}                         # In the order, the fist element contains the time, the second one contains fps
                self.measure_bottleneck[camera.id]      = {"time":0,"fps":0}                         # In the order, the fist element contains the time, the second one contains fps
                self.measure_input_frame2ai[camera.id]  = {"time":0,"fps":0,'nb_skip_frame':0}       # In the order, the fist element contains the time, the second one contains fps
                self.measure_tracking[camera.id]        = {"time":0,"fps":0} 
                self.measure_vms_fps[camera.id]         = {"time":0,"fps":0} 
                self.measure_license_plate[camera.id]   = {"time":0,"fps":0}
                # self.measure_vms_output                 = {"time":0,"fps":0}

                # self.measure_application[camera.id]['license_plate'] = {"time":0,"fps":0}
                for feature in self.features:
                    self.measure_neckhead[camera.id][feature]        = {"time":0,"fps":0}
                
                for application in self.__available_application:
                    self.measure_application[camera.id][application] = {"time":0,"fps":0}
                    

        self.config             = AiProcessorConfig()
        self.time_to_print      = self.config.time_to_print     # Contain the period of time to print
        self.batch_size         = batch_size                    # Contain the number of frame at each processing, will be assigned in base_ai_backbone
        self.calculation_thread = InfiniteTimer(self.time_to_print, self.calculation_system_time)
        
    
    #===================================
    # This function is used to calculate
    # the average fps and time of the 
    # system at different function blocks
    #-----------------------------------
    def calculation_system_time(self):
        aiLogger.info("\n \n \n")
        aiLogger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        aiLogger.info("=         THIS LOG BELONGS TO THE AI TEAM             =")
        aiLogger.info("=    MEASURING AFTER EACH {}S AT THE PROCESS {}        ".format(self.time_to_print,os.getpid()))
        aiLogger.info("=             AI DEVICE: {}                            ".format(self.device))     
        aiLogger.info("=        THE NUMBER OF CAMERAS: {} CAMS                ".format(len(self.cameras)))                                 
        aiLogger.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        try:
            for camera in self.cameras:
                #----------------------------------
                # Print what cameras are showed
                #----------------------------------
                aiLogger.info("#-----------------------------")
                aiLogger.info("#  Measure " + camera.name + " " + camera.id)
                aiLogger.info("#-----------------------------")

                if self.measure_vms_fps[camera.id]['fps'] == 0:
                    aiLogger.info("The {} was disconnected, please reconnect again to continue your programe".format(camera.id))
                else:    
                    
                    #----------------------------------
                    # Print information vms fps
                    #----------------------------------
                    # print("Time of vms input            :{:.5f}s".format(self.measure_vms_fps[camera.id]['time']/self.measure_vms_fps[camera.id]['fps']))
                    aiLogger.info("Fps of vms input             :{:.5f} frames\n".format(self.measure_vms_fps[camera.id]['fps']/self.time_to_print))
                    
                    #----------------------------------
                    # Print information of get_frame
                    #----------------------------------
                    aiLogger.info("Time of input_frame2ai       :{:.5f}s".format(self.measure_input_frame2ai[camera.id]['time']/self.measure_input_frame2ai[camera.id]['fps']))
                    aiLogger.info("Fps of input_frame2ai        :{:.5f} frames".format(self.measure_input_frame2ai[camera.id]['fps']/self.time_to_print))
                    aiLogger.info("The number of skip frame     :{}\n".format(self.measure_input_frame2ai[camera.id]['nb_skip_frame']))

                    #----------------------------------
                    # Print information of backbone
                    #----------------------------------
                    if not camera.run_ai:
                        aiLogger.info("AI in this camera was disabled\n")
                        self.measure_input_frame2ai[camera.id]       =   {"time":0,"fps":0}
                        self.measure_vms_fps[camera.id]              = {"time":0,"fps":0}
                        continue
                    elif self.measure_backbone[camera.id]['fps'] == 0:
                        backboneFps = 0
                        aiLogger.info("Time of backbone :can't measure")
                        aiLogger.info("Fps of backbone  :0 frames\n")
                    else: 
                        backboneFps = self.measure_backbone[camera.id]['fps']*self.batch_size/self.time_to_print
                        aiLogger.info("Time of backbone             :{:.5f}s (per process)".format(self.measure_backbone[camera.id]['time']/self.measure_backbone[camera.id]['fps']))
                        aiLogger.info("Fps of backbone              :{:.5f} frames\n".format(backboneFps))
                    # print("Time of backbone             :{:.5f}s".format(self.measure_backbone[camera.id]['time']/self.measure_backbone[camera.id]['fps']))
                    # print("Fps of backbone              :{:.5f} frames".format(self.measure_backbone[camera.id]['fps']*self.batch_size/self.measure_backbone[camera.id]['time']))
                    
                    
                    #----------------------------------
                    # Print information of Releasing 
                    # bottleneck
                    #----------------------------------    
                    if self.measure_bottleneck[camera.id]['fps'] == 0:
                        backbone2featureFps = 0
                        aiLogger.info("Time of release_bottleneck   :can't measure")
                        aiLogger.info("Fps of release_bottleneck    :0 frames \n")
                        
                    else:
                        backbone2featureFps = self.measure_bottleneck[camera.id]['fps']*self.batch_size/self.time_to_print
                        aiLogger.info("Time of release_bottleneck   :{:.5f}s (per process)".format(self.measure_bottleneck[camera.id]['time']/self.measure_bottleneck[camera.id]['fps']))
                        aiLogger.info("Fps of release_bottleneck    :{:.5f} frames \n".format(backbone2featureFps)) 
                    
                    
                    #----------------------------------
                    # Print information of Yolo5
                    #----------------------------------
                    neckheadYolo5Fps = 0
                    if camera.run['yolo5'] and self.measure_neckhead[camera.id]['yolo5']['fps']!= 0:
                        neckheadYolo5Fps = self.measure_neckhead[camera.id]['yolo5']['fps']*self.batch_size/self.time_to_print
                        aiLogger.info("Time of Yolo5's neckhead     :{:.5f}s (per process)".format(self.measure_neckhead[camera.id]['yolo5']['time']/self.measure_neckhead[camera.id]['yolo5']['fps']))
                        aiLogger.info("Fps of Yolo5's neckhead      :{:.5f} frames \n".format(neckheadYolo5Fps))
                    
                
                    
                    #----------------------------------
                    # Print information of RetinaFace
                    #----------------------------------
                    neckheadRetinaFaceFps = 0
                    if camera.run['retinaface'] and self.measure_neckhead[camera.id]['retinaface']['fps']!=0:
                        neckheadRetinaFaceFps = self.measure_neckhead[camera.id]['retinaface']['fps']*self.batch_size/self.time_to_print
                        aiLogger.info("Time of retinaFace's neckhead:{:.5f}s (per process)".format(self.measure_neckhead[camera.id]['retinaface']['time']/self.measure_neckhead[camera.id]['retinaface']['fps']))
                        aiLogger.info("Fps of retinaFace's neckhead :{:.5f} frames \n".format(neckheadRetinaFaceFps))
                    
                    #----------------------------------
                    # Print information of Fire Classification
                    #----------------------------------
                    neckheadFireFps = 0
                    if camera.run['fire_classification'] and self.measure_neckhead[camera.id]['fire_classification']['fps']!=0:
                        neckheadFireFps = self.measure_neckhead[camera.id]['fire_classification']['fps']*self.batch_size/self.time_to_print
                        aiLogger.info("Time of fire_classification's neckhead:{:.5f}s (per process)".format(self.measure_neckhead[camera.id]['fire_classification']['time']/self.measure_neckhead[camera.id]['fire_classification']['fps']))
                        aiLogger.info("Fps of fire_classification's neckhead :{:.5f} frames \n".format(neckheadFireFps))
                    

                    #----------------------------------
                    # Print information of Deeplabv3+
                    #----------------------------------
                    if camera.run['deeplabv3p'] and self.measure_neckhead[camera.id]['deeplabv3p']['fps']!=0:
                        aiLogger.info("Time of deeplabv3p's neckhead :{:.5f}s".format(self.measure_neckhead[camera.id]['deeplabv3p']['time']/self.measure_neckhead[camera.id]['deeplabv3p']['fps']))
                        aiLogger.info("Fps of deeplabv3p's neckhead  :{:.5f} frames \n".format(self.measure_neckhead[camera.id]['deeplabv3p']['fps']*self.batch_size/self.time_to_print))
                    


                    #----------------------------------
                    # Print information of Yolo application
                    #----------------------------------
                    trackingFps = 0
                    if camera.run_track_in_yolo5 and self.measure_tracking[camera.id]['fps']!=0:
                        trackingFps = self.measure_tracking[camera.id]['fps']*self.batch_size/self.time_to_print
                        aiLogger.info("Time of tracking              :{:.5f}s (per process)".format(self.measure_tracking[camera.id]['time']/self.measure_tracking[camera.id]['fps']))
                        aiLogger.info("Fps of tracking               :{:.5f} frames \n".format(trackingFps))
                    
                    # statisticTime = 0
                    # if camera.run['yolo5'] and  self.measure_application[camera.id]['statistic']['fps'] != 0:
                    #     statisticTime = self.measure_application[camera.id]['statistic']['fps']*self.batch_size/self.time_to_print
                    #     aiLogger.info("Time of statistic              :{:.5f}s (per process)".format(self.measure_application[camera.id]['statistic']['time']/self.measure_application[camera.id]['statistic']['fps']))
                    #     aiLogger.info("Fps of statistic               :{:.5f} frames \n".format(statisticTime))
                
                    if camera.run_track_in_yolo5 and self.measure_updating_memory[camera.id]['fps']!=0:
                        updateMemFps = self.measure_updating_memory[camera.id]['fps']*self.batch_size/self.time_to_print
                        aiLogger.info("Time of updating memory       :{:.5f}s (per process)".format(self.measure_updating_memory[camera.id]['time']/self.measure_updating_memory[camera.id]['fps']))
                        aiLogger.info("Fps of updating memory        :{:.5f} frames \n".format(updateMemFps))

                    
                                            
                    # --------------------------------
                    # Measure License Plate
                    # --------------------------------
                    licensePlateFps = 0
                    if camera.run['yolo5'] and camera.run['license_plate_recognition'] and self.measure_license_plate[camera.id]['fps'] != 0:
                        licensePlateFps = self.measure_license_plate[camera.id]['fps']*self.batch_size/self.time_to_print
                        aiLogger.info("Time of license plate          :{:.5f}s (per process)".format(self.measure_license_plate[camera.id]['time']/self.measure_license_plate[camera.id]['fps']))
                        aiLogger.info("Fps of license plate           :{:.5f} frames \n".format(licensePlateFps))

                    # --------------------------------
                    # Measure Yolo app(Object counter,
                    # velocity,...)
                    # --------------------------------
                    if self.config.measuringYoloApp:
                        for app in self.__available_application:
                            if camera.run['yolo5'] and camera.run[app] and self.measure_application[camera.id][app]['fps'] != 0:
                                yoloAppFps = self.measure_application[camera.id][app]['fps']*self.batch_size/self.time_to_print
                                aiLogger.info("Time of {}         :{:.5f}s (per process)".format(app,self.measure_application[camera.id][app]['time']/self.measure_application[camera.id][app]['fps']))
                                aiLogger.info("Fps of {}          :{:.5f} frames \n".format(app,yoloAppFps))

                        # # ----------------------------
                        # # Measure object counter
                        # if camera.run['yolo5'] and camera.run['object_counter'] and self.measure_application[camera.id]['object_counter']['fps'] != 0:
                        #     objectCounterFps = self.measure_application[camera.id]['object_counter']['fps']*self.batch_size/self.time_to_print
                        #     aiLogger.info("Time of object_counter         :{:.5f}s (per process)".format(self.measure_application[camera.id]['object_counter']['time']/self.measure_application[camera.id]['object_counter']['fps']))
                        #     aiLogger.info("Fps of object_counter          :{:.5f} frames \n".format(objectCounterFps))

                        # # ----------------------------
                        # # Measure heatmap
                        # if camera.run['yolo5'] and camera.run['heat_map'] and self.measure_application[camera.id]['heat_map']['fps'] != 0:
                        #     heatMapFps = self.measure_application[camera.id]['heat_map']['fps']*self.batch_size/self.time_to_print
                        #     aiLogger.info("Time of heat_map         :{:.5f}s (per process)".format(self.measure_application[camera.id]['heat_map']['time']/self.measure_application[camera.id]['heat_map']['fps']))
                        #     aiLogger.info("Fps of heat_map          :{:.5f} frames \n".format(heatMapFps))
                    

                    # --------------------------------
                    # Measure the vms output
                    # --------------------------------
                    if self.measure_vms_output['fps'] != 0:
                        aiOutputFps = self.measure_vms_output['fps']/self.time_to_print
                        aiLogger.info("Time of AiOutput   :{:.5f}s (per process)".format(self.measure_vms_output['time']/self.measure_vms_output['fps']))
                        aiLogger.info("Fps of AiOutput    :{:.5f} frames \n".format(aiOutputFps)) 


                    #----------------------------------
                    # Print the time of the whole system
                    #----------------------------------
                    if camera.priority_feature == "fire_classification" \
                        and backboneFps         != 0    \
                        and backbone2featureFps != 0    \
                        and neckheadFireFps     != 0:
                        systemTime = 1/backboneFps + 1/neckheadFireFps
                        aiLogger.info("Time of system               :{:.5f}s".format(systemTime))
                        aiLogger.info("Fps of system                :{:.5f}s".format(1/systemTime))

                    elif camera.priority_feature == "retinaface" \
                        and backboneFps             != 0    \
                        and backbone2featureFps     != 0    \
                        and neckheadRetinaFaceFps   != 0:
                        systemTime = 1/backboneFps + 1/neckheadRetinaFaceFps
                        aiLogger.info("Time of system               :{:.5f}s".format(systemTime))
                        aiLogger.info("Fps of system                :{:.5f}s".format(1/systemTime))

                    elif camera.priority_feature == "yolo5" \
                        and backboneFps         != 0    \
                        and backbone2featureFps != 0    \
                        and neckheadYolo5Fps    != 0    \
                        and trackingFps        != 0    \
                        and licensePlateFps    != 0:
                        systemTime = 1/backboneFps + 1/neckheadYolo5Fps + 1/trackingFps + 1/licensePlateFps
                        aiLogger.info("Time of system               :{:.5f}s".format(systemTime))
                        aiLogger.info("Fps of system                :{:.5f}s".format(1/systemTime))


                #----------------------------------
                # Reset values after printing
                #----------------------------------           
                self.measure_backbone[camera.id]                =   {"time":0,"fps":0}
                self.measure_bottleneck[camera.id]              =   {"time":0,"fps":0}
                self.measure_input_frame2ai[camera.id]          =   {"time":0,"fps":0}
                self.measure_vms_fps[camera.id]                 =   {"time":0,"fps":0} 
                self.measure_tracking[camera.id]                =   {"time":0,"fps":0}
                self.measure_updating_memory[camera.id]         =   {"time":0,"fps":0}
                self.measure_license_plate[camera.id]           =   {"time":0,"fps":0}
                self.measure_vms_output                         =   {"time":0,"fps":0}

                for feature in self.features:
                    self.measure_neckhead[camera.id][feature]   =   {"time":0,"fps":0}
                for application in self.__available_application:
                    self.measure_application[camera.id][application] = {"time":0,"fps":0}
                    
                
        except ZeroDivisionError:
            aiLogger.exception("Get ZeroDivisionError Exception")
            self.measure_backbone[camera.id]                =   {"time":0,"fps":0}
            self.measure_bottleneck[camera.id]              =   {"time":0,"fps":0}
            self.measure_input_frame2ai[camera.id]          =   {"time":0,"fps":0}
            self.measure_vms_fps[camera.id]                 =   {"time":0,"fps":0} 
            self.measure_tracking[camera.id]                =   {"time":0,"fps":0}
            self.measure_updating_memory[camera.id]         =   {"time":0,"fps":0}
            self.measure_license_plate[camera.id]           =   {"time":0,"fps":0}
            self.measure_vms_output                         =   {"time":0,"fps":0}
            for feature in self.features:
                self.measure_neckhead[camera.id][feature] = {"time":0,"fps":0}
            for application in self.__available_application:
                self.measure_application[camera.id][application] = {"time":0,"fps":0}
        except KeyError as e:
            aiLogger.exception("Key Exception" + str(e))
            self.measure_backbone[camera.id]                =   {"time":0,"fps":0}
            self.measure_bottleneck[camera.id]              =   {"time":0,"fps":0}
            self.measure_input_frame2ai[camera.id]          =   {"time":0,"fps":0}
            self.measure_vms_fps[camera.id]                 =   {"time":0,"fps":0} 
            self.measure_tracking[camera.id]                =   {"time":0,"fps":0}
            self.measure_updating_memory[camera.id]         =   {"time":0,"fps":0}
            self.measure_license_plate[camera.id]           =   {"time":0,"fps":0}
            self.measure_vms_output                         =   {"time":0,"fps":0}
            for feature in self.features:
                self.measure_neckhead[camera.id][feature]   = {"time":0,"fps":0}
            for application in self.__available_application:
                self.measure_application[camera.id][application] = {"time":0,"fps":0}
               

    #===================================
    #
    #-----------------------------------
    def start(self):
        self.calculation_thread.start()

    #===================================
    # 
    #-----------------------------------
    def stop(self):
        self.running = False
    
    #===================================
    # This function is used to add cameras
    # into the FpsLog
    #-----------------------------------
    def add_cameras(self,added_cameras):
        for camera in added_cameras:
            self.measure_neckhead[camera.id]        = {}
            self.measure_application[camera.id]     = {}
            self.measure_backbone[camera.id]        = {"time":0,"fps":0}      # In the order, the fist element contains the time, the second one contains fps
            self.measure_bottleneck[camera.id]      = {"time":0,"fps":0}      # In the order, the fist element contains the time, the second one contains fps
            self.measure_input_frame2ai[camera.id]  = {"time":0,"fps":0,'nb_skip_frame':0}      # In the order, the fist element contains the time, the second one contains fps
            self.measure_vms_fps[camera.id]         = {"time":0,"fps":0} 
            self.measure_tracking[camera.id]        = {"time":0,"fps":0}
            self.measure_updating_memory[camera.id] = {"time":0,"fps":0}
            self.measure_license_plate[camera.id]   = {"time":0,"fps":0}
            for feature in self.features :
                self.measure_neckhead[camera.id][feature] = {"time":0,"fps":0}
            
            for application in self.__available_application:
                self.measure_application[camera.id][application] = {"time":0,"fps":0}

        self.cameras+= added_cameras
    #===================================
    # This function is used to remove
    # cameras out of the FpsLog
    #-----------------------------------         
    def remove_cameras(self,rm_camera_id):
        for cur_cam in self.cameras[:]:
            if cur_cam.id in rm_camera_id:
                self.cameras.remove(cur_cam)
                del self.measure_tracking[cur_cam.id] 
                del self.measure_updating_memory[cur_cam.id] 
                del self.measure_application[cur_cam.id]
                del self.measure_neckhead[cur_cam.id]   
                del self.measure_backbone[cur_cam.id]   
                del self.measure_bottleneck[cur_cam.id] 
                del self.measure_license_plate[cur_cam.id]
                del self.measure_input_frame2ai[cur_cam.id]  
        
#===================================
# This class deals with problems in
# relation to fps of the system
#===================================
class InfiniteTimer():
    def __init__(self, seconds, target):
        self._should_continue = False
        self.is_running = False
        self.seconds = seconds
        self.target = target
        self.thread = None

    #===================================
    # 
    #-----------------------------------
    def _handle_target(self):
        self.is_running = True
        self.target()
        self.is_running = False
        self._start_timer()

    #===================================
    # 
    #-----------------------------------
    def _start_timer(self):
        if self._should_continue: 
            self.thread = Timer(self.seconds, self._handle_target)
            self.thread.start()
    #===================================
    # 
    #-----------------------------------
    def start(self):
        if not self._should_continue and not self.is_running:
            self._should_continue = True
            self._start_timer()
        else:
            print("Timer already started or running, please wait if you're restarting.")

    #===================================
    # 
    #-----------------------------------
    def cancel(self):
        if self.thread is not None:
            self._should_continue = False # Just in case thread is running and cancel fails.
            self.thread.cancel()
        else:
            print("Timer never started or failed to initialize.")