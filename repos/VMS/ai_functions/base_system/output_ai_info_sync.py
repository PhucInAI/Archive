
from ai_functions.utility.ai_logger  import aiLogger
import threading
import time
import torch
from ai_functions.utility.utils_image import extract_boxxy, draw_object_detection
from ai_functions.utility.config      import BufferConfig
from ai_functions.utility.config      import AiProcessorConfig

class OutputAiInfoSync():
    def __init__(self,
                 fps_log,
                 available_features,
                 batch_size,
                 device):
        self.device = device
                
        #----------------------------
        # Create thread and others
        #----------------------------
        self.cameras        = []
        self.ai_information = threading.Thread(target=self.__ai_info_output, daemon=False)       
        self.running        = False
        self.batch_size     = batch_size
        self.__fps_log      = fps_log
        #----------------------------
        # Other variables
        #----------------------------
        aiConfig                        = AiProcessorConfig()
        self.__available_features       = available_features
        self.__systemMetaType           = aiConfig.systemMetaType
        self.dict_sync                  = {}                        # Contain the data from different feature to synchonize the result         
        
        for camera in self.cameras:
            self.dict_sync[camera.id] = {}
            for feature in self.__available_features:
                self.dict_sync[camera.id][feature]   = {'frame':[],'frame_time':[],'uuid':[],'meta':[]}
            
        #----------------------------
        # Buffer
        #----------------------------
        self.buffConfig  = BufferConfig()

        self.processed_ai_info_buffer         = {}
        self.raw_ai_info_buffer               = {}
        for feature in self.__available_features:
            self.raw_ai_info_buffer[feature]  = self.buffConfig.rawBuffer
        
        #----------------------------
        # Condition object
        #----------------------------
        self.c_raw_ai_info_output           = None
        self.c_processed_ai_info            = {}
        

    

    #==========================================
    # This function is used to summary the info
    # before merged with VMS team
    #------------------------------------------        
    def __ai_info_output(self,):
        while True:
            try:
            #----------------------------------
            # Wait until get new frame
            #----------------------------------
                with self.c_raw_ai_info_output:
                    self.c_raw_ai_info_output.wait()
                    
                curTime = self.time_synchronized()
                #----------------------------------
                # Get all elements out of buffers
                #----------------------------------
                for curFeature in self.__available_features:

                    while not self.raw_ai_info_buffer[curFeature].empty():
                        # --------------------------------
                        # Get the result untill it's empty
                        # --------------------------------
                        base_info, aiMeta = self.raw_ai_info_buffer[curFeature].get()

                        #----------------------------------
                        # Append info into a stable dict
                        #----------------------------------    
                        # aiLogger.debug("==============================={}".format(aiMeta))   
                        self.dict_sync = self.__append_stable_ai_output(base_info,
                                                                        aiMeta,
                                                                        curFeature
                                                                        )
                
                #----------------------------------
                # Extract the batch,frame and send
                # info
                #---------------------------------- 
                for camera in self.cameras:
                                      
                    #------------------------------
                    # using 'without_ai' info if ai 
                    # wasn't unabled
                    #------------------------------
                    priority_feature = camera.priority_feature
                    
                    #------------------------------
                    # Check the length of frames 
                    # of priority_feature_2_sync
                    # If existing more than one
                    # element, get all and sent 
                    #------------------------------       
                    cur_batch_list_length =  len(self.dict_sync[camera.id][priority_feature]['frame'])     
                    
                    for _ in range(cur_batch_list_length):
                        sending_info = self.__extract_info2_send(camera,
                                                                priority_feature,
                                                                self.dict_sync,
                                                                self.batch_size
                                                                )
                        # aiLogger.debug(sending_info)
                        #----------------------------
                        # Put frame and get if it's full
                        #----------------------------
                        for idx_batch in range(self.batch_size):
                            self.processed_ai_info_buffer[camera.id].put(sending_info[idx_batch])
                            # aiLogger.debug("Camera {} Sending info ==================== {}".format(camera.id,sending_info[idx_batch]))
                            
                            if (self.processed_ai_info_buffer[camera.id].full()):
                                self.processed_ai_info_buffer[camera.id].get()
                            #----------------------------
                            # Notify that this buffer was
                            # ready to get out
                            #----------------------------
                            with self.c_processed_ai_info[camera.id]:
                                if (self.processed_ai_info_buffer[camera.id].qsize() >= 1):
                                    self.c_processed_ai_info[camera.id].notifyAll()
                        
                
                self.__fps_log.measure_vms_output['time'] += time.time()-curTime
                self.__fps_log.measure_vms_output['fps']  += 1 

                if not self.running:
                    break    

            except Exception as e:
                aiLogger.exception("__ai_info_output Exception" + str(e))
            
        
    #==========================================
    # This function is used to get the frame 
    # with ai option
    #------------------------------------------    
    def __append_stable_ai_output(  self,
                                    base_info,
                                    aiMeta,
                                    curFeature
                                ):
        #----------------------------------
        # Update all frames into 
        # stable dictionary
        #----------------------------------
        max_buf_len = self.buffConfig.buffer_size
        
        camera      = base_info['camera']
        frames      = base_info['frame']
        frame_times = base_info['frame_time']
        uuids       = base_info['uuid']

        self.dict_sync[camera.id][curFeature]['frame'].append(frames)
        self.dict_sync[camera.id][curFeature]['frame_time'].append(frame_times)
        self.dict_sync[camera.id][curFeature]['uuid'].append(uuids)
        
        # -----------------------------
        # Update the meta of each 
        # feature
        # -----------------------------
        self.dict_sync[camera.id][curFeature]['meta'].append(aiMeta)

        #------------------------
        # Get if full
        #------------------------    
        if (len(self.dict_sync[camera.id][curFeature]['frame']) >= max_buf_len):
            aiLogger.warning("Lost frames at output!!!")
            self.dict_sync[camera.id][curFeature]['frame'].pop(0)

        if (len(self.dict_sync[camera.id][curFeature]['frame_time']) >= max_buf_len):
            self.dict_sync[camera.id][curFeature]['frame_time'].pop(0)

        if (len(self.dict_sync[camera.id][curFeature]['uuid']) >= max_buf_len):
            self.dict_sync[camera.id][curFeature]['uuid'].pop(0)

        if (len(self.dict_sync[camera.id][curFeature]['meta']) >= max_buf_len):
            aiLogger.warning("Lost meta at output!!!")
            self.dict_sync[camera.id][curFeature]['meta'].pop(0)
                   
        # aiLogger.debug("meta: {}".format(aiMeta))

        return self.dict_sync
    
    
    #==========================================
    # This function is used to extract the info
    # before sent
    #------------------------------------------  
    def __extract_info2_send(self,
                             camera,
                             priority_feature,
                             dict_sync,
                             batch_size):
        
        list_processed_info = []
       

        # if not len(dict_sync[camera.id]['yolo5']['meta']):
        #     aiLogger.debug("dict sync {}".format(dict_sync[camera.id]['yolo5']))
        #     a =1 
        #----------------------------------
        # Extract batch of base info from 
        # features
        #----------------------------------
        frames          = dict_sync[camera.id][priority_feature]['frame'].pop(0)
        frame_times     = dict_sync[camera.id][priority_feature]['frame_time'].pop(0)
        uuids           = dict_sync[camera.id][priority_feature]['uuid'].pop(0)

        

        #----------------------------------
        # Get all batch of meta from features
        # and send at the first time received 
        # notification from backbone
        #----------------------------------
        meta  = {}
        for feature in self.__available_features:
            if camera.run[feature]:
                meta[feature] = []
                while len(dict_sync[camera.id][feature]['meta']):         
                    # aiLogger.debug(dict_sync[camera.id][feature]['meta'])
                    temMeta = dict_sync[camera.id][feature]['meta'].pop(0)
                    meta[feature].append(temMeta)                               # "feature": [{meta1},{meta2}]

                # if not len(meta[feature]):
                #     aiLogger.debug("dict sync {}".format(dict_sync[camera.id]))
                #     a=1

        #----------------------------------
        # Extract info from each batch
        #----------------------------------
        for idx_batch in range(batch_size):
            processed_info          = {}
            processed_info['meta']  = []
            
            #--------------------------
            # Extract base info from 
            # __priority_feature_2_sync
            #--------------------------
            processed_info['frame']          = frames[idx_batch].copy()
            processed_info['ts']             = frame_times[idx_batch]
            processed_info['frame_id']       = uuids[idx_batch]
            processed_info['camera_id']      = camera.id
            
            # -------------------------
            # Compress metas from buffer 
            # into one
            # -------------------------
            mergedMeta = {}
            for feature in self.__available_features:
                if camera.run[feature] and len(meta[feature]):
                    # --------------------
                    # Using comprehension 
                    # to compress values from
                    # various  
                    # --------------------
                    # aiLogger.debug(meta[feature]) 
                    # for iMeta in meta[feature]:        #{meta1}
                    
                    # aiLogger.debug(meta[feature])
                    for key in meta[feature][0].keys():
                        temp            = [mergedMeta[key] for mergedMeta in meta[feature]]
                        mergedMeta[key] = [splitList for batchList in temp for splitList in batchList]    
      
            # -------------------------
            # Assign the meta for sending
            # -------------------------
            if len(mergedMeta):
                processed_info['meta'] = [{'feature_id':metaType,'meta': mergedMeta[metaType]} for metaType in self.__systemMetaType if len(mergedMeta[metaType])]

            list_processed_info.append(processed_info)

        return list_processed_info

    

    #==========================================
    # This function is used to start the thread
    #------------------------------------------  
    def start(self,):
        self.running = True
        self.ai_information.start()
        
    #==========================================
    # This function is used to stop the thread
    #------------------------------------------  
    def stop(self,):
        self.running = False
    
    #==========================================
    # This function is used synchronize gpu before
    # measuring
    #------------------------------------------  
    def time_synchronized(self,): 
        # pytorch-accurate time 
        if torch.cuda.is_available(): 
            torch.cuda.synchronize(self.device) 
        return time.time() 
        
    #==========================================
    # This function is used add cameras into 
    # output
    #------------------------------------------  
    def add_cameras(self,
                    added_cameras,
                    out_buffer,
                    out_condition):
        #---------------------------
        # Update OutputAiInfoSync
        #---------------------------
        for camera in added_cameras:
            self.processed_ai_info_buffer[camera.id]     = out_buffer[camera.id]
            self.c_processed_ai_info[camera.id]          = out_condition[camera.id]
            self.dict_sync[camera.id]                    = {}
            for feature in self.__available_features:
                self.dict_sync[camera.id][feature]   = {'frame':[],'frame_time':[],'uuid':[],'meta':[]}
        self.cameras  += added_cameras        
                
    #==========================================
    # This function is used remove cameras out
    #  of output
    #------------------------------------------
    def remove_cameras(self,rm_camera_id):
        for cur_cam in self.cameras.copy():
            if cur_cam.id in rm_camera_id:
                self.cameras.remove(cur_cam)  
                time.sleep(0.5)  
                del self.processed_ai_info_buffer[cur_cam.id] 
                del self.c_processed_ai_info[cur_cam.id]  
                del self.dict_sync[cur_cam.id]
                
        
            
            
    