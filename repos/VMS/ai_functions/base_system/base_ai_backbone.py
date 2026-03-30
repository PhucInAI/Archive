import  os
import  queue
import  time
from    ai_core.utility.ai_logger import aiLogger
import  threading
# from    ai_core.utility.config import BackboneThreadConfig
from    ai_core.utility.config import BufferConfig
import torch
#=========================================
# This class create thread for backbone and 
# process frame 
#=========================================
class BaseAiBackbone:
    def __init__(self, fps_log,device):

        self.device             = device
        buffConfig              = BufferConfig()
                        
        self._ai_thread         = threading.Thread(target=self.__ai_backbone_processing,daemon=False)   # Declare backbone_thread with __ai_process function
        self._running           = False  
        self.__fps_log          = fps_log
        #--------------------------------
        # Contain features from Resnet
        # backbone 
        #--------------------------------

        self.c_raw_backbone         = None
        self.c_processed_backbone   = threading.Condition()    
        
        #--------------------------------
        # Buffer and frame list
        #--------------------------------
        self.processed_backbone_buffer   = buffConfig.processedBuffer         # Contain raw_frame, layer and camera id 
        self.raw_backbone_buffer = None
        self.frame = {}
        
    #==========================================
    #
    #------------------------------------------
    def start(self):
        raise RuntimeError(' start() method need to be overrided by child class.')
    #==========================================
    #
    #------------------------------------------
    def stop(self):
        raise RuntimeError('stop() method need to be overrided by child class.')
    
    def __ai_backbone_processing(self):
        while True:
            try:
                # curTimeTest = self.time_synchronized()
                #--------------------------------------
                with self.c_raw_backbone:
                    if self.raw_backbone_buffer.empty():
                        self.c_raw_backbone.wait()
                #------------------------
                # Run backbone thread with
                # run_ai = True
                #------------------------
                if not self.raw_backbone_buffer.empty():
                    # curTimeTest = self.time_synchronized()
                    base_info, utility_info = self.raw_backbone_buffer.get()
                    # print("xxxxxxxxxxxxxxxxxx",self.time_synchronized()-curTimeTest)
                    
                    # print("The time of entire backbone xxxxxxxxxxxxxxxxxxxxx",self.time_synchronized()-curTimeTest)
                    curTime = self.time_synchronized()
                    #------------------------
                    # Extract info
                    #------------------------
                    frame  = base_info['frame']
                    camera = base_info['camera']
                    motion = utility_info['motion']
                    
                    
                    #------------------------
                    # Process backbone
                    #------------------------
                    
                    if camera.run_ai and motion is True:
                        layer = self.__process_frame(frame)    
                        # utility_info['motion'] = False   
                    else:
                        layer = []   
                    
                    #------------------------
                    # Compress info
                    #------------------------
                    utility_info['layer'] = layer   
                    
                    
                    #------------------------
                    # Sending
                    #------------------------
                    self.processed_backbone_buffer.put((base_info,utility_info)) 
                    if self.processed_backbone_buffer.full():
                        self.processed_backbone_buffer.get()
                    
                    #------------------------
                    # Notify for releasing
                    # bottleneck
                    #------------------------        
                    with self.c_processed_backbone:
                        if (self.processed_backbone_buffer.qsize() >= 1):
                            self.c_processed_backbone.notifyAll()
                            
                    #-------------------------------
                    # Get process time and frame count 
                    # to calculate FPS
                    #-------------------------------

                    if camera.id in self.__fps_log.measure_backbone.keys():
                        self.__fps_log.measure_backbone[camera.id]['time']  += self.time_synchronized()-curTime
                        self.__fps_log.measure_backbone[camera.id]['fps']   += 1  
                        
                #--------------------------------------
                if not self._running:
                    break
                
                
            except Exception as e:
                aiLogger.exception("__ai_backbone_processing Exception" + str(e))
                # print("------- __ai_backbone_processing Exception " + str(e))             
    
    #==========================================
    #
    #------------------------------------------
    def __process_frame(self, raw_frame):
        try:
            layers = self._ai_backbone(raw_frame)
            
            return layers
        except BaseException as ex:
            if str(ex).find('out of memory') >= 0:
                print("Out of memory!!!")
                # logger.exception(ex)
                #self.out_of_memory_callback()
            aiLogger.exception(ex)
            os._exit(1)
            return None
    #==========================================
    #
    #------------------------------------------
    # def _ai_backbone(self,):
    #     raise RuntimeError('process_frame in BaseAiBackbone need to be override by child class.')

    #----------------------------
    # Using this one to synchonize
    # before measuring time
    #----------------------------
    def time_synchronized(self,): 
        # pytorch-accurate time 
        if torch.cuda.is_available(): 
            torch.cuda.synchronize(device=self.device) 
        return time.time() 