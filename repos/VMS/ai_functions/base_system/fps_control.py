import time

#===================================
# This class deals with problems in
# relation to fps of the system
#===================================
class FpsControl():
    def __init__(self,processRate=15,
                     time_to_get_new_skip_frame = 30):
        
        #----------------------------------
        # Init for get_skip_frame
        #----------------------------------
        self.__stable_skip_frame    = False         # Check whether the fps is stable or not yet
        self.__count_frame_fps      = 0     
        self.__stop_measuring_fps   = 0             # Using this one to stop measure fps after get stable  
        self.__system_fps           = 0             # Contain the fps of the system after measuring
        self.__skip_frame_time      = time.time()
        self.__nb_skip_frame        = 0             # Contain the number of skip frames at input
         
        
        #----------------------------------
        # Init for get_frame_now
        #----------------------------------
        self.skip_frames            = 0
        self.__stable               = False
        self.__cur_frame_counter    = 0
        self.processRate            = processRate 
        self.__get_frame_time       = time.time()
        self.__skip_frame_base_on_counter = False
        self.__skip_frame_base_on_time    = False
        
        self.get_new_skip_frame_time = time.time()
        self.__time_to_get_new_skip_frame = time_to_get_new_skip_frame
        
    #===================================
    # This function is used to notify
    # that this time to get frame
    #----------------------------------- 
    def get_frame_now(self):
        #--------------------------
        # Get the nb of skip_frame
        # if not stable and curTime
        #--------------------------
        
        if not self.__stable:
            self.skip_frames, self.__stable = self.__get_skip_frame(self.processRate)
            self.__cur_frame_counter  = 0            # Reset counter after getting skip_frames
            
        curTime = time.time()
        # print(self.skip_frames, self.__stable)
        #--------------------------
        # Controlling fps bases on
        # counter and time
        #--------------------------
        self.__skip_frame_base_on_counter = self.__stable and (self.__cur_frame_counter == self.skip_frames)
        self.__skip_frame_base_on_time    = (curTime-self.__get_frame_time) >= 1/self.processRate
           
        
        #--------------------------
        # Get new skip_frame after
        # each a specific time space
        #--------------------------
        if (curTime - self.get_new_skip_frame_time) >= self.__time_to_get_new_skip_frame:
            self.__stable = False
            self.get_new_skip_frame_time = curTime
            
        
        #--------------------------
        # Return the point to get
        # frame based on signals
        #--------------------------
        if self.__skip_frame_base_on_time or self.__skip_frame_base_on_counter:
            self.__get_frame_time       = curTime
            self.__cur_frame_counter    = 0
            return True
        else:
            self.__cur_frame_counter    +=1
            return False
                
                

    #===================================
    # This function is used for calculating 
    # the number of skipping frame in 
    # gst_stream_source and combine with 
    # time spaces to control the fps of the
    # system
    #----------------------------------- 
    def __get_skip_frame(self,processRate):
        self.__stable_skip_frame    = False
        self.__count_frame_fps     +=1
        cur_skip_time               = time.time()
        
        #---------------------
        # Get the skip frame
        # after each 2s
        #---------------------
        if (cur_skip_time - self.__skip_frame_time) >= 2:
            self.__skip_frame_time = cur_skip_time
            self.__system_fps = self.__count_frame_fps/2
            self.__count_frame_fps = 0
            
            #---------------------
            # Get the nb_skip_frame
            # basing on processRate
            #---------------------
            if self.__system_fps <= processRate:
                self.__nb_skip_frame = 0
            else:
                self.__nb_skip_frame = round(self.__system_fps/processRate) -1 
            
            self.__stop_measuring_fps += 1
            
            #---------------------
            # Checking 3 times before
            # getting the result
            #---------------------
            if self.__stop_measuring_fps == 3:
                self.__stop_measuring_fps = 0
                self.__stable_skip_frame = True
        # print('aaaaaaaaaaaaaaaaaaaaaaaaaaa', self.__nb_skip_frame,self.__stable_skip_frame)
        return self.__nb_skip_frame,self.__stable_skip_frame
    
    
    
    
    


