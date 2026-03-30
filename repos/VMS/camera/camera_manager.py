#=========================================
# Name_class: Camera
# Purpose   : This class is used to adjust
# features of the camera from option user at
# "listcam.py"
#=========================================

#-----------------------------------------   
# Warning!!!: Don't modify this class if 
# don't understand all the system because 
# it can make the system work wrong
#-----------------------------------------
# from utility.utils_logger import logger
# from ai_core.utility.ai_logger import aiLogger
from ai_functions.utility.ai_logger import aiLogger

class Camera:
    # TODO: clearly features of AI system each functions
    def __init__(self, raw_camera):
        # print(raw_camera)
        #----------------------------------
        # Id of camera and general Ai flag
        #---------------------------------- 
        self.id                 = raw_camera['stream_id'] 
        self.run_ai             = raw_camera['run_ai']
        self.name               = raw_camera['name']
        # self.run_ai = False

        #----------------------------------
        # Features of Ai
        #---------------------------------- 
        self.run                = {}
        self.run['statistic']   = {}
        self.config             = {}

