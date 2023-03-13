##########################################################################
# Check camera reposition from videos
# Author        :   Phuc Nguyen Thanh
# Created       :   Mar 07th, 2023
# Last editted  :   Mar 07th, 2023
##########################################################################


##########################################################################
# Import necessary library and define paths
##########################################################################


import  os
import  glob
import  shutil
import  cv2
import  sys
sys.path.append(os.path.dirname(os.getcwd()))

from    functions.reposition.check_reposition   import  CameraReposition
from    functions.utils.get_video_info          import  VideoInfo
from    functions.ai_logging.ai_logging         import  logger

VIDEO_DIR   =   '../data/input/check_reposition'
OUTPUT_DIR  =   '../data/output/check_reposition'

if  os.path.exists(OUTPUT_DIR):
    shutil  .rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)


##########################################################################
# Test on VIDEO_DIR
##########################################################################


def test_video_reposition(video_path, map_rate=0.05, check_type='depthmap', check_threshold=0.5, check_patient=3, draw = False):
    """
        Test flashlight functions given video 
    """

    #---------------------------------------------------------------------
    # Reposition object
    #---------------------------------------------------------------------
    reposition  =   CameraReposition(map_rate=map_rate, check_type=check_type, check_threshold=check_threshold, check_patient=check_patient)


    #---------------------------------------------------------------------
    # Load video object
    #---------------------------------------------------------------------
    
    video                           =   VideoInfo(video_path)
    score_lst                       =   []
    patient_lst                     =   []
    status_lst                      =   []


    #---------------------------------------------------------------------
    # Compute difference score over time
    #---------------------------------------------------------------------
    video_lenght                =   video.get_video_lenght()
    while   video.current_time  < video_lenght:
        ret, frame              =   video.get_frame_by_time()
        
        if ret:
            score,\
            current_patient,\
            current_status      =   reposition.get_different_score(frame=frame)

            score_lst           .append(score)
            patient_lst         .append(current_patient)
            status_lst          .append(current_status)
    
    #---------------------------------------------------------------------
    # Draw
    #---------------------------------------------------------------------
    if draw:
        video.draw_change_by_time(score_lst, patient_lst, status_lst, output_folder=OUTPUT_DIR)
        

def main():
    
    #---------------------------------------------------------------------
    # Log video run
    #---------------------------------------------------------------------
    
    logger.info("==========================================================================")
    logger.info("Test flashligh log")
    logger.info("==========================================================================")


    #---------------------------------------------------------------------
    # Run each video
    #---------------------------------------------------------------------
    
    video_path_lst  =   list(sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))))
    
    for index, video_path in enumerate(video_path_lst):
        logger.info(os.path.basename(video_path))
        if index    ==  1:
            map_rate    =   [0,0,0,0.5]
            test_video_reposition(video_path, map_rate=map_rate, draw =True)
        elif index  ==  4:
            check_threshold =   0.1
            test_video_reposition(video_path, check_threshold=check_threshold, draw =True)
        else:
            test_video_reposition(video_path, draw =True)
        


if  __name__    ==  "__main__":
    main()