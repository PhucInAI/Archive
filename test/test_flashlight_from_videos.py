##########################################################################
# Check flashlight from camera
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

from    functions.flashlight.check_flashlight   import  check_flashlight
from    functions.ai_logging.ai_logging         import  logger

VIDEO_DIR   =   '../data/input/check_flashlight'
OUTPUT_DIR  =   '../data/output/check_flashlight'

if  os.path.exists(OUTPUT_DIR):
    shutil  .rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)


##########################################################################
# Test on VIDEO_DIR
##########################################################################


def test_video_flashlight(video_path):
    """
        Test flashlight functions given video 
    """
    video_cap   =   cv2.VideoCapture(video_path)
    ret, frame  =   video_cap.read()
    count       =   0
    
    save_folder =   os.path.join(OUTPUT_DIR, os.path.basename(video_path).split('.')[0])
    os.makedirs(save_folder)

    while ret:
        num_lights, frame=  check_flashlight(frame, draw=True)
        cv2.imwrite(os.path.join(save_folder, 'frame_{}.jpg'.format(str(count).zfill(5))),frame)


def main():
    #---------------------------------------------------------------------
    # Log this run
    #---------------------------------------------------------------------
    logger.info("==========================================================================")
    logger.info("Test flashligh log")
    logger.info("==========================================================================")

    video_path_lst  =   list(sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))))
    for video_path in video_path_lst:
        logger.info(os.path.basename(video_path))
        test_video_flashlight(video_path)


if  __name__    ==  "__main__":
    main()