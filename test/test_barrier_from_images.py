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
from    tqdm                                        import  tqdm
import  torch
torch.cuda.is_available()                           # trick to enable cuda before SAM
import  sys
sys.path.append(os.path.dirname(os.getcwd()))

from    functions.segment_anything.check_barrier    import  BarrierMissing
from    functions.ai_logging.ai_logging             import  logger

INPUT_DIR   =   '../data/input/barriermissing/2023-03-27_50_clean'
OUTPUT_DIR  =   '../data/output/barriermissing/2023-03-27_50_clean'

if  os.path.exists(OUTPUT_DIR):
    shutil  .rmtree(OUTPUT_DIR)
os.mkdir(OUTPUT_DIR)


##########################################################################
# Test on VIDEO_DIR
##########################################################################


# def test_video_flashlight(video_path):
#     """
#         Test flashlight functions given video 
#     """
#     video_cap   =   cv2.VideoCapture(video_path)
#     ret, frame  =   video_cap.read()
#     count       =   0
    
#     save_folder =   os.path.join(OUTPUT_DIR, os.path.basename(video_path).split('.')[0])
#     os.makedirs(save_folder)

#     while ret:
#         num_lights, frame=  check_flashlight(frame, draw=True)
#         cv2.imwrite(os.path.join(save_folder, 'frame_{}.jpg'.format(str(count).zfill(5))),frame)


def main():
    #---------------------------------------------------------------------
    # Log this run
    #---------------------------------------------------------------------
    logger.info("==========================================================================")
    logger.info("Test barrier missing")
    logger.info("==========================================================================")
    
    check_barrier   =   BarrierMissing()


    img_path_lst    =   list(sorted(glob.glob(os.path.join(INPUT_DIR, '*'))))

    for img_path in tqdm(img_path_lst):
        img             =   cv2.imread(img_path)
        output          =   check_barrier.predict_barrier(img)
        for line in output:
            print(line)
            point_1, point_2        =   line
            point_1                 =   [int(coor) for coor in point_1]
            point_2                 =   [int(coor) for coor in point_2]
            point_1[1]              =   point_1[1]-100
            point_2[1]              =   point_2[1]+100
            # cv2.line(img, point_1, point_2, color=[255,0,0], thickness=5)
            cv2.rectangle(img, point_1, point_2,color=[0,0,255], thickness=5)
        #     cv2.rectangle(img, (left,top), (right, bottom),color=[255,0,0], thickness=5)

        # barrier                     =   check_barrier.barriers[0]
        # left_top                    =   barrier[0]
        # left_bottom                 =   barrier[1]
        # right_top                   =   barrier[2]
        # right_bottom                =   barrier[3]

        # point_floor_1               =   [int((left_top[0]+left_bottom[0])/2), int((left_top[1]+left_bottom[1])/2)]
        # point_floor_2               =   [int((right_top[0]+right_bottom[0])/2), int((right_top[1]+right_bottom[1])/2)]

        # cv2.line(img, point_floor_1, point_floor_2, color=[255,0,0], thickness=5)

        cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_path)), img)
    
    

    # video_path_lst  =   list(sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))))
    # for video_path in video_path_lst:
    #     logger.info(os.path.basename(video_path))
    #     test_video_flashlight(video_path)


if  __name__    ==  "__main__":
    main()