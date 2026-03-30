import os
import numpy as np
import csv
import cv2
from ai_core.statistic.utils.draw_frame import draw_grid
import time

def draw_from_csv(file_lst, type_draw, image_path="", classname="car", size = (640, 480), width_division = 32, height_division = 32, 
                alpha = 0.4, colorMap = 2, save_directory = "./data"):
    """
        Draw heatmap from csv files
        Input:
            + file_lst: List of files' path of 1 class, remember to check classname
            + type_draw:
                . 0 mean only heatmap
                . 1 mean heatmap + image
            + image: image_path
            + classname: None: do not care classname (mean sum all classes), else only 1 class 
            + size: Desire output size (h, w)
            Note that this size is apply scaling without padding
            + width_division, height_division: must as same as value in statistic.py
            + clas_lst: class of heatmap, must as same as input for statistic.py (elements and indexs must equal)
            + alpha: Parameter of adding
            + colorMap: see here https://learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/
        Output:
            + heatmap: sum_time when sum_classname.txt
    """

    # check type, path for drawing
    assert type_draw in [0,1], "Do not understand type for drawing heatmap"
    if type_draw == 1:
        assert os.path.exists(image_path), "Image path does not exists"

    # load csv files to numpy array
    heatmap = np.zeros((height_division, width_division))
    for file_path in file_lst:
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            line = 0
            for row in csv_reader:
                if line == 0:
                    line +=1
                    continue
                x = int(row[0])
                y = int(row[1])
                v = int(row[2])
                heatmap[y, x] += v
                line +=1
            
    # draw heatmap
    heatmap_output = cv2.resize(heatmap, size)
    heatmap_output = heatmap_output / np.max(heatmap_output)
    heatmap_output = np.uint8(heatmap_output*255)
    heatmap_output = cv2.applyColorMap(heatmap_output, colorMap)

    if type_draw == 1:
        base_image = cv2.imread(image_path)
        base_image = cv2.resize(base_image, size)
        heatmap_output = cv2.addWeighted(heatmap_output, alpha, base_image, 1-alpha, 0, base_image)
    
    # save heatmap
    assert os.path.exists(save_directory), "Save directory not exists"
    if classname is not None:
        heatmap_name = "heatmap_" + str(time.time()) + '_' + classname + '.jpg'
    else:
        heatmap_name = "heatmap_" + str(time.time()) + '_' + '.jpg'
    heatmap_path = os.path.join(save_directory, heatmap_name)
    cv2.imwrite(heatmap_path, heatmap_output)
    return heatmap_path