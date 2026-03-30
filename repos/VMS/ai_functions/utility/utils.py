

import numpy as np
import cv2
from ai_functions.utility.ai_logger import aiLogger

"""
    Check if a point inside a polygon or not
"""

# define infinity, 1M seem enough, cause I am poor, and will never become a milionare, right? 
INT_MAX = 1000000
 
# Given three collinear points p, q, r, 
# the function checks if point q lies
# on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
    """
        Check if points q lies on segment "pr" or not
        Input: 3 collinear points p, q, r
        Output: result of if
    """

    status_flag = False
     
    if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        status_flag = True
         
    return status_flag


# ===================================
# This function is used to get available
# GPU
# ===================================

def get_GPU_available():
    import nvidia_smi
    nvidia_smi.nvmlInit()

    _1Gb =  1000000000
    #-----------------------
    # Get the current device
    #-----------------------
    default_gpu_index = 0
    freeMem = []

    deviceCount = nvidia_smi.nvmlDeviceGetCount()                               # Get the number of gpu
    handle      = nvidia_smi.nvmlDeviceGetHandleByIndex(default_gpu_index)      # Get the default device
    info        = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)                    # Get the info of gpu

    
    device      = 'server' if (info.total >= 8*_1Gb and deviceCount>1)  else 'laptop'
    is_server   = True if device == "server" else False
    thr_gpu     = 4.5*_1Gb if device == "server" else 2.3*_1Gb    
    
    #-----------------------
    # Check free space and get info
    #-----------------------
    for i in range(deviceCount):
        handle  = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info    = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        aiLogger.warning("Device {}: {}, Memory : ({:.2f}% free): {}Gb(total), {}Gb(free), {}Gb (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total/_1Gb, info.free/_1Gb, info.used/_1Gb))

        freeMem.append(info.free)

    #-----------------------
    # Get the max free space 
    # and their position
    #-----------------------
    maxFreMem = max(freeMem)
    maxpos = freeMem.index(max(freeMem))
    
    gpu_index = maxpos if (maxFreMem > thr_gpu) else -1

    aiLogger.warning("The GPU {} is chosen for processing AI feature!!!".format(maxpos))
    
    nvidia_smi.nvmlShutdown()
    return gpu_index,is_server


# ===================================
# This function is used to synchronize 
# GPU before measuring time
# ===================================

import torch
import time
def time_synchronized(device): 
    
    # pytorch-accurate time 
    if torch.cuda.is_available(): 
        torch.cuda.synchronize(device) 
    return time.time() 



# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p:tuple, q:tuple, r:tuple) -> int:
    """
        Find the orientation ordered triplet (p, q, r).
        Input: 3 points p, q, r
        Output: 0 if p,q, and r are collinear else 1 if clockwise else 2 (counterclock)
    """
     
    val = (((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))
    
    status_value = -1

    if val == 0:
        status_value = 0
    if val > 0:
        status_value = 1
    else:
        status_value = 2
    return status_value
 
def doIntersect(p1: tuple, q1: tuple, p2: tuple, q2: tuple) -> int:
    """
        Find the four orientations needed for general and special cases
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    status_flag = False

    # General case
    if (o1 != o2) and (o3 != o4):
        status_flag = True
    # Special Cases
    # p1, q1 and p2 are collinear and
    # p2 lies on segment p1q1
    elif (o1 == 0) and (onSegment(p1, p2, q1)):
        status_flag = True 
    # p1, q1 and p2 are collinear and
    # q2 lies on segment p1q1
    elif (o2 == 0) and (onSegment(p1, q2, q1)):
        status_flag = True
    # p2, q2 and p1 are collinear and
    # p1 lies on segment p2q2
    elif (o3 == 0) and (onSegment(p2, p1, q2)):
        status_flag = True
    # p2, q2 and q1 are collinear and
    # q1 lies on segment p2q2
    elif (o4 == 0) and (onSegment(p2, q1, q2)):
        status_flag = True
    else:
        status_flag = False
    return status_flag

def is_inside_polygon(points:list, p:tuple) -> bool:
    """
        Check if point p lies inside polygon or not
        Input:
            + polygon: list of points
            + point
        Output:
            True if points inside polygon else False  
    """
     
    n = len(points)
     
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
         
    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
    count = i = 0
     
    while True:
        next = (i + 1) % n
         
        # Check if the line segment from 'p' to 
        # 'extreme' intersects with the line 
        # segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i],
                        points[next],
                        p, extreme)):
                             
            # If the point 'p' is collinear with line 
            # segment 'i-next', then check if it lies 
            # on segment. If it lies, return true, otherwise false
            if orientation(points[i], p,
                           points[next]) == 0:
                return onSegment(points[i], p,
                                 points[next])
                                  
            count += 1
             
        i = next
         
        if (i == 0):
            break
         
    # Return true if count is odd, false otherwise
    return (count % 2 == 1)

def draw_danger_zone(frames, zones):
    ret_frames = []
    for i in range(len(frames)):
        convert_tumple2array = lambda x: np.array(x)
        for z in range(len(zones)):
            points = np.array([convert_tumple2array(x) for x in zones[z]], dtype="int32")
            frames[i] = cv2.polylines(frames[i], [points], isClosed= True, color=[0,0,255], thickness=1) 
        ret_frames.append(frames[i])
    return ret_frames

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def find_most_duplicate(p_list):
    p_dict = {}
    for p in p_list:
        if (p not in p_dict):
            p_dict[p] = 1
        else:
            p_dict[p] += 1

    # print(p_dict)
    max_count = 1
    max_len = 0

    ret = ''
    for p in p_dict:
        if (p_dict[p] > max_count):
            max_count = p_dict[p]
            ret = p
        elif (p_dict[p] == max_count):
            if len(p) > max_len:
                max_len = len(p)
                ret = p
    return ret
# def findstem(arr):
 
#     # Determine size of the array
#     n = len(arr)
 
#     # Take first word from array
#     # as reference
#     s = arr[0]
#     l = len(s)
 
#     res = ""
 
#     for i in range(l):
#         for j in range(i + 1, l + 1):
 
#             # generating all possible substrings
#             # of our reference string arr[0] i.e s
#             stem = s[i:j]
#             k = 1
#             for k in range(1, n):
 
#                 # Check if the generated stem is
#                 # common to all words
#                 if stem not in arr[k]:
#                     break
 
#             # If current substring is present in
#             # all strings and its length is greater
#             # than current result
#             if (k + 1 == n and len(res) < len(stem)):
#                 res = stem
 
#     return res