##########################################################################
# 2D space functions
# Author        :   Phuc Thanh Nguyen
# Created       :   Apr 12th, 2023
# Last edited   :   Apr 17th, 2023
##########################################################################


import  numpy       as  np


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def intersect(A,B,C,D):
    """
        Return true if line segments AB and CD intersect
    """
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def distance_point_point(point1, point2):
    """
        Compute distance from 2 points
    """ 
    point1  =   np.array(point1)
    point2  =   np.array(point2)
    d       =   np.sqrt(np.sum(np.subtract(point1,point2)**2))
    return  d


def distance_point_n_line(point, line):
    """
        Compute distance from point [x, y and line [[x1, y1], [x2, y2]]
    """
    p1  =   np.array(line[0])
    p2  =   np.array(line[1])
    p3  =   np.array(point)
    d   =   np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
    return  d


def line_intersection(line1, line2):
    """
        Find intersection between 2 lines
        Line: [(x1, y1), (x2, y2)]
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def compute_iou(boxA, boxB):
	
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