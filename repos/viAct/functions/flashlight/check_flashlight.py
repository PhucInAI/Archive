##########################################################################
# Check flashlight from camera
# Author        :   Phuc Nguyen Thanh
# Created       :   Mar 07th, 2023
# Last editted  :   Mar 07th, 2023
##########################################################################


import  cv2
import  numpy       as      np
import  imutils
from    imutils     import  contours
from    skimage     import  measure

def check_flashlight(img, draw=False):
    """
        Check flashlight from input by detecting bright spot in an image
        Intput: BRG Image
        Output: Countours in image
    """
    img_gray    =   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred =   cv2.GaussianBlur(img_gray, (11,11), 0)

    #---------------------------------------------------------------------
    # Threshold the image to find out where is light regions
    #---------------------------------------------------------------------
    thresh      =   cv2.threshold(img_blurred, thresh=200, maxval=255, type=cv2.THRESH_BINARY)[1]

    #---------------------------------------------------------------------
    # Peform a series of erosions and delations to remove any small blobs
    # of noise from threshold images
    #---------------------------------------------------------------------
    thresh      =   cv2.erode(thresh, kernel=None, iterations=2)
    thresh      =   cv2.dilate(thresh, kernel=None, iterations=2)

    #---------------------------------------------------------------------
    # Perform a conntected component analysis on the thresholded image,
    # then initilaize a mask to store only the "large" components
    #---------------------------------------------------------------------
    labels      =   measure.label(thresh, background=0)
    mask        =   np.zeros(thresh.shape, dtype='uint8')

    for label in np.unique(labels):     # loop over the unique components
        #-----------------------------------------------------------------
        # If this is the background label, ignore it
        if  label   ==  0:
            continue

        #-----------------------------------------------------------------
        # If this is not the background label, construct the label mask
        # and contour 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        
        #-----------------------------------------------------------------
        # If the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)


    #---------------------------------------------------------------------
    # Find the contours in the mask, then sort them from left to right
    #---------------------------------------------------------------------
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    
    #---------------------------------------------------------------------
    # Loop over the contours to draw if needed
    #---------------------------------------------------------------------
    
    if draw:

        for (i, c) in enumerate(cnts):
            
            #-----------------------------------------------------------------
            # Draw the bright spot on the image
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(img, (int(cX), int(cY)), int(radius),
                (0, 0, 255), 3)
            cv2.putText(img, "#{}".format(i + 1), (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    return len(cnts), mask
