"""Detect QR code from image"""

# pylint: disable=import-error


import numpy as np
import cv2
from pyzbar.pyzbar import decode
import time


def detect_qr_code_pyzbar(image_path):
    """Detect QR code by pyzbar"""
    results = []

    # --------------------------------------------------------------------
    # Read the image
    # Use pyzbar to detect QR code
    # --------------------------------------------------------------------
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    qr_codes = decode(gray)

    # --------------------------------------------------------------------
    # Post process
    # --------------------------------------------------------------------
    for qr_code in qr_codes:
        qr_code_data = qr_code.data.decode('utf-8')
        print(qr_code_data)
        # Get the bounding box coordinates
        points = qr_code.polygon
        if len(points) == 4:
            pts = np.array(points, dtype=np.int32)
            rect = cv2.boundingRect(pts)

            # Extract the QR code as a sub-image
            x, y, w, h = rect
            qr_code_img = img[y:y+h, x:x+w]

            results.append(qr_code_img)

    return results

def detect_qr_code_opencv(image_path):
    """Detect QR code by OpenCV"""
    results = []

    # --------------------------------------------------------------------
    # Read the image
    # Use OpenCV to detect QR code
    # --------------------------------------------------------------------
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.QRCodeDetector()
    _, points = detector.detect(gray)

    if points is not None:
        # Convert points to integers
        print(points[0])
        points = points[0].astype(int)

        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(points)

        # Extract the QR code as a sub-image
        qr_code_img = img[y:y+h, x:x+w]

        results.append(qr_code_img)

    return results


def detect_qr_code_pyzbar_opencv(image_path):
    """Detect QR code by pyzbar with some process by opencv"""
    results = []

    # --------------------------------------------------------------------
    # Read the image
    # Use pyzbar to detect QR code
    # --------------------------------------------------------------------
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    qr_codes = decode(gray)

    # --------------------------------------------------------------------
    # dsa
    # --------------------------------------------------------------------
    for qr_code in qr_codes:
        # Get the bounding box coordinates
        points = qr_code.polygon
        if len(points) == 4:
            # Create a mask with the QR code polygon
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)

            # Bitwise-and to extract the QR code
            qr_code_img = cv2.bitwise_and(img, img, mask=mask)

            # Crop the image to the bounding box
            rect = cv2.boundingRect(np.array(points, dtype=np.int32))
            x, y, w, h = rect
            qr_code_img = qr_code_img[y:y+h, x:x+w]

            results.append(qr_code_img)

    return results


def detect_qr_code(image_path, method):
    """Detect QR code"""
    if method == 'pyzbar':
        result = detect_qr_code_pyzbar(image_path)
    elif method == 'opencv':
        result = detect_qr_code_opencv(image_path)
    elif method == 'pyzbar_opencv':
        result = detect_qr_code_pyzbar_opencv(image_path)

    cv2.imwrite('tmp.jpg', result[0])

    return result

# time
# img_path = '/home/ptn/Storage/MTI_Explore_code/QR_code/data/yolo/Dubska_QR/dataset1/images/20110817_001.jpg'
img_path = '/home/ptn/Storage/MTI_Explore_code/QR_code/example3.jpg'
# img_path = '/home/ptn/Storage/MTI_Explore_code/QR_code/data/raw/Dubska_QR/dataset1/20110817_070.jpg'
print(detect_qr_code(img_path, method='pyzbar')[0].shape)