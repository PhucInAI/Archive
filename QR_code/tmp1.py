# import cv2
# import numpy as np
# from pyzbar.pyzbar import decode

# def extract_qr_code(image_path):
#     # Read the image
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Use pyzbar to detect QR code
#     qr_codes = decode(gray)

#     for qr_code in qr_codes:
#         # Get the bounding box coordinates
#         points = qr_code.polygon
#         if len(points) == 4:
#             pts = np.array(points, dtype=np.int32)
#             rect = cv2.boundingRect(pts)

#             # Extract the QR code as a sub-image
#             x, y, w, h = rect
#             qr_code_img = img[y:y+h, x:x+w]

#             # Display the QR code image
#             cv2.imshow("QR Code", qr_code_img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

#             return qr_code_img

#     return None

# # Example usage
# qr_code_image = extract_qr_code('path_to_your_image.jpg')


# import cv2

# def extract_qr_code_opencv(image_path):
#     # Read the image
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Initialize the QRCodeDetector
#     detector = cv2.QRCodeDetector()

#     # Detect and decode the QR code
#     data, points, _ = detector.detectAndDecode(gray)

#     if points is not None:
#         # Convert points to integers
#         points = points[0].astype(int)

#         # Get the bounding box coordinates
#         x, y, w, h = cv2.boundingRect(points)

#         # Extract the QR code as a sub-image
#         qr_code_img = img[y:y+h, x:x+w]

#         # Display the QR code image
#         cv2.imshow("QR Code", qr_code_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         return qr_code_img

#     return None

# # Example usage
# qr_code_image = extract_qr_code_opencv('path_to_your_image.jpg')


# ------------------
# import cv2
# import numpy as np
# from pyzbar.pyzbar import decode

# def extract_qr_code_pyzbar(image_path):
#     # Read the image
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Use pyzbar to detect QR code
#     qr_codes = decode(gray)

#     for qr_code in qr_codes:
#         # Get the bounding box coordinates
#         points = qr_code.polygon
#         if len(points) == 4:
#             # Create a mask with the QR code polygon
#             mask = np.zeros_like(gray)
#             cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)

#             # Bitwise-and to extract the QR code
#             qr_code_img = cv2.bitwise_and(img, img, mask=mask)

#             # Crop the image to the bounding box
#             rect = cv2.boundingRect(np.array(points, dtype=np.int32))
#             x, y, w, h = rect
#             qr_code_img = qr_code_img[y:y+h, x:x+w]

#             # Display the QR code image
#             cv2.imshow("QR Code", qr_code_img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

#             return qr_code_img

#     return None

# # Example usage
# qr_code_image = extract_qr_code_pyzbar('path_to_your_image.jpg')
