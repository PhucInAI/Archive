# import cv2
# from pyzbar.pyzbar import decode

# def extract_qr_code_content(image_path):
#     # Read the image
#     img = cv2.imread(image_path)

#     # Decode the QR code
#     qr_codes = decode(img)

#     for qr_code in qr_codes:
#         # Extract QR code data
#         qr_code_data = qr_code.data.decode('utf-8')
#         print(f"QR Code Content: {qr_code_data}")
#         return qr_code_data

#     return None

# # Example usage
# qr_code_content = extract_qr_code_content('path_to_your_image.jpg')
# if qr_code_content:
#     print(f"Extracted QR Code Content: {qr_code_content}")
# else:
#     print("No QR code found.")


import cv2

def extract_qr_code_content_opencv(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the QRCodeDetector
    detector = cv2.QRCodeDetector()

    # Detect and decode the QR code
    data, points, _ = detector.detectAndDecode(gray)

    if data:
        print(f"QR Code Content: {data}")
        return data
    else:
        print("No QR code found.")
        return None

# Example usage
qr_code_content = extract_qr_code_content_opencv('path_to_your_image.jpg')
if qr_code_content:
    print(f"Extracted QR Code Content: {qr_code_content}")
else:
    print("No QR code found.")
