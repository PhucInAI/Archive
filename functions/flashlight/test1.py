import      os
import      glob
import      shutil
import      cv2
import      numpy           as      np
from        skimage.metrics import  structural_similarity

INPUT_DIR       =   '/home/ptn/Extension/Work/viAct/data/output/check_flashlight/tta_v01-cam02_2023-03-08_19-56-00/raw'
OUTPUT_DIR      =   '/home/ptn/Extension/Work/viAct/data/output/check_flashlight/tta_v01-cam02_2023-03-08_19-56-00/test1'
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

patient         =   10

img_path_lst    =   list(sorted(glob.glob(os.path.join(INPUT_DIR, '*'))))
print(img_path_lst)
print(len(img_path_lst))
pre_img_path    =   img_path_lst[0]

contours_lst    =   []
count           =   0
for img_path in img_path_lst[1:]:
    pre_img         =   cv2.imread(pre_img_path, cv2.IMREAD_GRAYSCALE)
    img             =   cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # diff            =   255 - cv2.absdiff(pre_img, img)
    # cv2             .imwrite(os.path.join(OUTPUT_DIR, 'frame_{}.jpg'.format(str(count).zfill(5))), diff)
    # count           +=  1



    (score, diff)   =   structural_similarity(pre_img, img, full=True)

    diff            =   (diff * 255).astype("uint8")
    diff_box        =   cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh          =   cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours        =   cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours        =   contours[0] if len(contours) == 2 else contours[1]

    if len(contours_lst) == 5:
        contours_lst    =   contours_lst[1:]
    contours_lst.append(contours)

    mask = np.zeros(pre_img.shape, dtype='uint8')
    filled_after = img.copy()

    for cnts in contours_lst:
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 40:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(pre_img, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.drawContours(mask, [c], 0, (255,255,255), -1)
                cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

    cv2             .imwrite(os.path.join(OUTPUT_DIR, 'frame_{}.jpg'.format(str(count).zfill(5))), img)
    count           +=  1

    pre_img_path    =   img_path

