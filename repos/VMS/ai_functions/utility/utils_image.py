import pickle as pkl
import random
import cv2
import os

from PIL import ImageFont, ImageDraw

# COLORS = pkl.load(open(os.path.dirname(__file__) + '/assets/pallete', "rb"))
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
THICKNESS = 1

classes = {'person':0,'car':1,'bird':2,'truck':3,'bench':4,'motorbike':5,'bicycle':6,'bus':7,'dog':8,'cat':9,'face':10,'plate':11, 'clothes':12, 'tree':13 , 'excavator':14, 'dump_truck':15, 'concrete_mixer_truck':16, 'van':17, 'container_truck':18, 'bulldozer':19, 'vehicle':20, 'boat':21, 'roller':22}

labels = {0:'person',1:'car',2:'bird',3:'truck',4:'bench',5:'motorbike',6:'bicycle',7:'bus',8:'dog',9:'cat',10:'face',11:'plate', 12:'clothes', 13:'tree' , 14:'excavator', 15:'dump_truck', 16:'concrete_mixer_truck', 17:'van', 18:'container_truck', 19:'bulldozer', 20:'vehicle', 21:'boat', 22:'roller'}


class Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c
    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def extract_boxxy(detection_result, iw, ih):
    # print('detection_result',detection_result)
    label, config, (x1, y1, x2, y2),text = detection_result
    x1 = int(x1 * iw)
    y1 = int(y1 * ih)
    x2 = int(x2 * iw)
    y2 = int(y2 * ih)
    return labels[label], config, (x1, y1), (x2, y2), text

def cv_draw_detection_result(results, img, index=None):
    ih, iw, _ = img.shape
    for r in results:
        if(index !=None):
            color=colors(index[results.index(r)], True)
        else:
            color = random.choice(COLORS)
        label, config, c1, c2, text = extract_boxxy(r, iw, ih)
    # print(label, c1, c2)
        if (label == 'plate'):
            # print('texxxtttt', text, c1,c2)
            t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, 5)
            cv2.putText(img, text, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 3, [225, 255, 255], 2)
        else:
            l_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            # c2 = c1[0] + l_size[0] + 3, c1[1] + l_size[1] + 4
            cv2.rectangle(img, c1, c2, color, 5)
            cv2.putText(img, label, (c1[0], c1[1] + l_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 3, [225, 255, 255], 2)

        
        # cv2.rectangle(img, c1, c2, color, 1)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        # cv2.rectangle(img, c1, c2, color, -1)
        # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
def draw_object_detection(img, results):
    ih, iw, _ = img.shape
    for r in results:
        if (r is not None):
            label, config, c1, c2, text = extract_boxxy(r, iw, ih)
            color=colors(classes[label], True)
            # cv2.rectangle(img, c1, c2, color, 1)
            
            # print(label, c1, c2)
            if (label == 'plate'):
                # print('texxxtttt', text)
                t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(img, c1, c2, color, -1)
                cv2.putText(img, text, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
            else:
                l_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                c2 = c1[0] + l_size[0] + 3, c1[1] + l_size[1] + 4
                cv2.rectangle(img, c1, c2, color, -1)
                cv2.putText(img, label, (c1[0], c1[1] + l_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)


def extract_box(detection_result, iw, ih):
    # print('detection_result', detection_result)
    label, config, (xc, yc, w, h),plate_info = detection_result
    xc = int(xc * iw)
    yc = int(yc * ih)
    w_ = int(w * iw / 2)
    h_ = int(h * ih / 2)
    x1, y1 = (xc - w_, yc - h_)
    x2, y2 = (xc + w_, yc + h_)
    # print(label, type(label))
    return labels[label], config, (x1, y1), (x2, y2),plate_info

def extract_box_plate(detection_result, iw, ih):
    # print('detection_result', detection_result)
    label, config, (xc, yc, w, h),plate_info = detection_result
    xc = int(xc * iw)
    yc = int(yc * ih)
    w_ = int(w * iw / 2)
    h_ = int(h * ih / 2)
    x1, y1 = (xc - w_, yc - h_)
    x2, y2 = (xc + w_, yc + h_)
    # print(label, type(label))
    return label, config, (x1, y1), (x2, y2),plate_info

def contains(r1, r2):
    # print(r1,r2)
    return r1[0] < r2[0] < r2[0]+r2[2] < r1[0]+r1[2] and r1[1] < r2[1] < r2[1]+r2[3] < r1[1]+r1[3]
