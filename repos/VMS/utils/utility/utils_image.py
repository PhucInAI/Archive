import pickle as pkl
import random
import cv2
import os

from PIL import ImageFont, ImageDraw

COLORS = pkl.load(open(os.path.dirname(__file__) + '/assets/pallete', "rb"))
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
THICKNESS = 1


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


def pil_draw_text_unicode(text, pil_img, c1, c2, box_color, color, font_size=14):
    path = os.path.dirname(os.path.abspath(__file__))
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([c1, c2], outline=box_color)
    if text:
        font = ImageFont.truetype(os.path.dirname(__file__) + '/assets/ARIALUNI.TTF', font_size)
        font.size = font_size
        ascent, descent = font.getmetrics()
        (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
        box = font.getmask(text).getbbox()
        draw.rectangle([c1, (c1[0] + box[2] + 4, c1[1] + box[3] + descent + 2)], outline=box_color, fill=box_color)
        draw.text(c1, text, font=font, fill=(255, 255, 255))

    del draw


def pil_draw_text_box(pil_img, box, box_color, text, text_color, padding=(0, 0, 0, 0), font_size=12):
    x1, y1, x2, y2 = box[0] - padding[0], box[1] - padding[1], box[2] + padding[2], box[3] + padding[3]
    w = x2 - x1
    pil_draw_text_unicode(text, pil_img, (x1, y1), (x2, y2), box_color, text_color, font_size=font_size)


def cv_draw_box(img, label, x1, y1, x2, y2, color=None, font=None, font_scale=None, thickness=None):
    if color is None:
        color = random.choice(COLORS)
    if font is None:
        font = FONT
    if font_scale is None:
        font_scale = FONT_SCALE
    if thickness is None:
        thickness = THICKNESS

    t_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x = x1
    y = y1 - t_size[1] - 4
    xt2 = x + t_size[0] + 3
    yt2 = y + t_size[1] + 4
    if label != '':
        cv2.rectangle(img, (x, y), (xt2, yt2), color, -1)
        cv2.putText(img, label, (x, yt2 -2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

def contains(r1, r2):
    # print(r1,r2)
    return r1[0] < r2[0] < r2[0]+r2[2] < r1[0]+r1[2] and r1[1] < r2[1] < r2[1]+r2[3] < r1[1]+r1[3]
    
def extract_box(detection_result, iw, ih):
    # print('detection_result', detection_result)
    label, confid, (xc, yc, w, h),text = detection_result
    xc = int(xc * iw)
    yc = int(yc * ih)
    w_ = int(w * iw / 2)
    h_ = int(h * ih / 2)
    x1, y1 = (xc - w_, yc - h_)
    x2, y2 = (xc + w_, yc + h_)
    return label, confid, (x1, y1), (x2, y2),text

def extract_boxxy(detection_result, iw, ih):
    # print('detection_result',detection_result)
    label, config, (x1, y1, x2, y2),text = detection_result
    x1 = int(x1 * iw)
    y1 = int(y1 * ih)
    x2 = int(x2 * iw)
    y2 = int(y2 * ih)
    return label, config, (x1, y1), (x2, y2), text

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


def cv_draw_text(text, img, c1, c2, color, font_scale=2, thickness=1):
    ih, iw, _ = img.shape
    cv2.rectangle(img, c1, c2, color, 1)
    if text:
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, text, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,
                    font_scale, [225, 255, 255], thickness)


def cv_resize_keep_ratio(frame, new_size, mode=None):
    '''
    * mode = None:
        set the smaler dimension to new_size
    * mode = 'h':
        set the height to new_size
    * mode = 'w'
        set the weight to new_size
    '''
    h, w, _ = frame.shape
    expected_ration = w / h

    if mode == 'h':
        new_h = new_size
        new_w = int(new_h * expected_ration)
    elif mode == 'w':
        new_w = new_size
        new_h = int(new_w / expected_ration)
    else:
        if h > w:
            new_w = new_size
            new_h = int(new_w / expected_ration)
        else:
            new_h = new_size
            new_w = int(new_h * expected_ration)

    return cv2.resize(frame, (new_w, new_h))


def get_frame(source):
    '''
    Source:
        - http, https, rtsp
        - list of file name of images
    '''
    if isinstance(source, list):
        for file in source:
            frame = cv2.imread(file)
            yield frame, file
    elif source.startswith('http') or \
            source.startswith('https') or \
            source.startswith('rtsp') or \
            source.endswith('mp4') or \
            source.endswith('avi') or \
            isinstance(source, int):

        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    yield None, None

                yield frame, None
