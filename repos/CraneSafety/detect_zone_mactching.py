##########################################################################
# Detect person under crane by yolo and matching template
##########################################################################


##########################################################################
# Import libraries and define path, constant parameters
##########################################################################


import  os
import  sys
sys.path.append(os.path.join(os.getcwd(), 'yolov7'))
import  argparse
import  random
import  time
import  cv2
from    pathlib import Path

import  torch

from    yolov7.models.experimental          import  attempt_load
from    yolov7.utils.datasets               import  LoadStreams, LoadImages
from    yolov7.utils.general                import  check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
                                                    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from    yolov7.utils.plots                  import  plot_one_box
from    yolov7.utils.datasets               import  LoadImages
from    yolov7.utils.torch_utils            import  select_device, load_classifier, time_synchronized
from    pattern_matching.matching           import  template_match

from    sort.sort                           import  *

##########################################################################
# 
##########################################################################

# def line_intersection(line1, line2):
# def line_intersection(x1,y1,x2,y2,x3,y3,x4,y4):
#     """
#         https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
#     """
#     px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
#     py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
#     return [px, py]
def lineLineIntersection(A, B, C, D):
    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1*(A[0]) + b1*(A[1])
 
    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2*(C[0]) + b2*(C[1])
 
    determinant = a1*b2 - a2*b1
 
    if (determinant == 0):
        # The lines are parallel. This is simplified
        # by returning a pair of FLT_MAX
        return 10**9, 10**9
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return x, y


def parse_args():
    """
        Parsing parameter
    """
    parser  =   argparse.ArgumentParser()
    parser  .add_argument('--weights'       , nargs='+', type=str, default='yolov7/weights/yolov7.pt', help='model.pt path(s)')
    parser  .add_argument('--source'        , type=str, default='data/input/zone_under_the_crane_6.avi', help='source')  # file/folder, 0 for webcam
    parser  .add_argument('--img-size'      , type=int, default=640, help='inference size (pixels)')
    parser  .add_argument('--conf-thres'    , type=float, default=0.25, help='object confidence threshold')
    parser  .add_argument('--augment', action='store_true', help='augmented inference')
    parser  .add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser  .add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser  .add_argument('--iou-thres'     , type=float, default=0.45, help='IOU threshold for NMS')
    parser  .add_argument('--device'        , default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt     =   parser.parse_args()

    return  opt


##########################################################################
# 
##########################################################################


def detect(opt):

    # Get parameter from opt
    source      =   opt.source
    weights     =   opt.weights
    imgsz       =   opt.img_size

    # Initialize
    device      =   select_device(opt.device)
    half        =   device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model       =   attempt_load(weights, map_location=device)  # load FP32 model
    stride      =   int(model.stride.max())  # model stride
    imgsz       =   check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model   .half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset     =   LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1


    #---------------------------------------------------------------------
    # Create sort object
    #---------------------------------------------------------------------
    mot_tracker     =   Sort()

    id_lst          =   {}
    #---------------------------------------------------------------------
    # Run inference path
    #---------------------------------------------------------------------
    index = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #---------------------------------------------------------
                # Pattern matching
                #---------------------------------------------------------
                crane_box   =   template_match(im0, 'pattern_matching/pattern/crane_001.png')
                if crane_box is not None:
                    x,y,w,h = crane_box
                    cv2.rectangle(im0, (x, y), (x+w, y+h), (255,0,0), 3)

                #---------------------------------------------------------
                # 90 with ground
                #---------------------------------------------------------
                x1, y1 = (658,350)
                x2, y2 = (694,583)
                # cv2.line(im0, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
                slope_line = (y1-y2)/(x1-x2)
                # find paralled line fo through center of bottom matching
                if crane_box is not None:
                    x,y,w,h = crane_box
                    x_b_c   = int(x+w/2)
                    y_b_c   = y+h
                    b_line  = y_b_c-slope_line*x_b_c
                    # finding point on top with known y
                    x_top       =   int((y-b_line)/slope_line)
                    y_top       = y
                    # cv2.line(im0, (x_top, y_top), (x_b_c, y_b_c), (0, 255, 0), thickness=3)
                    point_top = np.asarray((x_top, y_top))
                    point_bottom = np.asarray((x_b_c, y_b_c))


                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    x1,y1,x2,y2 = [int(coor.cpu().detach().numpy()) for coor in xyxy]
                    x_c   = int((x1+x2)/2)
                    y_c   = int((y1+y2)/2)
                    point_check =   np.asarray((x_c, y_c))
                    
                    if crane_box is not None:
                        d = np.linalg.norm(np.cross(point_bottom-point_top, point_top-point_check))/np.linalg.norm(point_bottom-point_top)
                        if d< (w/2)*1.25:
                            cv2.rectangle(im0, (x1, y1), (x2, y2), (0,0,255), 5)

                            # draw zone under (247, 247, 237) BGR
                            # vanishing point in image
                            x_van, y_van = (1202, 272)

                            # line from vanishing point to top, right of box
                            # find point cut "center line of box"
                            top_ground = lineLineIntersection((x_top,y_top), (x_b_c, y_b_c), (x_van, y_van), (x2, y1))
                            
                            slope_line_1 = (y_van-y1)/(x_van-x2)
                            b_line_1  = y_van-slope_line_1*x_van

                            top_left_ground_x = top_ground[0]-w/2
                            top_left_ground_y = top_left_ground_x*slope_line_1 + b_line_1

                            top_right_ground_x = top_ground[0]+w/2
                            top_right_ground_y = top_right_ground_x*slope_line_1 + b_line_1
                            # print((int(top_left_ground_x), int(top_left_ground_y)), (int(top_right_ground_x), int(top_right_ground_y)))
                            # cv2.line(im0, (int(top_left_ground_x), int(top_left_ground_y)), (int(top_right_ground_x), int(top_right_ground_y)), (247, 247, 237), thickness=3)
                        

                            # line from vanishing point to top, right of box
                            # find point cut "center line of box"
                            bottom_ground = lineLineIntersection((x_top,y_top), (x_b_c, y_b_c), (x_van, y_van), (x2, y2))
                            
                            slope_line_2 = (y_van-y2)/(x_van-x2)
                            b_line_2  = y_van-slope_line_2*x_van

                            bottom_left_ground_x = bottom_ground[0]-w/2
                            bottom_left_ground_y = bottom_left_ground_x*slope_line_2 + b_line_2

                            bottom_right_ground_x = bottom_ground[0]+w/2
                            bottom_right_ground_y = bottom_right_ground_x*slope_line_2 + b_line_2

                            pts = np.array([[top_left_ground_x, top_left_ground_y], [top_right_ground_x, top_right_ground_y],
                                            [bottom_right_ground_x, bottom_right_ground_y],[bottom_left_ground_x, bottom_left_ground_y]], np.int32)
                            
                            pts = pts.reshape((-1, 1, 2))
                            
                            isClosed = True
                            
                            # Blue color in BGR
                            color = (247, 247, 237)
                            
                            # Line thickness of 2 px
                            thickness = 5
                            
                            # Using cv2.polylines() method
                            # Draw a Blue polygon with
                            # thickness of 1 px
                            cv2.polylines(im0, [pts],
                                                isClosed, color, thickness)
                    

            # # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Save results (image with detections)
            save_path   =   'data/output/'
            save_path   =   os.path.join(save_path, 'img_{}.png'.format(str(index).zfill(5)))
            # if dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
            index       +=  1
                # print(f" The image with the result is saved in: {save_path}")


def map_det_n_track(dets, tracks):
    """
        Mapping 2 list by IoU, if no track, id = -1
    """
    
    result = np.zeros((len(dets), 6))-1
    result[:,:4] = dets[:,:4]
    result[:,4]  = dets[:, -1]

    tracks_left  = list(tracks.copy())

    for index_det, box in enumerate(result):
        if len(tracks_left) ==  0:
            break
        iou_lst =   [get_iou(box[:4], track[:4]) for track in tracks_left]
        iou_max         =   np.max(iou_lst)
        index_choose    =   np.argmax(iou_lst)
        if iou_max>0.5:
            result[index_det, -1]   =   tracks_left[index_choose][-1]
            del tracks_left[index_choose]

    return result

def get_iou(bb1, bb2):
    # print(bb1, bb2)
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def main():
    opt     =   parse_args()

    with torch.no_grad():
        detect(opt)


if __name__ ==  '__main__':
    main()
