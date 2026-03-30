import cv2
import numpy as np

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Regions:
    @staticmethod
    def to_pixel(width, height, rectangle):
        x1, y1 = int(rectangle[0] * width), int(rectangle[1] * height)
        x2, y2 = int(rectangle[2] * width), int(rectangle[3] * height)
        return x1, y1, x2, y2

    @staticmethod
    def calculate_regions(width, height, regions):
        new_regions = []
        x_max, x_min, y_max, y_min = 0, width, 0, height
        for r in regions:
            points = []
            for point in r:
                x1 = point[0] * width
                y1 = point[1] * height
                points.append((x1, y1))
                if x1 > x_max:
                    x_max = x1
                if x1 < x_min:
                    x_min = x1
                if y1 > y_max:
                    y_max = y1
                if y1 < y_min:
                    y_min = y1

            new_regions.append(points)
        rectangle_merge_regions = (x_min / width, y_min / height, x_max / width, y_max / height)
        # calculate regions coordinate on rectangle_merge_regions
        updated_regions = []
        nw = x_max - x_min + 1
        nh = y_max - y_min + 1
        for r in new_regions:
            points = []
            for p in r:
                points.append(((p[0] - x_min) / nw, (p[1] - y_min) / nh))

            updated_regions.append(points)
        return updated_regions, rectangle_merge_regions

    @staticmethod
    def is_inside(x, y, regions):
        region_indexes = []
        if regions is None or len(regions) == 0:
            return region_indexes

        point = Point(x, y)
        for i, r in enumerate(regions):
            polygon = Polygon(r)
            if polygon.contains(point):
                region_indexes.append(i)

        return region_indexes

    @staticmethod
    def draw_regions(frame, regions, color=(0, 0, 255), draw_label=False):
        fh, fw, _ = frame.shape
        for i, region in enumerate(regions):
            if region is None:
                continue

            poly = [(int(r[0] * fw), int(r[1] * fh)) for r in region]
            cv2.polylines(frame, [np.asarray(poly)], 1, color, 2)
            if draw_label:
                cv2.putText(frame, str(i + 1), (poly[0][0], poly[0][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color,
                            lineType=cv2.LINE_AA)

    @staticmethod
    def split_counting_rectangles(rectangle):
        poly = [(r[0], r[1]) for r in rectangle]
        cx1, cy1 = (poly[0][0] + poly[3][0]) / 2, (poly[0][1] + poly[3][1]) / 2
        cx2, cy2 = (poly[1][0] + poly[2][0]) / 2, (poly[1][1] + poly[2][1]) / 2
        cx3, cy3 = (poly[0][0] + poly[1][0]) / 2, (poly[0][1] + poly[1][1]) / 2
        cx4, cy4 = (poly[2][0] + poly[3][0]) / 2, (poly[2][1] + poly[3][1]) / 2
        r1 = [poly[0], poly[1], (cx2, cy2), (cx1, cy1)]
        r2 = [(cx1, cy1), (cx2, cy2), poly[2], poly[3]]
        return [r1, r2]

    @staticmethod
    def draw_counting_rectangles(frame, rectangles, color=(0, 0, 255)):
        fh, fw, _ = frame.shape
        for rectangle in rectangles:
            poly = [(int(r[0] * fw), int(r[1] * fh)) for r in rectangle]
            cx1, cy1 = int((poly[0][0] + poly[3][0]) / 2), int((poly[0][1] + poly[3][1]) / 2)
            cx2, cy2 = int((poly[1][0] + poly[2][0]) / 2), int((poly[1][1] + poly[2][1]) / 2)
            cx3, cy3 = int((poly[0][0] + poly[1][0]) / 2), int((poly[0][1] + poly[1][1]) / 2)
            cx4, cy4 = int((poly[2][0] + poly[3][0]) / 2), int((poly[2][1] + poly[3][1]) / 2)

            cv2.line(frame, (cx1, cy1), (cx2, cy2), color, 2)
            Regions.draw_arrow_line(frame, (cx3, cy3), (cx4, cy4), color, 2, tipLength=0.1)

            overlay = frame.copy()
            cv2.rectangle(overlay, (cx1, cy1), poly[2], (0, 0, 255), -1)
            alpha = 0.15
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def draw_arrow_line(frame, pt1, pt2, color, thickness, line_type=8, shift=0, tipLength=0.1):
        tipSize = np.linalg.norm(np.asarray(pt1) - np.asarray(pt2)) * tipLength;
        cv2.line(frame, pt1, pt2, color, thickness, line_type, shift)
        angle = np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])

        p = (int(pt2[0] + tipSize * np.cos(angle + np.pi / 4)),
             int(pt2[1] + tipSize * np.sin(angle + np.pi / 4)))
        cv2.line(frame, p, pt2, color, thickness, line_type, shift)

        p = (int(pt2[0] + tipSize * np.cos(angle - np.pi / 4)),
             int(pt2[1] + tipSize * np.sin(angle - np.pi / 4)))
        cv2.line(frame, p, pt2, color, thickness, line_type, shift)

    @staticmethod
    def filter_detection_results(detection_results, regions, mode=2):
        '''
            mode = 1:
                check (center_x, center_y) point in regions
            mode = 2:
                check (center_x, fh) point in regions
        '''
        if regions is None or len(regions) == 0:
            return detection_results
        new_results = []
        for r in detection_results:
            _, _, (xc, yc, w, h) = r
            x, y = xc, yc
            if mode == 2:
                y = yc + h / 2
            if len(Regions.is_inside(x, y, regions)) > 0:
                new_results.append(r)
        return new_results
