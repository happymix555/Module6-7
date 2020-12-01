import numpy as np
import cv2
from all_aruco import *

# def order_points(pts):
# 	# initialzie a list of coordinates that will be ordered
# 	# such that the first entry in the list is the top-left,
# 	# the second entry is the top-right, the third is the
# 	# bottom-right, and the fourth is the bottom-left
# 	rect = np.zeros((4, 2), dtype = "float32")
# 	# the top-left point will have the smallest sum, whereas
# 	# the bottom-right point will have the largest sum
# 	s = pts.sum(axis = 1)
# 	rect[0] = pts[np.argmin(s)]
# 	rect[2] = pts[np.argmax(s)]
# 	# now, compute the difference between the points, the
# 	# top-right point will have the smallest difference,
# 	# whereas the bottom-left will have the largest difference
# 	diff = np.diff(pts, axis = 1)
# 	rect[1] = pts[np.argmin(diff)]
# 	rect[3] = pts[np.argmax(diff)]
# 	# return the ordered coordinates
# 	return rect

def order_points(pts):
    if len(pts) == 4:
        top_left = pts[0]
        bottom_right = pts[1]
        top_right = pts[2]
        bottom_left = pts[3]
        for p in range(len(pts)):
            sum = pts[p][0] + pts[p][1]
            diff = pts[p][0] - pts[p][1]
            sum_tl = top_left[0] + top_left[1]
            sum_br = bottom_right[0] + bottom_right[1]
            diff_tr = top_right[0] - top_right[1]
            diff_bl = bottom_left[0] - bottom_left[1]
            if sum < sum_tl:
                top_left = pts[p]
            if sum > sum_br:
                bottom_right = pts[p]
            if diff > diff_tr:
                top_right = pts[p]
            if diff < diff_bl:
                bottom_left = pts[p]
    else:
        return None
    return [top_left, top_right, bottom_right, bottom_left]



def four_point_transform(image, pts):#input is ordered point
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def find_max_wh(set_of_image, list_of_corner_id):
    side_length = 0
    for img in set_of_image:
        corners, ids = detect_aruco(img)
        corner_point = get_all_corner(corners, ids, list_of_corner_id)
        if len(corner_point) == 4:
            ordered_point = order_points(corner_point)
            top_left = ordered_point[0]
            top_right = ordered_point[1]
            bottom_right = ordered_point[2]
            bottom_left = ordered_point[3]
            width1 = np.sqrt(abs((top_left[1] - top_right[1]) ** 2) + abs((top_left[0] - top_right[0]) ** 2))
            width2 = np.sqrt(abs((bottom_left[1] - bottom_right[1]) ** 2) + abs((bottom_left[0] - bottom_right[0]) ** 2))
            width = max(width1, width2)
            height1 = np.sqrt(abs((top_left[1] - bottom_left[1]) ** 2) + abs((top_left[0] - bottom_left[0]) ** 2))
            height2 = np.sqrt(abs((top_right[1] - bottom_right[1]) ** 2) + abs((top_right[0] - bottom_right[0]) ** 2))
            height = max(height1, height2)
            side_length1 = max(width, height)
            if side_length1 > side_length:
                side_length = side_length1
        else:
            None
    return int(side_length)

def line_intersection(line1, line2):
    if line1 != None and line2 != None:
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
        return [int(x), int(y)]
    else:
        return None

def distance_between_point(p1, p2):
    return np.sqrt(pow(p1[0] - p2[0],2) + pow(p1[1] - p2[1], 2))

def find_the_farest_point(list_of_point):
    max = 0
    p1 = None
    p2 = None
    for p in list_of_point:
        for p_ref in list_of_point:
            distance = distance_between_point(p, p_ref)
            if distance > max:
                max = distance
                p1 = p
                p2 = p_ref
    if p1 != None and p2 != None:
        return [p1, p2]
    else:
        return None

def find_all_corner(original_image, corner_id, side1_id, side2_id, side3_id, side4_id):
    real_corner = []
    compute_corner = []
    all_side = [[], [], [], []]
    line = [[], [], [], []]
    distance_ref = 0
    corners, ids = detect_aruco(original_image)
    for id in range(len(ids)):
        center = compute_aruco_center(corners[id])
        if ids[id] in corner_id:
            real_corner.append(center)
        if ids[id] in side1_id:
            all_side[0].append(center)
        if ids[id] in side2_id:
            all_side[1].append(center)
        if ids[id] in side3_id:
            all_side[2].append(center)
        if ids[id] in side4_id:
            all_side[3].append(center)
    if len(real_corner) == 4:
        return real_corner
    else:
        for side in range(len(all_side)):
            distance_ref = 0
            l1 = None
            l2 = None
            for point in all_side[side]:
                for point1 in all_side[side]:
                    distance = distance_between_point(point, point1)
                    if distance > distance_ref:
                        distance_ref = distance
                        l1 = point
                        l2 = point1
            line[side].append(l1)
            line[side].append(l2)
        compute_corner.append(line_intersection(line[0], line[1]))
        compute_corner.append(line_intersection(line[1], line[2]))
        compute_corner.append(line_intersection(line[2], line[3]))
        compute_corner.append(line_intersection(line[0], line[3]))
        return compute_corner

def find_corner(image, wanted_corner, side1_id, side2_id):
    corners, ids = detect_aruco(image)
    side1 = []
    side2 = []
    line1 = []
    line2 = []
    corner = []
    for id in range(len(ids)):
        center = compute_aruco_center(corners[id])
        if ids[id] == wanted_corner:
            return center
        else:
            if ids[id] in side1_id:
                side1.append(center)
            if ids[id] in side2_id:
                side2.append(center)
    line1 = find_the_farest_point(side1)
    line2 = find_the_farest_point(side2)
    corner = line_intersection(line1, line2)
    if corner != None:
        return corner
    else:
        return None

def find_center_of_side(corner1, corner2):
    x = int(abs(corner1[0] - corner2[0]) / 2)
    y = int(abs(corner1[1] - corner2[1]) / 2)
    if corner1[0] < corner2[0]:
        x = corner1[0] + x
    else:
        x = corner2[0] + x
    if corner1[1] < corner2[1]:
        y = corner1[1] + y
    else:
        y = corner2[1] + y
    return [x, y]

def find_field_center(corners):
    ordered_corner = order_points(corners)
    c1 = find_center_of_side(corners[0], corners[1])
    c2 = find_center_of_side(corners[2], corners[3])
    c3 = find_center_of_side(corners[0], corners[3])
    c4 = find_center_of_side(corners[1], corners[2])
    center = line_intersection((c1, c2), (c3, c4))
    return center

def perspective_transform(original_image, before_point, width, height):
    #for point from ARUCO to transform
    top_left = before_point[0]
    top_right = before_point[1]
    bottom_right = before_point[2]
    bottom_left = before_point[3]

    before_transform = np.float32([top_left, top_right, bottom_right, bottom_left])

    transform_to = np.float32([[0,0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(before_transform, transform_to)
    after_perspective_t = cv2.warpPerspective(original_image, M, (width, height))
    return after_perspective_t

def find_relationship(base_point, ref_point):
    x = base_point[0] - ref_point[0]
    y = base_point[1] - ref_point[1]
    return [x, y]

def point_from_relationship(base_point, relationship):
    x = base_point[0] - relationship[0]
    y = base_point[1] - relationship[1]
    return [x, y]

def is_out_of_image(point, image):
    if 0 <= point[0] <= image.shape[1] and 0 <= point[1] <= image.shape[0]:
        return True
    else:
        return False
