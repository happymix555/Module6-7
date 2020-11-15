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



def four_point_transform(image, pts):
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


def perspective_transform(original_image, before_point, side_length):
    #for point from ARUCO to transform
    top_left = before_point[0]
    top_right = before_point[1]
    bottom_right = before_point[2]
    bottom_left = before_point[3]

    before_transform = np.float32([top_left, top_right, bottom_right, bottom_left])

    transform_to = np.float32([[0,0], [side_length - 1, 0], [side_length - 1,side_length - 1], [0, side_length - 1]])
    M = cv2.getPerspectiveTransform(before_transform, transform_to)
    after_perspective_t = cv2.warpPerspective(original_image, M, (side_length, side_length))
    return after_perspective_t
