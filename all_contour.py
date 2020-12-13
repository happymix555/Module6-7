import cv2
from cv2 import aruco
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.morphology import medial_axis
import imutils
from skimage.morphology import skeletonize
from skimage.morphology import thin
from skimage.util import invert
from skimage import filters
from PIL import Image as PImage
import skimage.graph

def find_contours(gray_image, canny_l, canny_h, kernel_size0, kernel_size1, contour_type):
    # canny_without_blur = cv2.Canny(gray_image, canny_l, canny_h)
    img = cv2.GaussianBlur(gray_image, (kernel_size0, kernel_size1), 0)
    img_canny = cv2.Canny(img, canny_l, canny_h)
    if contour_type == 'external':
        contours, hierarchy = cv2.findContours(img_canny,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
    elif contour_type == 'tree':
        contours, hierarchy = cv2.findContours(img_canny,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def find_square_contours(contours, image_for_area):
    full_area = len(image_for_area) * len(image_for_area[0])
    checkpoint_area = int(full_area / 12)
    min_checkpoint_area = int(checkpoint_area / 18)
    square_contours = []
    area = []
    peri_list = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.15 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
		# a square will have an aspect ratio that is approximately
		# equal to one, otherwise, the shape is a rectangle
            if ar >= 0.9 and ar <= 1.1:
                # area.append(cv2.contourArea(cnt))
                if min_checkpoint_area <= cv2.contourArea(cnt) <= checkpoint_area: #and cv2.contourArea(cnt) > (checkpoint_area / 3):
                    if peri <= 700:
                        area.append(cv2.contourArea(cnt))
                        square_contours.append(cnt)
                        peri_list.append(peri)
			# shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    return square_contours, area, peri_list

def blank_image_with_same_size(img):
    return(np.zeros((img.shape[0], img.shape[1], 1), dtype = np.uint8))

def draw_contours(image, contours, thickness):
    # blank = blank_image_with_same_size(image)
    contour_img = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness)
    return contour_img

def find_contour_center(contour):
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return[cx, cy]

def find_roi(contour, image):
    x, y, width, height = cv2.boundingRect(contour)
    roi = image[y:y+height, x:x+width]
    return roi

def resize(input_image ,target_image):#wait for some sharpening method
    x_scale = target_image.shape[0] / input_image.shape[0]
    y_scale = target_image.shape[1] / input_image.shape[1]
    if x_scale < y_scale:
        scale = x_scale
    else:
        scale = y_scale
    # scale = min(int(x_scale), int(y_scale))
    width = int(input_image.shape[0] * scale)
    height = int(input_image.shape[1] * scale)
    dim = (height, width)
    return cv2.resize(input_image, dim, interpolation = cv2.INTER_AREA)

def template_matching_with_roi(template_image, checkpoint_roi, checkpoint_location, rotate_iter): #checkpoint_roi = list of roi image
    each_roi_found = []
    for roi_image in checkpoint_roi:
        found = None
        resized_template = resize(template_image, roi_image)
        for rotation in np.linspace(0, 359, rotate_iter):
            rotate = imutils.rotate(resized_template, angle = rotation)
            result = cv2.matchTemplate(roi_image, rotate, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if found == None or max_val > found:
                found = max_val
        each_roi_found.append(found)
    result = each_roi_found[0]
    for p in each_roi_found:
        if p > result:
            result = p
    # result = max[each_roi_found]
    template_coordinate_on_path = checkpoint_location[each_roi_found.index(result)]
    return template_coordinate_on_path

# def find_endpoint(checkpoint_location, gray_field):
#     ret,thresh_field = cv2.threshold(gray_field,200,255,cv2.THRESH_BINARY)
#     for point in checkpoint_location:
#         if thresh_field[point[1]][point[0]] == 255:
#             return point

def find_endpoint(checkpoint_location, field_image):
    c_field = field_image.copy()
    # ret,thresh_field = cv2.threshold(gray_field,200,255,cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(c_field, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    sensitivity = 90
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(c_field,c_field, mask= mask)
    max = 0
    endpoint = None
    for point in checkpoint_location:
        sum = int(res[point[1]][point[0]][0]) + int(res[point[1]][point[0]][1]) + int(res[point[1]][point[0]][2])
        if sum > max:
            max = sum
            endpoint = point
    return endpoint, res

# def find_palette_by_checkpoint_area(contours, checkpoint_area):
#     max_checkpoint_area = checkpoint_area[0]
#     for a in checkpoint_area:
#         if a > max_checkpoint_area:
#             max_checkpoint_area = a
#     palette_contour = []
#     for cnt in contours:
#         if (max_checkpoint_area * 1.5) < cv2.contourArea(cnt) <= (max_checkpoint_area * 4):
#             palette_contour.append(cnt)
#     return palette_contour

def find_palette_by_checkpoint_area(contours, checkpoint_area, field_image_for_pixel):
    max_checkpoint_area = checkpoint_area[0]
    for a in checkpoint_area:
        if a > max_checkpoint_area:
            max_checkpoint_area = a
    r = []
    palette_contour = []
    for cnt in contours:
        pixel_count = 0
        sum = 0
        if (max_checkpoint_area * 1) < cv2.contourArea(cnt) <= (max_checkpoint_area * 4):
            for x in range(len(field_image_for_pixel[0])):
                for y in range(len(field_image_for_pixel)):
                    dist = cv2.pointPolygonTest(cnt,(x,y),False)
                    if dist == 1:
                        sum += field_image_for_pixel[y][x]
                        pixel_count += 1
                    else:
                        None
            result = int(sum / pixel_count)
            r.append(result)
            if result < 150:
                palette_contour.append(cnt)
            # palette_contour.append(cnt)
    return palette_contour, r
