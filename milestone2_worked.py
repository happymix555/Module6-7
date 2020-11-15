import cv2
from cv2 import aruco
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.morphology import medial_axis
import imutils

#reseize input image to target image's size
def resize_img(input_img, target_img):
    x_scale = target_img.shape[0] / input_img.shape[0]
    y_scale = target_img.shape[1] / input_img.shape[1]
    scale = min(x_scale, y_scale)
    width = int(input_img.shape[0] * scale)
    # print('original_width = ' + str(input_img.shape[0]))
    # print('resize_width = ' + str(width))
    # print('width ratio = ' + str(width / input_img.shape[0]))
    height = int(input_img.shape[1] * scale)
    # print('original_height = ' + str(input_img.shape[1]))
    # print('resize_height = ' + str(height))
    # print('height ratio = ' + str(height / input_img.shape[1]))
    dim = (height, width)
    return cv2.resize(input_img, dim, interpolation = cv2.INTER_AREA)

def template_match3(img, template, scale_iter, rotate_iter):
    found = None
    img_2 = img.copy()
    edge_point = []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 40, 100)
    img_contours, hierarchy = cv2.findContours(img,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    img_black = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img = cv2.drawContours(img_black, img_contours, -1, (255, 255, 255), 1)
    # cv2.imshow('img_black_contours', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.GaussianBlur(template, (5, 5), 0)
    template = cv2.Canny(template, 40, 100)
    template_contours, hierarchy = cv2.findContours(template,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    template_black = np.zeros((template.shape[0], template.shape[1], 1), np.uint8)
    template = cv2.drawContours(template_black, template_contours, -1, (255, 255, 255), 1)

    fullsize_template = resize_img(template, img)
    for scale in np.linspace(0.1, 1, scale_iter)[::-1]:
        resized = imutils.resize(   fullsize_template,
                                    width = int(fullsize_template.shape[1] * scale),
                                    height = int(fullsize_template.shape[0] * scale))
        w, h = resized.shape[::-1]
        for rotation in np.linspace(0, 359, rotate_iter):
            rotate = imutils.rotate(resized, angle = rotation)
            result = cv2.matchTemplate(img, rotate, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if found == None or max_val > found:
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                found = max_val
                edge_point = []
                edge_point.append(top_left)
                edge_point.append(bottom_right)

    cv2.rectangle(img_2, top_left, bottom_right, 255, 1)
    cv2.imshow("result", img_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge_point

field3 = cv2.imread('test_field2.JPG')
tem3 = cv2.imread('star.PNG')
point = template_match3(field3, tem3, 20, 30)
point

field3 = cv2.imread('test_field.JPG')
tem3 = cv2.imread('triangle.PNG')
point = template_match3(field3, tem3, 20, 40)
point
