import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import numpy as np
from numpy import asarray
from PIL import Image

from delete_rail import *
from perspective import *
from all_aruco import *
from all_contour import *

def find_template_contours(contours, image_for_area):
    full_area = len(image_for_area) * len(image_for_area[0])
    checkpoint_area = int(full_area)
    min_checkpoint_area = int(checkpoint_area / 20)
    template_contour = []
    area = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.1 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
		# a square will have an aspect ratio that is approximately
		# equal to one, otherwise, the shape is a rectangle
            if ar >= 0.9 and ar <= 1.1:
                # area.append(cv2.contourArea(cnt))
                if min_checkpoint_area <= cv2.contourArea(cnt) <= checkpoint_area: #and cv2.contourArea(cnt) > (checkpoint_area / 3):
                # if cv2.isContourConvex(cnt) == True:
                    # area.append(cv2.contourArea(cnt))
                    template_contour.append(cnt)
			# shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    return template_contour

def close_contour(contours):
    for cnt in contours:
        if cv2.isContourConvex(cnt) == True:
            return cnt


vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
vid.set(3, 1280) # set the resolution
vid.set(4, 720)
while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    # if key == ord('c'):
    #     name_and_path = 'rail_image/rail_image_loop/' + str(image_count) + '.jpg'
    #     cv2.imwrite(name_and_path, frame)
    #     image_count += 1
    # elif key & 0xFF == ord('q'):
    #     break
    if key == ord('c'):
        name_and_path = 'raw_template/' + str(0) + '.jpg'
        cv2.imwrite(name_and_path, frame)
    elif key & 0xFF == ord('q'):
        break
raw_template = cv2.imread('raw_template/0.jpg')
raw_template = resize_percent(raw_template, 50)
cv2.imshow('raw template', raw_template)
cv2.waitKey(0)
cv2.destroyAllWindows()

raw_template_gray = cv2.cvtColor(raw_template, cv2.COLOR_BGR2GRAY)
contours, hierarchy = find_contours(raw_template_gray, 40, 100, 3, 1, 'tree')
first_contour_img = draw_contours(blank_image_with_same_size(raw_template_gray), contours, 10)
cv2.imshow('first contour', first_contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours2, hierarchy2 = find_contours(first_contour_img, 40, 100, 3, 1, 'tree')
second_contour_img = draw_contours(blank_image_with_same_size(raw_template_gray), contours2, 3)
cv2.imshow('second contour', second_contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

pre_square_contour, pre_square_hie = find_contours(second_contour_img, 40, 100, 3, 1, 'external')
pre_square_contour_img = draw_contours(blank_image_with_same_size(raw_template_gray), pre_square_contour, 3)
cv2.imshow('second contour', pre_square_contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

template_contour = find_template_contours(pre_square_contour, raw_template_gray)
template_contour_img = draw_contours(blank_image_with_same_size(raw_template_gray), template_contour, 1)
cv2.imshow('template contour img', template_contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


prepared_template = find_roi(template_contour[0], raw_template)
cv2.imshow('prepared template', prepared_template)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('prepared_template/0.jpg', prepared_template)
