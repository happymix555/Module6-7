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

#read multiplt image for Median
images = read_multiple_image('rail_image')
#resize image for viewing ability
for img in images:
    img = resize_percent(img, 60)
side_length = find_max_wh(images, [33, 22, 44, 11])
corners, ids = detect_aruco(images[2])
frame_markers = aruco.drawDetectedMarkers(images[2].copy(), corners, ids)
cv2.imshow('marker', frame_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()

after_perspective_transform = []
for img in images:
    corners, ids = detect_aruco(img)
    frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
    cv2.imshow('marker', frame_markers)
    corner_point = get_all_corner(corners, ids, [33, 22, 44, 11])
    ordered_point = order_points(corner_point)
    if ordered_point != None:
        after_pt = perspective_transform(img, ordered_point , side_length)
        after_perspective_transform.append(after_pt)
        cv2.imshow('apt', after_pt)
        cv2.waitKey(0)
    else:
        None
cv2.destroyAllWindows()
cv2.imshow('apt', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
after_perspective_transform
set_of_image_to_stack = tuple(after_perspective_transform)
for img in after_perspective_transform:
    sequence = np.stack(set_of_image_to_stack, axis=3)
result = np.median(sequence, axis=3).astype(np.uint8)
cv2.imwrite('rail1.jpg', result)

# tf = four_point_transform(images[0].copy(), corner_point)
