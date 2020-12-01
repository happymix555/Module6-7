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

# corner_id = [16, 4, 8, 12]
# side1_id = [16, 1, 2, 3, 4]
# side2_id = [4, 5, 6, 7, 8]
# side3_id = [8, 9, 10, 11, 12]
# side4_id = [12, 13, 14, 15, 16]
corner_id = [16, 4, 8, 12]
side1_id = [8, 9, 10, 15, 16]
side2_id = [16, 1, 2, 3, 4]
side3_id = [4, 5, 6, 11, 12]
side4_id = [12, 13, 14, 7, 8]

#read multiplt image for Median
images = read_multiple_image('rail_image/rail_image10')
#resize image for viewing ability

# cv2.imshow('i', images[0])
# corners, ids = detect_aruco(images[0])
# frame_markers = aruco.drawDetectedMarkers(images[0].copy(), corners, ids)
# cv2.imshow('i2', frame_markers)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


for img in images:
    img = resize_percent(img, 60)

width = 600
height = 600
# corners, ids = detect_aruco(images[2])
# frame_markers = aruco.drawDetectedMarkers(images[2].copy(), corners, ids)
# cv2.imshow('marker', frame_markers)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# after_perspective_transform = []
# for img in images:
#     # corners, ids = detect_aruco(img)
#     # frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
#     # cv2.imshow('marker', frame_markers)
#     # corner_point = get_all_corner(corners, ids, [0, 1, 2, 3])
#     corner_point = find_all_corner(img, [0, 1, 2, 3], side1_id, side2_id, side3_id, side4_id)
#     ordered_point = order_points(corner_point)
#     if ordered_point != None:
#         after_pt = perspective_transform(img, ordered_point , width, height)
#         after_perspective_transform.append(after_pt)
#         cv2.imshow('apt', after_pt)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         None

# after_perspective_transform
# set_of_image_to_stack = tuple(after_perspective_transform)
# # for img in after_perspective_transform:
# sequence = np.stack(set_of_image_to_stack, axis=3)
# result = np.median(sequence, axis=3).astype(np.uint8)
# cv2.imshow('apt', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('rail1.jpg', result)
#find relationship
full_field_image = images[0]
img = full_field_image.copy()
corner_point = find_all_corner(img, [0, 1, 2, 3], side1_id, side2_id, side3_id, side4_id)
ordered_point = order_points(corner_point)
side2_center = find_center_of_side(ordered_point[1], ordered_point[2])
side4_center = find_center_of_side(ordered_point[0], ordered_point[3])
# img = cv2.circle(img, (side2_center[0], side2_center[1]), radius=0, color=(0, 0, 255), thickness=10)
# img = cv2.circle(img, (side4_center[0], side4_center[1]), radius=0, color=(255, 0, 0), thickness=10)
TL2Four = find_relationship(ordered_point[0], side4_center)
TR2Two = find_relationship(ordered_point[1], side2_center)
BL2Four = find_relationship(ordered_point[3], side4_center)
BR2Two = find_relationship(ordered_point[2], side2_center)
#find relationship
top_half = []
bottom_half = []
for img in images:
    TL = find_corner(img, 3, side1_id, side4_id)
    TR = find_corner(img, 1, side1_id, side2_id)
    BR = find_corner(img, 0, side2_id, side3_id)
    BL = find_corner(img, 2, side4_id, side3_id)
    if TL != None and TR != None:
        recon4 = point_from_relationship(TL, TL2Four)
        recon2 = point_from_relationship(TR, TR2Two)
        if is_out_of_image(recon4, img) and is_out_of_image(recon2, img):
            before_point = [TL, TR, recon2, recon4]
            top_half.append(perspective_transform(img, before_point, 600, 300))
            # cv2.imshow('t', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    if BR != None and BL != None:
        recon4 = point_from_relationship(BL, BL2Four)
        recon2 = point_from_relationship(BR, BR2Two)
        # print(BR)
        # print(recon2)
        # print(img.shape[0])
        # print(img.shape[1])
        if is_out_of_image(recon4, img) and is_out_of_image(recon2, img):
            before_point = [recon4, recon2, BR, BL]
            bottom_half.append(perspective_transform(img, before_point, 600, 300))
            # cv2.imshow('b', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
top_half_stack = tuple(top_half)
bottom_half_stack = tuple(bottom_half)
# for img in after_perspective_transform:
top_sequence = np.stack(top_half_stack, axis=3)
bottom_sequence = np.stack(bottom_half_stack, axis=3)
top_result = np.median(top_sequence, axis=3).astype(np.uint8)
bottom_result = np.median(bottom_sequence, axis=3).astype(np.uint8)
vertical = np.concatenate((top_result, bottom_result), axis = 0)
cv2.imshow('top', top_result)
cv2.imshow('bottom', bottom_result)
cv2.imshow('full_sequence', vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()


# full_field_image2 = images[3]
# img2 = full_field_image2.copy()
# corner_point2 = find_all_corner(img2, [0, 1, 2, 3], side1_id, side2_id, side3_id, side4_id)
# ordered_point2 = order_points(corner_point2)
# recon2 =  point_from_relationship(ordered_point2[1], side2_rela)
# recon4 =  point_from_relationship(ordered_point2[0], side4_rela)
# img2 = cv2.circle(img2, (recon2[0], recon2[1]), radius=0, color=(0, 0, 255), thickness=10)
# img2 = cv2.circle(img2, (recon4[0], recon4[1]), radius=0, color=(255, 0, 0), thickness=10)
# cv2.imshow('img', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# tf = four_point_transform(images[0].copy(), corner_point)
