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
# corner_id = [16, 4, 8, 12]
# side1_id = [8, 9, 10, 15, 16]
# side2_id = [16, 1, 2, 3, 4]
# side3_id = [4, 5, 6, 11, 12]
# side4_id = [12, 13, 14, 7, 8]
corner_id = [16, 8, 32, 24]
side1_id = []
for i in range(16, 7, -1):
    side1_id.append(i)
side2_id = [32]
for i in range(8, 0, -1):
    side2_id.append(i)
side3_id = []
for i in range(24, 33):
    side3_id.append(i)
side4_id = []
for i in range(16, 25):
    side4_id.append(i)
#read multiplt image for Median

images = read_multiple_image('rail_image/rail_image_loop')

# test_aruco = images[0]
# test_aruco = resize_percent(test_aruco, 60)
# cv2.imshow('i', test_aurco)
# corners, ids = detect_aruco(test_aruco)
# frame_markers = aruco.drawDetectedMarkers(test_aruco.copy(), corners, ids)
# cv2.imshow('i2', frame_markers)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# for img in images:
#     img = resize_percent(img, 60)

# width = 600
# height = 600
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
#
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
corner_point = find_all_corner(img, corner_id, side1_id, side2_id, side3_id, side4_id)
ordered_point = order_points(corner_point)
side1_center = find_center_of_side(ordered_point[0], ordered_point[1])
side2_center = find_center_of_side(ordered_point[1], ordered_point[2])
side3_center = find_center_of_side(ordered_point[2], ordered_point[3])
side4_center = find_center_of_side(ordered_point[0], ordered_point[3])
field_center = find_field_center(ordered_point)
# img = cv2.circle(img, (side2_center[0], side2_center[1]), radius=0, color=(0, 0, 255), thickness=10)
# img = cv2.circle(img, (side4_center[0], side4_center[1]), radius=0, color=(255, 0, 0), thickness=10)
TL2Four = find_relationship(ordered_point[0], side4_center)
TL2One = find_relationship(ordered_point[0], side1_center)
TL2C = find_relationship(ordered_point[0], field_center)

TR2One = find_relationship(ordered_point[1], side1_center)
TR2Two = find_relationship(ordered_point[1], side2_center)
TR2C = find_relationship(ordered_point[1], field_center)

BR2Two = find_relationship(ordered_point[2], side2_center)
BR2Three = find_relationship(ordered_point[2], side3_center)
BR2C = find_relationship(ordered_point[2], field_center)

BL2Three = find_relationship(ordered_point[3], side3_center)
BL2Four = find_relationship(ordered_point[3], side4_center)
BL2C = find_relationship(ordered_point[3], field_center)

q1_count = 0
q2_count = 0
q3_count = 0
q4_count = 0
#find relationship
Q1 = []
Q2 = []
Q3 = []
Q4 = []
for img in images:
    TL = find_corner(img, 16, side1_id, side4_id)
    TR = find_corner(img, 8, side1_id, side2_id)
    BR = find_corner(img, 32, side2_id, side3_id)
    BL = find_corner(img, 24, side4_id, side3_id)
    if TL != None:
        recon1 = point_from_relationship(TL, TL2One)
        reconC = point_from_relationship(TL, TL2C)
        recon4 = point_from_relationship(TL, TL2Four)
        if is_out_of_image(recon1, img) and is_out_of_image(reconC, img) and is_out_of_image(recon4, img):
            before_point = [TL, recon1, reconC, recon4]
            Q1.append(perspective_transform(img, before_point, 300, 300))
            q1_count += 1
    if TR != None:
        recon1 = point_from_relationship(TR, TR2One)
        reconC = point_from_relationship(TR, TR2C)
        recon2 = point_from_relationship(TR, TR2Two)
        if is_out_of_image(recon1, img) and is_out_of_image(reconC, img) and is_out_of_image(recon2, img):
            before_point = [recon1, TR, recon2, reconC]
            Q2.append(perspective_transform(img, before_point, 300, 300))
            q2_count += 1
    if BR != None:
        recon2 = point_from_relationship(BR, BR2Two)
        reconC = point_from_relationship(BR, BR2C)
        recon3 = point_from_relationship(BR, BR2Three)
        if is_out_of_image(recon2, img) and is_out_of_image(reconC, img) and is_out_of_image(recon3, img):
            before_point = [reconC, recon2, BR, recon3]
            Q3.append(perspective_transform(img, before_point, 300, 300))
            q3_count += 1
    if BL != None:
        recon3 = point_from_relationship(BL, BL2Three)
        reconC = point_from_relationship(BL, BL2C)
        recon4 = point_from_relationship(BL, BL2Four)
        if is_out_of_image(recon3, img) and is_out_of_image(reconC, img) and is_out_of_image(recon4, img):
            before_point = [recon4, reconC, recon3, BL]
            Q4.append(perspective_transform(img, before_point, 300, 300))
            q4_count += 1
Q1_stack = tuple(Q1)
Q2_stack = tuple(Q2)
Q3_stack = tuple(Q3)
Q4_stack = tuple(Q4)
# for img in after_perspective_transform:
Q1_sequence = np.stack(Q1_stack, axis=3)
Q2_sequence = np.stack(Q2_stack, axis=3)
Q3_sequence = np.stack(Q3_stack, axis=3)
Q4_sequence = np.stack(Q4_stack, axis=3)
Q1_result = np.median(Q1_sequence, axis=3).astype(np.uint8)
Q2_result = np.median(Q2_sequence, axis=3).astype(np.uint8)
Q3_result = np.median(Q3_sequence, axis=3).astype(np.uint8)
Q4_result = np.median(Q4_sequence, axis=3).astype(np.uint8)
vertical1 = np.concatenate((Q1_result, Q4_result), axis = 0)
vertical2 = np.concatenate((Q2_result, Q3_result), axis = 0)
horizontal = np.concatenate((vertical1, vertical2), axis = 1)
cv2.imshow('full_sequence', horizontal)
cv2.imwrite('prepared_field/ready_field.jpg', horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('q1 = ' + str(q1_count) + '\n')
print('q2 = ' + str(q2_count) + '\n')
print('q3 = ' + str(q3_count) + '\n')
print('q4 = ' + str(q4_count) + '\n')

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
