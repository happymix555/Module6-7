import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import numpy as np
from numpy import asarray
from PIL import Image
from skimage.morphology import skeletonize
import skimage.graph

from delete_rail import *
from perspective import *
from all_aruco import *
from all_contour import *

def fill_contour(contour, contour_image_for_size):
    filled = blank_image_with_same_size(contour_image_for_size)
    filled = cv2.fillPoly(filled, contour, color=(255,255,255))
    return filled

def skeleton_with_erotion(filled_image, erode_iter):
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(filled_image, kernel, iterations = erode_iter)
    skeleton_image = skeletonize(erosion, method='lee')
    return skeleton_image

def skeleton_coordinate2(skeleton_image):
    coordinate = []
    for y in range(len(skeleton_image)):
        for x in range(len(skeleton_image[0])):
            if skeleton_image[y][x] == 255:
                coordinate.append([x, y])
    return coordinate

def find_opened_end(skeleton_coordinate, skeleton): #point = skeleton_coordinate, image = skeleton image, return in [y, x]
    opened_point = []
    t = 0
    t2 = 0
    for p in skeleton_coordinate:
        x = p[0]
        y = p[1]
        # kernel = [[x-1, y-1], [x, y-1], [x+1, y-1],
        #           [x-1, y], [x, y], [x+1, y],
        #           [x-1, y+1], [x, y+1], [x+1, y+1]]
        kernel = [[x-1, y-1], [x-1, y], [x-1, y+1],
                  [x, y-1], [x, y], [x, y+1],
                  [x+1, y-1], [x+1, y], [x+1, y+1]]
        sum = 0
        t2 += 1
        for k in kernel:
            if skeleton[k[1]][k[0]] == 255:
                sum += 1
        if sum <= 2:
            opened_point.append([x, y])
        else:
            t += 1
    return opened_point

def distance_between_point(p1, p2):
    return np.sqrt(pow(p1[0] - p2[0],2) + pow(p1[1] - p2[1], 2))

def find_connected_point(checkpoint, all_about_palette, distance_threshold):
    connected_point = []
    for cp in checkpoint:
        for aap in all_about_palette:
            for op in aap[2]:
                if distance_between_point(cp, [op[0], op[1]]) < distance_threshold:
                    connected_point.append([cp, op])
    return connected_point

def draw_connected_path(skeleton, connected_point):
    connected_path = skeleton.copy()
    for cp in connected_point:
        connected_path = cv2.line(connected_path, (cp[0][0], cp[0][1]), (cp[1][0], cp[1][1]),(255,255,255), 1)
    return connected_path

def draw_full_path(connected_point, all_about_palette, field_image_for_size):
    blank = blank_image_with_same_size(field_image_for_size)
    for p in all_about_palette:
        for sc in p[1]:
            blank[sc[1]][sc[0]] = 255
    for cp in connected_point:
        blank = cv2.line(blank, (cp[0][0], cp[0][1]), (cp[1][0], cp[1][1]),(255,255,255), 1)
    return blank

# def find_min_max(full_path_image, all_about_palette, denoised_gray_image):
#     full_path_coordinate = skeleton_coordinate(full_path_image)
#     # test_coordinate = []
#     for palette_set in all_about_palette:
#         min = 255
#         max = 0
#         palette_contour = palette_set[3]
#         for point in full_path_coordinate:
#             dist = cv2.pointPolygonTest(palette_contour, (point[0], point[1]), False)
#             if dist == 1:
#                 # test_coordinate.append(point)
#                 color_dense = denoised_gray_image[point[1]][point[0]]
#                 if color_dense > max:
#                     max = color_dense
#                 if color_dense < min:
#                     min = color_dense
#         palette_set.append([min, max])
#     # return test_coordinate

def find_min_max(full_path_image, all_about_palette, denoised_gray_image):
    full_path_coordinate = skeleton_coordinate2(full_path_image)
    # test_coordinate = []
    for palette_set in all_about_palette:
        min = 255
        max = 0
        palette_skeleton = palette_set[1]
        for point in full_path_coordinate:
            if point in palette_skeleton:
                # test_coordinate.append(point)
                color_dense = denoised_gray_image[point[1]][point[0]]
                if color_dense > max:
                    max = color_dense
                if color_dense < min:
                    min = color_dense
        palette_set.append([min, max])
    # return test_coordinate

def map_value(min1, max1, min2, max2, current):
    result = (current - min1) / (max1 - min1)
    result = result * (max2 - min2)
    result = int(result + min2)
    return result

# def find_path_height(full_path_image, all_about_palette, denoised_gray_image):
#     full_path_coordinate = skeleton_coordinate(full_path_image)
#     for point in full_path_coordinate:
#         for palette_set in all_about_palette:
#             palette_contour = palette_set[3]
#             min = palette_set[4][0]
#             max = palette_set[4][1]
#             dist = cv2.pointPolygonTest(palette_contour, (point[0], point[1]), False)
#             if dist == 1:
#                 color_dense = denoised_gray_image[point[1]][point[0]]
#                 height = map_value(min, max, 100, 200, color_dense)
#                 point.append(height)
#             else:
#                 point.append(None)
#     return full_path_coordinate

# def find_path_height(full_path_image, all_about_palette, denoised_gray_image):
#     full_path_coordinate = skeleton_coordinate2(full_path_image)
#     # test = []
#     test = 0
#     for point in full_path_coordinate:
#         for palette_set in all_about_palette:
#             palette_skeleton = palette_set[1]
#             min = palette_set[4][0]
#             max = palette_set[4][1]
#             if point in palette_skeleton:
#                 test += 1
#                 # test.append(point)
#                 color_dense = denoised_gray_image[point[1]][point[0]]
#                 if min <= color_dense <= max:
#                     height = map_value(min, max, 100, 200, color_dense)
#                     point.append(height)
#                 else:
#                     point.append(None)
#             else:
#                 point.append(None)
#                 test += 1
#     return full_path_coordinate, test

def find_path_height(full_path_image, all_about_palette, denoised_gray_image):
    full_path_coordinate = skeleton_coordinate2(full_path_image)
    for point in full_path_coordinate:
        for palette_set in all_about_palette:
            palette_skeleton = palette_set[1]
            min = palette_set[4][0]
            max = palette_set[4][1]
            if point in palette_skeleton:
                # test.append(point)
                color_dense = denoised_gray_image[point[1]][point[0]]
                height = map_value(min, max, 100, 200, color_dense)
                point.append(300 - height)
    for point in full_path_coordinate:
        if len(point) != 3:
            point.append(None)
    return full_path_coordinate

# def fullfill_height(shortest_path, traject_point, full_path_with_height):

# def shortest_path(start_point, end_point, skeleton_image):
#     # kernel = [[x-1, y-1], [x-1, y], [x-1, y+1],
#     #           [x, y-1], [x, y], [x, y+1],
#     #           [x+1, y-1], [x+1, y], [x+1, y+1]]
#     # kernel = [[x-1, y+1], [x, y+1], [x+1, y+1],
#     #           [x-1, y], [x, y], [x+1, y],
#     #           [x-1, y-1], [x, y-1], [x+1, y-1]]
#     path = []
#     path.append(start_point)
#     ref = start_point
#     while ref != end_point:
#         x = ref[0]
#         y = ref[1]
#         kernel = [[x-1, y+1], [x, y+1], [x+1, y+1],
#                   [x-1, y], [x, y], [x+1, y],
#                   [x-1, y-1], [x, y-1], [x+1, y-1]]
#         for k in kernel:
#             if skeleton_image[k[1]][k[0]] == [255]:
#                 # print('kk')
#                 if k in path:
#                     None
#                 else:
#                     path.append([k[0], k[1]])
#             else:
#                 None
#         ref = path[-1]
#     return path

def is_next_to(point1, point2):
    ref1 = abs(point1[0] - point2[0])
    ref2 = abs(point1[1] - point2[1])
    if (ref1 + ref2) <= 2:
        if ref1 < 2 and ref2 < 2:
            return True
    else:
        return False

def is_point_in_list(point, list):
    count = 0
    for p in list:
        if p[0] == point[0] and p[1] == point[1]:
            count += 1
    if count == 0:
        return False
    else:
        return True

def shortest_pathh(start_point, end_point, full_path_skeleton_image):
    skeleton_co = skeleton_coordinate2(full_path_skeleton_image)
    ref_for_count = len(skeleton_coordinate2(full_path_skeleton_image))
    list1 = []
    test_count2 = []
    test_path2 = []
    for co in skeleton_co:
        if co[0] != start_point[0] or co[1] != start_point[1]:
            if is_next_to(start_point, co):
                list1.append(co)
                skeleton_co.remove(co)
    for p in list1:
        sk2 = skeleton_co
        path1 = []
        path1.append(start_point)
        path1.append(p)
        count = 0
        count2 = 0
        while len(sk2) != 0 and count <= ref_for_count:
            if path1[-1][0] == end_point[0] and path1[-1][1] == end_point[1]:
                print('mix sud lhor')
                break
            else:
                count += 1
                for co2 in sk2:
                    if is_next_to(path1[-1], co2):

                        # if is_point_in_list(co2, list1) == False:
                        if is_point_in_list(co2, path1) == False:
                            count2 +=1
                            path1.append(co2)
                            sk2.remove(co2)
        test_count2.append(count2)
        test_path2.append(path1)
    for path in test_path2:
        if path[0][0] == start_point[0] and path[0][1] == start_point[1]:
            if path[-1][0] == end_point[0] and path[-1][1] == end_point[1]:
                return path

def find_real_traject_point(start_point, end_point, original_traject_point, full_path_skeleton_image):
    traject_point2 = []
    for p in original_traject_point:
        if traject_point2 != []:
            if is_point_in_list(p[0], traject_point2) == False:
                traject_point2.append(p[0])
        else:
            traject_point2.append(p[0])
    short_path= shortest_pathh(start_point, end_point, full_path_skeleton_image)
    short_traject_path = []
    for point in short_path:
        for tp2 in traject_point2:
            if point[0] == tp2[0] and point[1] == tp2[1]:
                short_traject_path.append(point)
    return short_traject_path

def center_of_2_point(point1, point2):
    x = int((point1[0] + point2[0]) / 2)
    y = int((point1[1] + point2[1]) / 2)
    return [x, y]

def short_path_with_height(shortest_path, full_path_with_height):
    short_path_with_height = []
    for sp in shortest_path:
        for fph in full_path_with_height:
            if sp[0] == fph[0] and sp[1] == fph[1]:
                short_path_with_height.append([sp[0], sp[1], fph[2]])
    real_path = []
    stack = []
    ref = None
    for sp in short_path_with_height:
        if sp[2] == None:
            if ref != None:
                sp[2] = ref
                real_path.append(sp)
            else:
                stack.append(sp)
        else:
            if stack == []:
                real_path.append(sp)
                ref = sp[2]
            else:
                ref = sp[2]
                for p in stack:
                    p[2] = ref
                    real_path.append(p)
                real_path.append(sp)
                stack = []
    return real_path
