import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import numpy as np
from numpy import asarray
from PIL import Image, ImageEnhance

from delete_rail import *
from perspective import *
from all_aruco import *
from all_contour import *
from path_finder import *
from prepare_rail_2 import *

# %matplotlib qt

# checkpoint_center = []
# checkpoint_roi = []
#
# template_image = cv2.imread('prepared_template/0.jpg')
# field_image = cv2.imread('prepared_field/ready_field.jpg')
# cv2.imshow('original field', field_image)
# # field_image = cv2.fastNlMeansDenoisingColored(field_image,None, 10, 10,7,21)
# # cv2.imshow('fastNimean field', field_image)
# # field_image = cv2.GaussianBlur(field_image,(3,3),0)
# cv2.imshow('gaussian field', field_image)
# kernel_sharpening = np.array([[-1,-1,-1],
#                               [-1, 9,-1],
#                               [-1,-1,-1]])
# sharpened_field_image = cv2.filter2D(field_image, -1, kernel_sharpening)
# cv2.imshow('sharpened field', sharpened_field_image)
# field_image_blur = cv2.medianBlur(field_image,31)
# field_gray_blur = cv2.cvtColor(field_image_blur, cv2.COLOR_BGR2GRAY)
# # field_image = cv2.GaussianBlur(field_image,(51,51),0)
# cv2.imshow('fastNimean + medianBlur field', field_image_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# def nothing(x):
#     pass
#
# cv2.namedWindow('first contour setting')
# cv2.createTrackbar('canny LOW','first contour setting',0,255,nothing)
# cv2.createTrackbar('canny HIGH','first contour setting',100,255,nothing)
# cv2.createTrackbar('Gaussian kernel size','first contour setting',1,21,nothing)
# cv2.createTrackbar('Thickness','first contour setting',1,30,nothing)
#
# while(1):
#     # cv2.imshow('image',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         cv2.destroyAllWindows()
#         break
#     canny_low = cv2.getTrackbarPos('canny LOW','first contour setting')
#     canny_high = cv2.getTrackbarPos('canny HIGH','first contour setting')
#     gs = cv2.getTrackbarPos('Gaussian kernel size','first contour setting')
#     gs = ((gs+1) * 2) - 1
#     thickness = cv2.getTrackbarPos('Thickness','first contour setting')
#     field_gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
#     contours, hierarchy = find_contours(field_gray, canny_low, canny_high, gs, gs, 'tree')
#     first_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours, thickness)
#     for_track = first_contour_img.copy() * 0
#     cv2.imshow('first contour', first_contour_img)
#     cv2.imshow('first contour setting', for_track)
#
#
# # field_gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
# # contours, hierarchy = find_contours(field_gray, 40, 100, 3, 1, 'tree')
# # first_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours, 10)
# # cv2.imshow('first contour', first_contour_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# contours2, hierarchy2 = find_contours(first_contour_img, 40, 100, 3, 1, 'tree')
# second_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours2, 2)
# cv2.imshow('second contour', second_contour_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# contours3, hierarchy3 = find_contours(second_contour_img, 40, 100, 3, 1, 'external')
# third_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours3, 1)
# cv2.imshow('third contour', third_contour_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# square_contours, square_areas, peri_l = find_square_contours(contours3, field_image)
# square_contour_img = draw_contours(blank_image_with_same_size(field_gray), square_contours, 1)
# cv2.imshow('square contour', square_contour_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# peri_l
#
# for sc in square_contours:
#     center = find_contour_center(sc)
#     checkpoint_center.append(center)
#     roi = find_roi(sc, field_image)
#     checkpoint_roi.append(roi)
# c_field = field_image.copy()
# for p in checkpoint_center:
#     c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(0, 0, 255), thickness=10)
# cv2.imshow('checkpoint center', c_field)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for r in checkpoint_roi:
#     cv2.imshow('checkpoint ROI', r)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# # template_image = cv2.imread('prepared_template/prepared_template.jpg')
# cv2.imshow('template', template_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# template_location = template_matching_with_roi(template_image, checkpoint_roi, checkpoint_center, 360)
# c_field = field_image.copy()
# c_field = cv2.circle(c_field, (template_location[0], template_location[1]), radius=0, color=(0, 0, 255), thickness=10)
# cv2.imshow('start point', c_field)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# ending_location, res = find_endpoint(checkpoint_center, field_image)
# c_field = field_image.copy()
# c_field = cv2.circle(c_field, (ending_location[0], ending_location[1]), radius=0, color=(0, 0, 255), thickness=10)
# cv2.imshow('res endpoint', res)
# cv2.imshow('End point', c_field)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# pre_palette_contour, r = find_palette_by_checkpoint_area(contours3, square_areas, field_gray_blur)
# pre_palette_contour_img = draw_contours(blank_image_with_same_size(field_gray), pre_palette_contour, 5)
# cv2.imshow('pre palette contour', pre_palette_contour_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# palette_contour, palette_hie = find_contours(pre_palette_contour_img, 40, 100, 3, 1, 'external')
# palette_contour_img = draw_contours(blank_image_with_same_size(field_gray), palette_contour, 1)
# for cnt in palette_contour:
#     palette_contour_img2 = cv2.drawContours(blank_image_with_same_size(field_gray), [cnt], 0, (255, 255, 255), 1)
#     cv2.imshow('palette contour', palette_contour_img2)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# test = []
# all_about_palette = []
# for cnt in palette_contour:
#     elipson = 0.01 * cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, elipson, True)
#     # cv2.drawContours(ppppp ,[approx], -1, (255, 255, 255), 3)
#     # filled_palette_image = fill_contour([approx], field_image)
#     filled_palette_image = fill_contour([cnt], field_image)
#     # filled_palette_image = cv2.fillPoly(field_image, [cnt])
#     # filled_palette_image = cv2.drawContours(blank_image_with_same_size(field_image), cnt, (255, 255, 255), -1)
#     cv2.imshow('filled palette contour', filled_palette_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     kernel = np.ones((3,3),np.uint8)
#     dilate = cv2.dilate(filled_palette_image, kernel, iterations = 3)
#     erosion = cv2.erode(dilate, kernel, iterations = 1)
#     cv2.imshow('dilate', dilate)
#     cv2.imshow('erode', erosion)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     skeleton_image = skeletonize(erosion, method='lee')
#     # skeleton_image = skeleton_with_erotion(filled_palette_image, 20, 1)
#     cv2.imshow('skeletonized palette', skeleton_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     test.append(skeleton_image)
#     skeleton_coordinate = skeleton_coordinate2(skeleton_image)
#     c_field = field_image.copy()
#     for sc in skeleton_coordinate:
#         c_field[sc[1]][sc[0]] = [0, 0, 255]
#     cv2.imshow('check skeleton co', c_field)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     opened_point = find_opened_end(skeleton_coordinate, skeleton_image)
#     for op in opened_point:
#         c_field = cv2.circle(c_field, (op[0], op[1]), radius=0, color=(255, 0, 0), thickness=10)
#     cv2.imshow('check opened end point', c_field)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     all_in_one_palette = [skeleton_image, skeleton_coordinate, opened_point, cnt]
#     all_about_palette.append(all_in_one_palette)
#
# all_connected_point = find_connected_point(checkpoint_center, all_about_palette, 180)
# for cp in all_connected_point:
#     c_field = cv2.circle(c_field, (cp[0][0], cp[0][1]), radius=0, color=(255, 0, 255), thickness=10)
#     c_field = cv2.circle(c_field, (cp[1][0], cp[1][1]), radius=0, color=(255, 0, 255), thickness=10)
# cv2.imshow('check connected point', c_field)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# full_path_image = draw_full_path(all_connected_point, all_about_palette, field_image)
# cv2.imshow('full skeleton path', full_path_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# denoised_gray_field = cv2.fastNlMeansDenoising(field_gray_blur, None, 10, 7, 21)
# cv2.imshow('test', denoised_gray_field)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# find_min_max(full_path_image, all_about_palette, denoised_gray_field)
# full_path_with_height = find_path_height(full_path_image, all_about_palette, denoised_gray_field)
#
# from mpl_toolkits import mplot3d
# fig = plt.figure()
# ax = plt.axes(projection="3d")
# x = []
# y = []
# z = []
# for co in full_path_with_height:
#     # for co in co_list:
#     x.append(int(co[1]))
#     y.append(int(co[0]))
#     if co[2] != None:
#         z.append(int(co[2]))
#     else:
#         z.append(200)
# ax.scatter3D(x, y, z, 'gray')
# plt.show()
#
# countours, hierarchy = cv2.findContours(full_path_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# trajec_point_image = blank_image_with_same_size(field_gray)
#
# for cnt in countours:
#     peri = cv2.arcLength(cnt, True)
#     traject_point = cv2.approxPolyDP(cnt, 0.0015 * peri, True)
#     trajec_point_image = cv2.drawContours(trajec_point_image, traject_point, -1, (255, 255, 255), 10)
#
# cv2.imshow("trajec point image", trajec_point_image)
# cv2.imshow('full skeleton path', full_path_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# def is_next_to2(point1, point2):
#     pre_next_to = []
#     x1 = point1[0]
#     y1 = point1[1]
#     x2 = point2[0]
#     y2 = point2[1]
#     result = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
#     if result < 3:
#         return True
#     else:
#         return False
#
# def shortest_path3(start_point, end_point, full_path_skeleton_image):
#     skeleton_co1 = skeleton_coordinate2(full_path_skeleton_image)
#     if is_point_in_list(start_point, skeleton_co1) and is_point_in_list(end_point, skeleton_co1):
#         print('all point in list')
#     else:
#         print('point not in path')
#     next_to_start = []
#     for p in skeleton_co1:
#         if is_next_to2(start_point, p):
#             next_to_start.append(p)
#             # print('next_to_start = ' + str(next_to_start)
#     for p in next_to_start:
#         short_image_test = blank_image_with_same_size(full_path_skeleton_image)
#         path1 = []
#         skeleton_co2 = skeleton_coordinate2(full_path_skeleton_image)
#         path1.append(start_point)
#         path1.append(p)
#         pre_next_to2 = []
#         count_ref = len(skeleton_coordinate2(full_path_skeleton_image))
#         current_count = 0
#         while True:
#             # print('path1 = ' + str(path1) + '\n')
#             if current_count > count_ref * 2:
#                 break
#             if path1[-1][0] == end_point[0] and path1[-1][1] == end_point[1]:
#                 print('mix sud lhor')
#                 return path1
#                 break
#             current_count += 1
#             for p2 in skeleton_co2:
#                 if is_next_to2(path1[-1], p2):
#                     if is_point_in_list(p2, path1) == False:
#                         if is_point_in_list(p2, next_to_start) == False:
#                             pre_next_to2.append(p2)
#                         else:
#                             None
#                     else:
#                         None
#                 else:
#                     None
#             real_next_to_point = pre_next_to2[0]
#             distance1 = abs(real_next_to_point[0] - path1[-1][0]) + abs(real_next_to_point[1] - path1[-1][1])
#             for p3 in pre_next_to2:
#                 distance2 = abs(p3[0] - path1[-1][0]) + abs(p3[1] - path1[-1][1])
#                 if distance2 < distance1:
#                     real_next_to_point = p3
#             path1.append(real_next_to_point)
#             skeleton_co2.remove(real_next_to_point)
#             pre_next_to2 = []
#     return path1
#
#
#
#     return start_point, next_to_start
#
#
#
# path1_test = shortest_path3(template_location, ending_location, full_path_image)
#
#
# stp, traject2 = find_real_traject_point(template_location, ending_location, traject_point, full_path_image)
# c_field = field_image.copy()
# for p in stp:
#     c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(255, 0, 255), thickness=10)
#     cv2.imshow('pp', c_field)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# for x in range(full_path_image.shape[0]):
#     for y in range(full_path_image.shape[1]):
#         if x == template_location[0] and y == template_location[1]:
#             if full_path_image[y][x] == 255:
#                 print('found template')
#         if x == ending_location[0] and y == ending_location[1]:
#             if full_path_image[y][x] == 255:
#                 print('found end point')
#
# short_path = shortest_pathh(template_location, ending_location, full_path_image)
# c_field = field_image.copy()
# for p in short_path:
#     c_field[p[1]][p[0]] = [255, 0, 255]
# cv2.imshow('mmm', c_field)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# short_path_height = short_path_with_height(short_path, full_path_with_height)
# traject_point_with_height = []
# for p in stp:
#     for p2 in short_path_height:
#         if p[0] == p2[0] and p[1] == p2[1]:
#             if traject_point_with_height != []:
#                 if distance_between_point([traject_point_with_height[-1][0], traject_point_with_height[-1][1]], p) > 30:
#                     traject_point_with_height.append([p[0], p[1], p2[2]])
#             else:
#                     traject_point_with_height.append([p[0], p[1], p2[2]])
#
# traject_point_with_height
# c_field = field_image.copy()
# for p in traject_point_with_height:
#     c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(0, 255, 255), thickness=10)
#     cv2.imshow('traject point with height', c_field)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# short_path_height
# from mpl_toolkits import mplot3d
# fig = plt.figure()
# ax = plt.axes(projection="3d")
# x = []
# y = []
# z = []
# for co in short_path_height:
#     # for co in co_list:
#     x.append(int(co[1]))
#     y.append(int(co[0]))
#     z.append(co[2])
# ax.scatter3D(x, y, z, 'gray')
# plt.show()
#
# world_traject_point = []
# for i in traject_point_with_height:
#     world_traject_point.append(image_to_world(i, 0, 0, 0, 600, 600, 430, 430))
# world_traject_point
#
# swap = []
# for p in world_traject_point:
#     x = p[1]
#     y = p[0]
#     z = p[2]
#     swap.append([x, y, z])
# swap
#
#
# c_field = field_image.copy()
# for p in world_traject_point:
#     c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(0, 255, 255), thickness=10)
#     cv2.imshow('traject point with height', c_field)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def all_main_demo2():
    delete_rail()
    checkpoint_center = []
    checkpoint_roi = []

    template_image = cv2.imread('prepared_template/0.jpg')
    field_image = cv2.imread('prepared_field/pera2.jpg')
    cv2.imshow('original field', field_image)
    # field_image = cv2.fastNlMeansDenoisingColored(field_image,None, 10, 10,7,21)
    # cv2.imshow('fastNimean field', field_image)
    # field_image = cv2.GaussianBlur(field_image,(3,3),0)
    cv2.imshow('gaussian field', field_image)
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    sharpened_field_image = cv2.filter2D(field_image, -1, kernel_sharpening)
    cv2.imshow('sharpened field', sharpened_field_image)
    field_image_blur = cv2.medianBlur(field_image,31)
    field_gray_blur = cv2.cvtColor(field_image_blur, cv2.COLOR_BGR2GRAY)
    # field_image = cv2.GaussianBlur(field_image,(51,51),0)
    cv2.imshow('fastNimean + medianBlur field', field_image_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    def nothing(x):
        pass

    cv2.namedWindow('first contour setting')
    cv2.createTrackbar('canny LOW','first contour setting',0,255,nothing)
    cv2.createTrackbar('canny HIGH','first contour setting',100,255,nothing)
    cv2.createTrackbar('Gaussian kernel size','first contour setting',1,21,nothing)
    cv2.createTrackbar('Thickness','first contour setting',1,30,nothing)

    while(1):
        # cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
        canny_low = cv2.getTrackbarPos('canny LOW','first contour setting')
        canny_high = cv2.getTrackbarPos('canny HIGH','first contour setting')
        gs = cv2.getTrackbarPos('Gaussian kernel size','first contour setting')
        gs = ((gs+1) * 2) - 1
        thickness = cv2.getTrackbarPos('Thickness','first contour setting')
        field_gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = find_contours(field_gray, canny_low, canny_high, gs, gs, 'tree')
        first_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours, thickness)
        for_track = first_contour_img.copy() * 0
        cv2.imshow('first contour', first_contour_img)
        cv2.imshow('first contour setting', for_track)


    # field_gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
    # contours, hierarchy = find_contours(field_gray, 40, 100, 3, 1, 'tree')
    # first_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours, 10)
    # cv2.imshow('first contour', first_contour_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours2, hierarchy2 = find_contours(first_contour_img, 40, 100, 3, 1, 'tree')
    second_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours2, 2)
    cv2.imshow('second contour', second_contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    contours3, hierarchy3 = find_contours(second_contour_img, 40, 100, 3, 1, 'external')
    third_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours3, 1)
    cv2.imshow('third contour', third_contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    square_contours, square_areas, perimiter = find_square_contours(contours3, field_image)
    square_contour_img = draw_contours(blank_image_with_same_size(field_gray), square_contours, 1)
    cv2.imshow('square contour', square_contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    for sc in square_contours:
        center = find_contour_center(sc)
        checkpoint_center.append(center)
        roi = find_roi(sc, field_image)
        checkpoint_roi.append(roi)
    c_field = field_image.copy()
    for p in checkpoint_center:
        c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(0, 0, 255), thickness=10)
    cv2.imshow('checkpoint center', c_field)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for r in checkpoint_roi:
        cv2.imshow('checkpoint ROI', r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # template_image = cv2.imread('prepared_template/prepared_template.jpg')
    cv2.imshow('template', template_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    template_location = template_matching_with_roi(template_image, checkpoint_roi, checkpoint_center, 360)
    c_field = field_image.copy()
    c_field = cv2.circle(c_field, (template_location[0], template_location[1]), radius=0, color=(0, 0, 255), thickness=10)
    cv2.imshow('start point', c_field)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ending_location, res = find_endpoint(checkpoint_center, field_image)
    c_field = field_image.copy()
    c_field = cv2.circle(c_field, (ending_location[0], ending_location[1]), radius=0, color=(0, 0, 255), thickness=10)
    cv2.imshow('res endpoint', res)
    cv2.imshow('End point', c_field)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pre_palette_contour, r = find_palette_by_checkpoint_area(contours3, square_areas, field_gray_blur)
    pre_palette_contour_img = draw_contours(blank_image_with_same_size(field_gray), pre_palette_contour, 5)
    cv2.imshow('pre palette contour', pre_palette_contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    palette_contour, palette_hie = find_contours(pre_palette_contour_img, 40, 100, 3, 1, 'external')
    palette_contour_img = draw_contours(blank_image_with_same_size(field_gray), palette_contour, 1)
    for cnt in palette_contour:
        palette_contour_img2 = cv2.drawContours(blank_image_with_same_size(field_gray), [cnt], 0, (255, 255, 255), 1)
        cv2.imshow('palette contour', palette_contour_img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    test = []
    all_about_palette = []
    for cnt in palette_contour:
        elipson = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, elipson, True)
        # cv2.drawContours(ppppp ,[approx], -1, (255, 255, 255), 3)
        filled_palette_image = fill_contour([approx], field_image)
        # filled_palette_image = cv2.fillPoly(field_image, [cnt])
        # filled_palette_image = cv2.drawContours(blank_image_with_same_size(field_image), cnt, (255, 255, 255), -1)
        cv2.imshow('filled palette contour', filled_palette_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        skeleton_image = skeleton_with_erotion(filled_palette_image, 18, 1)
        cv2.imshow('skeletonized palette', skeleton_image)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        test.append(skeleton_image)
        skeleton_coordinate = skeleton_coordinate2(skeleton_image)
        c_field = field_image.copy()
        for sc in skeleton_coordinate:
            c_field[sc[1]][sc[0]] = [0, 0, 255]
        cv2.imshow('check skeleton co', c_field)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        opened_point = find_opened_end(skeleton_coordinate, skeleton_image)
        for op in opened_point:
            c_field = cv2.circle(c_field, (op[0], op[1]), radius=0, color=(255, 0, 0), thickness=10)
        cv2.imshow('check opened end point', c_field)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        all_in_one_palette = [skeleton_image, skeleton_coordinate, opened_point, cnt]
        all_about_palette.append(all_in_one_palette)

    all_connected_point = find_connected_point(checkpoint_center, all_about_palette, 180)
    for cp in all_connected_point:
        c_field = cv2.circle(c_field, (cp[0][0], cp[0][1]), radius=0, color=(255, 0, 255), thickness=10)
        c_field = cv2.circle(c_field, (cp[1][0], cp[1][1]), radius=0, color=(255, 0, 255), thickness=10)
    cv2.imshow('check connected point', c_field)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    full_path_image = draw_full_path(all_connected_point, all_about_palette, field_image)
    cv2.imshow('full skeleton path', full_path_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    denoised_gray_field = cv2.fastNlMeansDenoising(field_gray_blur, None, 10, 7, 21)
    cv2.imshow('test', denoised_gray_field)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    find_min_max(full_path_image, all_about_palette, denoised_gray_field)
    full_path_with_height = find_path_height(full_path_image, all_about_palette, denoised_gray_field)
    full_path_with_height
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x = []
    y = []
    z = []
    for co in full_path_with_height:
        # for co in co_list:
        x.append(int(co[1]))
        y.append(int(co[0]))
        if co[2] != None:
            z.append(int(co[2]))
        else:
            z.append(200)
    ax.scatter3D(x, y, z, 'gray')
    plt.show()

    countours, hierarchy = cv2.findContours(full_path_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    trajec_point_image = blank_image_with_same_size(field_gray)

    for cnt in countours:
        peri = cv2.arcLength(cnt, True)
        traject_point = cv2.approxPolyDP(cnt, 0.0015 * peri, True)
        trajec_point_image = cv2.drawContours(trajec_point_image, traject_point, -1, (255, 255, 255), 10)
    traject_point
    pre_traject_point = []
    for ap in traject_point:
        pre_traject_point.append(ap.tolist())
    pre_traject_point
    for ap in all_about_palette:
        for p in ap[2]:
            pre_traject_point.append([p])
    pre_traject_point

    cv2.imshow("trajec point image", trajec_point_image)
    cv2.imshow('full skeleton path', full_path_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    full_path_list_test = []
    for x in range(full_path_image.shape[0]):
        for y in range(full_path_image.shape[1]):
            if full_path_image[y][x] == 255:
                full_path_list_test.append([x, y])

    full_path_list_test
    template_location
    ending_location
    for p in full_path_list_test:
        if p[0] == template_location[0] and p[1] == template_location[1]:
            print('template is on path')
        if p[0] == ending_location[0] and p[1] == ending_location[1]:
            print('ending is on path')

    stp, traject2 = find_real_traject_point(template_location, ending_location, pre_traject_point, full_path_image)
    c_field = field_image.copy()
    for p in stp:
        c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(255, 0, 255), thickness=10)
    cv2.imshow('pp', c_field)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    def find_nearest_point(point, list_of_point):
        plus_point = []
        multiply_point = []
        condition_point = []
        for p in list_of_point:
            distance = abs(point[0] - p[0]) + abs(point[1] - p[1])
            if distance == 1:
                plus_point.append(p)
            if distance == 2:
                multiply_point.append(p)
        if multiply_point == []:
            return plus_point
        else:
            for p in plus_point:
                condition_point.append(p)
            for p in multiply_point:
                count = 0
                for p2 in plus_point:
                    distance2 = abs(p[0] - p2[0]) + abs(p[1] - p2[1])
                    if distance2 == 1:
                        count += 1
                if count == 0:
                    condition_point.append(p)
            return condition_point

    def shortest_path4(start_point, end_point, skeleton_path_image):
        skeleton_co1 = skeleton_coordinate2(skeleton_path_image)
        nest_start = find_nearest_point(start_point, skeleton_co1)
        for np in nest_start:
            skeleton_co2 = skeleton_coordinate2(skeleton_path_image)
            count = 0
            path1 = []
            path1.append(start_point)
            path1.append(np)
            skeleton_co2.remove(start_point)
            skeleton_co2.remove(np)
            while True:
                count += 1
                print(count)
                if count > len(skeleton_coordinate2(skeleton_path_image)):
                    break
                if path1[-1][0] == end_point[0] and path1[-1][1] == end_point[1]:
                    print('mix sud lhor')
                    return path1
                    break
                np2 = find_nearest_point(path1[-1], skeleton_co2)
                path1.append(np2[0])
                skeleton_co2.remove(np2[0])
            print(count)



#     import skimage.graph
# ### give start (y1,x1) and end (y2,x2) and the binary maze image as input
#     # full_path_image[115][489]
#     template_location
#     ending_location
#     def shortest_path(start,end,binary):
#         costs=np.where(binary,255,1000)
#         path, cost = skimage.graph.route_through_array(
#             costs, start=start, end=end, fully_connected=True)
#         return path,cost
#     p, c = shortest_path((115, 489), (332, 333), full_path_image)
    # sp_test = shortest_path4(template_location, ending_location, full_path_image)
    # template_location
    # ending_location
    short_path_test = shortest_pathh(template_location, ending_location, full_path_image)
    short_path_test[0]
    short_path = shortest_pathh(template_location, ending_location, full_path_image)
    c_field = field_image.copy()
    for p in short_path:
        c_field[p[1]][p[0]] = [255, 0, 255]
    cv2.imshow('path', c_field)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    short_path_height = short_path_with_height(short_path, full_path_with_height)
    traject_point_with_height = []
    for p in stp:
        for p2 in short_path_height:
            if p[0] == p2[0] and p[1] == p2[1]:
                if traject_point_with_height != []:
                    if distance_between_point([traject_point_with_height[-1][0], traject_point_with_height[-1][1]], p) > 30:
                        traject_point_with_height.append([p[0], p[1], p2[2]])
                else:
                        traject_point_with_height.append([p[0], p[1], p2[2]])

    traject_point_with_height
    c_field = field_image.copy()
    for p in traject_point_with_height:
        c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(0, 255, 255), thickness=10)
        cv2.imshow('traject point with height', c_field)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    short_path_height
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x = []
    y = []
    z = []
    for co in short_path_height:
        # for co in co_list:
        x.append(int(co[1]))
        y.append(int(co[0]))
        z.append(co[2])
    ax.scatter3D(x, y, z, 'gray')
    plt.show()

    threshold_traject_point = []
    for p in traject_point_with_height:
        pre_point = []
        pre_point.append(p[0])
        pre_point.append(p[1])
        if p[2] > 150:
            pre_point.append(200)
        else:
            pre_point.append(100)
        threshold_traject_point.append(pre_point)
    world_traject_point = []
    before_first_point = image_to_world([threshold_traject_point[0][0], threshold_traject_point[0][1], 400], 0, 0, 0, 600, 600, 430, 430)
    world_traject_point.append(before_first_point)
    for i in threshold_traject_point:
        world_traject_point.append(image_to_world(i, 0, 0, 0, 600, 600, 430, 430))
    correct_some_point = []
    correct_some_point.append(world_traject_point[0])
    for p in range(len(world_traject_point)):
        last = correct_some_point[-1]
        pre_point = []
        if p != 0:
            if abs(world_traject_point[p][0] - last[0]) < 5:
                pre_point.append(last[0])
            else:
                pre_point.append(world_traject_point[p][0])
            if abs(world_traject_point[p][1] - last[1]) < 5:
                pre_point.append(last[1])
            else:
                pre_point.append(world_traject_point[p][1])
            pre_point.append(world_traject_point[p][2])
            correct_some_point.append(pre_point)

    return world_traject_point, correct_some_point, traject_point_with_height, threshold_traject_point



# w, c, t, tt = all_main_demo2()
# w
# c
# t
# tt
