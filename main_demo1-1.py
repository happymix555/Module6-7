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

%matplotlib qt

checkpoint_center = []
checkpoint_roi = []

field_image = cv2.imread('prepared_field/ready_field.jpg')
cv2.imshow('original field', field_image)
field_image = cv2.fastNlMeansDenoisingColored(field_image,None,10,10,7,21)
cv2.imshow('fastNimean field', field_image)
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
sharpened_field_image = cv2.filter2D(field_image, -1, kernel_sharpening)
cv2.imshow('sharpened field', sharpened_field_image)
field_image = cv2.medianBlur(field_image,31)
# field_image = cv2.GaussianBlur(field_image,(51,51),0)
cv2.imshow('fastNimean + medianBlur field', field_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
def nothing(x):
    pass

cv2.namedWindow('first contour')
cv2.createTrackbar('canny LOW','first contour',0,255,nothing)
cv2.createTrackbar('canny HIGH','first contour',100,255,nothing)
cv2.createTrackbar('Gaussian kernel size','first contour',1,21,nothing)

while(1):
    # cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break
    canny_low = cv2.getTrackbarPos('canny LOW','first contour')
    canny_high = cv2.getTrackbarPos('canny HIGH','first contour')
    gs = cv2.getTrackbarPos('Gaussian kernel size','first contour')
    gs = ((gs+1) * 2) - 1
    field_gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = find_contours(field_gray, canny_low, canny_high, gs, gs, 'tree')
    first_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours, 10)
    cv2.imshow('first contour', first_contour_img)


# field_gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
# contours, hierarchy = find_contours(field_gray, 40, 100, 3, 1, 'tree')
# first_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours, 10)
# cv2.imshow('first contour', first_contour_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours2, hierarchy2 = find_contours(first_contour_img, 40, 100, 3, 1, 'tree')
second_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours2, 3)
cv2.imshow('second contour', second_contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


contours3, hierarchy3 = find_contours(second_contour_img, 40, 100, 3, 1, 'external')
third_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours3, 1)
cv2.imshow('second contour', third_contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

square_contours, square_areas = find_square_contours(contours3, field_image)
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

template_image = cv2.imread('prepared_template/prepared_template.jpg')
cv2.imshow('template', template_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

template_location = template_matching_with_roi(template_image, checkpoint_roi, checkpoint_center, 360)
c_field = field_image.copy()
c_field = cv2.circle(c_field, (template_location[0], template_location[1]), radius=0, color=(0, 0, 255), thickness=10)
cv2.imshow('start point', c_field)
cv2.waitKey(0)
cv2.destroyAllWindows()

ending_location = find_endpoint(checkpoint_center, field_image)
c_field = field_image.copy()
c_field = cv2.circle(c_field, (ending_location[0], ending_location[1]), radius=0, color=(0, 0, 255), thickness=10)
cv2.imshow('End point', c_field)
cv2.waitKey(0)
cv2.destroyAllWindows()

pre_palette_contour = find_palette_by_checkpoint_area(contours3, square_areas)
pre_palette_contour_img = draw_contours(blank_image_with_same_size(field_gray), pre_palette_contour, 3)
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
    filled_palette_image = fill_contour([cnt], field_image)
    cv2.imshow('filled palette contour', filled_palette_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    skeleton_image = skeleton_with_erotion(filled_palette_image, 15)
    cv2.imshow('skeletonized palette', skeleton_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    test.append(skeleton_image)
    skeleton_coordinate = skeleton_coordinate2(skeleton_image)
    c_field = field_image.copy()
    for sc in skeleton_coordinate:
        c_field[sc[1]][sc[0]] = [0, 0, 255]
    cv2.imshow('check skeleton co', c_field)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    opened_point = find_opened_end(skeleton_coordinate, skeleton_image)
    for op in opened_point:
        c_field = cv2.circle(c_field, (op[0], op[1]), radius=0, color=(255, 0, 0), thickness=10)
    cv2.imshow('check opened end point', c_field)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    all_in_one_palette = [skeleton_image, skeleton_coordinate, opened_point, cnt]
    all_about_palette.append(all_in_one_palette)

all_connected_point = find_connected_point(checkpoint_center, all_about_palette, 200)
for cp in all_connected_point:
    c_field = cv2.circle(c_field, (cp[0][0], cp[0][1]), radius=0, color=(255, 0, 0), thickness=10)
    c_field = cv2.circle(c_field, (cp[1][0], cp[1][1]), radius=0, color=(255, 0, 0), thickness=10)
cv2.imshow('check connected point', c_field)
cv2.waitKey(0)
cv2.destroyAllWindows()

full_path_image = draw_full_path(all_connected_point, all_about_palette, field_image)
cv2.imshow('full skeleton path', full_path_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

denoised_gray_field = cv2.fastNlMeansDenoising(field_gray, None, 10, 7, 21)
cv2.imshow('test', denoised_gray_field)
cv2.waitKey(0)
cv2.destroyAllWindows()
find_min_max(full_path_image, all_about_palette, denoised_gray_field)
full_path_with_height = find_path_height(full_path_image, all_about_palette, denoised_gray_field)

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
    traject_point = cv2.approxPolyDP(cnt, 0.01 * peri, True)
    trajec_point_image = cv2.drawContours(trajec_point_image, traject_point, -1, (255, 255, 255), 10)

cv2.imshow("trajec point image", trajec_point_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

stp = find_real_traject_point(template_location, ending_location, traject_point, full_path_image)
c_field = field_image.copy()
for p in stp:
    c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(255, 0, 255), thickness=10)
    cv2.imshow('pp', c_field)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

short_test = shortest_pathh(template_location, ending_location, full_path_image)
c_field = field_image.copy()
for p in short_test:
    c_field[p[1]][p[0]] = [255, 0, 255]
cv2.imshow('pp', c_field)
cv2.waitKey(0)
cv2.destroyAllWindows()

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
    return real_path, short_path_with_height
len(short_test)
len(full_path_with_height)
short_path_height_test, test_for_short = short_path_with_height(short_test, full_path_with_height)
len(short_path_height_test)
len(test_for_short)
test_for_short
traject_point_with_height = []
for p in stp:
    for p2 in short_path_height_test:
        if p[0] == p2[0] and p[1] == p2[1]:
            traject_point_with_height.append([p[0], p[1], p2[2]])
traject_point_with_height
short_path_height_test
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection="3d")
x = []
y = []
z = []
for co in short_path_height_test:
    # for co in co_list:
    x.append(int(co[1]))
    y.append(int(co[0]))
    z.append(co[2])
ax.scatter3D(x, y, z, 'gray')
plt.show()


c_field = field_image.copy()
for p in short_test:
    c_field[p[1]][p[0]] = [255, 0, 255]
cv2.imshow('pp', c_field)
cv2.waitKey(0)
cv2.destroyAllWindows()
