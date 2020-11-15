#import library and load field image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.morphology import medial_axis
import imutils
from skimage.morphology import skeletonize
from skimage.morphology import thin
from skimage.util import invert
from skimage import filters
from PIL import Image
# from mpl_toolkits import mplot3d
%matplotlib qt

# from shapely.geometry import Polygon
# from centerline.geometry import Centerline

field = cv2.imread('test_field.JPG')
cv2.imshow('field', field)
cv2.waitKey(0)
cv2.destroyAllWindows()

#convert field to grayscale image
gray_field = cv2.imread('test_field.JPG',0)
cv2.imshow('gray field', gray_field)
cv2.waitKey(0)
cv2.destroyAllWindows()

#blur image and perform canny edge detection
gray_field = cv2.imread('test_field.JPG',0)
blur_gray_field = cv2.GaussianBlur(gray_field, (5, 5), 0)
# cv2.imshow('field_blur', blur_gray_field)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
field_canny_edge = cv2.Canny(blur_gray_field, 40, 50)
# cv2.imshow('field_canny_edge', field_canny_edge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#find External Contours
contours, hierarchy = cv2.findContours(field_canny_edge,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

blank2 = np.zeros((gray_field.shape[0], gray_field.shape[1], 1), np.uint8)
external_contour = cv2.drawContours(blank2, contours, -1,  (255, 255, 255), 1)


# #find External Contours
# contours, hierarchy = cv2.findContours(field_canny_edge,
#                                         cv2.RETR_EXTERNAL,
#                                         cv2.CHAIN_APPROX_SIMPLE)
#
# blank2 = np.zeros((gray_field.shape[0], gray_field.shape[1], 1), np.uint8)
# external_contour = cv2.drawContours(blank2, contours, -1,  (255, 255, 255), 1)
# # cv2.drawContours(blank2, contours, 1, (255, 255, 255), 1)
# # cv2.drawContours(blank2, contours, 8, (255, 255, 255), 1)
# cv2.imshow('external_contour', external_contour)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ret3,blank2_binary = cv2.threshold(blank2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('blank_binary', blank2_binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
blank_img = np.zeros((gray_field.shape[0], gray_field.shape[1], 1), np.uint8)
filled_contour = cv2.fillPoly(blank_img, contours, color=(255,255,255))
# cv2.imshow('filled contour', filled_contour)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(filled_contour.shape)

# filled_contour_binary = filled_contour > filters.threshold_otsu(filled_contour_binary)
ret3,filled_contour_binary = cv2.threshold(filled_contour,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print(filled_contour_binary.shape)

from skimage.morphology import thin
thin = thin(filled_contour_binary)
# print(filled_contour_binary.shape)
# plt.imshow(thin, cmap='gray')

thin = thin.astype(np.uint8)
# print(thin.shape)
contours, hierarchy = cv2.findContours(thin,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
field = cv2.imread('test_field.JPG')
blank_img = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
thin = cv2.drawContours(blank_img, contours, -1,  (255, 255, 255), 1)
plt.imshow(thin)


contours, hierarchy = cv2.findContours(field_canny_edge,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
path = []
parent = 1
for hie in range(len(hierarchy[0])):
    hie2 = hierarchy[0][hie]
    # print(hie2)
    if hie2[0] == -1:
        if hie2[1] == -1:
            if hie2[2] == -1:
                if hierarchy[0][hie2[3]][3] == 1:
                    path.append(hie)
path
field = cv2.imread('test_field.JPG')
blank_img = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
for cnt in path:
    palette = cv2.drawContours(blank_img, contours, cnt, (255, 0, 0), 2)
plt.imshow(palette, cmap='gray')

blur_palette = cv2.GaussianBlur(palette, (5, 5), 0)
palette_canny = cv2.Canny(blur_palette, 40, 100)
plt.imshow(palette_canny, cmap='gray')
contours_for_palette, hierarchy = cv2.findContours(palette_canny,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

def map_value(input, min1, max1, min2, max2):
    return(input * ((max2 - min2) / (max1 - min1)))

field = cv2.imread('test_field.JPG')
gray_field = cv2.imread('test_field.JPG', 0)
plt.imshow(gray_field, cmap='gray')
red = [0,0,255]
palette_plot = []
blank_img2 = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
indices = np.where(thin == [255])
for pixel in range(len(indices[0])):
    x = indices[0][pixel] #x coordinate of particular pixel
    y = indices[1][pixel] #y coordinate of particular pixel
    for cnt in range(len(contours_for_palette)):
        if_in_contour = cv2.pointPolygonTest(contours_for_palette[cnt],(y,x),False)
        if if_in_contour == 1:
            color = gray_field[y, x]
            gray_field[x, y] = 255
            z = map_value(color, 0, 255, 0, 20)
            palette_plot.append([x, y, color])
plt.imshow(cv2.cvtColor(gray_field, cv2.COLOR_BGR2RGB))

from mpl_toolkits import mplot3d
range(len(palette_plot))
palette_plot[0]
fig = plt.figure()
ax = plt.axes(projection="3d")
x_line = []
y_line = []
z_line = []
for co in palette_plot:
    x_line.append(int(co[0]))
    y_line.append(int(co[1]))
    z_line.append(int(co[2]))
ax.scatter3D(x_line, y_line, z_line, 'gray')
plt.show()




blur_palette = cv2.GaussianBlur(palette, (5, 5), 0)
palette_canny = cv2.Canny(blur_palette, 40, 100)
plt.imshow(palette_canny, cmap='gray')
contours, hierarchy = cv2.findContours(palette_canny,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
blank_img2 = np.zeros((field.shape[0], field.shape[1], 1), np.uint8)
filled_contour = cv2.fillPoly(blank_img2, contours, color=(255,255,255))
cv2.imshow('filled contour', filled_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()

# filled_contour_binary = filled_contour > filters.threshold_otsu(filled_contour_binary)
ret3,filled_contour_binary = cv2.threshold(filled_contour,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(filled_contour_binary.shape)

from skimage.morphology import thin
skeleton = skeletonize(filled_contour, method='lee')
print(filled_contour_binary.shape)
plt.imshow(skeleton, cmap='gray')

thin = thin.astype(np.uint8)
print(thin.shape)
contours, hierarchy = cv2.findContours(skeleton,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
field = cv2.imread('test_field.JPG')
blank_img = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
skeleton_on_field = cv2.drawContours(field, contours, -1,  (255, 255, 255), 1)
plt.imshow(skeleton_on_field)


contours, hierarchy = cv2.findContours(field_canny_edge,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
path = []
parent = 1
for hie in range(len(hierarchy[0])):
    hie2 = hierarchy[0][hie]
    # print(hie2)
    if hie2[0] == -1:
        if hie2[1] == -1:
            if hie2[2] == -1:
                if hierarchy[0][hie2[3]][3] == 1:
                    path.append(hie)
path
blank_img = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
for cnt in path:
    palette = cv2.drawContours(blank_img, contours, cnt, (255, 255, 255), 1)
plt.imshow(palette, cmap='gray')
# palette = palette.astype(np.uint8)
# plt.imshow(palette)
blur_palette = cv2.GaussianBlur(palette, (5, 5), 0)
palette_canny = cv2.Canny(blur_palette, 40, 100)
plt.imshow(palette_canny, cmap='gray')
contours_for_palette, hierarchy = cv2.findContours(palette_canny,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
blank_img2 = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
palette2 = cv2.drawContours(blank_img2, contours_for_palette, -1, (255, 255, 255), 1)
plt.imshow(palette2)

red = [0,0,255]
blank_img2 = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
indices = np.where(thin == [255])
for pixel in range(len(indices[0])):
    x = indices[0][pixel]
    y = indices[1][pixel]
    for cnt in range(len(contours_for_palette)):
        if_in_contour = cv2.pointPolygonTest(contours_for_palette[cnt],(y,x),False)
        if if_in_contour == 1:
            field[x, y] = red
plt.imshow(cv2.cvtColor(field, cv2.COLOR_BGR2RGB))

for pixel in range(len(indices[0])):
    x = indices[0][pixel]
    y = indices[1][pixel]
    if_in_contour = cv2.pointPolygonTest(contours_for_palette[0],(y,x),False)
    if if_in_contour == True:
        blank_img[x, y] = red
plt.imshow(cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB))

contours, hierarchy = cv2.findContours(field_canny_edge,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
path = []
parent = 1
for hie in range(len(hierarchy[0])):
    hie2 = hierarchy[0][hie]
    # print(hie2)
    if hie2[0] == -1:
        if hie2[1] == -1:
            if hie2[2] == -1:
                if hierarchy[0][hie2[3]][3] == 1:
                    path.append(hie)
path

#create black blank image
blank_img = np.zeros((field.shape[0], field.shape[1], 1), np.uint8)
for cnt in path:
    palette = cv2.drawContours(blank_img, contours, cnt, (255, 255, 255), 1)
plt.imshow(palette, cmap='gray')

contours, hierarchy = cv2.findContours(blank_img,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

#gaussianblur and otsu thresholding blank_img with contours drawn on it
blank_blur = cv2.GaussianBlur(blank_img,(5,5),0)
ret3,blank_binary = cv2.threshold(blank_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('blank_binary', blank_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

#filling contours
cv2.fillPoly(blank_binary, contours, color=(255,255,255))
cv2.imshow('blank_binary', blank_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(blank_binary,kernel,iterations = 10)
cv2.imshow('erosion', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#find medial axis from Contours
print(blank_binary.shape)
skeleton = medial_axis(erosion).astype(np.uint8)
# Show
cv2.imshow("skeleton", skeleton*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

#read and process template image
star = cv2.imread('star.PNG')
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(star)
ax.set_title("Original star template")

star = cv2.cvtColor(star, cv2.COLOR_BGR2GRAY)
cv2.imshow("star", star)
cv2.waitKey(0)
cv2.destroyAllWindows()

star = cv2.Canny(star, 100, 250)
cv2.imshow("star", star)
cv2.waitKey(0)
cv2.destroyAllWindows()

#reseize template image to equal to field image
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

field = cv2.imread('test_field.JPG')
# re = cv2.resize(star, (int(1 * star.shape[0]), int(1 * star.shape[1])))
star = cv2.imread('star.PNG', 0)
fullsize_star = resize_img(star, field)
cv2.imshow("star" ,star)
cv2.imshow('full size star', fullsize_star)
cv2.waitKey(0)
cv2.destroyAllWindows()

resized = imutils.resize(fullsize_star, width = int(fullsize_star.shape[1] * 0.1))
cv2.imshow('resize star', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Get template(star) height and weight
(tH, tW) = star.shape[:2]
tH
tW
cv2.imshow("star", star)
cv2.waitKey(0)
cv2.destroyAllWindows()

rgb_field = cv2.imread('test_field.JPG')
field = cv2.imread('test_field.JPG', 0)
star = cv2.imread('star.PNG', 0)
w, h = star.shape[::-1]
result = cv2.matchTemplate(field, star, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
min_val
max_val
min_loc
max_loc
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
print(top_left)
print(buttom_right)
cv2.rectangle(rgb_field,top_left, bottom_right, 255, 2)
cv2.imshow("result", rgb_field)
cv2.waitKey(0)
cv2.destroyAllWindows()


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

def template_match(img, template, scale_iter):
    found = None
    #edge_point = [top_left, buttom_right]
    #debugging
    img_2 = img.copy()
    #debugging
    edge_point = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    fullsize_template = resize_img(template, img)
    for scale in np.linspace(0.1, 1, scale_iter)[::-1]:
        resized = imutils.resize(   fullsize_template,
                                    width = int(fullsize_template.shape[1] * scale),
                                    height = int(fullsize_template.shape[0] * scale))
        w, h = resized.shape[::-1]
        result = cv2.matchTemplate(img, resized, cv2.TM_CCOEFF_NORMED)
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
    return edge_point

field = cv2.imread('test_field.JPG')
tem = cv2.imread('star.PNG')
point = template_match(field, tem, 20)
point

field = cv2.imread('test_field.JPG')
tem = cv2.imread('triangle.PNG')
point = template_match(field, tem, 20)
point

field = cv2.imread('test_field.JPG')
tem = cv2.imread('double_triangle.PNG')
point = template_match(field, tem, 20)
point


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

def template_match2(img, template, scale_iter, rotate_iter):
    found = None
    #edge_point = [top_left, buttom_right]
    #debugging
    img_2 = img.copy()
    #debugging
    edge_point = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    fullsize_template = resize_img(template, img)
    for scale in np.linspace(0.1, 1, scale_iter)[::-1]:
        resized = imutils.resize(   fullsize_template,
                                    width = int(fullsize_template.shape[1] * scale),
                                    height = int(fullsize_template.shape[0] * scale))
        w, h = resized.shape[::-1]
        for rotation in np.linspace(0, 359, rotate_iter):
            rotate = imutils.rotate(resized, angle = rotation)
            rotate[np.where((rotate==[0]).all(axis=1))] = [255]
            cv2.imshow("rotate", rotate)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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
    return edge_point

field2 = cv2.imread('test_field2.JPG')
tem2 = cv2.imread('star.PNG')
point = template_match2(field2, tem2, 20, 20)
point


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
    cv2.imshow('template_black_contours', template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

field3 = cv2.imread('test_field3.JPG')
tem3 = cv2.imread('star.PNG')
point = template_match3(field3, tem3, 20, 30)
point

field3 = cv2.imread('test_field3.JPG')
tem3 = cv2.imread('triangle.PNG')
point = template_match3(field3, tem3, 20, 40)
point

field3 = cv2.imread('test_field2.JPG')
tem3 = cv2.imread('double_triangle.PNG')
point = template_match3(field3, tem3, 20, 30)
point
