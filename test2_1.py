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
source_image = 'test_field2.JPG'
gray_field = cv2.imread(source_image,0)
plt.imshow(gray_field)
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
# plt.imshow(external_contour)

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
plt.imshow(thin, cmap='gray')

thin = thin.astype(np.uint8)
# print(thin.shape)
contours, hierarchy = cv2.findContours(thin,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
field = cv2.imread('test_field2.JPG')
# plt.imshow(field)
# print(field.shape)
blank_img = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
print(blank_img.shape)
thin_img = cv2.drawContours(blank_img, contours, -1,  (255, 255, 255), 1)
plt.imshow(thin_img)
print(thin_img.shape)

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
# field = cv2.imread('test_field2.JPG')
blank_img = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
for cnt in path:
    palette = cv2.drawContours(blank_img, contours, cnt, (255, 0, 0), 1)
# palette = cv2.drawContours(blank_img, contours, -1, (255, 0, 0), 1)
plt.imshow(palette, cmap='gray')

blur_palette = cv2.GaussianBlur(palette, (5, 5), 0)
palette_canny = cv2.Canny(blur_palette, 40, 100)
plt.imshow(palette_canny, cmap='gray')
contours_for_palette, hierarchy = cv2.findContours(palette_canny,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
contours_for_palette
blank_img = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
blank_img = cv2.drawContours(blank_img, contours_for_palette, -1, (255, 255, 255), 1)
plt.imshow(blank_img)
def map_value(input, min1, max1, min2, max2):
    return(input * ((max2 - min2) / (max1 - min1)))

field = cv2.imread('test_field2.JPG')
gray_field = cv2.imread('test_field2.JPG', 0)
color = gray_field[289, 494]
color
red = [0,0,255]
palette_plot = []
blank_img2 = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
# indices = None
# indices = np.where(thin == [255])
# indices
coordinate = []
print(thin_img.shape)
gray_thin = cv2.cvtColor(thin_img, cv2.COLOR_BGR2GRAY)
print(gray_thin.shape)
for x in range(thin_img.shape[0]):
    for y in range(thin_img.shape[1]):
        color = gray_thin[x][y]
        if color == 255:
            coordinate.append([x, y])
coordinate

field = cv2.imread('test_field2.JPG')
plt.imshow(gray_thin)
for pixel in coordinate:
    field[pixel[0], pixel[1]] = [0, 255, 0]
plt.imshow(cv2.cvtColor(field, cv2.COLOR_BGR2RGB))

blank_img3 = np.zeros((field.shape[0], field.shape[1], 3), np.uint8)
img_contour = cv2.drawContours(blank_img3, contours_for_palette, -1,  (255, 255, 255), 1)
plt.imshow(img_contour)

filled_contour = cv2.fillPoly(blank_img3, contours_for_palette, color=(255,255,255))
plt.imshow(filled_contour)
filled_contour_binary = cv2.cvtColor(filled_contour, cv2.COLOR_RGB2GRAY)
print(filled_contour.shape)
print(filled_contour_binary.shape)
plt.imshow(filled_contour_binary, cmap='gray')

field = cv2.imread('test_field2.JPG')
gray_field = cv2.imread('test_field2.JPG', 0)
plot_co = []
for pixel in coordinate:
    x = pixel[0]
    y = pixel[1]
    if filled_contour_binary[x, y] == 255:
        field[x, y] = (0, 0, 255)
        z = gray_field[x, y]
        plot_co.append([x, y, 255 - z])
plt.imshow(field)
plot_co
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection="3d")
x = []
y = []
z = []
for co in plot_co:
    x.append(int(co[0]))
    y.append(int(co[1]))
    z.append(int(co[2]))
ax.scatter3D(x, y, z, 'gray')
plt.show()
