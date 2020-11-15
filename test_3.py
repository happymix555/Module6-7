# Love(Mix.error);
import cv2
from cv2 import aruco
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.morphology import medial_axis
import imutils
from skimage.morphology import skeletonize
from skimage.morphology import thin
from skimage.util import invert
from skimage import filters
from PIL import Image as PImage
# from mpl_toolkits import mplot3d
%matplotlib qt

source = 'test_field.JPG'
field = cv2.imread(source)
cv2.imshow('original field', field)

def find_gradient2(image, area_threshold, noise_reduction = None, erode_iter = 1):
    thick_contour, _, _, _, _ = self.find_and_draw_contour(5, 5, 40, 100, 'tree', 5)
    # plt.imshow(thick_contour)
    thick = Image(image_object = thick_contour)
    # plt.imshow(thick.raw_image)
    high_res_image, high_res_contour, high_res_hierarchy, _, _ = thick.find_and_draw_contour(5, 5, 40, 100, 'tree', 1)
    for hie in range(len(high_res_hierarchy[0])):
        hie2 = high_res_hierarchy[0][hie]
        if hie2[2] == -1:
            # if high_res_hierarchy[0][hie2[3]][3] == 1:
            self.palette.append(hie)

    if noise_reduction == None:
        gray = cv2.cvtColor(self.raw_image.copy(), cv2.COLOR_BGR2GRAY)
    elif noise_reduction == True:
        # gray = cv2.cvtColor(self.noise_reduction(), cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    pre_fill = []
    blank_img = self.blank_image_with_same_size()
    for cnt in self.palette:
        if cv2.contourArea(high_res_contour[cnt]) > area_threshold:
            coordinate = []
            pre_coordinate_plot = []
            # pre_fill.append(high_res_contour[cnt])
            palette = cv2.drawContours(blank_img.copy(), high_res_contour, cnt, (255, 0, 0), 1)
            filled_inner = cv2.fillPoly(palette.copy(), pre_fill, color=(255,255,255))
            from skimage.morphology import skeletonize
            # from skimage.morphology import thin
            from skimage import morphology, filters
            # binary = filled_inner > filters.threshold_otsu(filled_inner)
            # thin = thin(filled_inner_binary)
            kernel = np.ones((3,3),np.uint8)
            erosion = cv2.erode(filled_inner, kernel, iterations = erode_iter)
            # binary = erosion > filters.threshold_otsu(erosion)
            # binary = filled_inner > filters.threshold_otsu(filled_inner)
            skeleton = skeletonize(erosion, method='lee')
            for x in range(skeleton.shape[0]):
                for y in range(skeleton.shape[1]):
                    color = skeleton[x][y]
                    if color == 255:
                        self.coordinate_palette.append([x, y])
            min = 255
            max = 0
            for pixel in self.coordinate_palette:
                x = pixel[0]
                y = pixel[1]
                z = gray[x, y]
                if z < min:
                    min = z
                elif z > max:
                    max = z
                pre_coordinate_plot.append([x, y, 255 - z])
            for co in pre_coordinate_plot:
                co[2] = map_value(min, max, 100, 200, co[2])
            self.coordinate_plot.append(pre_coordinate_plot)
    gradient_path = self.raw_image.copy()
    gray_gradient_path = gray.copy()
    cv2.imwrite('gray_for_slide.jpg', gray_gradient_path)
    gray_gradient_path2 = cv2.imread('gray_for_slide.jpg')
    layer2 = np.zeros((gray_gradient_path.shape[0], gray_gradient_path.shape[1], 3))
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x = []
    y = []
    z = []
    for co_list in self.coordinate_plot:
        for co in co_list:
            x.append(int(co[0]))
            y.append(int(co[1]))
            z.append(int(co[2]))
    ax.scatter3D(x, y, z, 'gray')
    plt.show()
    return self.coordinate_plot


cv2.waitKey(0)
cv2.destroyAllWindows()
