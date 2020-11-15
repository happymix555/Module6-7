# Love(Mix.error);
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
from PIL import Image as PImage
# from mpl_toolkits import mplot3d
%matplotlib qt

class Image:
    # raw_image = None
    # matched_template = []

    def __init__(self, image_location):
        self.raw_image = cv2.imread(image_location)
        self.name = image_location
        self.template_location = []
        # plt.imshow(self.raw_image)

    def show_raw_image(self): #show raw image
        # return self.raw_image
        plt.imshow(self.raw_image)

    def RGB_image(self):
        raw = self.raw_image
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        return rgb

    def show_RGB_image(self):
        plt.imshow(self.RGB_image())

    def gray_image(self):
        return cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)

    def resize(self, input_image ,target_image):#wait for some sharpening method
        x_scale = target_image.shape[0] / input_image.shape[0]
        y_scale = target_image.shape[1] / input_image.shape[1]
        scale = min(x_scale, y_scale)
        width = int(input_image.shape[0] * scale)
        height = int(input_image.shape[1] * scale)
        dim = (height, width)
        return cv2.resize(input_image, dim, interpolation = cv2.INTER_AREA)

    def template_matching(self, raw_template, scale_iter, rotate_iter): #matching 'template' with this object image
        found = None
        edge_point = []
        centerpoint = []
        template = raw_template.raw_image.copy()
        img2 = self.RGB_image().copy()
        # plt.title('img2')
        # plt.imshow(img2)
        # plt.title('template')
        # plt.imshow(template)

        img = self.gray_image().copy()
        # plt.title('img')
        # plt.imshow(img)
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

        fullsize_template = self.resize(template, img)
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

        cv2.rectangle(img2, top_left, bottom_right, (255, 0, 0), 1)
        centerpoint_x = (bottom_right[0] - top_left[0]) / 2 + top_left[0]
        centerpoint_y = (bottom_right[1] - top_left[1]) / 2 + top_left[1]
        centerpoint.append(centerpoint_x)
        centerpoint.append(centerpoint_y)
        img2 = cv2.circle(img2, (int(centerpoint_x), int(centerpoint_y)), radius=0, color=(0, 0, 255), thickness=10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = '(' + str(centerpoint_x) + ', ' + str(centerpoint_y) + ')'
        cv2.putText(img2, text, (int(centerpoint_x), int(centerpoint_y)), font, 1, (0,0,255), 2, cv2.LINE_AA, False)
        plt.imshow(img2)
        pre_template_location = [raw_template.name, centerpoint_x, centerpoint_y]
        self.template_location.append(pre_template_location)
        return centerpoint

    def draw_template_location(self, pre_background):
        background = pre_background.copy()
        for tem in self.template_location:
            x = int(tem[1])
            y = int(tem[2])
            text1 = tem[0]
            text2 = '(' + str(x) + ', ' + str(y) + ')'
            font = cv2.FONT_HERSHEY_SIMPLEX
            background = cv2.circle(background, (x, y), radius=0, color=(0, 0, 255), thickness=10)
            cv2.putText(background, text1, (x - 100, y), font, 1, (0,0,255), 1, cv2.LINE_AA, False)
            cv2.putText(background, text2, (x - 100, y + 30), font, 1, (0,0,255), 1, cv2.LINE_AA, False)
        plt.imshow(background)



field = Image('test_field.JPG')
# field.name
triangle = Image('triangle.PNG')
# plt.imshow(triangle.raw_image)
# plt.imshow(field.raw_image)
field.template_matching(triangle, 20, 20)
# field.template_location
double_triangle = Image('double_triangle.PNG')
star = Image('star.PNG')
field.template_matching(double_triangle, 20, 20)
field.template_matching(star, 20, 20)
field.draw_template_location(field.RGB_image())
# field.show_raw_image()
# field.show_RGB_image()
# plt.imshow(field.RGB_image())
# plt.imshow(field.gray_image(), cmap='gray')
# template = Image('triangle.PNG')
# template.show_raw_image()
# plt.imshow(template.resize(template.raw_image, field.raw_image))
