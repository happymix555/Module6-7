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
import skimage.graph
# from mpl_toolkits import mplot3d
%matplotlib qt

def map_value(min1, max1, min2, max2, current):
    result = (current - min1) / (max1 - min1)
    result = result * (max2 - min2)
    result = int(result + min2)
    return result

def find_square_contours(contours, min_area_threshold, max_area_threshold):
    square_contours = []
    area = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
		# a square will have an aspect ratio that is approximately
		# equal to one, otherwise, the shape is a rectangle
            if ar >= 0.8 and ar <= 1.2:
                # area.append(cv2.contourArea(cnt))
                if min_area_threshold < cv2.contourArea(cnt) and cv2.contourArea(cnt) < max_area_threshold:
                    area.append(cv2.contourArea(cnt))
                    square_contours.append(cnt)
			# shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    return square_contours, area

def find_square_contours2(contours, image_for_area):
    full_area = len(image_for_area) * len(image_for_area[0])
    checkpoint_area = int(full_area / 5)
    min_checkpoint_area = int(checkpoint_area / 8)
    square_contours = []
    area = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.1 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
		# a square will have an aspect ratio that is approximately
		# equal to one, otherwise, the shape is a rectangle
            if ar >= 0.9 and ar <= 1.1:
                # area.append(cv2.contourArea(cnt))
                if min_checkpoint_area <= cv2.contourArea(cnt) <= checkpoint_area: #and cv2.contourArea(cnt) > (checkpoint_area / 3):
                    area.append(cv2.contourArea(cnt))
                    square_contours.append(cnt)
			# shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    return square_contours, area


def blank_image_with_same_size(img):
    return(np.zeros((img.shape[0], img.shape[1], 1), dtype = np.uint8))

def find_contours(gray_image, canny_l, canny_h, kernel_size0, kernel_size1, contour_type):
    # canny_without_blur = cv2.Canny(gray_image, canny_l, canny_h)
    img = cv2.GaussianBlur(gray_image, (kernel_size0, kernel_size1), 0)
    img_canny = cv2.Canny(img, canny_l, canny_h)
    if contour_type == 'external':
        contours, hierarchy = cv2.findContours(img_canny,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
    elif contour_type == 'tree':
        contours, hierarchy = cv2.findContours(img_canny,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def draw_contours(image, contours, thickness):
    # blank = blank_image_with_same_size(image)
    contour_img = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness)
    return contour_img

def find_contour_center(contour):
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return[cx, cy]

def find_palette_contour_by_area(contours, hierarchy, area_threshold):
    inner_contours = []
    palette_contours = []
    for hie in range(len(hierarchy[0])):
        hie2 = hierarchy[0][hie]
        if hie2[2] == -1:
            # if high_res_hierarchy[0][hie2[3]][3] == 1:
            inner_contours.append(hie)
    for cnt in inner_contours:
        if cv2.contourArea(contours[cnt]) > area_threshold:
            palette_contours.append([contours[cnt]])
    return palette_contours

def find_palette_contour(contours, hierarchy):
    inner_contours = []
    palette_contours = []
    for hie in range(len(hierarchy[0])):
        hie2 = hierarchy[0][hie]
        if hie2[2] == -1:
            # if high_res_hierarchy[0][hie2[3]][3] == 1:
            inner_contours.append(hie)
    for cnt in inner_contours:
        palette_contours.append([contours[cnt]])
    return palette_contours

def find_palette_contour_by_area(contour, min_area_threshold, max_area_threshold):
    palette_contours = []
    for cnt in contour:
        if cv2.contourArea(cnt) > min_area_threshold and cv2.contourArea(cnt) < max_area_threshold:
            palette_contours.append(cnt)
    return palette_contours

def find_max_contour_area(contour):
    max = cv2.contourArea(contour[0])
    for cnt in contour:
        if cv2.contourArea(cnt) > max:
            max = cv2.contourArea(cnt)
    return max

def find_contour_by_area(contour, min_area_threshold, max_area_threshold):
    found_contours = []
    for cnt in contour:
        if cv2.contourArea(cnt) > min_area_threshold and cv2.contourArea(cnt) < max_area_threshold:
            found_contours.append(cnt)
    for cnt in found_contours:
        if cv2.contourArea(cnt) > min_area_threshold and cv2.contourArea(cnt) < max_area_threshold:
            None
        else:
            found_contours.remove(cnt)
    return found_contours

def find_skeleton(gray_image, erode_iter): #return in [x, y]
    skeleton_coordinate = []
    filled_pc = np.zeros((gray_image.shape[0], gray_image.shape[1], 1), dtype = np.uint8)
    blank = filled_pc.copy()
    c1, _ = find_contours(gray_image, 40, 100, 5, 5, 'tree')
    c2 = draw_contours(blank, c1, 5)
    c3, hie = find_contours(c2, 40, 100, 5, 5, 'tree')
    pc = find_palette_contour(c3, hie, 15000)
    for p in pc:
        filled_pc = cv2.fillPoly(filled_pc, p, color=(255,255,255))
    from skimage.morphology import skeletonize
    from skimage import morphology, filters
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(filled_pc, kernel, iterations = erode_iter)
    skeleton = skeletonize(erosion, method='lee')
    # print(skeleton)
    # for x in range(skeleton.shape[0]):
    #     for y in range(skeleton.shape[1]):
    #         if skeleton[x][y] == 255:
    #             skeleton_coordinate.append([x, y])
    for x in range(len(skeleton)):
        for y in range(len(skeleton[0])):
            if skeleton[x][y] == 255:
                skeleton_coordinate.append([y, x])
    return skeleton_coordinate, skeleton

def create_skeleton(palette_contour, erode_iter, size_ref_image):
    filled_pc = np.zeros((size_ref_image.shape[0], size_ref_image.shape[1], 1), dtype = np.uint8)
    skeleton_coordinate = []
    drawn_palette = draw_contours(blank_image_with_same_size(size_ref_image).copy(), palette_contour, 1)
    c3, hie = find_contours(drawn_palette, 40, 100, 5, 5, 'tree')
    pc = find_palette_contour(c3, hie)
    for p in pc:
        filled_pc = cv2.fillPoly(filled_pc, p, color=(255,255,255))
    from skimage.morphology import skeletonize
    from skimage import morphology, filters
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(filled_pc, kernel, iterations = erode_iter)
    skeleton = skeletonize(erosion, method='lee')
    for x in range(len(skeleton)):
        for y in range(len(skeleton[0])):
            if skeleton[x][y] == 255:
                skeleton_coordinate.append([y, x])
    return skeleton, skeleton_coordinate

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

def find_connected_path(checkpoint, opened_point, distance_threshold):
    connected_point = []
    for c in checkpoint:
        cp = c
        for op in opened_point:
            if distance_between_point(cp, [op[0], op[1]]) < distance_threshold:
                connected_point.append([cp, op])
    return connected_point

def draw_connected_path(skeleton, connected_point):
    connected_path = skeleton.copy()
    for cp in connected_point:
        connected_path = cv2.line(connected_path, (cp[0][0], cp[0][1]), (cp[1][0], cp[1][1]),(255,255,255), 1)
    return connected_path

def implement_connected_path(gray_image):
    skeleton_co, skeleton_image = find_skeleton(gray_image, 11)
    op_point = find_opened_end(skeleton_co, skeleton_image)
    connected_point = find_connected_path(field2.checkpoint_location, op_point, 250)
    cnp = draw_connected_path(skeleton_image, connected_point)
    return cnp

def resize(input_image ,target_image):#wait for some sharpening method
    x_scale = target_image.shape[0] / input_image.shape[0]
    y_scale = target_image.shape[1] / input_image.shape[1]
    if x_scale < y_scale:
        scale = x_scale
    else:
        scale = y_scale
    # scale = min(int(x_scale), int(y_scale))
    width = int(input_image.shape[0] * scale)
    height = int(input_image.shape[1] * scale)
    dim = (height, width)
    return cv2.resize(input_image, dim, interpolation = cv2.INTER_AREA)

def template_matching(image, raw_template, scale_iter, rotate_iter): #image = tuned image, raw_template = gray_scale template
    found = None
    edge_point = []
    centerpoint = []
    img = image
    tem_contour, _ = find_contours(raw_template, 40, 100, 3, 3, 'tree')
    template = draw_contours(blank_image_with_same_size(raw_template), tem_contour, 1)


    fullsize_template = resize(template, img)
    for scale in np.linspace(0.1, 0.4, scale_iter)[::-1]:
        resized = imutils.resize(   fullsize_template,
                                    width = int(fullsize_template.shape[1] * scale),
                                    height = int(fullsize_template.shape[0] * scale))
        w, h = resized.shape[::-1]
        for rotation in np.linspace(0, 359, rotate_iter):
            rotate = imutils.rotate(resized, angle = rotation)
            result = cv2.matchTemplate(img, rotate, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if found == None or max_val > found:
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                found = max_val
                edge_point = []
                edge_point.append(top_left)
                edge_point.append(bottom_right)
    centerpoint_x = (bottom_right[0] - top_left[0]) / 2 + top_left[0]
    centerpoint_y = (bottom_right[1] - top_left[1]) / 2 + top_left[1]
    centerpoint.append(int(centerpoint_x))
    centerpoint.append(int(centerpoint_y))
    return centerpoint

def template_matching_with_roi(template_image, checkpoint_roi, checkpoint_location, rotate_iter): #checkpoint_roi = list of roi image
    each_roi_found = []
    for roi_image in checkpoint_roi:
        found = None
        resized_template = resize(template_image, roi_image)
        for rotation in np.linspace(0, 359, rotate_iter):
            rotate = imutils.rotate(resized_template, angle = rotation)
            result = cv2.matchTemplate(roi_image, rotate, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if found == None or max_val > found:
                found = max_val
        each_roi_found.append(found)
    result = each_roi_found[0]
    for p in each_roi_found:
        if p > result:
            result = p
    # result = max[each_roi_found]
    template_coordinate_on_path = checkpoint_location[each_roi_found.index(result)]
    return template_coordinate_on_path

def resize_percent(input_image ,percent):#wait for some sharpening method
    # x_scale = target_image.shape[0] / input_image.shape[0]
    # x_scale = (percent/100)
    # y_scale = (percent/100)
    # y_scale = target_image.shape[1] / input_image.shape[1]
    scale = (percent/100)
    width = int(input_image.shape[0] * scale)
    height = int(input_image.shape[1] * scale)
    dim = (height, width)
    return cv2.resize(input_image, dim, interpolation = cv2.INTER_AREA)

def find_endpoint(path_image, checkpoint_location):
    cnp = path_image
    skeleton_co = []
    for x in range(len(cnp)):
        for y in range(len(cnp[0])):
            if cnp[x][y] == 255:
                skeleton_co.append([y, x])
    opened = find_opened_end(skeleton_co, cnp)
    checkpoint = []
    for p in opened:
            if p not in checkpoint_location:
                return p

# def find_endpoint_demo(checkpoint_location, template_location):
#     for p in checkpoint_location:
#         if template


def shortest_path(start, end, skeleton):
    costs=np.where(skeleton, 1, 1000)
    path, cost = skimage.graph.route_through_array(
        costs, start=start, end=end, fully_connected=True)
    return path,cost

def find_plane_path(start, end, skeleton):
    plane_path = []
    pre_path, _ = shortest_path(start, end, skeleton)
    for co in pre_path:
        x_co = co[1]
        y_co = co[0]
        plane_path.append([x_co, y_co])
    return plane_path


def find_palette_by_checkpoint_area(contours, checkpoint_area):
    max_checkpoint_area = checkpoint_area[0]
    for a in checkpoint_area:
        if a > max_checkpoint_area:
            max_checkpoint_area = a
    palette_contour = []
    for cnt in contours:
        if (max_checkpoint_area) < cv2.contourArea(cnt) <= (max_checkpoint_area * 3):
            palette_contour.append(cnt)
    return palette_contour

def find_endpoint2(checkpoint_location, gray_field):
    ret,thresh_field = cv2.threshold(gray_field,127,255,cv2.THRESH_BINARY)
    for point in checkpoint_location:
        if thresh_field[point[1]][point[0]] == 255:
            return point, thresh_field

def find_min_max_palette(palette_contour_for_path_height, gray_denoised_field_image):
    min = gray_denoised_field_image[0][0]
    max = gray_denoised_field_image[0][1]
    min_co = []
    max_co = []
    for y in range(len(gray_denoised_field_image)):
        for x in range(len(gray_denoised_field_image)):
            dist = cv2.pointPolygonTest(palette_contour_for_path_height[0], (x, y), False)
            if dist == 1:
                if gray_denoised_field_image[y][x] > max:
                    max = gray_denoised_field_image[y][x]
                    max_co = [x, y]
                if gray_denoised_field_image[y][x] < min:
                    min = gray_denoised_field_image[y][x]
                    min_co = [x, y]
    return [min, max]
# def find_path(start_point, end_point, skeleton_image):#start_point and end_point must be in skeleton_image
#     skeleton_co = []
#     for x in range(len(skeleton_image)):
#         for y in range(len(skeleton_image[0])):
#             if skeleton_image[x][y] == 255:
#                 skeleton_co.append([y, x])
#     opened_point = find_opened_end(skeleton_co, skeleton_image)
#     lastpoint = start_point
#     first_set = []
#     x = start_point[0]
#     y = start_point[1]
#     kernel = [[x-1, y-1], [x-1, y], [x-1, y+1],
#               [x, y-1], [x, y], [x, y+1],
#               [x+1, y-1], [x+1, y], [x+1, y+1]]
#     for k in kernel:
#         if skeleton_image[k[1]][k[0]] == 255:
#             first_set.append(k)
#     path = []
#     path.append(start_point)
#     for p in first_set:
#         if p not in path:
#             x = p[0]
#             y = p[1]
#             pre_path = []
#             pre_path.append(start_point)
#             pre_path.append([x, y])
#             while(1):
#                 for p1 in skeleton_co:
#                     if abs(x - p1[0]) == 1 or abs(y - p1[1] == 1):
#                         pre_path.append(p1)
#                 x = pre_path[-1][0]
#                 y = pre_path[-1][1]
#                 if [x, y] == end_point:
#                     return pre_path
#                     break
#                 for op in opened_point:
#                     if [x, y] == op:
#                         break
#             # count = len()
#     # while(lastpoint != end_point):
#     return open_point
#
#
# list1 = find_path([76, 400], [324, 140], cnp2)
# list1
# cnp3 = cv2.circle(cnp2.copy(), (76, 400), radius=0, color=(255, 255, 255), thickness=10)
# cv2.imshow('cnp3', cnp3)
# # cv2.imshow('cnp', cnp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


class Image: #!!!must read image with opencv to obtain 'BGR' format!!!
    # raw_image = None
    # matched_template = []

    def __init__(self, image_location = None, image_object = None):
        if image_location != None:
            self.raw_image = cv2.imread(image_location)
            self.name = image_location
        else:
            self.raw_image = image_object
            self.name = None
        self.template_location = []
        self.after_perspective_t = None
        self.palette = []
        self.coordinate_palette = []
        self.coordinate_plot = []
        self.checkpoint_location = []
        self.endpoint_location = []
        self.skeleton_point = []
        self.skeleton_coordinate = []
        # plt.imshow(self.raw_image)

    def clear_template_location(self):
        self.template_location = []

    def noise_reduction(self):
        # if image == []:
        result = self.raw_image.copy()
        result = cv2.fastNlMeansDenoisingColored(result,None,10,10,7,21)
        # result = cv2.bilateralFilter(result,20,75,75)
        # result = cv2.GaussianBlur(result,(45,45),0)
        # result = cv2.medianBlur(result,55)
        # result = cv2.blur(result,(5,5))
        # result = cv2.medianBlur(result,5)
        # result = cv2.fastNlMeansDenoisingColored(result,None,10,10,7,21)
        # result = cv2.bilateralFilter(result,9,75,75)
        # else:
        #     result = cv2.fastNlMeansDenoisingColored(image.copy(),None,10,10,7,21)
        return result

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
        if self.raw_image.shape[2] == 1:
            return self.raw_image
        else:
            return cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)

    def resize(self, input_image ,target_image):#wait for some sharpening method
        x_scale = target_image.shape[0] / input_image.shape[0]
        y_scale = target_image.shape[1] / input_image.shape[1]
        scale = min(x_scale, y_scale)
        width = int(input_image.shape[0] * scale)
        height = int(input_image.shape[1] * scale)
        dim = (height, width)
        return cv2.resize(input_image, dim, interpolation = cv2.INTER_AREA)

    def find_and_draw_contour(self, kernel_size0, kernel_size1, canny_l, canny_h,  contour_type, thickness):
        img = self.gray_image().copy()
        # kernel_sharpening = np.array([[-1,-1,-1],
        #                       [-1, 9,-1],
        #                       [-1,-1,-1]])
        # sharpened = cv2.filter2D(img, -1, kernel_sharpening)
        canny_without_blur = cv2.Canny(img, canny_l, canny_h)
        img = cv2.GaussianBlur(img, (kernel_size0, kernel_size1), 0)
        img_canny = cv2.Canny(img, canny_l, canny_h)
        # plt.imshow(img)
        img_black = self.blank_image_with_same_size()
        if contour_type == 'external':
            img_contours, hierarchy = cv2.findContours(img_canny,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.drawContours(img_black, img_contours, -1, (255, 255, 255), thickness)
        elif contour_type == 'tree':
            img_contours, hierarchy = cv2.findContours(img_canny,
                                                    cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.drawContours(img_black, img_contours, -1, (255, 255, 255), thickness)
            img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img, img_contours, hierarchy, img_canny, canny_without_blur

    def blank_image_with_same_size(self):
        img = self.raw_image
        return(np.zeros((img.shape[0], img.shape[1], 1), dtype = np.uint8))

    def template_matching(self, raw_template, scale_iter, rotate_iter): #matching 'template' with this object image
        found = None
        edge_point = []
        centerpoint = []
        # template = raw_template.raw_image.copy()
        img2 = self.RGB_image().copy()
        # plt.title('img2')
        # plt.imshow(img2)
        # plt.title('template')
        # plt.imshow(template)

        # img = self.gray_image().copy()
        # # plt.title('img')
        # # plt.imshow(img)
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        # img = cv2.Canny(img, 40, 100)
        # img_contours, hierarchy = cv2.findContours(img,
        #                                         cv2.RETR_TREE,
        #                                         cv2.CHAIN_APPROX_SIMPLE)
        # img_black = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        # img = cv2.drawContours(img_black, img_contours, -1, (255, 255, 255), 1)
        img, _, _, _, _ = self.find_and_draw_contour(5, 5, 40, 100, 'tree', 1)
        # cv2.imshow('img_black_contours', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # template = cv2.GaussianBlur(template, (5, 5), 0)
        # template = cv2.Canny(template, 40, 100)
        # template_contours, hierarchy = cv2.findContours(template,
        #                                         cv2.RETR_TREE,
        #                                         cv2.CHAIN_APPROX_SIMPLE)
        # template_black = np.zeros((template.shape[0], template.shape[1], 1), np.uint8)
        # template = cv2.drawContours(template_black, template_contours, -1, (255, 255, 255), 1)
        template, _, _, _, _ = raw_template.find_and_draw_contour(5, 5, 40, 100, 'tree', 1)

        fullsize_template = self.resize(template, img)
        for scale in np.linspace(0.1, 0.4, scale_iter)[::-1]:
            resized = imutils.resize(   fullsize_template,
                                        width = int(fullsize_template.shape[1] * scale),
                                        height = int(fullsize_template.shape[0] * scale))
            w, h = resized.shape[::-1]
            for rotation in np.linspace(0, 359, rotate_iter):
                rotate = imutils.rotate(resized, angle = rotation)
                result = cv2.matchTemplate(img, rotate, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

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
        # plt.imshow(img2)
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
        return background

    def perspective_t(self):
    	field_original = self.raw_image.copy()
    	gray_field = self.gray_image().copy()
    	blur_gray_field = cv2.GaussianBlur(gray_field, (5, 5), 0)
    	# plt.imshow(blur_gray_field, cmap='gray')
    	field_canny_edge = cv2.Canny(blur_gray_field, 20, 50)
    	# plt.imshow(field_canny_edge, cmap='gray')

    	#find External Contours
    	# contours, hierarchy = cv2.findContours(field_canny_edge,
    	#                                         cv2.RETR_LIST,
    	#                                         cv2.CHAIN_APPROX_SIMPLE)
    	contours = cv2.findContours(field_canny_edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    	grabed_contour = contours[0] if len(contours) == 2 else contours[1]
    	sorted_contour = sorted(grabed_contour, key = cv2.contourArea, reverse = True)[:5]

    	# loop over the contours
    	for c in sorted_contour:
    		# approximate the contour
    		perimeter = cv2.arcLength(c, True)
    		approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    		# if our approximated contour has four points, then we
    		# can assume that we have found our screen
    		if len(approx) == 4:
    			field_contour_vertices = approx
    			break
    	#approx = top_left, bottom_left, bottom_right, top_right
    	top_left = [approx[0][0][0], approx[0][0][1]]
    	bottom_left = [approx[1][0][0], approx[1][0][1]]
    	bottom_right = [approx[2][0][0], approx[2][0][1]]
    	top_right = [approx[3][0][0], approx[3][0][1]]
    	before_transform = np.float32([top_left, top_right, bottom_right, bottom_left])
    	img = field_original.copy()
    	# for i in before_transform:
    	# 	img = cv2.circle(img, (int(i[0]), int(i[1])), radius=0, color=(0, 0, 255), thickness=10)
    	# plt.imshow(img)
    	width1 = np.sqrt(abs((top_left[1] - top_right[1]) ** 2) + abs((top_left[0] - top_right[0]) ** 2))
    	width2 = np.sqrt(abs((bottom_left[1] - bottom_right[1]) ** 2) + abs((bottom_left[0] - bottom_right[0]) ** 2))
        # if(int(width1) < int(width2)):
        #     width = width1
        # else:
        #     width = width2
    	width = int(min(width1, width2))
    	height1 = np.sqrt(abs((top_left[1] - bottom_left[1]) ** 2) + abs((top_left[0] - bottom_left[0]) ** 2))
    	height2 = np.sqrt(abs((top_right[1] - bottom_right[1]) ** 2) + abs((top_right[0] - bottom_right[0]) ** 2))
        # if height1 < height2:
        #     height = height1
        # else:
        #     height = height2
    	height = int(min(height1, height2))
    	# height
    	ratio = height1 / width1
    	# ratio
    	# if height < width:xs
    	# 	transform_to = np.float32([[0,0], [height / ratio - 1, 0], [height / ratio - 1, height - 1], [0, height - 1]])
    	# else:
    	# 	transform_to = np.float32([[0,0], [width - 1, 0], [width - 1, width * ratio - 1], [0, width * ratio - 1]])
    	transform_to = np.float32([[0,0], [width - 1, 0], [width - 1, width * ratio - 1], [0, width * ratio - 1]])
    	M = cv2.getPerspectiveTransform(before_transform, transform_to)
    	self.after_perspective_t = cv2.warpPerspective(field_original, M, (width, height))
    	return(self.after_perspective_t)

    def find_gradient(self, area_threshold, noise_reduction = None, erode_iter = 1):
        thick_contour, _, _, _, _ = self.find_and_draw_contour(5, 5, 40, 100, 'tree', 5)
        # plt.imshow(thick_contour)
        thick = Image(image_object = thick_contour)
        # plt.imshow(thick.raw_image)
        high_res_image, high_res_contour, high_res_hierarchy, _, _ = thick.find_and_draw_contour(5, 5, 40, 100, 'tree', 1)
        # plt.imshow(high_res_contour)
        # print(high_res_hierarchy)
        for hie in range(len(high_res_hierarchy[0])):
            hie2 = high_res_hierarchy[0][hie]
            # print(hie2)
            # if hie2[0] == -1:
            # if hie2[1] == -1:
            if hie2[2] == -1:
                # if high_res_hierarchy[0][hie2[3]][3] == 1:
                self.palette.append(hie)
        pre_fill = []
        blank_img = self.blank_image_with_same_size()
        for cnt in self.palette:
            if cv2.contourArea(high_res_contour[cnt]) > area_threshold:
                pre_fill.append(high_res_contour[cnt])
                palette = cv2.drawContours(blank_img, high_res_contour, cnt, (255, 0, 0), 1)
        #     filled_palette = cv2.fillPoly(blank_img.copy(), high_res_contour, cnt, color=(255,255,255))
        # plt.imshow(high_res_image)
        # filled_palette = cv2.fillPoly(blank_img.copy(), self.palette, color=(255,255,255))
        # filled_palette = cv2.fillPoly(blank_img.copy(), high_res_contour, color=(255,255,255))
        # plt.imshow(palette)
        # inner_palette = Image(image_object = palette)
        # inner_image, inner_contour, _, _, _ = inner_palette.find_and_draw_contour(5, 5, 40, 100, 'external', 1)
        filled_inner = cv2.fillPoly(palette.copy(), pre_fill, color=(255,255,255))
        # plt.imshow(filled_inner)
        # _,filled_inner_binary = cv2.threshold(filled_inner,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # plt.imshow(filled_inner_binary)
        # from skimage.morphology import thin
        from skimage.morphology import skeletonize
        # from skimage.morphology import thin
        from skimage import morphology, filters
        # binary = filled_inner > filters.threshold_otsu(filled_inner)
        # thin = thin(filled_inner_binary)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(filled_inner, kernel, iterations = erode_iter)
        binary = erosion > filters.threshold_otsu(erosion)
        # binary = filled_inner > filters.threshold_otsu(filled_inner)
        skeleton = skeletonize(binary, method='lee')
        # plt.imshow(skeleton, cmap = 'gray')
        coordinate = []
        for x in range(skeleton.shape[0]):
            for y in range(skeleton.shape[1]):
                color = skeleton[x][y]
                if color == 255:
                    self.coordinate_palette.append([x, y])

        if noise_reduction == None:
            gray = cv2.cvtColor(self.raw_image.copy(), cv2.COLOR_BGR2GRAY)
        elif noise_reduction == True:
            # gray = cv2.cvtColor(self.noise_reduction(), cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            # gray = cv2.GaussianBlur(gray,(5,5),0)
            # gray = cv2.fastNlMeansDenoising(gray, None, 3, 7, 21)

        for pixel in self.coordinate_palette:
            x = pixel[0]
            y = pixel[1]
            z = gray[x, y]
            self.coordinate_plot.append([x, y, 255 - z])

        gradient_path = self.raw_image.copy()
        gray_gradient_path = gray.copy()
        cv2.imwrite('gray_for_slide.jpg', gray_gradient_path)
        gray_gradient_path2 = cv2.imread('gray_for_slide.jpg')
        layer2 = np.zeros((gray_gradient_path.shape[0], gray_gradient_path.shape[1], 3))

        for co in self.coordinate_plot:
            gradient_path[co[0], co[1]] = [0, 0, 255]
            gray_gradient_path2[co[0], co[1]] = [0, 0, 255]
            layer2[co[0], co[1]] = [0, 0, 255]
        # plt.imshow(gradient_path)
        from mpl_toolkits import mplot3d
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        x = []
        y = []
        z = []
        for co in self.coordinate_plot:
            x.append(int(co[0]))
            y.append(int(co[1]))
            z.append(int(co[2]))
        ax.scatter3D(x, y, z, 'gray')
        plt.show()
        return thick_contour, high_res_image, palette, filled_inner, skeleton, gradient_path, gray_gradient_path2

    def find_gradient2(self, area_threshold, noise_reduction = None, erode_iter = 1):
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
        l = 0
        test_filled_inner = []
        min_max = []
        for cnt in self.palette:
            if cv2.contourArea(high_res_contour[cnt]) > area_threshold:
                coordinate = []
                pre_coordinate_plot = []
                pre_coordinate_plot2 = []
                # pre_fill.append(high_res_contour[cnt])
                palette = cv2.drawContours(blank_img.copy(), high_res_contour, cnt, (255, 0, 0), 1)
                filled_inner = cv2.fillPoly(palette.copy(), [high_res_contour[cnt]], color=(255,255,255))
                test_filled_inner.append(filled_inner)
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
                            pre_coordinate_plot2.append([x, y])
                min = gray[pre_coordinate_plot2[0][0],  pre_coordinate_plot2[0][1]]
                max = gray[pre_coordinate_plot2[1][0],  pre_coordinate_plot2[1][1]]
                for pixel in pre_coordinate_plot2:
                    x = pixel[0]
                    y = pixel[1]
                    z = gray[x, y]
                    if z < min:
                        min = z
                    if z > max:
                        max = z
                    pre_coordinate_plot.append([x, y, abs(255 - z)])
                min_max.append([min, max])
                if max - min >= 10:
                    for co in pre_coordinate_plot:
                        co[2] = map_value((255-min), (255-max), 100, 200, co[2])
                        self.coordinate_plot.append(co)
                else:
                    sum = 0
                    for co in pre_coordinate_plot:
                        sum += co[2]
                        # co[2] = map_value(255, (255-max), 100, 200, co[2])
                    mean = int(sum / len(pre_coordinate_plot))
                    for co in pre_coordinate_plot:
                        co[2] = map_value(255, 0, 100, 200, mean)
                        self.coordinate_plot.append(co)

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
        for co in self.coordinate_plot:
            # for co in co_list:
            x.append(int(co[0]))
            y.append(int(co[1]))
            z.append(int(co[2]))
        ax.scatter3D(x, y, z, 'gray')
        plt.show()
        return high_res_image

    def defind_checkpoint_location(self):
        blank = self.blank_image_with_same_size()
        c_image = draw_contours(blank.copy(), find_contours(self.gray_image(), 40, 100, 5, 5, 'tree')[0], 5)
        s_contour, _= find_square_contours(find_contours(c_image, 40, 100, 5, 5, 'tree')[0], 10000, 22000)
        s_image = draw_contours(blank.copy(), s_contour, 5)
        s_contour2 = find_contours(s_image, 40, 100, 5, 5, 'external')[0]
        for cnt in s_contour2:
            center = find_contour_center(cnt)
            check = 0
            if self.checkpoint_location != []:
                for p in self.checkpoint_location:
                    if p[0][0] != center[0] and p[0][1] != center[1]:
                        check = 1
                if check == 1:
                    self.checkpoint_location.append([center, [cnt]])
                    check = 0
            else:
                self.checkpoint_location.append([center, [cnt]])
        return None

    def implement_connected_path(self):
        skeleton_co, skeleton_image = find_skeleton(self.gray_image(), 11)
        op_point = find_opened_end(skeleton_co, skeleton_image)
        f2 = self.raw_image.copy()
        # self.defind_checkpoint_location()
        connected_point = find_connected_path(self.checkpoint_location, op_point, 100)
        cnp = draw_connected_path(skeleton_image, connected_point)
        return cnp, connected_point

    def find_endpoint(self):
        cnp, _ = self.implement_connected_path()
        skeleton_co = []
        for x in range(len(cnp)):
            for y in range(len(cnp[0])):
                if cnp[x][y] == 255:
                    skeleton_co.append([y, x])
        opened = find_opened_end(skeleton_co, cnp)
        checkpoint = []
        for p in field2.checkpoint_location:
            checkpoint.append(p[0])
        for p in opened:
                if p not in checkpoint:
                    self.endpoint_location.append(p)
        return opened

    def feature_match(self, template):
        img1, _, _, _, _= template.find_and_draw_contour(3, 3, 40, 100, 'tree', 5)      # queryImage
        img2, _, _, _, _ = self.find_and_draw_contour(3, 3, 40, 100, 'tree', 5)  # trainImage
        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

        return img3

# field1 = Image('test_field.JPG')
# plt.imshow(field.find_and_draw_contour(5, 5, 40, 100, 'external', 5)[0])
# # field.name
triangle = Image('triangle.PNG')
# # plt.imshow(triangle.raw_image)
# # plt.imshow(field.raw_image)
# field.template_matching(triangle, 20, 20)
# # field.template_location
double_triangle = Image('double_triangle.PNG')
star = Image('star.PNG')
# field.template_matching(double_triangle, 20, 20)
# field.template_matching(star, 20, 20)
# field.draw_template_location(field.RGB_image())
# star.template_location
# field.show_raw_image()
# field.show_RGB_image()
# plt.imshow(field.RGB_image())
# plt.imshow(field.gray_image(), cmap='gray')
# template = Image('triangle.PNG')
# template.show_raw_image()
# plt.imshow(template.resize(template.raw_image, field.raw_image))
# field_before_perspective_t = Image('field_from_cam2.jpg')
# field_after_perspective_t = Image(image_object = field_before_perspective_t.perspective_t())
# print(field_after_perspective_t.find_gradient(15000))
field = Image('field_from_cam2.jpg')
# plt.imshow(field.raw_image)
field1 = Image(image_object = field.perspective_t())
# field2 = Image(image_object = resize_percent(field1.raw_image, 60))
# field2 = Image('wawa2.jpg')
# field2 = Image('wawa.png')
# field2 = Image('wawa3.png')
field2 = Image('rail1.jpg')
# img3 = field2.feature_match(star)
# x = field2.find_gradient2(15000, noise_reduction = True, erode_iter = 11)
# x
# black = np.zeros((field2.raw_image.shape[0], field2.raw_image.shape[1], 1), dtype = np.uint8)
# f = cv2.fillPoly(black.copy(), x, (255, 255, 255))
# find_contours(field2.gray_image(), 40, 100, 5, 5, 'tree')[0]
# len(s_contour)
# cv2.imshow('c_image', c_image)
# cv2.imshow('s_image', s_image)
# cv2.imshow('s1', s1)
# cv2.imshow('image1', image1)
# bg = field2.blank_image_with_same_size()
# g = field2.gray_image()
# skeleton_co, skeleton_image, c2 = find_skeleton(g, 11)
# print(skeleton_image[959][1262])
# skeleton_image.shape[0]
# skeleton_image.shape[1]
# len(skeleton_image)
# len(skeleton_image[0])
# len(skeleton_image[1])
# skeleton_co
# op_point = find_opened_end(skeleton_co, skeleton_image)
# op_point
# f2 = field2.raw_image.copy()
# for s_co in skeleton_co:
#     f2[s_co[0], s_co[1]] = [0, 0, 255]
# for p in op_point:
#     f2 = cv2.circle(f2, (p[1], p[0]), radius=0, color=(255, 255, 255), thickness=10)
# field2.defind_checkpoint_location()
# connected_point = find_connected_path(field2.checkpoint_location, op_point, 250)
# for cp in connected_point:
#     f2 = cv2.circle(f2, (cp[0][0], cp[0][1]), radius=0, color=(255, 0, 0), thickness=10)
#     f2 = cv2.circle(f2, (cp[1][1], cp[1][0]), radius=0, color=(255, 0, 0), thickness=10)
# connected_point
# bg = field2.raw_image.copy()
# implement_connected_path(field2.gray_image())
# cnp = field2.implement_connected_path()
# for x in range(len(cnp[0])):
    # for y in range(len(cnp[1])):
    #     if cnp[x][y] == [255, 255, 255]:
    #         bg[x][y] = [0, 0, 255]
# field2.defind_checkpoint_location()
# field2.checkpoint_location
# cnp2, connected_point = field2.implement_connected_path()
# connected_point
# opened = field2.find_endpoint()
# opened
# field2.endpoint_location
# checkpoint = []
# for p in field2.checkpoint_location:
#     checkpoint.append(p[0])
# checkpoint
# t = []
# for p in opened:
#     if p not in checkpoint:
#         t.append(p)
# t
# cnp = cv2.circle(cnp2.copy(), (324, 140), radius=0, color=(255, 255, 255), thickness=10)
# cnp = cv2.circle(cnp, (76, 400), radius=0, color=(255, 255, 255), thickness=10)
#
#
# c1, _ = find_contours(field2.gray_image().copy(), 30, 100, 3, 3, 'tree')
# c1_img = draw_contours(field2.blank_image_with_same_size().copy(), c1, 1)
# kernel = np.ones((3,3),np.uint8)
# erosion = cv2.erode(c1_img, kernel, iterations = 2)
# gradient = cv2.morphologyEx(c1_img, cv2.MORPH_GRADIENT, kernel)
# c2, _ = find_contours(gradient.copy(), 40, 100, 9, 9, 'tree')
# c2_img = draw_contours(field2.blank_image_with_same_size().copy(), c2, 1)
# sc, _ = find_square_contours(c2, 10000, 20000)
# sc
# sc_img = field2.blank_image_with_same_size().copy()
# sc_img = draw_contours(sc_img, sc, 1)
# s_co, s_img = find_skeleton(field2.gray_image().copy(), 11)
# palette = find_palette_contour_by_area(c2, 6000, 10000)
# p = draw_contours(field2.blank_image_with_same_size().copy(), palette, 1)
# hull = cv2.convexHull(palette)
# square_contour = find_palette_contour_by_area(c2, 10000, 13000)
# s = draw_contours(field2.blank_image_with_same_size().copy(), square_contour, 1)
# cv2.imshow('c1', c1_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
checkpoint_roi = []
endpoint_location = []

def nothing(x):
    pass

# Create a black image, a window
# img = np.zeros((len(field2.raw_image[0]),len(field2.raw_image[1]), 1), np.uint8)
img = np.zeros((len(field2.raw_image),len(field2.raw_image[0]), 1), np.uint8)
img2 = img.copy()
cv2.namedWindow('Draw Contours')
cv2.namedWindow('Select Only Checkpoint')
cv2.namedWindow('Select Only Palette')
# create trackbars for color change
cv2.createTrackbar('canny LOW','Draw Contours',0,255,nothing)
cv2.createTrackbar('canny HIGH','Draw Contours',100,255,nothing)
cv2.createTrackbar('Gaussian kernel size','Draw Contours',1,21,nothing)
cv2.createTrackbar('contour thickness', 'Draw Contours', 1, 10, nothing)
cv2.createTrackbar('Dilation Iteration', 'Draw Contours', 0, 10, nothing)
# cv2.createTrackbar('Min area', 'image', 1, 50000, nothing)
# cv2.createTrackbar('Max area', 'image', 1, 50000, nothing)

#create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'Draw Contours',0,3,nothing)
# cv2.createTrackbar(switch, 'Select Only Checkpoint',3,4,nothing)
# cv2.createTrackbar(switch, 'Select Only Palette',4,5,nothing)

#kernel for dilation
dilation_kernel = np.ones((3,3), np.uint8)

while(1):
    # cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break

    # get current positions of four trackbars
    canny_low = cv2.getTrackbarPos('canny LOW','Draw Contours')
    canny_high = cv2.getTrackbarPos('canny HIGH','Draw Contours')
    gs = cv2.getTrackbarPos('Gaussian kernel size','Draw Contours')
    gs = ((gs+1) * 2) - 1
    contour_thickness = cv2.getTrackbarPos('contour thickness', 'Draw Contours')
    dilate_iter = cv2.getTrackbarPos('Dilation Iteration', 'Draw Contours')
    min_area = cv2.getTrackbarPos('Min area','Select Only Checkpoint') #for contour selection
    max_area = cv2.getTrackbarPos('Max area','Select Only Checkpoint') #for contour selection
    min_area2 = cv2.getTrackbarPos('Min area2','Select Only Palette') #for palette selection
    max_area2 = cv2.getTrackbarPos('Max area2','Select Only Palette') #for palette selection
    erode_iter = cv2.getTrackbarPos('Erosion Iteration', 'Select Only Palette')
    s = cv2.getTrackbarPos(switch,'Draw Contours')


    # c1, _ = find_contours(field2.gray_image().copy(), canny_low, canny_high, gs, gs, 'tree')
    if s == 0:
        img[:] = 0
        cv2.imshow('Draw Contours',img)
    if s == 1:
        c1, _ = find_contours(field2.gray_image().copy(), canny_low, canny_high, gs, gs, 'tree')
        img = draw_contours(field2.blank_image_with_same_size().copy(), c1, contour_thickness)
        # for_template_matching = draw_contours(field2.blank_image_with_same_size().copy(), c1, 1)
        img = cv2.dilate(img, dilation_kernel, iterations=dilate_iter)
        tuned_image1 = img.copy()
        cv2.imshow('Draw Contours',img)
        cv2.imshow('Select Only Checkpoint',img)
    if s == 2:
        c2, _ = find_contours(tuned_image1, canny_low, canny_high, gs, gs, 'tree')
        img = draw_contours(field2.blank_image_with_same_size().copy(), c2, contour_thickness)
        img = cv2.dilate(img, dilation_kernel, iterations=dilate_iter)
        tuned_image2 = img.copy()
        cv2.imshow('Draw Contours',img)
        # cv2.createTrackbar('Min area', 'Select Only Checkpoint', 1, int(find_max_contour_area(c2)) + 1, nothing)
        # cv2.createTrackbar('Max area', 'Select Only Checkpoint', 1, int(find_max_contour_area(c2)) + 1, nothing)
        cv2.createTrackbar(switch, 'Select Only Checkpoint',3,5,nothing)
    if s == 3: #for define checkpoint contour
        s = cv2.getTrackbarPos(switch,'Select Only Checkpoint')
        # for_find_square_contour = draw_contours(field2.blank_image_with_same_size().copy(), c2, 1)
        pre_checkpoint_contour, checkpoint_area = find_square_contours2(c2, field2.raw_image) #checkpoint_contour = [contours]
        img = draw_contours(field2.blank_image_with_same_size().copy(), pre_checkpoint_contour, 1)
        cv2.imshow('Select Only Checkpoint',img)
        count_for_4 = 0
    if s == 4: #save checkpoint contour
        checkpoint_roi = []
        img2[ : : ] = 255
        cv2.imshow('Select Only Checkpoint',img2)
        checkpoint_contour, _ = find_contours(img, 40, 100, 3, 3, 'external')
        for cnt in checkpoint_contour:
            if find_contour_center(cnt) not in field2.checkpoint_location:
                field2.checkpoint_location.append(find_contour_center(cnt))
            if count_for_4 < len(checkpoint_contour):
                x, y, width, height = cv2.boundingRect(cnt)
                roi = field2.raw_image[y:y+height, x:x+width]
                checkpoint_roi.append(roi)
                count_for_4 += 1
            else:
                None
        # for cnt in checkpoint_contour:
        #     if count_for_4 < len(checkpoint_contour):
        #         x, y, width, height = cv2.boundingRect(cnt)
        #         roi = field2.raw_image[y:y+height, x:x+width]
        #         checkpoint_roi.append(roi)
        #         count_for_4 += 1
        img = draw_contours(field2.blank_image_with_same_size().copy(), checkpoint_contour, 1)
        cv2.imshow('Select Only Checkpoint',img)
        # cv2.createTrackbar('Min area2', 'Select Only Palette', 1, int(find_max_contour_area(c2)) + 1, nothing)
        # cv2.createTrackbar('Max area2', 'Select Only Palette', 1, int(find_max_contour_area(c2)) + 1, nothing)
        cv2.createTrackbar('Erosion Iteration', 'Select Only Palette', 1, 20, nothing)
        cv2.createTrackbar(switch, 'Select Only Palette',5,8,nothing)
    if s == 5:
        s = cv2.getTrackbarPos(switch,'Select Only Palette')
        # palette_contour = find_contour_by_area(c2, min_area2, max_area2)
        palette_contour = find_palette_by_checkpoint_area(c2, checkpoint_area)
        img = draw_contours(field2.blank_image_with_same_size().copy(), palette_contour, 3)
        cv2.imshow('Select Only Palette',img)
    if s == 6:
        img2[ : : ] = 255
        cv2.imshow('Select Only Palette',img2)
        palette_contour_for_path_height, palette_hie = find_contours(img, 40, 100, 3, 3, 'tree')
        skeleton_image, field2.skeleton_coordinate = create_skeleton(palette_contour, erode_iter, field2.raw_image)
        cv2.imshow('Select Only Palette',skeleton_image)
    if s == 7:
        open_point = find_opened_end(field2.skeleton_coordinate, skeleton_image)
        background = skeleton_image.copy()
        for op in open_point:
            background = cv2.circle(background, (op[0], op[1]), radius=0, color=(255, 255, 255), thickness=10)
        cv2.imshow('Select Only Palette',background)
    if s == 8:
        connected_point = find_connected_path(field2.checkpoint_location, open_point, 150)
        path = draw_connected_path(skeleton_image, connected_point)
        cv2.imshow('Select Only Palette',path)

# cv2.destroyAllWindows()
start_point = template_matching_with_roi(triangle.raw_image, checkpoint_roi, field2.checkpoint_location, 10)
# lo
# checkpoint_area
# max(checkpoint_area)
start_point
field2.checkpoint_location
end_point, tf = find_endpoint2(field2.checkpoint_location, field2.gray_image())
end_point
### give start (y1,x1) and end (y2,x2) and the binary maze image as input
# path2, cost = shortest_path([start_point[1], start_point[0]], [end_point[1], end_point[0]], path)
plane_path = find_plane_path([start_point[1], start_point[0]], [end_point[1], end_point[0]], path)
plane_path #is in (y, x) fornat
bg = field2.raw_image.copy()
bg = cv2.circle(bg, (80, 367), radius=0, color=(0, 0, 255), thickness=10)
bg = cv2.circle(bg, (353, 55), radius=0, color=(0, 0, 255), thickness=10)
for p in plane_path:
    bg[p[1]][p[0]] = [0, 0, 255]
len(palette_contour_for_path_height[0])

palette_hie
len(palette_hie[0])
real_palette = []
l1 = []
for hie in range(len(palette_hie[0])):
    hie2 = palette_hie[0][hie]
    if hie2[2] == -1:
        # if high_res_hierarchy[0][hie2[3]][3] == 1:
        real_palette.append(hie)
real_palette
denoised_img = cv2.fastNlMeansDenoising(field2.gray_image(), None, 10, 7, 21)
min_max = []
for hie in real_palette:
    pre_coordinate_plot2 = [] #correct
    palette = cv2.drawContours(field2.blank_image_with_same_size().copy(), palette_contour_for_path_height, hie, (255, 255, 255), 1)
    filled_inner = cv2.fillPoly(field2.blank_image_with_same_size().copy(), [palette_contour_for_path_height[hie]], color=(255,255,255))
    from skimage.morphology import skeletonize
    # from skimage.morphology import thin
    from skimage import morphology, filters
    # binary = filled_inner > filters.threshold_otsu(filled_inner)
    # thin = thin(filled_inner_binary)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(filled_inner, kernel, iterations = 1)
    # binary = erosion > filters.threshold_otsu(erosion)
    # binary = filled_inner > filters.threshold_otsu(filled_inner)
    skeleton = skeletonize(erosion, method='lee')
    for x in range(skeleton.shape[0]):
        for y in range(skeleton.shape[1]):
            color = skeleton[x][y]
            if color == 255:
                pre_coordinate_plot2.append([x, y])
    min = denoised_img[pre_coordinate_plot2[0][0],  pre_coordinate_plot2[0][1]]
    max = denoised_img[pre_coordinate_plot2[1][0],  pre_coordinate_plot2[1][1]]
    for pixel in pre_coordinate_plot2:
        x = pixel[0]
        y = pixel[1]
        z = denoised_img[x, y]
        if z < min:
            min = z
        if z > max:
            max = z
    min_max.append([min, max])
min_max
min
max
denoised_img[160,  351]
pre_coordinate_plot2
bg = field2.raw_image.copy()
for p in pre_coordinate_plot2:
    bg[p[0], p[1]] = [0, 0, 255]
l1
[min, max] = find_min_max_palette(palette_contour_for_path_height, field2.gray_image())
min
max

path22 = []
for p in plane_path:
    x = p[0]
    y = p[1]
    z = None
    found = 0
    dist = cv2.pointPolygonTest(palette_contour_for_path_height[3], (x, y), False)
    if dist == 1:
        bg[y][x] = [0, 0, 255]
        # [min, max] = find_min_max_palette([palette_contour_for_path_height[3]], field2.gray_image())
        current = denoised_img[y][x]
        z = map_value(95, 138, 100, 200, current)
        z = 300 - z
    if z == None:
        z = None
    path22.append([x, y, z])
path22
real2 = []
near = None
pre_real2 = []
for co in path22:
    if co[2] != None:
        near = co[2]
        if pre_real2 == []:
            real2.append(co)
        else:
            for co2 in pre_real2:
                co2[2] = near
                real2.append(co2)
            real2.append(co)
            pre_real2 = []
    if co[2] == None:
        if near == None:
            pre_real2.append(co)
        else:
            co[2] = near
            real2.append(co)

mmm = cv2.drawContours(field2.blank_image_with_same_size().copy(),  palette_contour_for_path_height, 3, (255, 255, 255), 1)
min
max
l1.append([min, max])
l1
real_palette
cccc = draw_contours(field2.blank_image_with_same_size().copy(), [palette_contour_for_path_height[3]], 1)
ppp = cv2.drawContours(field2.blank_image_with_same_size().copy(), palette_contour_for_path_height, 3, (255, 255, 255), 1)
# def find_real_path(plane_path, palette_contour, palette_hie, gray_denoised_field_image):
#     real_path = []
#     for p in plane_path:
#         x = p[0]
#         y = p[1]
#         z = None
#         found = 0
#         # for palette_c in palette_contour:
#         for hie in range(len(palette_hie[0])):
#             dist = cv2.pointPolygonTest(palette_contour_for_path_height[hie], (x, y), False)
#             if dist == 1 and found == 0:
#                 [min, max] = find_min_max_palette([palette_contour_for_path_height[hie]], gray_denoised_field_image)
#                 current = gray_denoised_field_image[y][x]
#                 z = map_value(min, max, 200, 100, current)
#                 found += 1
#             else:
#                 None
#         real_path.append([x, y, z])
#     return real_path

# r = find_real_path(plane_path, palette_contour_for_path_height, palette_hie, field2.gray_image())
# r
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection="3d")
x = []
y = []
z = []
for co in real2:
    # for co in co_list:
    x.append(int(co[1]))
    y.append(int(co[0]))
    z.append(int(co[2]))
ax.scatter3D(x, y, z, 'gray')
plt.show()
# for p in path2:

# [min, max] = find_min_max_palette(palette_contour_for_path_height, field2.gray_image())
# min
# max
# min
# max
# point = []
# # for y in range(len())
# end_point = find_endpoint2(field2.checkpoint_location, field2.gray_image())
# end_point


# [min, max] = find_mic_max_palette()


# square_contour, area = find_square_contours2(c2, field2.raw_image)
# len(square_contour)
# bg2 = blank_image_with_same_size(field2.raw_image)
# bg3 = draw_contours(bg2.copy(), square_contour, 1)
#
# palette_contour = find_palette_by_checkpoint_area(c2, area)
# bg2 = blank_image_with_same_size(field2.raw_image)
# bg3 = draw_contours(bg2.copy(), palette_contour, 1)


# centerpoint = template_matching(for_template_matching, triangle.gray_image(), 15, 20)
# x = centerpoint[0]
# y = centerpoint[1]
# background = cv2.circle(path.copy(), (391, 357), radius=0, color=(255, 255, 255), thickness=10)
cv2.imshow('tune2', bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
