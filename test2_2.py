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

%matplotlib qt
#
# field_source = 'field_from_cam2.jpg'
# field_original = cv2.imread(field_source)
# gray_field = cv2.imread(field_source,0)
# blur_gray_field = cv2.GaussianBlur(gray_field, (5, 5), 0)
# # plt.imshow(blur_gray_field, cmap='gray')
# field_canny_edge = cv2.Canny(blur_gray_field, 20, 50)
# # plt.imshow(field_canny_edge, cmap='gray')
#
# #find External Contours
# # contours, hierarchy = cv2.findContours(field_canny_edge,
# #                                         cv2.RETR_LIST,
# #                                         cv2.CHAIN_APPROX_SIMPLE)
# contours = cv2.findContours(field_canny_edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# grabed_contour = contours[0] if len(contours) == 2 else contours[1]
# sorted_contour = sorted(grabed_contour, key = cv2.contourArea, reverse = True)[:5]
#
# # loop over the contours
# for c in sorted_contour:
# 	# approximate the contour
# 	perimeter = cv2.arcLength(c, True)
# 	approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
# 	# if our approximated contour has four points, then we
# 	# can assume that we have found our screen
# 	if len(approx) == 4:
# 		field_contour_vertices = approx
# 		break
# #approx = top_left, bottom_left, bottom_right, top_right
# top_left = [approx[0][0][0], approx[0][0][1]]
# bottom_left = [approx[1][0][0], approx[1][0][1]]
# bottom_right = [approx[2][0][0], approx[2][0][1]]
# top_right = [approx[3][0][0], approx[3][0][1]]
# before_transform = np.float32([top_left, top_right, bottom_right, bottom_left])
# before_transform
# img = field_original.copy()
# # for i in before_transform:
# # 	img = cv2.circle(img, (int(i[0]), int(i[1])), radius=0, color=(0, 0, 255), thickness=10)
# # plt.imshow(img)
# width1 = np.sqrt(abs((top_left[1] - top_right[1]) ** 2) + abs((top_left[0] - top_right[0]) ** 2))
# width2 = np.sqrt(abs((bottom_left[1] - bottom_right[1]) ** 2) + abs((bottom_left[0] - bottom_right[0]) ** 2))
# width = int(min(width1, width2))
# width
# height1 = np.sqrt(abs((top_left[1] - bottom_left[1]) ** 2) + abs((top_left[0] - bottom_left[0]) ** 2))
# height2 = np.sqrt(abs((top_right[1] - bottom_right[1]) ** 2) + abs((top_right[0] - bottom_right[0]) ** 2))
# height = int(min(height1, height2))
# height
# ratio = height / width
# ratio
# # if height < width:xs
# # 	transform_to = np.float32([[0,0], [height / ratio - 1, 0], [height / ratio - 1, height - 1], [0, height - 1]])
# # else:
# # 	transform_to = np.float32([[0,0], [width - 1, 0], [width - 1, width * ratio - 1], [0, width * ratio - 1]])
# transform_to = np.float32([[0,0], [width - 1, 0], [width - 1, width * ratio - 1], [0, width * ratio - 1]])
# M = cv2.getPerspectiveTransform(before_transform, transform_to)
# dst = cv2.warpPerspective(field_original, M, (width, height))
# dst

def perspective_t(self):
	field_original = self.raw_image().copy()
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
	before_transform
	img = field_original.copy()
	# for i in before_transform:
	# 	img = cv2.circle(img, (int(i[0]), int(i[1])), radius=0, color=(0, 0, 255), thickness=10)
	# plt.imshow(img)
	width1 = np.sqrt(abs((top_left[1] - top_right[1]) ** 2) + abs((top_left[0] - top_right[0]) ** 2))
	width2 = np.sqrt(abs((bottom_left[1] - bottom_right[1]) ** 2) + abs((bottom_left[0] - bottom_right[0]) ** 2))
	width = int(min(width1, width2))
	width
	height1 = np.sqrt(abs((top_left[1] - bottom_left[1]) ** 2) + abs((top_left[0] - bottom_left[0]) ** 2))
	height2 = np.sqrt(abs((top_right[1] - bottom_right[1]) ** 2) + abs((top_right[0] - bottom_right[0]) ** 2))
	height = int(min(height1, height2))
	height
	ratio = height / width
	ratio
	# if height < width:xs
	# 	transform_to = np.float32([[0,0], [height / ratio - 1, 0], [height / ratio - 1, height - 1], [0, height - 1]])
	# else:
	# 	transform_to = np.float32([[0,0], [width - 1, 0], [width - 1, width * ratio - 1], [0, width * ratio - 1]])
	transform_to = np.float32([[0,0], [width - 1, 0], [width - 1, width * ratio - 1], [0, width * ratio - 1]])
	M = cv2.getPerspectiveTransform(before_transform, transform_to)
	dst = cv2.warpPerspective(field_original, M, (width, height))
	return(dst)
