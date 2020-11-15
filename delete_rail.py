import glob
import cv2
import numpy as np
from PIL import Image

def resize_percent(input_image ,percent):
    scale = (percent/100)
    width = int(input_image.shape[0] * scale)
    height = int(input_image.shape[1] * scale)
    dim = (height, width)
    return cv2.resize(input_image, dim, interpolation = cv2.INTER_AREA)

def read_multiple_image(path):
    real_path = path + '/*.jpg'
    images = [cv2.imread(file) for file in glob.glob(real_path)]
    images2 = []
    for i in images:
        i = resize_percent(i, 60)
        images2.append(i)
    return images2

#
# img = read_multiple_image('rail_image')
# # img
# for i in img:
#     cv2.imshow('i' ,i)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
