import cv2

from delete_rail import resize_percent
from canny_and_contour import *


original_symbol = cv2.imread('symbol/real_symbol1.jpg')
original_symbol = resize_percent(original_symbol, 60)
cv2.imshow('original_symbol', original_symbol)
cv2.waitKey(0)
cv2.destroyAllWindows()

#turn image to gray scale
gray_original = cv2.cvtColor(original_symbol, cv2.COLOR_BGR2GRAY)
#find contours
contours, hierarchy = find_contours(gray_original, 40, 100, 3, 3, 'tree')
#select only square contour
square_contours, area = find_square_contours(contours, gray_original, 10, 20)
#perspective transform corner of square contour
#save for further usage
