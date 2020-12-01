import cv2

from all_aruco import *
from perspective import *

video_object = cv2.VideoCapture(0)
original_image = []
capture_counter = []
while(True):
    ret,frame = video_object.read()
    cv2.imshow('Capturing Video',frame)
    corners, ids = detect_aruco(frame)
    corner_point = get_all_corner(corners, ids, [0, 1, 2, 3]])
    ordered_point = order_points(corner_point)
    if ordered_point != None:
        after_pt = perspective_transform(frame, ordered_point , side_length)
    key = cv2.waitKey(1)
    if key == 27:
        video_object.release()
        cv2.destroyAllWindows()
        break
side_length = 
