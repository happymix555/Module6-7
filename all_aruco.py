import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl

#detect ARUCO --> corners, ids of all ARUCO in an image
def detect_aruco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return corners, ids

#take ARUCO corner as input, Compute center of that ARUCO and return as Output
def compute_aruco_center(corner):
    c = corner[0]
    center = [int(c[:, 0].mean()), int(c[:, 1].mean())]
    return center

#compute point for perspective transfrom
def get_all_corner(corners, ids, list_of_corner_id):
    corner_point = []
    for c in range(len(corners)):
        if ids[c] in list_of_corner_id:
            center = compute_aruco_center(corners[c])
            corner_point.append(center)
    return corner_point
