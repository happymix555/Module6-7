from prepare_rail_2 import *
from PC2PIC2 import *
from find_template import *

import cv2
import os
import glob
import time
from time import sleep

from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import asarray
from PIL import Image, ImageEnhance

from delete_rail import *
from perspective import *
from all_aruco import *
from all_contour import *
from path_finder import *
from main_demo2 import *
from find_template import *
# from loop_picture import *
# import setting

import serial
import math
check = True
# buf1 = []
package = []
#------------------------------------------------------------------------------------------------------------------------------------------------
#====================================== THIS PROGRAM IS FOR UART COMMUNICATION WITH DSPIC33FJ64MC802 ============================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#======================================                This is lnwModule's PACKAGE                   ============================================
#------------------------------------------------------------------------------------------------------------------------------------------------
#======================================             Author by Natmatee Pituwarajirakul              ============================================
#-------------------------------------------------------------------------------------- ----------------------------------------------------------
#   HEAD  | COMMAND | POSx_MSB | POSx_LSB | POSy_MSB | POSy_LSB | POSz_MSB | POSz_LSB | Degree |  Grip  |               CHECKSUM                |
#   0xFF  | 1 Byte  |    2 Byte for int   |    2 Byte for int   |    2 Byte for int   | 1 Byte | 1 Byte | Checksum = (SUMMATION OF PACKAGE)%256 |
#------------------------------------------------------------------------------------------------------------------------------------------------
#       FUNCTION IN THIS PROGRAM
#       1. InitUART >> Init about UART sth like baudrate, port ,parity bit, brah brah...
#       2. Acknowledge >> Wait for dspic33fj64mc802 response to PC that have a little package
#       HEAD  |  ACK   |               CHECKSUM                |
#       0xFF  |  0x69  | Checksum = (SUMMATION OF PACKAGE)/256 |
#       3. Sethome >> set position of X,Y,Z Axis to (0,0,0)
#       4. Go to pos >> go to position that you want to go
#       5. Rotate >> rotate Axis of Gripper
#       6. Grip >> Pick or Place the object in the mission
#======================================                       Good Luck Have Fun                     ============================================
#------------------------------------------------------------------------------------------------------------------------------------------------
    #Pairing with dspic33fj64mc802
    #   HEAD  |  PAIR   | POSx_MSB | POSx_LSB | POSy_MSB | POSy_LSB | POSz_MSB | POSz_LSB | Degree |  Grip  |               CHECKSUM                |
    #   0xFF  |  0x96   |   0x00   |   0x00   |   0x00   |   0x00   |   0x00   |   0x00   |  0x00  |  0x00  | Checksum = (SUMMATION OF PACKAGE)/256 |
    #------------------------------------------------------------------------------------------------------------------------------------------------
#    package = [255,150,0 0,0,0,0,0,0,0,0]
#    package[-1] = sum(package)%256
#    while(check):
#        lnwmodule.write(package)
#        lnwmodule_acknowledge()


def lnwmodule_acknowledge():
    arr = []
    arr = lnwmodule.read(4)
    if(arr[1] == 255):
        if(arr[2] == 105):
            if(arr[3] == 104):
                # print(arr)
                return True
            else:
                arr = []
        else:
            arr = []
    else:
        arr = []



def lnwmodule_sethome():
    buf1 = []
    package_home = [255,1,0,0,0,0,0,0,0,0]
    for i in range(len(package_home)):
        buf1.append(package_home[i])
    print(package_home)
    package_home[-1] = sum(package_home)%256
    lnwmodule.write(package_home)
    return buf1


# def lnwmodule_go2pos(PosX, PosY, PosZ):
#     package = [255,2,0,0,0,0,0,0,0,0,0]
#     Rho =
#     if(PosX == 0 or PosY == 0 or PosZ ==0):
#         package[2] = 0
#         package[3] = 0
#         package[4] = 0
#         package[5] = 0
#         package[6] = 0
#         package[7] = 0
#     else:
#         Rho_H = int(Rho >> 8)
#         Rho_L = int(Rho % 256 )
#         Theta_H = int(Theta >> 8)
#         Theta_L = int(Theta % 256)
#         Phi_H = int(Phi >> 8)
#         Phi_L = int(Phi % 256)
#         package[2] = Rho_H
#         package[3] = Rho_L
#         package[4] = Theta_H
#         package[5] = Theta_L
#         package[6] = Phi_H
#         package[7] = Phi_L
#     package[-1] = sum(package)%256
#     lnwmodule.write(package)

# def image_to_world(point):
#     x = point[0]
#     y = point[1]
#     z = point[2]
#     return [x, y, z]

# def lnwmodule_go2pos(list_of_point, offset_x, offset_y):
#     package = [255,2,0,0,0,0,0,0,0,0,0]
#     ref_x = 0
#     ref_y = 0
#     ref_z = 0
#     for point in list_of_point:
#         real_point = image_to_world(point)
#         current_x = real_point[0]
#         current_y = real_point[1]
#         current_z = real_point[2]
#         delta_x = current_x - ref_x
#         delta_y = current_y - ref_y
#         delta_z = current_z - ref_z
#         Rho = sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
#         Theta = math.atan(delta_y / delta_x)
#         Theta = int(Theta * 180 / math.pi)
#         Phi = math.acos(delta_z / Rho)
#         if delta_x < 0 and delta_y >= 0:
#             Theta = 180 + Theta
#         elif delta_x < 0 and delta_y < 0:
#             Theta = 270 + Theta
#         elif delta_x >= 0 and delta_y < 0:
#             Theta = 360 + Theta
#
#         if delta_z >= 0:
#             Phi = 90 - Phi
#
#         Rho_H = int(Rho >> 8)
#         Rho_L = int(Rho % 256 )
#         Theta_H = int(Theta >> 8)
#         Theta_L = int(Theta % 256)
#         Phi_H = int(Phi >> 8)
#         Phi_L = int(Phi % 256)
#         package[2] = Rho_H
#         package[3] = Rho_L
#         package[4] = Theta_H
#         package[5] = Theta_L
#         package[6] = Phi_H
#         package[7] = Phi_L
#         package[-1] = sum(package)%256
#         lnwmodule.write(package)
# print(math.atan(1 / -1))

def lnwmodule_go2pos(point, ref_x, ref_y, ref_z, offset_x, offset_y, offset_z):
    buf2 = []
    package_g2 = [255,2,0,0,0,0,0,0,0,0]
    # real_point = image_to_world(point, 0, 0, 0, 600, 600, 400, 400)
    if point[1] < offset_y:
        offset_y = 0
    if point[0] < offset_x:
        offset_x = 0
    current_x = point[0] + offset_x
    current_y = point[1] + offset_y
    current_z = point[2] + offset_z
    delta_x = current_x - ref_x
    delta_y = current_y - ref_y
    delta_z = current_z - ref_z
    Rho = math.sqrt(delta_x ** 2 + delta_y ** 2)
    if delta_x == 0 and delta_y > 0:
        Theta = 90
    elif delta_x == 0 and delta_y < 0:
        Theta = 270
    elif delta_x > 0 and delta_y == 0:
        Theta = 0
    elif delta_x < 0 and delta_y == 0:
        Theta = 180
    elif delta_x == 0 and delta_y == 0:
        Theta = 0
    # elif delta_x > 0 and delta_y < 0:
    #     Theta = math.atan(delta_y / delta_x)
    #     Theta = int(Theta * 180 / math.pi)
    #     Theta = Theta + 360
    # elif delta_x < 0 and delta_y > 0:
    #     Theta = math.atan(delta_y / delta_x)
    #     Theta = int(Theta * 180 / math.pi)
    #     Theta = Theta + 180
    else:
        Theta = math.atan(delta_y / delta_x)
        Theta = int(Theta * 180 / math.pi)
    # Phi = math.acos(delta_z / Rho)
    # Phi = int(Phi * 180 / math.pi)
    # if delta_z < 0:
    #     Phi = Phi - 90
    # elif delta_z == 0:
    #     Phi = 90
    # else:
    #     Phi = Phi

    if delta_x < 0 and delta_y > 0:
        Theta = 180 + Theta
    elif delta_x < 0 and delta_y < 0:
        Theta = 180 + Theta
    elif delta_x > 0 and delta_y < 0:
        Theta = 360 + Theta


    Rho_H = int(Rho) >> 8
    Rho_L = int(Rho) % 256
    Theta_H = int(Theta) >> 8
    Theta_L = int(Theta) % 256
    if point[2] < abs(offset_z):
        offset_z = 0
    Phi_H = int(point[2] + offset_z) >> 8
    Phi_L = int(point[2] + offset_z) % 256
    package_g2[2] = Rho_H
    package_g2[3] = Rho_L
    package_g2[4] = Theta_H
    package_g2[5] = Theta_L
    package_g2[6] = Phi_H
    package_g2[7] = Phi_L
    package_g2[-1] = sum(package_g2)%256
    for p in package_g2:
        buf2.append(p)
    lnwmodule.write(package_g2)
    print('rho = ' + str(Rho))
    print('theta = ' + str(Theta))
    print('phi(Z) = ' + str(ref_z))
    print('delta x = ' + str(delta_x))
    print('delta y = ' + str(delta_y))
    print('delta z = ' + str(delta_z))
    return current_x ,current_y, current_z, package_g2, buf2

def resetPic():
    lnwmodule.rts = 1
    lnwmodule.rts = 0
#
# def waitingPic():
#     state_wait = 0
#     while(lnwmodule.in_waiting):
#         pass
#     arr = []
#     arr2 = []
#     arr = lnwmodule.read(3)
#     print(arr)
#     if(arr[0] == 255):
#         if(arr[1] == 105):
#             if(arr[2] == 104):
#                 state_wait = 1
#             else:
#                 arr = []
#         else:
#             arr = []
#     else:
#         arr = []
#     if(state_wait == 1):
#         while(lnwmodule.in_waiting):
#             pass
#         arr2 = lnwmodule.read(3)
#         if(arr2[0] == 255):
#             if(arr2[1] == 160):
#                 if(arr2[2] == 159):
#                     resetPic()
#                 else:
#                     arr2 = []
#             else:
#                 arr2 = []
#         else:
#             arr2 = []
#     else:
#         resetPic()

# def lnwmodule_rotate(Degree):
#     package = [255,3,0,0,0,0,0,0,0,0,0]
#     package[8] = int(Degree)
#     package[-1] = sum(package)%256
#     lnwmodule.write(package)


def lnwmodule_grip(grip):
    buf3 = []
    package_grip = [255,3,0,0,0,0,0,0,0,0]
    package_grip[8] = grip
    package_grip[-1] = sum(package_grip)%256
    print('package_grip = ' + str(package_grip))
    for p in package_grip:
        buf3.append(p)
    lnwmodule.write(package_grip)
    return buf3

def lnwmodule_done():
    done = []
    done = lnwmodule.read(11)
    num = 0
    state = 0
    for i in range(len(package)):
        if(package[i] == done[i]):
            num += 1
    if(num == 11):
        state = 1
    return state

def lnwmodule_done2(serial_obj, ref_buffer):
    print('in done function')
    done = []
    c = 0
    while(lnwmodule.in_waiting):
        if c == 0:
            print('while Lnwmodule inwaiting')
            c = 1
        pass
    done = serial_obj.read(10)
    num = 0
    state = 0
    print('done = ' + str(done))
    for i in range(len(ref_buffer)):
        # print('package = ' + str(buf1))
        # print('done = ' + str(done))
        if(ref_buffer[i] == done[i]):
            print('package = ' + str(ref_buffer))
            num += 1
    print('num = ' + str(num))
    if(num == 10):
        return True
        print('done222')
    else:
        return False

# buf = []
# lnwmodule = serial.Serial()
# lnwmodule.baudrate = 19200
# lnwmodule.port = 'COM14'
# lnwmodule.rts = 0
# lnwmodule.open()
# lnwmodule_grip(1)
# while(lnwmodule.in_waiting):
#      pass
# lnwmodule_acknowledge()
# while(lnwmodule.in_waiting):
#      pass
# buf = lnwmodule.read(10)
# print('buf = ' + str(buf))
# inwait


# current_x, current_y, current_z, package = lnwmodule_go2pos([100 ,100, 0], 0, 0, 0, 0, 0)
# resetPic()

# buf_h = lnwmodule_sethome()
# arr = []
# arr = lnwmodule.read(4)
# print('buf_h = ' + str(buf_h))
# print('arr = ' + str(arr))
# while True:
#     if lnwmodule_done2(lnwmodule, buf_h) == True:
#         break

def demo1_home():
    buf_h = lnwmodule_sethome()
    arr = []
    # while(lnwmodule.in_waiting):
    #     pass
    arr = lnwmodule.read(4)
    print('buf_h = ' + str(buf_h))
    print('arr = ' + str(arr))
    while True:
        if lnwmodule_done2(lnwmodule, buf_h) == True:
            break

def demo2_gotoPos(list_of_pos, offset_x, offset_y, offset_z):
    print('in demo1_gotoPos')
    position_flag = 0
    position_count = 0
    image_count = 0
    record_flag = 0
    ref_x = 0
    ref_y = 0
    ref_z = list_of_pos[0][2]
    cc = 0
    while(True):
        if cc == 0:
            print('in gotoPos while')
        cc += 1
        # if cc > 1000:
        #     break
        if position_flag == 0: #send position
            current_x, current_y, current_z, package_g, buf_g = lnwmodule_go2pos(list_of_pos[position_count], ref_x, ref_y, ref_z, offset_x, offset_y, offset_z)
            ref_x = current_x
            ref_y = current_y
            ref_z = current_z
            position_flag = 1
            print('state 1')
            print('ref_x = ' + str(ref_x))
            print('ref_y = ' + str(ref_y))
            print('ref_z = ' + str(ref_z))
            print('package g = ' + str(package_g))
        if position_flag == 1: #waiting for Acknowledge and Done sign from Low level
            arr_ack = []
            # while(lnwmodule.in_waiting):
            #     pass
            arr_ack = lnwmodule.read(4)
            print('state 2')
            if lnwmodule_done2(lnwmodule, buf_g) == True:
                arr_ack = []
                position_count += 1
                position_flag = 0
                sleep(0.5)
                print('state 3')
                if position_count == 1:
                    record_flag = 1
                    print('state 4')
                if position_count == len(list_of_pos):
                    print(position_count)
                    # vid.release()
                    # cv2.destroyAllWindows()
                    print('state 5')
                    # state = 2
                    sleep(0.2)
                    break
# vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
# vid.set(3, 1280) # set the resolution
# vid.set(4, 720)



def gripper_grip(flag):
    if flag == 0:
        buf_gr = lnwmodule_grip(1)
        arr = []
        arr = lnwmodule.read(4)
        while(True):
            if lnwmodule_done2(lnwmodule, buf_gr) == True:
                break
    # return 1

def gripper_release(flag):
    if flag == 1:
        buf_gr = lnwmodule_grip(0)
        arr = []
        arr = lnwmodule.read(4)
        while(True):
            if lnwmodule_done2(lnwmodule, buf_gr) == True:
                break
    # return 0

def loop_picture(list_of_pos):
    state = 0
    position_flag = 0
    position_count = 0
    image_count = 0
    record_flag = 0
    ref_x = 0
    ref_y = 0
    ref_z = list_of_pos[0][2]

    path = 'rail_image/rail_image_loop'
    files = glob.glob(path + '/*')
    for f in files:
        os.remove(f)

    vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    vid.set(3, 1280) # set the resolution
    vid.set(4, 720)

    while (True):
        if state == 0:
            print('in loop picture state 0')
            if position_flag == 0: #send position
                current_x, current_y, current_z, package_g, buf_g = lnwmodule_go2pos(list_of_pos[position_count], ref_x, ref_y, ref_z, 0, 0, 0)
                ref_x = current_x
                ref_y = current_y
                ref_z = current_z
                position_flag = 1
                print('state 1')
                print('ref_x = ' + str(ref_x))
                print('ref_y = ' + str(ref_y))
                print('ref_z = ' + str(ref_z))
                print('package g = ' + str(package_g))
            if position_flag == 1: #waiting for Acknowledge and Done sign from Low level
                arr_ack = []
                # while(lnwmodule.in_waiting):
                #     pass
                arr_ack = lnwmodule.read(4)
                print('state 2')
                if lnwmodule_done2(lnwmodule, buf_g) == True:
                    arr_ack = []
                    position_count += 1
                    position_flag = 0
                    sleep(0.5)
                    state = 1
                    print('state 3')
                    # if position_count == 1:
                    #     record_flag = 1
                    #     print('state 4')
                    # if position_count == len(list_of_pos): #finish all position
                    #     print(position_count)
                    #     # vid.release()
                    #     # cv2.destroyAllWindows()
                    #     print('state 5')
                    #     # state = 2
                    #     sleep(0.5)
                    #     break
        if state == 1:
            ref_time = time.perf_counter()
            while(True):
                current_time = time.perf_counter()
                ret, frame = vid.read()
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                elif current_time - ref_time > 1:
                    name_and_path = 'rail_image/rail_image_loop/' + str(image_count) + '.jpg'
                    cv2.imwrite(name_and_path, frame)
                    image_count += 1
                    break
            # vid.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
            state = 2

        if state == 2:
            if position_count == len(list_of_pos): #finish all position
                # print(position_count)
                # vid.release()
                # cv2.destroyAllWindows()
                # print('state 5')
                # state = 2
                # sleep(0.5)
                break
            else:
                state = 0


def go_to_grip_and_home():
    point = [[42,30,0],[42,30,280]]
    demo2_gotoPos(point, 0, 0, 0)
    grip_flag = gripper_grip(0)
    point2 = [[0,0,0]]
    demo2_gotoPos(point2, 0, 0, 20)
    # demo1_home()

def grip2():
    gripper_grip(0)

def release2():
    gripper_release(1)

# lnwmodule = serial.Serial()
# sleep(2)
# lnwmodule.baudrate = 19200
# lnwmodule.port = 'COM12'
# lnwmodule.rts = 0
# lnwmodule.open()

# demo1_home()

# go_to_grip_and_home()

# record_trigger = 0
# list_of_position = [[220, 340, 100]]
# demo1_gotoPos(list_of_position)
# if __name__ == "__main__":
#     record_trigger = 1

# list_of_position = [[220, 360, 0]]
# demo1_gotoPos(list_of_position, 0, 0, 0)
# pre = []
# for i in range(210, 20, -20):
#     pre2 = []
#     pre2.append(i)
#     pre2.append(390)
#     pre2.append(0)
#     pre.append(pre2)
# pre
#
# pre = []
# for i in range(390, 150, -20):
#     pre2 = []
#     pre2.append(20)
#     pre2.append(i)
#     pre2.append(0)
#     pre.append(pre2)
# pre
#
# pre = []
# for i in range(20, 390, 20):
#     pre2 = []
#     pre2.append(i)
#     pre2.append(150)
#     pre2.append(0)
#     pre.append(pre2)
# pre
#
# pre = []
# for i in range(150, 390, 20):
#     pre2 = []
#     pre2.append(390)
#     pre2.append(i)
#     pre2.append(0)
#     pre.append(pre2)
# pre
#
# pre = []
# for i in range(390, 20, -20):
#     pre2 = []
#     pre2.append(i)
#     pre2.append(340)
#     pre2.append(0)
#     pre.append(pre2)
# pre

# exec(open('loop_picture.py').read())
# list_of_position = [[220, 340, 0],[210,390,0],[20,390,0],[20,150,0],[390,150,0],[390,390,0],[20,390,0],[220, 340, 0]]
list_of_position = [[220, 330, 0],
 [210, 390, 0],
 [190, 390, 0],
 [170, 390, 0],
 [150, 390, 0],
 [130, 390, 0],
 [110, 390, 0],
 [90, 390, 0],
 [70, 390, 0],
 [50, 390, 0],
 [30, 390, 0],[20,390,0],[20, 370, 0],
 [20, 350, 0],
 [20, 330, 0],
 [20, 310, 0],
 [20, 290, 0],
 [20, 270, 0],
 [20, 250, 0],
 [20, 230, 0],
 [20, 210, 0],
 [20, 190, 0],
 [20, 170, 0],[20,150,0],[40, 150, 0],
 [60, 150, 0],
 [80, 150, 0],
 [100, 150, 0],
 [120, 150, 0],
 [140, 150, 0],
 [160, 150, 0],
 [180, 150, 0],
 [200, 150, 0],
 [220, 150, 0],
 [240, 150, 0],
 [260, 150, 0],
 [280, 150, 0],
 [300, 150, 0],
 [320, 150, 0],
 [340, 150, 0],
 [360, 150, 0],
 [380, 150, 0],[390,150,0],[390, 170, 0],
 [390, 190, 0],
 [390, 210, 0],
 [390, 230, 0],
 [390, 250, 0],
 [390, 270, 0],
 [390, 290, 0],
 [390, 310, 0],
 [390, 330, 0],
 [390, 350, 0],
 [390, 370, 0],[390,390,0], [370, 340, 0],
 [350, 340, 0],
 [330, 340, 0],
 [310, 340, 0],
 [290, 340, 0],
 [270, 340, 0],
 [250, 340, 0],
 [230, 340, 0],
 [210, 340, 0],
 [190, 340, 0],
 [170, 340, 0],
 [150, 340, 0],
 [130, 340, 0],
 [110, 340, 0],
 [90, 340, 0],
 [70, 340, 0],
 [50, 340, 0],
 [30, 340, 0],[220, 340, 0]]
# loop_picture(list_of_position)
def loop_pic_ui():
    list_of_position = [[220, 330, 0],
     [210, 390, 0],
     [170, 390, 0],
     [130, 390, 0],
     [90, 390, 0],
     [50, 390, 0],[20,390,0],[20, 370, 0],
     [20, 330, 0],
     [20, 290, 0],
     [20, 250, 0],
     [20, 210, 0],
     [20, 170, 0],[20,150,0],[40, 150, 0],
     [80, 150, 0],
     [120, 150, 0],
     [160, 150, 0],
     [200, 150, 0],
     [240, 150, 0],
     [320, 150, 0],
     [360, 150, 0],
     [380, 150, 0],[390,150,0],
     [390, 190, 0],
     [390, 230, 0],
     [390, 270, 0],
     [390, 310, 0],
     [390, 350, 0],
     [390, 370, 0],[390,390,0],
     [350, 390, 0],
     [310, 390, 0],
     [270, 390, 0],
     [230, 390, 0],
     [220, 340, 0]]
    # list_of_position = [[220, 330, 0],
    #  [210, 390, 0],
    #  [190, 390, 0],
    #  [170, 390, 0],
    #  [150, 390, 0],
    #  [130, 390, 0],
    #  [110, 390, 0],
    #  [90, 390, 0],
    #  [70, 390, 0],
    #  [50, 390, 0],
    #  [30, 390, 0],[20,390,0],[20, 370, 0],
    #  [20, 350, 0],
    #  [20, 330, 0],
    #  [20, 310, 0],
    #  [20, 290, 0],
    #  [20, 270, 0],
    #  [20, 250, 0],
    #  [20, 230, 0],
    #  [20, 210, 0],
    #  [20, 190, 0],
    #  [20, 170, 0],[20,150,0],[40, 150, 0],
    #  [60, 150, 0],
    #  [80, 150, 0],
    #  [100, 150, 0],
    #  [120, 150, 0],
    #  [140, 150, 0],
    #  [160, 150, 0],
    #  [180, 150, 0],
    #  [200, 150, 0],
    #  [220, 150, 0],
    #  [240, 150, 0],
    #  [260, 150, 0],
    #  [280, 150, 0],
    #  [300, 150, 0],
    #  [320, 150, 0],
    #  [340, 150, 0],
    #  [360, 150, 0],
    #  [380, 150, 0],[390,150,0],[390, 170, 0],
    #  [390, 190, 0],
    #  [390, 210, 0],
    #  [390, 230, 0],
    #  [390, 250, 0],
    #  [390, 270, 0],
    #  [390, 290, 0],
    #  [390, 310, 0],
    #  [390, 330, 0],
    #  [390, 350, 0],
    #  [390, 370, 0],[390,390,0], [370, 340, 0],
    #  [350, 340, 0],
    #  [330, 340, 0],
    #  [310, 340, 0],
    #  [290, 340, 0],
    #  [270, 340, 0],
    #  [250, 340, 0],
    #  [230, 340, 0],
    #  [210, 340, 0],
    #  [190, 340, 0],
    #  [170, 340, 0],
    #  [150, 340, 0],
    #  [130, 340, 0],
    #  [110, 340, 0],
    #  [90, 340, 0],
    #  [70, 340, 0],
    #  [50, 340, 0],
    #  [30, 340, 0],[220, 340, 0]]
    loop_picture(list_of_position)
# loop_pic_ui()

# list_of_position = [[20,150,0],[390,150,0],[390,390,0],[20,390,0],[220, 340, 0]]
# # lnwmodule.write([255,2,0,100,0,90,0,0,0,191])


# point = [[77,0,0],[77,0,300]]
# demo2_gotoPos(point, 0, 0, 0)

# grip_flag = gripper_grip(0)


# grip_flag = gripper_release(1)

# point = [[0,0,0]]
# demo2_gotoPos(point, 0, 0, 20)

list_of_point = [[77, 355, 0],
 [77, 355, 200],
 [77, 276, 200],
 [77, 191, 300],
 [89, 163, 300],
 [165, 88, 300],
 [187, 80, 300],
 [275, 80, 200],
 [352, 80, 200],
 [352, 148, 200],
 [352, 351, 300],
 [218, 351, 300],
 [218, 300, 300],
 [218, 232, 300]]
# demo2_gotoPos(list_of_point, 0, -35, -30)
#
# all_find_template()
#
# delete_rail()
#
# point = [[75,0,0],[75,0,300]]
# demo1_gotoPos(point, 0, 0, 0)
#
# grip_flag = 0
# grip_flag = gripper_grip(grip_flag)
#
# grip_flag = gripper_release(grip_flag)
#
# point2 = [[100,100,100]]
# demo1_gotoPos(point2, 0, 0, 0)
#
# point3 = all_main_demo2()
# demo1_gotoPos(point3, 0, 0, 0)


# list_of_position = [[220, 340, 100],[210,390,100],[20,390,100],[20,150,100],[390,150,100],[390,390,100],[20,390,100],[220, 340, 100]]
# len(list_of_position)
# position_flag = 0
# position_count = 0
# image_count = 0
# record_flag = 0
# ref_x = 0
# ref_y = 0
# ref_z = 0
# cc = 0
# while(True):
#     cc += 1
#     # if cc > 1000:
#     #     break
#     if position_flag == 0: #send position
#         current_x, current_y, current_z, package_g, buf_g = lnwmodule_go2pos(list_of_position[position_count], ref_x, ref_y, ref_z, 0, 0)
#         ref_x = current_x
#         ref_y = current_y
#         ref_z = current_z
#         position_flag = 1
#         print('state 1')
#         print('ref_x = ' + str(ref_x))
#         print('ref_y = ' + str(ref_y))
#         print('ref_z = ' + str(ref_z))
#         print('package g = ' + str(package_g))
#     if position_flag == 1: #waiting for Acknowledge and Done sign from Low level
#         arr_ack = []
#         arr_ack = lnwmodule.read(4)
#         print('state 2')
#         if lnwmodule_done2(lnwmodule, buf_g) == True:
#             arr_ack = []
#             position_count += 1
#             position_flag = 0
#             print('state 3')
#             if position_count == 1:
#                 record_flag = 1
#                 print('state 4')
#             if position_count == len(list_of_position):
#                 print(position_count)
#                 # vid.release()
#                 # cv2.destroyAllWindows()
#                 print('state 5')
#                 # state = 2
#                 break
    # if record_flag == 1:
    #     ret, frame = vid.read()
    #     cv2.imshow('frame', frame)
    #     name_and_path = 'rail_image/rail_image_loop/' + str(image_count) + '.jpg'
    #     cv2.imwrite(name_and_path, frame)
    #     image_count += 1
    #     print('state 6')

# point = [[80,0,0],[80,0,300]]
# len(list_of_position)
# position_flag = 0
# position_count = 0
# image_count = 0
# record_flag = 0
# ref_x = 0
# ref_y = 0
# ref_z = 0
# cc = 0
# while(True):
#     cc += 1
#     # if cc > 1000:
#     #     break
#     if position_flag == 0: #send position
#         current_x, current_y, current_z, package_g, buf_g = lnwmodule_go2pos(point[position_count], ref_x, ref_y, ref_z, 0, 0)
#         ref_x = current_x
#         ref_y = current_y
#         ref_z = current_z
#         position_flag = 1
#         print('state 1')
#         print('ref_x = ' + str(ref_x))
#         print('ref_y = ' + str(ref_y))
#         print('ref_z = ' + str(ref_z))
#         print('package g = ' + str(package_g))
#     if position_flag == 1: #waiting for Acknowledge and Done sign from Low level
#         arr_ack = []
#         arr_ack = lnwmodule.read(4)
#         print('state 2')
#         if lnwmodule_done2(lnwmodule, buf_g) == True:
#             arr_ack = []
#             position_count += 1
#             position_flag = 0
#             print('state 3')
#             if position_count == 1:
#                 record_flag = 1
#                 print('state 4')
#             if position_count == len(point):
#                 print(position_count)
#                 # vid.release()
#                 # cv2.destroyAllWindows()
#                 print('state 5')
#                 # state = 2
#                 break

# buf_gr = lnwmodule_grip(1)
# arr = []
# arr = lnwmodule.read(4)
# while(True):
#     if lnwmodule_done2(lnwmodule, buf_gr) == True:
#         break
#
# buf_gr = lnwmodule_grip(0)
# arr = []
# arr = lnwmodule.read(4)
# while(True):
#     if lnwmodule_done2(lnwmodule, buf_gr) == True:
#         break


# point = [[100,100,100]]
# # len(list_of_position)
#
# point = [[328, 75, 202],
#  [328, 188, 204],
#  [315, 209, 200],
#  [200, 277, 280],
#  [160, 298, 300],
#  [90, 334, 300]]
# len(list_of_position)

import tkinter as tk
import tkinter.ttk as ttk
#from playsound import playsound

def print_mix():
    print('mix')

class LnwApp:
    def __init__(self, master=None):
        # build ui
        master.title("LNWmodule GUI")
        self.frame_1 = tk.Frame(master)
        self.frame_2 = tk.Frame(self.frame_1)
        self.label_1 = tk.Label(self.frame_2)
        self.logolek_png = tk.PhotoImage(file='logolek.png')
        self.label_1.config(background='#ffff80', image=self.logolek_png)
        self.label_1.pack(padx='5', pady='5', side='top')
        self.button_1 = tk.Button(self.frame_2)
        self.button_1.config(font='{Impact} 16 {}', justify='left', text='START', command=loop_pic_ui)
        self.button_1.pack(padx='5', pady='5', side='top')
        self.button_2 = tk.Button(self.frame_2)
        self.button_2.config(font='{Impact} 16 {}', justify='left', takefocus=False, text='Capture template', command=all_find_template)
        self.button_2.pack(padx='5', pady='5', side='top')
        self.button_3 = tk.Button(self.frame_2)
        self.button_3.config(font='{Impact} 16 {}', justify='left', text='Image processing', command=self.point_storage)
        self.button_3.pack(padx='5', pady='5', side='top')
        self.button_4 = tk.Button(self.frame_2)
        self.button_4.config(font='{Impact} 16 {}', justify='left', text='Go to grip', command=go_to_grip_and_home)
        self.button_4.pack(padx='5', pady='5', side='top')
        self.button_5 = tk.Button(self.frame_2)
        self.button_5.config(font='{Impact} 16 {}', justify='left', text='HOME', command=demo1_home)
        self.button_5.pack(padx='5', pady='5', side='top')
        self.button_6 = tk.Button(self.frame_2)
        self.button_6.config(font='{Impact} 16 {}', justify='left', text='Grip', command=grip2)
        self.button_6.pack(padx='5', pady='5', side='top')
        self.button_7 = tk.Button(self.frame_2)
        self.button_7.config(font='{Impact} 16 {}', justify='left', text='Release', command=release2)
        self.button_7.pack(padx='5', pady='5', side='top')
        self.button_8 = tk.Button(self.frame_2)
        self.button_8.config(font='{Impact} 16 {}', justify='left', text='Lnw Module', command=self.perform_lnw)
        self.button_8.pack(padx='5', pady='5', side='top')
        self.button_9 = tk.Button(self.frame_2)
        self.button_9.config(activebackground='#0080ff', background='#ff0000', font='{Impact} 24 {}', foreground='#ffffff')
        self.button_9.config(justify='left', text='Do not click !!!!', command=self.Popup)
        self.button_9.pack(padx='5', pady='5', side='top')
        self.button_10 = tk.Button(self.frame_2)
        self.button_10.config(activebackground='#0080ff', background='#ff0000', bitmap='error', font='{Impact} 24 {}')
        self.button_10.config(foreground='#ffffff', justify='left',command=self.finishes)
        self.button_10.pack(padx='5', pady='5', side='top')
        self.frame_2.config(background='#ffff80', height='200', width='200')
        self.frame_2.pack(padx='10', pady='10', side='top')
        self.frame_1.config(background='#ffff00', height='200', width='200')
        self.frame_1.pack(side='top')

        # Main widget
        self.mainwindow = self.frame_1
        self.tp = []



    def run(self):
        self.mainwindow.mainloop()

    def point_storage(self):
        _, self.tp, _, _ = all_main_demo2()
        print(self.tp)

    def perform_lnw(self):
        demo2_gotoPos(self.tp, 5, -35, -40)
        self.finishes()

    def Popup(self):
        #playsound('LopingSting.mp3')
        self.image = cv2.imread('joke.jpg')
        cv2.imshow('joke!!!!',self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def finishes(self):
        #playsound('LopingSting.mp3')
        self.image2 = cv2.imread('finishes.jpg')
        width = int(self.image2.shape[1] * 1/2)
        height = int(self.image2.shape[0] * 1/2)
        dsize = (width, height)
        self.image2 = cv2.resize(self.image2, dsize)
        cv2.imshow('finishes!!!!',self.image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import tkinter as tk
    lnwmodule = serial.Serial()
    sleep(2)
    lnwmodule.baudrate = 19200
    lnwmodule.port = 'COM12'
    lnwmodule.rts = 0
    lnwmodule.open()
    root = tk.Tk()
    app = LnwApp(root)
    app.run()
