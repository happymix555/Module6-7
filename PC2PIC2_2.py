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
def lnwmodule_initUART():
    lnwmodule = serial.Serial()
    lnwmodule.baudrate = 19200
    lnwmodule.port = 'COM4'
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
        Theta = 270 + Theta
    elif delta_x > 0 and delta_y < 0:
        Theta = 360 + Theta


    Rho_H = int(Rho) >> 8
    Rho_L = int(Rho) % 256
    Theta_H = int(Theta) >> 8
    Theta_L = int(Theta) % 256
    Phi_H = int(ref_z) >> 8
    Phi_L = int(ref_z) % 256
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
    arr = lnwmodule.read(4)
    print('buf_h = ' + str(buf_h))
    print('arr = ' + str(arr))
    while True:
        if lnwmodule_done2(lnwmodule, buf_h) == True:
            break

def demo1_gotoPos(list_of_pos, offset_x, offset_y, offset_z):
    position_flag = 0
    position_count = 0
    image_count = 0
    record_flag = 0
    ref_x = 0
    ref_y = 0
    ref_z = list_of_pos[0][2]
    cc = 0
    while(True):
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
            arr_ack = lnwmodule.read(4)
            print('state 2')
            if lnwmodule_done2(lnwmodule, buf_g) == True:
                arr_ack = []
                position_count += 1
                position_flag = 0
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
                    sleep(0.5)
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
    return 1

def gripper_release(flag):
    if flag == 1:
        buf_gr = lnwmodule_grip(0)
        arr = []
        arr = lnwmodule.read(4)
        while(True):
            if lnwmodule_done2(lnwmodule, buf_gr) == True:
                break
    return 0


lnwmodule = serial.Serial()
sleep(2)
lnwmodule.baudrate = 19200
lnwmodule.port = 'COM12'
lnwmodule.rts = 0
lnwmodule.open()

demo1_home()

# record_trigger = 0
# list_of_position = [[220, 340, 100]]
# demo1_gotoPos(list_of_position)
# if __name__ == "__main__":
#     record_trigger = 1

# list_of_position = [[220, 360, 0]]
# demo1_gotoPos(list_of_position, 0, 0, 0)

# exec(open('loop_picture.py').read())
list_of_position = [[220, 360, 0],[210,390,0],[20,390,0],[20,150,0],[390,150,0],[390,390,0],[20,390,0],[220, 340, 0]]
demo1_gotoPos(list_of_position, 0, 0, 0)

all_find_template()

delete_rail()

point = [[75,0,0],[75,0,300]]
demo1_gotoPos(point, 0, 0, 0)

grip_flag = 0
grip_flag = gripper_grip(grip_flag)

grip_flag = gripper_release(grip_flag)

point2 = [[100,100,100]]
demo1_gotoPos(point2, 0, 0, 0)

point3 = all_main_demo2()
demo1_gotoPos(point3, 0, 0, 0)


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
