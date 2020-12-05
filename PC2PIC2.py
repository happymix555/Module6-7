import serial
import math
check = True
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
    package = [255,1,0,0,0,0,0,0,0,0,0]
    print(package)
    package[-1] = sum(package)%256
    lnwmodule.write(package)


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

def lnwmodule_go2pos(point, ref_x, ref_y, ref_z, offset_x, offset_y):
    package = [255,2,0,0,0,0,0,0,0,0]
    real_point = image_to_world(point)
    current_x = real_point[0]
    current_y = real_point[1]
    current_z = real_point[2]
    delta_x = current_x - ref_x
    delta_y = current_y - ref_y
    delta_z = current_z - ref_z
    Rho = sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
    Theta = math.atan(delta_y / delta_x)
    Theta = int(Theta * 180 / math.pi)
    Phi = math.acos(delta_z / Rho)
    if delta_x < 0 and delta_y >= 0:
        Theta = 180 + Theta
    elif delta_x < 0 and delta_y < 0:
        Theta = 270 + Theta
    elif delta_x >= 0 and delta_y < 0:
        Theta = 360 + Theta

    if delta_z >= 0:
        Phi = 90 - Phi

    Rho_H = int(Rho >> 8)
    Rho_L = int(Rho % 256 )
    Theta_H = int(Theta >> 8)
    Theta_L = int(Theta % 256)
    Phi_H = int(Phi >> 8)
    Phi_L = int(Phi % 256)
    package[2] = Rho_H
    package[3] = Rho_L
    package[4] = Theta_H
    package[5] = Theta_L
    package[6] = Phi_H
    package[7] = Phi_L
    package[-1] = sum(package)%256
    lnwmodule.write(package)
    return current_x ,current_y, current_z, package

# def resetPic():
#     lnwmodule.rts = 1
#     lnwmodule.rts = 0
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
    package = [255,3,0,0,0,0,0,0,0,0]
    package[8] = grip
    package[-1] = sum(package)%256
    print('package = ' + str(package))
    lnwmodule.write(package)

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
