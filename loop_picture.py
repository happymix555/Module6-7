# import the opencv library
import cv2
import os
import glob
import time
# import setting
import test_variable
# from PC2PIC import *
#remove all image from folder
path = 'rail_image/rail_image_loop'
files = glob.glob(path + '/*')
for f in files:
    os.remove(f)

# define a video capture object
# vid = cv2.VideoCapture(1)
vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
vid.set(3, 1280) # set the resolution
vid.set(4, 720)
image_count = 0
flag = 0
# test_variable.record_trigger
while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('c'):
        flag = 1
        # name_and_path = 'rail_image/rail_image_loop/' + str(image_count) + '.jpg'
        # cv2.imwrite(name_and_path, frame)
        # image_count += 1
    # elif key & 0xFF == ord('q'):
    #     break
    # if setting.record_trigger == 1:
    #     name_and_path = 'rail_image/rail_image_loop/' + str(image_count) + '.jpg'
    #     cv2.imwrite(name_and_path, frame)
    #     image_count += 1
    # if key == ord('c'):
    #     flag = 1
    elif key & 0xFF == ord('q'):
        break
    if flag == 1:
        name_and_path = 'rail_image/rail_image_loop/' + str(image_count) + '.jpg'
        cv2.imwrite(name_and_path, frame)
        image_count += 1
    # if loop_count == 0:
    #     point = list_of_point(image_count)
    #     ref_x, ref_y, ref_z = lnwmodule_go2pos(point, ref_x, ref_y, ref_z, offset_x, offset_y)
    #     timer1 = time.perf_counter()
    #     while(lnwmodule.in_waiting):
    #         if time.perf_counter() - timer1 > 3:
    #             #reset PIC
    #             loop_count = 0
    #             reset_flag = 1
    #             break
    #         else:
    #             None
    #     if reset_flag == 1:
    #         None\
    #     else:
    #
    #         if acknoeledge:
    #             loop_count += 1
    #             while(lnwmodule_done() != 1):
    #                 None
    #             name_and_path = 'rail_image/rail_image_loop/' + str(image_count) + '.jpg'
    #             cv2.imwrite(name_and_path, frame)
    #             image_count += 1
    #             break
    #         elif time.perf_counter() - timer1 > 3:
    #             #reset PIC
    #             loop_count = 0
    #             break
    # if image_count = len(list_of_point) - 1:
    #     break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
# os.system('python main_demo1-1.py')
# def loop_picture():
#     path = 'rail_image/rail_image_loop'
#     files = glob.glob(path + '/*')
#     for f in files:
#         os.remove(f)
#     vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
#     vid.set(3, 1280) # set the resolution
#     vid.set(4, 720)
#     image_count = 0
#     flag = 0
#     while(True):
#         ret, frame = vid.read()
#
#     # Display the resulting frame
#         cv2.imshow('frame', frame)
#         key = cv2.waitKey(1)
#         if setting.record_trigger == 1:
#             name_and_path = 'rail_image/rail_image_loop/' + str(image_count) + '.jpg'
#             cv2.imwrite(name_and_path, frame)
#             image_count += 1
#         # if key == ord('c'):
#         #     flag = 1
#         elif key & 0xFF == ord('q'):
#             break
#     vid.release()
#     # Destroy all the windows
#     cv2.destroyAllWindows()
