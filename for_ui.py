from prepare_rail_2 import *
from PC2PIC2 import *
from find_template import *

import cv2
import os
import glob
import time

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

lnwmodule = serial.Serial()
lnwmodule.baudrate = 19200
lnwmodule.port = 'COM12'
lnwmodule.rts = 0
lnwmodule.open()

state = 0

    if state == 0: #capture template
        path = 'raw_template'
        files = glob.glob(path + '/*')
        try:
            for f in files:
                os.remove(f)
        except:
            pass
        vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        vid.set(3, 1280) # set the resolution
        vid.set(4, 720)
        while(True):

            # Capture the video frame
            # by frame
            ret, frame = vid.read()

            # Display the resulting frame
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            # if key == ord('c'):
            #     name_and_path = 'rail_image/rail_image_loop/' + str(image_count) + '.jpg'
            #     cv2.imwrite(name_and_path, frame)
            #     image_count += 1
            # elif key & 0xFF == ord('q'):
            #     break
            if key == ord('c'):
                name_and_path = 'raw_template/' + str(0) + '.jpg'
                cv2.imwrite(name_and_path, frame)
            elif key & 0xFF == ord('q'):
                break
        raw_template = cv2.imread('raw_template/0.jpg')
        raw_template = resize_percent(raw_template, 50)
        cv2.imshow('raw template', raw_template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # big_white = detect_big_white(raw_template)
        # cv2.imshow('big white', big_white)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        raw_template_gray = cv2.cvtColor(raw_template, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = find_contours(raw_template_gray, 40, 100, 3, 1, 'tree')
        first_contour_img = draw_contours(blank_image_with_same_size(raw_template_gray), contours, 15)
        cv2.imshow('first contour', first_contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # contours, hierarchy = find_contours(first_contour_img, 40, 100, 3, 1, 'tree')
        # large = largest_contour(contours)
        # first_contour_img = draw_contours(blank_image_with_same_size(raw_template_gray), large, 1)
        # cv2.imshow('first contour', first_contour_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        contours2, hierarchy2 = find_contours(first_contour_img, 40, 100, 3, 1, 'tree')
        second_contour_img = draw_contours(blank_image_with_same_size(raw_template_gray), contours2, 3)
        cv2.imshow('second contour', second_contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        pre_square_contour, pre_square_hie = find_contours(second_contour_img, 40, 100, 3, 1, 'external')
        pre_square_contour_img = draw_contours(blank_image_with_same_size(raw_template_gray), pre_square_contour, 3)
        cv2.imshow('pre square contour', pre_square_contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        template_contour = find_template_contours(pre_square_contour, raw_template_gray)
        template_contour_img = draw_contours(blank_image_with_same_size(raw_template_gray), template_contour, 1)
        cv2.imshow('template contour img', template_contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        prepared_template = find_roi(template_contour[0], raw_template)
        cv2.imshow('prepared template', prepared_template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('prepared_template/0.jpg', prepared_template)

        state = 1
    if state == 1: #take multiple field image
        lnwmodule_sethome()
        while True:
            if lnwmodule_acknowledge() == True:
                break

        vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        vid.set(3, 1280) # set the resolution
        vid.set(4, 720)

        list_of_position = [[200, 200, 0]]
        position_flag = 0
        position_count = 0
        image_count = 0
        record_flag = 0
        x_ref = 0
        y_ref = 0
        z_ref = 0
        while(True):
            if position_flag == 0:
                current_x, current_y, current_z = lnwmodule_go2pos(list_of_position[position_count], ref_x, ref_y, ref_z, 0, 0)
                ref_x = current_x
                ref_y = current_y
                ref_z = current_z
                position_flag = 1
            if position_flag == 1:
                if lnwmodule_acknowledge() == True:
                    position_count += 1
                    if image_count == 0:
                        record_flag = 1
                    if position_count == len(list_of_position):
                        vid.release()
                        cv2.destroyAllWindows()
                        # state = 2
                        break
                    else:
                        position_count += 1
            if record_flag == 1:
                ret, frame = vid.read()
                cv2.imshow('frame', frame)
                name_and_path = 'rail_image/rail_image_loop/' + str(image_count) + '.jpg'
                cv2.imwrite(name_and_path, frame)
                image_count += 1
            else:
                pass

    if state == 2: #prepare rail image
        delete_rail()

        state = 3
    if state == 3:
        checkpoint_center = []
        checkpoint_roi = []

        template_image = cv2.imread('prepared_template/prepared_template.jpg')
        field_image = cv2.imread('prepared_field/ready_field.jpg')
        cv2.imshow('original field', field_image)
        # field_image = cv2.fastNlMeansDenoisingColored(field_image,None, 10, 10,7,21)
        # cv2.imshow('fastNimean field', field_image)
        # field_image = cv2.GaussianBlur(field_image,(3,3),0)
        cv2.imshow('gaussian field', field_image)
        kernel_sharpening = np.array([[-1,-1,-1],
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
        sharpened_field_image = cv2.filter2D(field_image, -1, kernel_sharpening)
        cv2.imshow('sharpened field', sharpened_field_image)
        field_image_blur = cv2.medianBlur(field_image,31)
        field_gray_blur = cv2.cvtColor(field_image_blur, cv2.COLOR_BGR2GRAY)
        # field_image = cv2.GaussianBlur(field_image,(51,51),0)
        cv2.imshow('fastNimean + medianBlur field', field_image_blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        def nothing(x):
            pass

        cv2.namedWindow('first contour')
        cv2.createTrackbar('canny LOW','first contour',0,255,nothing)
        cv2.createTrackbar('canny HIGH','first contour',100,255,nothing)
        cv2.createTrackbar('Gaussian kernel size','first contour',1,21,nothing)
        cv2.createTrackbar('Thickness','first contour',1,30,nothing)

        while(1):
            # cv2.imshow('image',img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
            canny_low = cv2.getTrackbarPos('canny LOW','first contour')
            canny_high = cv2.getTrackbarPos('canny HIGH','first contour')
            gs = cv2.getTrackbarPos('Gaussian kernel size','first contour')
            gs = ((gs+1) * 2) - 1
            thickness = cv2.getTrackbarPos('Thickness','first contour')
            field_gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = find_contours(field_gray, canny_low, canny_high, gs, gs, 'tree')
            first_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours, thickness)
            cv2.imshow('first contour', first_contour_img)


        # field_gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
        # contours, hierarchy = find_contours(field_gray, 40, 100, 3, 1, 'tree')
        # first_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours, 10)
        # cv2.imshow('first contour', first_contour_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        contours2, hierarchy2 = find_contours(first_contour_img, 40, 100, 3, 1, 'tree')
        second_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours2, 3)
        cv2.imshow('second contour', second_contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        contours3, hierarchy3 = find_contours(second_contour_img, 40, 100, 3, 1, 'external')
        third_contour_img = draw_contours(blank_image_with_same_size(field_gray), contours3, 1)
        cv2.imshow('second contour', third_contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        square_contours, square_areas = find_square_contours(contours3, field_image)
        square_contour_img = draw_contours(blank_image_with_same_size(field_gray), square_contours, 1)
        cv2.imshow('square contour', square_contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        for sc in square_contours:
            center = find_contour_center(sc)
            checkpoint_center.append(center)
            roi = find_roi(sc, field_image)
            checkpoint_roi.append(roi)
        c_field = field_image.copy()
        for p in checkpoint_center:
            c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(0, 0, 255), thickness=10)
        cv2.imshow('checkpoint center', c_field)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for r in checkpoint_roi:
            cv2.imshow('checkpoint ROI', r)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # template_image = cv2.imread('prepared_template/prepared_template.jpg')
        cv2.imshow('template', template_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        template_location = template_matching_with_roi(template_image, checkpoint_roi, checkpoint_center, 360)
        c_field = field_image.copy()
        c_field = cv2.circle(c_field, (template_location[0], template_location[1]), radius=0, color=(0, 0, 255), thickness=10)
        cv2.imshow('start point', c_field)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ending_location = find_endpoint(checkpoint_center, field_image)
        c_field = field_image.copy()
        c_field = cv2.circle(c_field, (ending_location[0], ending_location[1]), radius=0, color=(0, 0, 255), thickness=10)
        cv2.imshow('End point', c_field)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        pre_palette_contour, r = find_palette_by_checkpoint_area(contours3, square_areas, field_gray_blur)
        pre_palette_contour_img = draw_contours(blank_image_with_same_size(field_gray), pre_palette_contour, 3)
        cv2.imshow('pre palette contour', pre_palette_contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        palette_contour, palette_hie = find_contours(pre_palette_contour_img, 40, 100, 3, 1, 'external')
        palette_contour_img = draw_contours(blank_image_with_same_size(field_gray), palette_contour, 1)
        for cnt in palette_contour:
            palette_contour_img2 = cv2.drawContours(blank_image_with_same_size(field_gray), [cnt], 0, (255, 255, 255), 1)
            cv2.imshow('palette contour', palette_contour_img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        test = []
        all_about_palette = []
        for cnt in palette_contour:
            elipson = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, elipson, True)
            # cv2.drawContours(ppppp ,[approx], -1, (255, 255, 255), 3)
            filled_palette_image = fill_contour([approx], field_image)
            cv2.imshow('filled palette contour', filled_palette_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            skeleton_image = skeleton_with_erotion(filled_palette_image, 10, 1)
            cv2.imshow('skeletonized palette', skeleton_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            test.append(skeleton_image)
            skeleton_coordinate = skeleton_coordinate2(skeleton_image)
            c_field = field_image.copy()
            for sc in skeleton_coordinate:
                c_field[sc[1]][sc[0]] = [0, 0, 255]
            cv2.imshow('check skeleton co', c_field)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            opened_point = find_opened_end(skeleton_coordinate, skeleton_image)
            for op in opened_point:
                c_field = cv2.circle(c_field, (op[0], op[1]), radius=0, color=(255, 0, 0), thickness=10)
            cv2.imshow('check opened end point', c_field)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            all_in_one_palette = [skeleton_image, skeleton_coordinate, opened_point, cnt]
            all_about_palette.append(all_in_one_palette)

        all_connected_point = find_connected_point(checkpoint_center, all_about_palette, 180)
        for cp in all_connected_point:
            c_field = cv2.circle(c_field, (cp[0][0], cp[0][1]), radius=0, color=(255, 0, 255), thickness=10)
            c_field = cv2.circle(c_field, (cp[1][0], cp[1][1]), radius=0, color=(255, 0, 255), thickness=10)
        cv2.imshow('check connected point', c_field)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        full_path_image = draw_full_path(all_connected_point, all_about_palette, field_image)
        cv2.imshow('full skeleton path', full_path_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        denoised_gray_field = cv2.fastNlMeansDenoising(field_gray_blur, None, 10, 7, 21)
        cv2.imshow('test', denoised_gray_field)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        find_min_max(full_path_image, all_about_palette, denoised_gray_field)
        full_path_with_height = find_path_height(full_path_image, all_about_palette, denoised_gray_field)

        from mpl_toolkits import mplot3d
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        x = []
        y = []
        z = []
        for co in full_path_with_height:
            # for co in co_list:
            x.append(int(co[1]))
            y.append(int(co[0]))
            if co[2] != None:
                z.append(int(co[2]))
            else:
                z.append(200)
        ax.scatter3D(x, y, z, 'gray')
        plt.show()

        countours, hierarchy = cv2.findContours(full_path_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        trajec_point_image = blank_image_with_same_size(field_gray)

        for cnt in countours:
            peri = cv2.arcLength(cnt, True)
            traject_point = cv2.approxPolyDP(cnt, 0.0015 * peri, True)
            trajec_point_image = cv2.drawContours(trajec_point_image, traject_point, -1, (255, 255, 255), 10)

        cv2.imshow("trajec point image", trajec_point_image)
        cv2.imshow('full skeleton path', full_path_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        stp, traject2 = find_real_traject_point(template_location, ending_location, traject_point, full_path_image)
        c_field = field_image.copy()
        for p in stp:
            c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(255, 0, 255), thickness=10)
            cv2.imshow('pp', c_field)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        short_path = shortest_pathh(template_location, ending_location, full_path_image)
        c_field = field_image.copy()
        for p in short_path:
            c_field[p[1]][p[0]] = [255, 0, 255]
        cv2.imshow('mmm', c_field)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        short_path_height = short_path_with_height(short_path, full_path_with_height)
        traject_point_with_height = []
        for p in stp:
            for p2 in short_path_height:
                if p[0] == p2[0] and p[1] == p2[1]:
                    if traject_point_with_height != []:
                        if distance_between_point([traject_point_with_height[-1][0], traject_point_with_height[-1][1]], p) > 10:
                            traject_point_with_height.append([p[0], p[1], p2[2]])
                    else:
                            traject_point_with_height.append([p[0], p[1], p2[2]])

        traject_point_with_height
        c_field = field_image.copy()
        for p in traject_point_with_height:
            c_field = cv2.circle(c_field, (p[0], p[1]), radius=0, color=(0, 255, 255), thickness=10)
            cv2.imshow('traject point with height', c_field)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        short_path_height
        from mpl_toolkits import mplot3d
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        x = []
        y = []
        z = []
        for co in short_path_height:
            # for co in co_list:
            x.append(int(co[1]))
            y.append(int(co[0]))
            z.append(co[2])
        ax.scatter3D(x, y, z, 'gray')
        plt.show()

        world_traject_point = []
        for i in traject_point_with_height:
            world_traject_point.append(image_to_world(i, 0, 0, 0, 600, 600, 400, 400))

        state = 4
    if state == 4:
        point = [[200,200,0],[0,0,200]]
        grip_num = 0
        ref_x4 = 0
        ref_y4 = 0
        ref_z4 = 0
        for i in range(len(point)):
            ref_x4 ,ref_y4 ,ref_z4 ,package4 = lnwmodule_go2pos(point[i],ref_x4,ref_y4,ref_z4,0,0)
            while(True):
                if lnwmodule_acknowledge() == True:
                    break
            while(lnwmodule.in_waiting):
                pass
            if lnwmodule_done2(lnwmodule) == True:
                grip_num + 1
                pass
            else:
                print("err")
                break
        if grip_num == len(point):
            lnwmodule_grip(1) #open gripper
            while(True):
                if lnwmodule_acknowledge() == True:
                    break
            while(lnwmodule.in_waiting):
                pass
            if lnwmodule_done2(lnwmodule) == True:
                lnwmodule_sethome()
                while(True):
                    if lnwmodule_acknowledge() == True:
                        break
                while(lnwmodule.in_waiting):
                    pass
                if lnwmodule_done2(lnwmodule) == True:
                    state = 5




    if state == 5:

        ref_x = 0
        ref_y = 0
        ref_z = 0
        buf = []
        num = 0
        for i in range(len(world_traject_point)):
            ref_x,ref_y,ref_z,package = lnwmodule_go2pos(world_traject_point[i],ref_x,ref_y,ref_z,0,0)
            if(lnwmodule_acknowledge() == True):
                while(lnwmodule.in_waiting):
                    pass
                buf = lnwmodule.read(10)
                for i in range(len(buf)):
                    if(buf[i] == package[i]):
                        num + 1
                if(num == 10):
                    pass
                else:
                    print("err")
                    break
            else:
                print("error")
                break
            buf = []
            num = 0
