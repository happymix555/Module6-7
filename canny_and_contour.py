import cv2

def find_contours(gray_image, canny_l, canny_h, kernel_size0, kernel_size1, contour_type):
    # canny_without_blur = cv2.Canny(gray_image, canny_l, canny_h)
    img = cv2.GaussianBlur(gray_image, (kernel_size0, kernel_size1), 0)
    img_canny = cv2.Canny(img, canny_l, canny_h)
    if contour_type == 'external':
        contours, hierarchy = cv2.findContours(img_canny,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
    elif contour_type == 'tree':
        contours, hierarchy = cv2.findContours(img_canny,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def find_square_contours(contours, image_for_area, percent_epsilon, error_range):
    epsilon = int(percent_epsilon / 100)
    min_error = 1 - (error_range / 100)
    max_error = 1 + (error_range + 100)
    full_area = len(image_for_area) * len(image_for_area[0])
    checkpoint_area = int(full_area / 16)
    min_checkpoint_area = int(checkpoint_area / 20)
    square_contours = []
    area = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
		# a square will have an aspect ratio that is approximately
		# equal to one, otherwise, the shape is a rectangle
            if ar >= min_error and ar <= max_error:
                # area.append(cv2.contourArea(cnt))
                if min_checkpoint_area <= cv2.contourArea(cnt) <= checkpoint_area: #and cv2.contourArea(cnt) > (checkpoint_area / 3):
                    area.append(cv2.contourArea(cnt))
                    square_contours.append(cnt)
			# shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    return square_contours, area

def blank_image_with_same_size(img):
    return(np.zeros((img.shape[0], img.shape[1], 1), dtype = np.uint8))

def only_roi(contours, original_image):
    
