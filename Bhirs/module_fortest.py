import numpy as np
import cv2
from numpy import ones,vstack
from numpy.linalg import lstsq
import math
import skimage.graph
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize,thin
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
%matplotlib qt
# cv2.namedWindow("gray")
def mouseclick_world(event,xw,yw,flags,pram):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("color",imgray[xw,yw])
cv2.setMouseCallback("gray",mouseclick_world)
source_file='test_field.JPG'
field = cv2.imread(source_file)
plt.imshow(field)
#
# ##หามุมุ
# img = cv2.imread(source_file)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)
#
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
# print(img[0][0])
# for i in range(0,len(img)-1):
#     a = np.where(dst>0.01*dst.max())
# print("a",a)
# for i in range(0,len(a)):
#     res = []
#     # printing original list
#     print("The original list is : " + str(a[i]))
#     # Identical Consecutive Grouping in list
#     # using groupby() + list comprehension
#     res = [list(y) for x, y in groupby(a[i])]
#     # printing result
#     print("List after grouping is : " + str(res))
# cv2.imshow('dst',img)


image = cv2.imread(source_file)
imgray = cv2.imread(source_file,0)
# cv2.imshow('gray', imgray)
gaussian = cv2.GaussianBlur(imgray, (7, 7), 0)
detection_symbol=cv2.imread(source_file)
detection_path=cv2.imread(source_file)
#cv2.imshow('Input', imgray)
# cv2.waitKey(0)
edged = cv2.Canny(gaussian, 40, 100)
kernel = np.ones((3,3))
# cv2.imshow('edged', edged)
# closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('closing', closing)
# cv2.waitKey(0)
kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(edged,kernel,iterations = 1)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(dilation, contours, 3, (0,255,0), 3)
# cv2.imshow('Canny Edges After Contouring', dilation)
#cv2.waitKey(0)
print("Number of Contours found = " + str(len(contours)))
#print(hierarchy[0]) #[Next contour in same hierarchy is contour,previous contours,Child, parent]
#detection_symbollllllllll
print("check symbol")
symbol = []
centorid=[]
#cv2.drawContours(detection_symbol, contours, -1, (0, 255, 0), 3)
for i in range(1,len(hierarchy[0])):
    x_y = []
    if hierarchy[0][i][2] >= 0 and hierarchy[0][i][0] == -1 and hierarchy[0][i][1] == -1:
        print(i,hierarchy[0][i])
        cv2.drawContours(detection_symbol, contours, i-1, (0, 255, 0), 3)
        symbol.append(contours[i-1])
        # calculate moments for each contour
        M = cv2.moments(contours[i])
        # calculate x,y coordinate of center0
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(detection_symbol, (cX, cY), 5, (0, 255, 0), -1)
        #cv2.putText(detection_symbol, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        x_y.append(cX)
        x_y.append(cY)
        centorid.append(x_y)
        print((cX)*400/image.shape[1],"cm", (cY)*400/image.shape[0],"cm")
# cv2.imshow('dafk',detection_symbol)
print(symbol[0][0][0][0])
for i in symbol:
    for j in range(0,len(i)-1):
        cv2.circle(image, (i[j][0][0], i[j][0][1]), 1, (125, 0, 255), -1)

path_data=[]
#detection_pathhhhhhhhhhh
print("check path")
# shape=[image.shape[0],image.shape[1]]
new_path = np.zeros( (704,703) ) # create a single channel  black image
for i in range(1,len(hierarchy[0])):
    if hierarchy[0][i][0] !=-1 and hierarchy[0][i][3] == 0 and hierarchy[0][i][2] == -1:
        print(i,hierarchy[0][i])
        cv2.drawContours(imgray, contours, i, (0, 255, 0), 3)
        path_data.append(contours[i])
cv2.fillPoly(new_path, pts=[contours[0]], color=(255, 255, 255))
new_path_gaussian = cv2.GaussianBlur(new_path, (7, 7), 0)
kernel = np.ones((9,9),np.uint8)
erosion = cv2.erode(new_path_gaussian,kernel,iterations = 1)
# cv2.imshow('path', erosion)

# skeletonnnnnnnnnnnnnnnnnnnnnnnnnn
print("x , y , number point contour",centorid[0][0],centorid[0][1],len(path_data))
(thresh, path_new) = cv2.threshold(erosion, 0, 1, cv2.THRESH_BINARY)
# cv2.imshow("hsthhs",path_new)
# print('skeletonize(path_new)',skeletonize(path_new))
#skeleton = thin(path_new)
skeleton= skeletonize(path_new, method='lee')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(path_new, cmap=plt.cm.gray)
ax[0].set_title('original')
ax[0].axis('off')
ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].set_title('skeletonize (Lee 94)')
ax[1].axis('off')
fig.tight_layout()
plt.show()
point_path=[]
point_path.append(np.where(skeleton != 0))
print('fgskefg')
skeleton=skeleton.astype(np.uint8)
wtf = np.zeros( (704,703) )
contoursa, hierarchya = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(wtf, contoursa, 0, (255,255,255), 3)
print("Number of Contours found = " + str(len(contoursa)))
all_point_path=contoursa[0]

cv2.circle(wtf,(all_point_path[0][0][0],all_point_path[0][0][1]), 10, (255,255,255), -1)

### give start (y1,x1) and end (y2,x2) and the binary maze image as input
# def shortest_path(start,end,binary):
#     costs=np.where(binary,1,1000)
#     path, cost = skimage.graph.route_through_array(costs, start=start, end=end, fully_connected=True)
#     return path,cost
# ret,thresh2 = cv2.threshold(skeleton_lee,127,255,cv2.THRESH_BINARY_INV)
# real_path = shortest_path((centorid[0][0],centorid[0][1]),(centorid[1][0],centorid[1][1]),thresh2)
# print(real_path)
# test = np.zeros( (704,703) )
# for i in real_path[0]:
#     cv2.circle(test, (i[0], i[1]), 10, (255, 255, 255), -1)
#     print(i)
# cv2.imshow("sDGbzdf",test)
# #find start point
# merge_point=[]
# for i in range(0,len(point_path[0][0])-1):
#     merge_point

for i in range(0,len(all_point_path)-1):
    # print(all_point_path[i][0])
    cv2.circle(image, (all_point_path[i][0][0], all_point_path[i][0][1]), 1, (255, 0, 255), -1)

print("check nearest")
point_ref=[centorid[0][0],centorid[0][1]]
# print(len(symbol))
sort_path=[]
count_point=0
print(all_point_path[0][0][1],contoursa[0][1])
# for i in symbol[0]:
#     if (i[0][1] == all_point_path[0][0][1]):
#         print(i[0],i[0][1],all_point_path[0][0][1])
#         count_point+=1
# print ('count_point = ',count_point)

intersection_point=[]
for k in range(0,len(all_point_path-1)):
    for i in symbol:
        for j in range(0,len(i)-1):
            a=[]
            if all_point_path[k][0][0]==i[j][0][0] and all_point_path[k][0][1]==i[j][0][1] and [i[j][0][0],i[j][0][1]] not in intersection_point :
                cv2.circle(image, (i[j][0][0], i[j][0][1]), 1, (5, 200,5), -1)
                a=[i[j][0][0], i[j][0][1]]
                intersection_point.append(a)
print(intersection_point)
cv2.circle(image, (616, 143), 5, (5, 0,5), -1)
# cv2.imshow("result",image)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x =[]
y =[]
z =[]
for i in range(0,len(all_point_path-1)):
    x.append(all_point_path[i][0][0])
    y.append(all_point_path[i][0][1])
    z.append(255 - imgray[all_point_path[i][0][1], all_point_path[i][0][0]])
    print(all_point_path[i][0][0],all_point_path[i][0][1],imgray[all_point_path[i][0][0], all_point_path[i][0][1]])

print(len(x),len(y),len(z))
ax.scatter(x, y, z, c='b', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
# cv2.waitKey(0)
# x = ['p','y','t','h','o','n']
# print(x.index('o'))



# closer=[20,20,50] #x,y,
# for i in range(0,len(all_point_path)-1):
#     x_distance = all_point_path[i][0][0]-point_ref[0]
#     y_distance = all_point_path[i][0][1]-point_ref[1]
#     distance = math.sqrt((x_distance**2)+(y_distance**2))
#     if closer[2]>distance:
#         closer[0] = all_point_path[i][0][0]
#         closer[1] = all_point_path[i][0][1]
#         closer[2] = distance
#         print((all_point_path[i][0]),x_distance,y_distance,distance)
# cv2.circle(image, (closer[0], closer[1]),5, (0, 255, 255), -1)
# print("centorid",point_ref[0])
# print("closer",closer)
