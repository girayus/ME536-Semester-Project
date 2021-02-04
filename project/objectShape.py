from __future__ import division
from objectColor import *
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("all_shapes_and_colors.png")
imgLast = img.copy()
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret , thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)



for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if cv2.contourArea(contour) > 400:
        if len(approx) == 3:
            
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            object_center_vec = [cX,cY]
            #print(object_center_vec)
            cv2.putText( imgLast, "triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0) )
            x, y , w, h = cv2.boundingRect(approx)
            imgSendColor = img[y-5:y+h+5, x-5:x+w+5]
            color_index_list = objectColorCluster(imgSendColor)
            found_prop_vec = ['t'] + color_index_list
            print(found_prop_vec)
            
            
        elif len(approx) == 4 :
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            object_center_vec = [cX,cY]
            #print(object_center_vec)
            cv2.putText(imgLast, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            x, y , w, h = cv2.boundingRect(approx)
            imgSendColor = img[y-10:y+h+10, x-10:x+w+10]
            color_index_list = objectColorCluster(imgSendColor)
            found_prop_vec = ['s'] + color_index_list
            print(found_prop_vec)

        elif len(approx) == 5 :
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            object_center_vec = [cX,cY]
            #print(object_center_vec)
            cv2.putText(imgLast, "pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            x, y , w, h = cv2.boundingRect(approx)
            imgSendColor = img[y-10:y+h+10, x-10:x+w+10]
            color_index_list = objectColorCluster(imgSendColor)
            found_prop_vec = ['p'] + color_index_list
            print(found_prop_vec)

        else:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            object_center_vec = [cX,cY]
            #print(object_center_vec)
            cv2.putText(imgLast, "unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            x, y , w, h = cv2.boundingRect(approx)
            imgSendColor = img[y-10:y+h+10, x-10:x+w+10]
            color_index_list = objectColorCluster(imgSendColor)
            found_prop_vec = ['u'] + color_index_list
            print(found_prop_vec)
cv2.imshow('img', img)
cv2.waitKey()
cv2.imshow('shapes', imgLast)
cv2.waitKey()