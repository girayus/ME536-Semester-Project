from __future__ import division
from objectColor import *
from objectMotion import *
from objectExpectancy import *
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2



###init system lists###
object_history = [[],[]]
node_name_list = ['t', 's', 'p', 'u', 'r', 'g', 'b', 'n'] #triangle, square, pentagon, unknown, red, green, blue, none
learned_props_vec = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) #binary list for properties whose expectations are learned, 1's-> not learned, 0's -> learned
node_mask_list = [] #mask of nodes
node_coef_list = [] #coefficients of nodes
prop_num = len(node_name_list) #number of properties

for i in range(prop_num): #initialize the the coefficient vectors and arrange the mask vectors
    mask_array = np.zeros(prop_num)
    coef_array = np.zeros(3)
    mask_array[i] = 1
    node_mask_list.append(mask_array)
    node_coef_list.append(coef_array)


scenario = np.array([[0, 0, 1],[0, 0, 0],[0, 1, 0],[0, 0, 0],[0, 0, 0],[0, 0, 1],[0, 0.5, 0.5],[0, 0, 0]])
pretrain_mode = False #take this as argument such as if args == '--p' pretrain mode is True, else False
if pretrain_mode: #if pretrained nodes are provided
    node_coef_list = []
    precoef_list = scenario #np.load('scenario.npy') #take path as argument and load the coefficient list
    for i in range(len(precoef_list)):
        node_coef_list.append(precoef_list[i]) #add node coefficient vectors to the node_coef_list
        if np.sum(precoef_list[i]) == 1.0:
            learned_props_vec[i] = 0


vidCapture = cv2.VideoCapture('/home/girayus/Desktop/python_scripts/project/video_animate/trial1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vidOut = cv2.VideoWriter('output.mp4', fourcc , 20.0, (640,480))

while (vidCapture.isOpened()):
    _, img = vidCapture.read()
    if img is None:
        break
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
                x, y , w, h = cv2.boundingRect(approx)
                imgSendColor = img[y-5:y+h+5, x-5:x+w+5]
                color_index_list = objectColorCluster(imgSendColor)
                found_prop_vec = ['t'] + color_index_list
                object_history, object_motion_vec = findMotionVector(object_history, found_prop_vec, object_center_vec)
                if object_motion_vec[0] == -1: #if previous center data not enough, do nothing
                    pass 
                else: #if enough, send it to neural network
                    expectancy, node_coef_list, learned_props_vec, expected_motion_vec = expectancyNN(found_prop_vec, object_motion_vec, prop_num, learned_props_vec, node_coef_list, node_mask_list, node_name_list) #put expectancyNN function here.
                    cv2.putText(imgLast, str(found_prop_vec) + str(object_center_vec), (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    cv2.putText(imgLast, str(object_motion_vec)+ str(expectancy)+ str(expected_motion_vec), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                
                
            elif len(approx) == 4 :
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                object_center_vec = [cX,cY]
                print(object_center_vec)
                x, y , w, h = cv2.boundingRect(approx)
                imgSendColor = img[y-10:y+h+10, x-10:x+w+10]
                color_index_list = objectColorCluster(imgSendColor)
                found_prop_vec = ['s'] + color_index_list
                #print(found_prop_vec)
                object_history, object_motion_vec = findMotionVector(object_history, found_prop_vec, object_center_vec)
                if object_motion_vec[0] == -1: #if previous center data not enough, do nothing
                    pass 
                else: #if enough, send it to neural network
                    expectancy, node_coef_list, learned_props_vec, expected_motion_vec = expectancyNN(found_prop_vec, object_motion_vec, prop_num, learned_props_vec, node_coef_list, node_mask_list, node_name_list) #put expectancyNN function here.
                    cv2.putText(imgLast, str(found_prop_vec) + str(object_center_vec), (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    cv2.putText(imgLast, str(object_motion_vec)+ str(expectancy)+ str(expected_motion_vec), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                

            elif len(approx) == 5:
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                object_center_vec = [cX,cY]
                #print(object_center_vec)
                x, y , w, h = cv2.boundingRect(approx)
                imgSendColor = img[y-10:y+h+10, x-10:x+w+10]
                color_index_list = objectColorCluster(imgSendColor)
                found_prop_vec = ['p'] + color_index_list
                object_history, object_motion_vec = findMotionVector(object_history, found_prop_vec, object_center_vec)
                if object_motion_vec[0] == -1: #if previous center data not enough, do nothing
                    pass 
                else: #if enough, send it to neural network
                    expectancy, node_coef_list, learned_props_vec, expected_motion_vec = expectancyNN(found_prop_vec, object_motion_vec, prop_num, learned_props_vec, node_coef_list, node_mask_list, node_name_list) #put expectancyNN function here.
                    cv2.putText(imgLast, str(found_prop_vec) + str(object_center_vec), (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    cv2.putText(imgLast, str(object_motion_vec)+ str(expectancy)+ str(expected_motion_vec), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

            else:
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                object_center_vec = [cX,cY]
                #print(object_center_vec)
                #cv2.putText(imgLast, "unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                x, y , w, h = cv2.boundingRect(approx)
                imgSendColor = img[y-10:y+h+10, x-10:x+w+10]
                color_index_list = objectColorCluster(imgSendColor)
                found_prop_vec = ['u'] + color_index_list
                #print(found_prop_vec)
                object_history, object_motion_vec = findMotionVector(object_history, found_prop_vec, object_center_vec)
                if object_motion_vec[0] == -1: #if previous center data not enough, do nothing
                    pass 
                else: #if enough, send it to neural network
                    expectancy, node_coef_list, learned_props_vec, expected_motion_vec = expectancyNN(found_prop_vec, object_motion_vec, prop_num, learned_props_vec, node_coef_list, node_mask_list, node_name_list) #put expectancyNN function here.
                    cv2.putText(imgLast, str(found_prop_vec) + str(object_center_vec), (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    cv2.putText(imgLast, str(object_motion_vec)+ str(expectancy)+ str(expected_motion_vec), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    
    #if node_coef_list:
    np_node_coef_list = np.round(np.asarray(node_coef_list, dtype=np.float32),3)
    cv2.putText(imgLast, "[t, s, p, u]:" + str(np_node_coef_list[0])+str(np_node_coef_list[1])+str(np_node_coef_list[2])+str(np_node_coef_list[3])   , (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))           
    cv2.putText(imgLast, "[r, g, b, n]:" + str(np_node_coef_list[4])+str(np_node_coef_list[5])+str(np_node_coef_list[6])+str(np_node_coef_list[7])   , (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))           
    cv2.imshow('img', img)
    cv2.imshow('shapes', imgLast)
    vidOut.write(imgLast)
    k = cv2.waitKey(5) & 0xFF
    if k == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed
    if k == 27:
        break
vidOut.release()
vidCapture.release()
cv2.destroyAllWindows()
