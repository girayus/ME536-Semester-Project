import numpy as np

def motionSVD(Mn,threshSigma = 2.5): #threshold value is related to the magnitude of the noise and magnitude of the shifted center coordinates (speed of the object)
    Mn = Mn - Mn[0] #shift the data matrix towards the origin
    motionVecSVD = np.array([0,0,0]) #initialize the motion vector [0D, 1D, 2D]
    try:
        sigmaSVD = np.linalg.svd(Mn, full_matrices=False, compute_uv=False) #find sigma vector using SVD
        #print(sigmaSVD)
        rankSVD = 0 #initialize rank of the denoised matrix
        for i in range(2): #for sigma values of the SVD
            if sigmaSVD[i] > threshSigma: #if sigma value larger than the threshold
                rankSVD = i+1 #increase the rank of the matrix
    except: #if any error occurs
        rankSVD = 0 #set rank as 0
    motionVecSVD[rankSVD] = 1 #set the proper element of the motion vector as 1 according to the found denoised matrix rank
    return motionVecSVD

def findMotionVector(objectHistory, foundPropVec, objectCenterVec):
    objectMotionVec = np.array([-1,0,0]) #create the motion vector array which rpresents the dimension of the motion, -1 for not identifieed motion type
    if foundPropVec in objectHistory[0]: #if object property is in the previously found object list
        history_index = objectHistory[0].index(foundPropVec) #get the index of the object properties
        objectHistory[1][history_index].append(objectCenterVec) #add its center position to the corresponding center list
        
        if len(objectHistory[1][history_index]) > 10: #if there are 10 center positions, go and calculate the dimension of the motion
            objectMotionVec = motionSVD(np.asarray(objectHistory[1][history_index])) #calculate the dimension of the motion using previous 3 center positions
            #print(objectHistory)
            #print(objectMotionVec)
            del objectHistory[1][history_index][0] #delete the most previous center coordinates since 3 points is enough for determining the dimension of the motion
        else: #until collect 3 center points do nothing
            pass
        
    else: #if not in the previously found object list, add it to the list
        objectHistory[0].append(foundPropVec) #add its properties
        objectHistory[1].append([objectCenterVec]) #add its center position

    return objectHistory, objectMotionVec
