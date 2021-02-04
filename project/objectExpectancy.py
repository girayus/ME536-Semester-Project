import numpy as np

def LearnExpectation(unknownPropVec, motionVec, nodeMaskList, nodeCoefList, learnedPropsVec):
    learningSpeed = 0.005
    learnInputVec = np.array([unknownPropVec, motionVec],dtype=object) #ex: unknownPropVec =  np.array([1,0,0,1,0,0,0,0]), motionVec = np.array([1,2,3])
    for i in range(len(nodeMaskList)): #for total number of parameters: shapes + colors
        nodeCoefList[i] = nodeCoefList[i] + learningSpeed*(np.dot(np.dot(learnInputVec[0],nodeMaskList[i]),learnInputVec[1]))
        if round(np.sum(nodeCoefList[i]),5)==1: #if sum of probabilities 1 
            learnedPropsVec[i] = 0 #make it learned
    return nodeCoefList, learnedPropsVec

def FindExpectation(findPropVec, nodeMaskList, nodeCoefList):
    expectedMotionVec1 = np.zeros(3) #initially zero for summing all outputs into this vector as a final output
    expectedMotionVec2 = np.zeros(3) #initially zero for summing all outputs into this vector as a final output
    for i in range(int(len(nodeMaskList)/2)): #for total number of parameters: shapes + colors
        expectedMotionVec1 = expectedMotionVec1 + np.multiply(nodeCoefList[i],np.dot(findPropVec,nodeMaskList[i])) #sum all outputs from each node
    if np.sum(expectedMotionVec1) == 0:
        expectedMotionVec1 = np.zeros(3)
    else:
        expectedMotionVec1 = expectedMotionVec1 / (np.sum(expectedMotionVec1)) #scale the color output so that total probability will be 1
    
    for i in range(int(len(nodeMaskList)/2),len(nodeMaskList)): #for total number of parameters: shapes + colors
        expectedMotionVec2 = expectedMotionVec2 + np.multiply(nodeCoefList[i],np.dot(findPropVec,nodeMaskList[i])) #sum all outputs from each node
    if np.sum(expectedMotionVec2) == 0:
        expectedMotionVec2 = np.zeros(3)
    else:
        expectedMotionVec2 = expectedMotionVec2 / (np.sum(expectedMotionVec2))
    
    expectedMotionVec = expectedMotionVec1 + expectedMotionVec2  #sum the color and shape expectations
    
    expectedMotionVec = expectedMotionVec / (np.sum(expectedMotionVec)) #scale the final output so that total probability will be 1
    return expectedMotionVec

def TheExpectationIs(expectedMotionVec, objectMotionVec):
    ### check probabilities in the expectedMotionVec and transform it in binary vector(?)
    expectedMotionVecBin = np.where(expectedMotionVec > 0.3, 1.0, 0.0) #probability higher than 0.3 is assumed to be expected, binarize the expectancy vector
    expectedMotionVal = np.multiply(expectedMotionVecBin,objectMotionVec) #bitwise multiplication with expectation and observed motion vec
    if round(np.sum(expectedMotionVal),5) == 1.0: #if there is element 1 in this vector, than the motion is expected
        expectancy = 1 #1 represents expected result
    else: #otherwise not expected
        expectancy = 0 #0 represents unexpected result
    return expectancy

def UpdateExpectation(findPropVec, nodeCoefList, objectMotionVec):
    updatingSpeed = 0.001
    for i in range(len(nodeCoefList)):

        if findPropVec[i] == 1: #update the coefficients of the properties found in the object
            
            nodeCoefList[i] = nodeCoefList[i] + updatingSpeed * objectMotionVec
            
            if np.count_nonzero(nodeCoefList[i]) == 2:
                divide = 1
            else:
                divide = 2
            
            nodeCoefList[i] = nodeCoefList[i] - (updatingSpeed/divide) * np.where((objectMotionVec==0)|(objectMotionVec==1), 1-objectMotionVec, objectMotionVec)
            nodeCoefList[i] = np.clip(nodeCoefList[i],0.0,1.0) #limit the coefficient vector between 0 and 1
    return nodeCoefList

def expectancyNN(found_prop_vec, object_motion_vec, prop_num, learned_props_vec, node_coef_list, node_mask_list, node_name_list):
    
    object_prop_vec = np.zeros(prop_num) #property binary vector, 1's-> found, 0's not found on the object
    for found_prop in found_prop_vec: #for found properties
        object_prop_vec[node_name_list.index(found_prop)] = 1.0 #make the corresponding index 1

    #print(object_prop_vec)

    unknown_prop_vec = np.multiply(object_prop_vec,learned_props_vec) #unknown property binary vector, 1's-> unknown, 0's -> known

    #print(unknown_prop_vec)

    if not np.all(unknown_prop_vec==0): #if there is at least 1 unknown expectation for a property, train it by updating the coefficient vectors
        node_coef_list, learned_props_vec = LearnExpectation(unknown_prop_vec, object_motion_vec, node_mask_list, node_coef_list, learned_props_vec)
        #print(node_coef_list)

    #print(learned_props_vec)

    ### find prop vector for properties which are trained
    notlearned_props_vec = np.where((learned_props_vec==0)|(learned_props_vec==1), 1-learned_props_vec, learned_props_vec) #vector that is binary inverse of learned_props_vec

    #print(notlearned_props_vec)

    find_prop_vec = np.multiply(object_prop_vec, notlearned_props_vec) #binary vector of properties which are learned at total probability 1, ready to find expectation

    if np.sum(find_prop_vec) != 0: #if there are at least 1 learned property found in the object for expectation, find the expectation
        expected_motion_vec = np.round(FindExpectation(find_prop_vec, node_mask_list, node_coef_list),3)
        
        #print(expected_motion_vec)
        expectancy = TheExpectationIs(expected_motion_vec, object_motion_vec)
        node_coef_list = UpdateExpectation(find_prop_vec, node_coef_list, object_motion_vec) #update the expectancy coefficients
    else: #dont have any expectation
        expected_motion_vec = np.zeros(3) #return zero vector
        expectancy = -1

    return expectancy, node_coef_list, learned_props_vec, expected_motion_vec