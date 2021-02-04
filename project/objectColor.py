import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

def calcDistance(x1,y1,a,b,c):
    d = abs((a*x1+b*y1+c)/(np.sqrt(a*a+b*b)))
    return d

def hsvColorFinder(hsvColors):
    colorIndexList = [] #color list to return
    #print(hsvColors)
    for hsvColor in hsvColors[0]: #for each found color values
        #print(hsvColor)

        if (hsvColor[1] <5) and (hsvColor[2] >150): #if it is white
            pass
        elif (hsvColor[2] < 5): #if it is black
            pass
        elif (165 <= hsvColor[0]) or (hsvColor[0] <5) : #if it is red
            colorIndexList.append('r')
        elif (45 <= hsvColor[0] <= 75): #if it is green
            colorIndexList.append('g')
        elif (90 <= hsvColor[0] <= 135): #if it is blue
            colorIndexList.append('b')
        else: #if another color (none)
            colorIndexList.append('n')
    return sorted(list(set(colorIndexList))) #return the found unique colors

def objectColorCluster(img):
    #img = cv2.imread("red_triangle")
    #cv2.imshow('img', img)
    #cv2.waitKey()
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img, (50,50), interpolation = cv2.INTER_AREA) #shrink the image for faster clustering
    #cv2.imshow('img1', img)
    #cv2.waitKey()
    img=img.reshape((img.shape[1]*img.shape[0],3)) #create a single vector of pixel color values for clustering

    #elbow method
    """
    md=[]
    K = range(1,7)
    for i in K:
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(img)
        o=kmeans.inertia_
        md.append(o)

    #plt.plot(list(np.arange(1,7)),md)
    #plt.show()


    a = md[0] - md[5]
    b = K[5] - K[0]
    c1 = K[0] * md[5]
    c2 = K[5] * md[0]
    c = c1- c2

    elbow_distances = []
    for i in range(6):
        elbow_distances.append(calcDistance(K[i],md[i], a, b, c))

    #plt.plot(K, elbow_distances)
    #plt.show()

    #print(elbow_distances.index(max(elbow_distances))+1)
    opt_k = elbow_distances.index(max(elbow_distances))+1
    
    #print(f"optimum k is {opt_k}")
    """
    
    opt_k=3 #for faster calculations maximum number of colors is selected as 3 (1 background + 2 object colors)
    kmeans=KMeans(n_clusters=opt_k) #give the number of clusters
    s=kmeans.fit(img) #cluster the color pixel array
    labels=kmeans.labels_ #label data points
    #print(labels)
    labels=list(labels) #list the labels
    centroid=kmeans.cluster_centers_ #find cluster centroids
    #print(centroid)
    
    percent=[]
    for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)
    
    colors=np.uint8([centroid]) #create a numpy array of found cluster centers (average color values found by kmeans)
    #print(colors)
    hsv_colors = cv2.cvtColor(colors,cv2.COLOR_BGR2HSV) #transform found colors into hsv
    #print(hsv_colors)
    colorindexList = hsvColorFinder(hsv_colors) #get the color names
    return colorindexList #return the unique colors of the object
    



