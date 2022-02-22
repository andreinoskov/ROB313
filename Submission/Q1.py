import numpy as np
import time
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset


def rootmeansquared(known, unknown):
    '''
    computes the root mean squared error
    '''
    return np.sqrt(np.average((known-unknown)**2))

def L1(x1, x2):
    '''
    computes the root manhattan distance
    '''
    return np.linalg.norm([x1-x2], ord = 1)

def L2(x1, x2):
    '''
    computes the root Eclidean error
    '''
    return np.linalg.norm([x1-x2], ord = 2)

def knearestneighbors(newpoint, xtest, ytest, k, distmetric):
    """
    This function takes a predifined value of k, a new point a test set
    and a string which specifies which distance metric to use and returns
    the predicted value for that point.
    """

    # if using Euler distance
    if distmetric == 'L2':
        # create a list of points by distance from test point
        listpos = []
        j = 0
        while j < len (xtest):
            listpos.append([xtest[j], ytest[j], eulerdist(newpoint, [xtest[j], ytest[j]])])
            j = j + 1
        # find the average value of K closest points in test set
        i = 0
        averageyvalue = 0
        
        while i < k:
            minimum = min(x[2] for x in listpos)
            minvalue = [x for x in listpos if (x[2] == minimum)]
            averageyvalue = averageyvalue + minvalue[0][1]
            i = i + 1
            listpos.remove(minvalue[0])
            
        averageyvalue = averageyvalue/k
        
    
    # if using Manhatten distance
    if distmetric == 'L1':
        # create a list of points by distance from test point
        listpos = []
        j = 0
        while j < len (xtest):
            listpos.append([xtest[j], ytest[j], mandist(newpoint, [xtest[j], ytest[j]])])
            j = j + 1
        # find the average value of K closest points in test set
        i = 0
        averageyvalue = 0
        
        while i < k:
            minimum = min(x[2] for x in listpos)
            minvalue = [x for x in listpos if (x[2] == minimum)]
            averageyvalue = averageyvalue + minvalue[0][1]
            i = i + 1
            listpos.remove(minvalue[0])

        averageyvalue = averageyvalue/k    
    return averageyvalue

def kfinder(xlist, ylist, dist):
    """
    This function takes a list of x-values and y-values and preforms a 5-cross v
    validation for 30 different values of K and returns a dictionary with the value
    of K and their RMSE.
    """
    loss = {}
    # randomize data
    np.random.seed(5)
    np.random.shuffle(xlist)
    np.random.seed(5)
    np.random.shuffle(ylist)

    
    length = len(xlist)//5

    # list of k values
    klist = list(range(0, min(30,length)))

    i = 0
    while i < 5:
        #split data into validation and testing sets
        xvalid = xlist[i * length: (i + 1) * length]
        xtrain = np.vstack([xlist[:i * length], xlist[(i + 1) * length :]])
        yvalid = ylist[i * length: (i + 1) * length]
        ytrain = np.vstack([ylist[:i * length], ylist[(i + 1) * length :]])
        
        i = i + 1

        j = 0
        y = {}
        while j < length:
            # create a list of closest points in training data set and take the average value computed by KNN
            lis = []
            l = 0
            while l < len(xvalid):
                lis.append([dist(xtrain[l], xvalid[l]), ytrain[l]])
                l = l + 1
            lis.sort(key = lambda x: x[0])
            for k in klist:
                kvalue = 0
                for elem in lis[:k+1]:
                    kvalue += elem[1]
                if k not in y:
                    y[k] = []

                y[k].append(kvalue/(k))
                
            j= j + 1
        
        for k in klist:
            # append root mean square loss of this set computed for 
            if k not in loss:
                loss[k] = [rootmeansquared(yvalid, y[k])]

        return loss


                 
            
    
