import numpy as np
import time
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
from scipy import stats
import sklearn.neighbors
from Q1 import *

def classificationfinder(xlist, ylist, xtest, ytest, dist, k):
    '''
    Returns accuracy for the KNN algorotihm at a fixed value of K
    '''
    ypreds = []
    tree = sklearn.neighbors.KDTree(x_train, metric = dist)
    d, k_nb = tree.query(xtest, k = k)
    ypreds = stats.mode(y_train[k_nb], axis=1)[0]
    ypreds = np.array(ypreds)
    result = (ypreds.flatten() == ytest).sum()
    result = result /len(y_test)
    return result

def kfinder(xlist, ylist, xtest, ytest, dist):
    '''
    Basically the same as Kfinder for Q1 only for classification. Notice how
    this is much neater... it is because I am learning.
    '''
    klist =  list(range(1, 30))
    minlist = []
    for k in klist:
            print(k)
            minlist.append([classificationfinder(xlist, ylist, xtest, ytest, dist, k), k])

    return max(minlist)


