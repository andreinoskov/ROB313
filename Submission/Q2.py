import numpy as np
import time
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
import sklearn.neighbors
from Q1 import *


def timetest(k, xtrain, ytrain, xvalid, yvalid, data, dist, dimension):
    # Take the start and final time for kfinder
    startbrute = time.time()
    kfinder(xvalid, yvalid, dist)
    finbrute = time.time()

    # Take the start and final time for KDTree
    startfine = time.time()
    tree = sklearn.neighbors.KDTree(xtrain)
    d, knb = tree.query(xvalid, k =k) 
    ypreds = []
    ypreds = np.sum(ytrain[knb], axis = 1)/k 
    finfine = time.time()

    # return time for the KD tree and brute force algorithms to run.
    return finbrute - startbrute,finfine - startfine


