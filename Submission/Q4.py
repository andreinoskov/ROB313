import numpy as np
import time
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
import sklearn.neighbors
from Q1 import *

def linregreg(xlist, ylist, x_test, y_test):
    '''
    Returns the root mean squared error of a linear regression algorithm.
    '''
    Xmatrix = np.ones((len(xlist), len(xlist[0]) + 1))
    Xmatrix[:, 1:] = xlist

    U, S, V = np.linalg.svd(Xmatrix)

    sig = np.diag(S)
    zero = np.zeros([len(xlist) - len(S), len(S)])
    invertsigm = np.linalg.pinv(np.vstack([sig, zero]))

    weight = np.dot(V.T, np.dot(invertsigm, np.dot(U.T, ylist)))

    testmatrix = np.ones((len(x_test), len(x_test[0]) + 1))
    testmatrix[:, 1:] = x_test
    guess = np.dot(testmatrix, weight)

    result = rootmeansquared(y_test, guess)
    return result
    
    
def linregclass(xlist, ylist, xtest, y_test):
        '''
    Returns the root mean squared error of a linear classification algorithm.
    '''
    Xmatrix = np.ones((len(xlist), len(xlist[0]) + 1))
    Xmatrix[:, 1:] = xlist

    U, S, V = np.linalg.svd(Xmatrix)

    sig = np.diag(S)
    zero = np.zeros([len(xlist) - len(S), len(S)])
    invertsigm = np.linalg.pinv(np.vstack([sig, zero]))

    weight = np.dot(V.T, np.dot(invertsigm, np.dot(U.T, ylist)))

    testmatrix = np.ones((len(x_test), len(x_test[0]) + 1))
    testmatrix[:, 1:] = x_test
    guess = np.argmax(np.dot(testmatrix, weight), axis = 1)
    ymax = np.argmax(y_test, axis =1)

    result = (guess == ymax).sum()
    result = result / len(y_test)
    return result


