import numpy as np
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve



def RMSE(set1, set2):
    '''
    Finds the RMSE value
    '''
    return np.sqrt(((set1-set2)**2).mean())
    

def Question3(x_train, x_valid, x_test, y_train, y_valid, y_test):
    # Initialize theta list and grid list
    theta = [0.05, 0.1, 0.5, 1, 2]
    grid = [0.001, 0.01, 0.1, 1]

    # Merge the validation and training set
    x_full = np.vstack([x_valid, x_train])

    y_full = np.vstack([y_valid, y_train])

    # store error for each theta-regularization pair
    error = np.zeros((len(grid), len(theta)))

    # test each combination of theta and lambda
    for bO, O in enumerate(theta): # get it cause O kinda looks like a theta?
        for blam, lam in enumerate(grid):
            # Find a prediction for the y value and get RMSE for that prediction
            X = gaussianrfb(x_train, x_train, O) + lam*np.identity(x_train.shape[0])
            a = solver(X, y_train)
            pred = gaussianrfb(x_valid, x_train, O).dot(a)
            error[blam, bO] = RMSE(pred, y_valid)
            
    # get combination with minimum error
    blamtrue, bOtrue = np.unravel_index(np.argmin(error), shape=error.shape)
    O, lam = theta[bOtrue], grid[blamtrue]

    # Get a predicted testing value
    X = gaussianrfb(x_full, x_full, O) + lam*np.identity(x_full.shape[0])
    a = solver(X, y_full)

    pred = gaussianrfb(x_test, x_full, O).dot(a)
    test = RMSE(y_test, pred)
    
    return O, lam, error[blamtrue, bOtrue], test

def gaussianrfb(x0, x1, theta = 1.):
    '''
    Creates a guassian kernel
    '''
    x0 = np.expand_dims(x0, axis = 1)
    x1 = x1[np.newaxis, ...]
    return np.exp(-np.sum(np.square(x0 - x1)/ theta, axis = 2, keepdims = False))

def solver(x_train, y_train):
    '''
    Returns the cholesky factorization
    '''
    Fact = cho_factor(x_train)
    a = cho_solve(Fact, y_train)
    return a
    

        

                
                    
