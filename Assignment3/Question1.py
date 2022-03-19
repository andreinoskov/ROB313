import numpy as np
import math
import time
from matplotlib import pyplot as plt
from data_utils import load_dataset
import random

def RMSE(set1, set2):
    '''
    Finds the RMSE value
    '''
    return np.sqrt(((set1-set2)**2).mean())

    
def Fullbatch(x_train, x_test, y_train, y_test, rates, optrmse):
    '''
    Function that is designed to do gradient decent on the first thousand points of
    the pumadyn32nm data set (Full-batchedly)
    '''
    # Define a stoping limit for the RMSE. (Optrmse = the optimal computed in Assignment 1)
    stoplimit = optrmse*1.02

    # train and test matrix
    Train = np.ones((len(x_train), len(x_train[0]) + 1))
    Train[:, 1:] = x_train

    Test = np.ones((len(x_test), len(x_test[0]) + 1))
    Test[:, 1:] = x_test

    # Return Variables
    times = []
    optweight = []
    test_rmses = []
    loss = {}

    for rate in rates:
        # Set the weights to zero initially
        w = np.zeros((33, 1))

        loss[rate] = []
        timed_in = False

        # Gradient descent starts here! Yay!
        total_time = 0
        t1 = time.time()
        for i in range(500):
            predictions = np.dot(Train, w)

            # Compute Full-Batch Gradient
            grad_L = np.insert(np.zeros(np.shape(x_train[0])), 0, 0)
            for j in range(len(predictions)):
                grad_L += (predictions[j]-y_train[j])*np.insert(x_train[j], 0, 1)
                
            grad_L = 2*grad_L/len(predictions)
            grad_L = grad_L.reshape((33, 1))

            # Update weights
            w = np.add(w, -rate*grad_L)

            # End Gradient Descent here! Yaya
            t2 = time.time()
            total_time += t2 - t1

            # Calculate Loss
            L = 0
            for k in range(len(predictions)):
                L += (predictions[k]-y_train[k])**2
            loss[rate].append(L/len(predictions))

            test_estimates = np.dot(Test, w)
            test_error = RMSE(test_estimates, y_test)

            # Test if the stoping condition has been met
            if test_error <= stoplimit and not timed_in:
                times.append((total_time, i + 1))
                timed_in = True

            # Restart Timer
            t1 = time.time()

        if not timed_in:

            t2 = time.time()
            total_time += t2 - t1
            times.append((total_time, 500))

        test_estimates = np.dot(Test, w)
        test_error = RMSE(test_estimates, y_test)
        test_rmses.append(test_error)
        optweight.append(w)

    # Find the minimum rmse and the time and learning rate to get it
    min_test_rmse = min(test_rmses)
    min_w = optweight[test_rmses.index(min_test_rmse)]
    min_it = np.inf
    for i in range(len(times)):
        if times[i][1] < min_it:
            min_it = times[i][1]
            preferred_rate = rates[i]
    return loss, times, min_test_rmse, min_w, preferred_rate

def Stoich(x_train, x_test, y_train, y_test, rates, optrmse):
    '''
    Function that is designed to do gradient decent on the first thousand points of
    the pumadyn32nm data set (stochiastically)
    '''
    stoplimit = optrmse*1.02

    # Train and Test matrix
    Train = np.ones((len(x_train), len(x_train[0]) + 1))
    Train[:, 1:] = x_train
    Test = np.ones((len(x_test), len(x_test[0]) + 1))
    Test[:, 1:] = x_test

    times = []
    optweight = []
    test_rmses = []
    loss = {}

    for rate in rates:
        # Set weights to zero
        w = np.zeros((33, 1))

        loss[rate] = []
        timed_in = False

        #Gradient Descent Starts here! Yay!
        total_time = 0
        t1 = time.time()
        for i in range(500):
            predictions = np.dot(Train, w)
            batch = random.randint(0, len(predictions)-1)
            grad_L = 2 * (predictions[batch] - y_train[batch])*np.insert(x_train[batch], 0, 1)
            grad_L = grad_L.reshape((33, 1))
            w = np.add(w, -rate*grad_L)

            t2 = time.time()
            total_time += t2 - t1

            # Loss
            L = 0
            for j in range(len(predictions)):
                L += (predictions[j]-y_train[j])**2
            loss[rate].append(L/len(predictions))

            # Prediction on test set
            test_estimates = np.dot(Test, w)
            test_error = RMSE(test_estimates, y_test)
            if test_error <= stoplimit and not timed_in:
                times.append((total_time, i + 1))
                timed_in = True

            # Restart Timer
            t1 = time.time()


        if not timed_in:
            # Record Training Time
            t2 = time.time()
            total_time += t2 - t1
            times.append((total_time, 500))

        test_estimates = np.dot(Test, w)
        test_error = RMSE(test_estimates, y_test)
        test_rmses.append(test_error)

        optweight.append(w)


    min_test_rmse = min(test_rmses)
    min_w = optweight[test_rmses.index(min_test_rmse)]
    preferred_rate = rates[test_rmses.index(min_test_rmse)]

    return loss, times, min_test_rmse, min_w, preferred_rate


    
'''
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
x_train = x_train[:1000]
y_train = y_train[:1000]

rates = [0.0001, 0.0005, 0.001, 0.005 ,0.01, 0.05 ,0.1]
opt_rmse = 0.86335124366
gd_loss, conv_times_gd, test_rmse_gd, opt_w_gd, prefrate = Stoich(x_train, x_test, y_train, y_test, rates, opt_rmse)
print(gd_loss)
print(conv_times_gd)
print(test_rmse_gd)
print(opt_w_gd)
print(prefrate)
'''
        
        
            
            
            
