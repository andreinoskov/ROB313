import numpy as np
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
import time
import random

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def accuracy_ratio(y_test, y_estimates):
    return (y_estimates == y_test).sum() / len(y_test)


def log_likelihood(estimates, actual):
    total = 0
    for i in range(len(estimates)):
        total += actual[i]*np.log(sigmoid(estimates[i])) + (1-actual[i])*np.log(1 - sigmoid(estimates[i]))
    return total


def logFullBatch(x_train, x_valid, x_test, y_train, y_valid, y_test, rates):
    '''
    Full Batch Gradient Descent to find weights of sigmodal function
    '''

    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    # Create Xtrain and Xtest matrix
    Xtrain = np.ones((len(x_train), len(x_train[0]) + 1))
    Xtrain[:, 1:] = x_train

    Xtest = np.ones((len(x_test), len(x_test[0]) + 1))
    Xtest[:, 1:] = x_test

    test_accuracies = []
    test_logs = []
    neg_log = {}

    for rate in rates:
        # Initialize minimizer
        w = np.zeros(np.shape(Xtrain[0, :]))

        neg_log[rate] = []

        for iteration in range(5000):
            estimates = np.dot(Xtrain, w)
            estimates = estimates.reshape(np.shape(y_train))

            # Compute Full-Batch Gradient
            grad_L = np.zeros(np.shape(w))
            for i in range(len(y_train)):
                grad_L += (y_train[i] - sigmoid(estimates[i])) * Xtrain[i, :]

            # Update weights
            w = np.add(w, rate*grad_L)

            # Calculate Full-Batch Log-Likelihood
            L = log_likelihood(estimates, y_train)
            neg_log[rate].append(-L)


        # Allocate Space and make classifications, record test accuracy ratio
        test_estimates = np.dot(Xtest, w)
        test_estimates = test_estimates.reshape(np.shape(y_test))
        predictions = np.zeros(np.shape(y_test))
        for i in range(len(predictions)):
            p = sigmoid(test_estimates[i])
            if p > 1/2:
                predictions[i] = 1
            elif p < 1/2:
                predictions[i] = 0
            else:
                predictions[i] = -1

        # Append Test Accuracy and Test Log-likelihood
        test_accuracies.append(accuracy_ratio(y_test, predictions))
        test_logs.append(log_likelihood(test_estimates, y_test))

    # Final Recordings (Best Accuracy, Log-likelihood, Preferred Rates)
    best_accuracy = max(test_accuracies)
    test_log = min(test_logs)
    min_rates = []
    min_rates.append(rates[test_accuracies.index(best_accuracy)])
    min_rates.append(rates[test_logs.index(test_log)])

    return neg_log, best_accuracy, min_rates, test_log

def logStoch(x_train, x_valid, x_test, y_train, y_valid, y_test, rates):
    '''
    Mini-batch stocastic gradient descent to find weights of sigmodal function. 
    '''
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    # Create training and testing matrix
    Xtrain = np.ones((len(x_train), len(x_train[0]) + 1))
    Xtrain[:, 1:] = x_train

    Xtest = np.ones((len(x_test), len(x_test[0]) + 1))
    Xtest[:, 1:] = x_test

    test_accuracies = []
    test_logs = []
    neg_log = {}

    for rate in rates:

        w = np.zeros(np.shape(Xtrain[0, :]))

        neg_log[rate] = []

        for iteration in range(5000):

            # LM Estimates (on training set)
            estimates = np.dot(Xtrain, w)
            estimates = estimates.reshape(np.shape(y_train))

            # Compute Mini-Batch (1) Gradient
            i = random.randint(0, len(y_train)-1)
            grad_L = (y_train[i] - sigmoid(estimates[i])) * Xtrain[i, :]

            # Update weights
            w = np.add(w, rate*grad_L)

            # Calculate Full-Batch Log-Likelihood
            L = log_likelihood(estimates, y_train)
            neg_log[rate].append(-L)

        # Allocate Space and make classifications, record test accuracy ratio
        test_estimates = np.dot(Xtest, w)
        test_estimates = test_estimates.reshape(np.shape(y_test))
        predictions = np.zeros(np.shape(y_test))
        for i in range(len(predictions)):
            p = sigmoid(test_estimates[i])
            if p > 1/2:
                predictions[i] = 1
            elif p < 1/2:
                predictions[i] = 0
            else:
                predictions[i] = -1

        # Append Test Accuracy and Test Log-likelihood
        test_accuracies.append(accuracy_ratio(y_test, predictions))
        test_logs.append(log_likelihood(test_estimates, y_test))

    # Final Recordings (Best Accuracy, Log-likelihood, Preferred Rates)
    best_accuracy = max(test_accuracies)
    test_log = min(test_logs)
    min_rates = []
    min_rates.append(rates[test_accuracies.index(best_accuracy)])
    min_rates.append(rates[test_logs.index(test_log)])

    return neg_log, best_accuracy, min_rates, test_log


    


