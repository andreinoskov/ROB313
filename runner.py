import numpy as np
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
import time
import random
from Question1 import *
from Question2 import *

def generate_lossplots(x_train, y_train, losses, opt_w, model):

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    X = np.ones((len(x_train), len(x_train[0]) + 1))
    X[:, 1:] = x_train

    opt_pred = np.dot(X, opt_w)
    L_opt = 0
    for i in range(len(opt_pred)):
        L_opt += (opt_pred[i]-y_train[i])**2
    L_opt = L_opt / len(opt_pred)

    x_axis = list(range(500))
    L = [L_opt] * 500
    print(L)
    if model == 'Fullbatch':
        plt.figure(1)
        plt.title('Full-Batch Least Exact Loss vs Iteration')
        plt.plot(x_axis, L, '--k', label='Optimal Loss')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Loss')
        for rate in losses:
            plt.plot(x_axis, losses[rate], label='Loss for n = ' + str(rate))
        plt.legend(loc='upper right')
        plt.show()

        plt.plot(x_axis, L, '--k', label='Optimal Loss')
    elif model == 'Stoich':
        plt.figure(2)
        plt.title('Stoich Exact Loss vs Iteration')
        plt.plot(x_axis, L, '--k', label='Optimal Loss')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Exact Loss')
        for rate in losses:
            plt.plot(x_axis, losses[rate], label='Loss for n = ' + str(rate))
        plt.legend(loc='upper right')
        plt.show()

def generate_loglossplots(losses, model):

    x_axis = list(range(5000))
    if model == 'Fullbatch':
        plt.figure(3)
        plt.title('Full-Batch Negative Log-Likelihood')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log-Likelihood')
        for rate in losses:
            plt.plot(x_axis, losses[rate], label='-Log-L for n = ' + str(rate))
        plt.legend(loc='upper right')
        plt.show()

    elif model == 'Stoch':
        plt.figure(4)
        plt.title('Stochastic Negative Log-Likelihood')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log-Likelihood')
        for rate in losses:
            plt.plot(x_axis, losses[rate], label='-Log-L for n = ' + str(rate))
        plt.legend(loc='upper right')
        plt.show()
        
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
x_train = x_train[:1000]
y_train = y_train[:1000]
optrmse = 0.86335124366
rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]

loss, times, testrmse, opt_w, rate = Fullbatch(x_train, x_test, y_train, y_test, rates, optrmse)
print(loss)
print(times)
print(testrmse)
print(opt_w)
print(rate)
generate_lossplots(x_train, y_train, loss, opt_w, 'Fullbatch')

loss2, times2, testrmse2, opt_w2, rate2 = Stoich(x_train, x_test, y_train, y_test, rates, optrmse)
print(loss2)
print(times2)
print(testrmse2)
print(opt_w2)
print(rate2)
generate_lossplots(x_train, y_train, loss2, opt_w2, 'Stoich')

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]

y_train = np.asarray(y_train, int)
y_valid = np.asarray(y_valid, int)
y_test = np.asarray(y_test, int)

rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]



loss, ratio_gd, pref_rGD, test_log_gd = logFullBatch(x_train, x_valid, x_test, y_train, y_valid, y_test, rates)
generate_loglossplots(loss ,'Fullbatch')

loss2, ratio_gd, pref_rGD, test_log_gd = logStoch(x_train, x_valid, x_test, y_train, y_valid, y_test, rates)
generate_loglossplots(loss2 ,'Stoch')
