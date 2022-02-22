import numpy as np
import time
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
from Q1 import *
from Q2 import *
from Q3 import *
from Q4 import *

# Question 1
# The kfinder algorithm is used to find the best k for regressions.
regression = ['mauna_loa', 'pumadyn32nm']

for reg in regression:

    # L1
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('reg') 
    xlist = np.vstack([x_valid, x_train])
    ylist = np.vstack([y_valid, y_train])
    k = kfinder(xlist, ylist, dist = L1)
    xvalue = []
    yvalue = []
    for i in k.keys():
        xvalue.append(i)
        yvalue.append(k[i])

    print(min(yvalue))
    print(xvalue[yvalue.index(min(yvalue))])


    #L2
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('reg') 
    xlist = np.vstack([x_valid, x_train])
    ylist = np.vstack([y_valid, y_train])
    k = kfinder(xlist, ylist, dist = L2)
    xvalue = []
    yvalue = []
    for i in k.keys():
        xvalue.append(i)
        yvalue.append(k[i])

    print(min(yvalue))
    print(xvalue[yvalue.index(min(yvalue))])      
    # a plot is only really needed for Mauna_loa L2 but I was too lazy to bring this out of the loop.
    plt.plot(xvalue, yvalue)
    plt.title(" Kvalue vs RMSE for L2 Distance")
    plt.xlabel("K-value")
    plt.ylabel("Average RMSE for L2 Distance Metric")
    plt.show()


# rosenbrock
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train = 1000, d = 2) 
xlist = np.vstack([x_valid, x_train])
ylist = np.vstack([y_valid, y_train])
k = kfinder(xlist, ylist, dist = L1)
xvalue = []
yvalue = []
for i in k.keys():
    xvalue.append(i)
    yvalue.append(k[i])

print(min(yvalue))
print(xvalue[yvalue.index(min(yvalue))])

xlist = np.vstack([x_valid, x_train])
ylist = np.vstack([y_valid, y_train])
k = kfinder(xlist, ylist, dist = L1)
xvalue = []
yvalue = []
for i in k.keys():
    xvalue.append(i)
    yvalue.append(k[i])

print(min(yvalue))
print(xvalue[yvalue.index(min(yvalue))])

# Preformance on Mauna_loa data set for k = 1, 5, 10

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa', n_train = 1000, d = 2) 
xlist = np.vstack([x_valid, x_train])
ylist = np.vstack([y_valid, y_train])
klist = [1, 5, 10]
for k in klist:
    xvalues = [1.3 + (1.8-1.3)/100*i for i in range(100)]
    yvalues = []
    for x in xvalues:
        lis = [x]
        yvalues.append(knearestneighbors(lis, x_test, y_test, k, 'L2'))
        
    string = 'Predicted plot for K = ' + str(k) 
    ax = plt.gca()
    ax.scatter(x_test, y_test)
    ax.plot(xvalues, yvalues)
    ax.set_title(string)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


# Question 2
dimensionlist = [2, 3, 4, 5, 6, 7, 8, 9]



brute = []
fine = []

for dimension in dimensionlist:
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train =5000, d =dimension) 
    xlist = np.vstack([x_valid, x_train])
    ylist = np.vstack([y_valid, y_train])
    data = np.hstack([xlist, ylist])
    timebrute, timefine = timetest(5, xlist, ylist, x_test, y_test, data, L2, dimension)
    brute.append(timebrute)
    fine.append(timefine)
ax = plt.gca()
ax.scatter(dimensionlist, brute)
ax.scatter(dimensionlist, fine)
ax.set_title('Time for brute force [blue] and KDtree[orange] for rosenbrock with d varying and n_train =5000. K =5')
ax.set_xlabel('Value for dimension')
ax.set_ylabel('Time to run KNN')
plt.show()


# Question 3
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
y_train, y_valid, y_test = np.argmax(y_train, axis = 1), np.argmax(y_valid, axis = 1),np.argmax(y_test, axis = 1) 
print(kfinder(x_train, y_train, x_test, y_test, "euclidean"))

# Question 4
regression = ['mauna_loa', 'pumadyn32nm']
classification = ['iris', 'mnist_small']
for i in regression:
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(i)
    xlist = np.vstack([x_train, x_valid])
    ylist = np.vstack([y_train, y_valid])
    print(i)
    print(linregreg(xlist, ylist, x_test, y_test))

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train = 1000, d = 2)
xlist = np.vstack([x_train, x_valid])
ylist = np.vstack([y_train, y_valid])
print(linregreg(xlist, ylist, x_test, y_test))

for i in classification:    
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(i)
    xlist = np.vstack([x_train, x_valid])
    ylist = np.vstack([y_train, y_valid])
    print(i)
    print(linregclass(xlist, ylist, x_test, y_test))




