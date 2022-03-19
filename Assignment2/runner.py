import numpy as np
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
from Question3 import *
from Question4 import *

print('Q3 ML')
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')

theta, reg, valid_error, test_err = Question3(x_train, x_valid, x_test, y_train, y_valid, y_test)

print(theta)
print(reg)
print(valid_error)
print(test_err)


print('Q3 RB')
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train = 1000, d = 2)

theta, reg, valid_error, test_err = Question3(x_train, x_valid, x_test, y_train, y_valid, y_test)

print(theta)
print(reg)
print(valid_error)
print(test_err)
        

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')

xfull = np.vstack([x_valid, x_train, x_test])
yfull = np.vstack([y_valid, y_train, y_test])


plt.title("mauna_loa")
plt.plot(xfull, yfull)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
dic = diccreator()
func, fact = greedyregression(x_train, x_valid, y_train, y_valid, dic)
print(func)
def fun(func , norm, x):
    dic = diccreator()
    funret = 0
    for f in func:
        funret += f[1]*dic[f[0]](x)[0]
    
    return funret[0]

pred = []
for x in x_test:
    pred.append( fun(func, fact, x))
print(RMSE(pred, y_test))

fig, ax = plt.subplots()
ax.scatter(x_test, y_test)
ax.plot(x_test, pred)
plt.title("Prediction function for test data set")
plt.ylabel("y")
plt.xlabel("x")
plt.show()

pred = []
for x in x_train:
    pred.append( fun(func, fact, x))

fig, ax = plt.subplots()
ax.plot(x_train, y_train)
ax.plot(x_train, pred)
plt.title("Prediction function for traing data set")
plt.ylabel("y")
plt.xlabel("x")
plt.show()



