import numpy as np
import math
from matplotlib import pyplot as plt
from data_utils import load_dataset
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from Question3 import *
'''
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')

xfull = np.vstack([x_valid, x_train, x_test])
yfull = np.vstack([y_valid, y_train, y_test])

plt.plot(xfull, yfull)
plt.show()
'''


def MDL(x_test, y_test, function, k):
    '''
    Finds MDL error
    '''
    # Intialize everything
    dic = diccreator()
    loss = []
    i = 0
    x_tester = list(x_test)
    y_tester = list(y_test)
    N = len(x_tester)
    remlist = []

    # Due to resonance near x=0, take out the middle of the set
    while i < len(x_tester)-1:
        if x_tester[i] < x_test.mean() + 1/10*max(x_test)/(max(x_test)-min(x_test)) or x_tester[i] > x_test.mean()-1/10*max(x_test)/(max(x_test)-min(x_test)):
            remlist.append(i)
        i = i + 1

    for rem in remlist:
        x_tester.remove(list(x_test)[rem])
        y_tester.remove(list(y_test)[rem])

    # Iterate through list of functions and find there prediction
    N = len(x_tester)
    for x in x_tester: 
        s = 0
        for fun in function:
            s += fun[1]*dic[fun[0]](x)
        loss.append(s)

    loss = np.array(loss)

    # Return MDL 
    return 1/2 * N * np.log ( np.sum((y_tester -loss)**2)) + k/2*np.log(N)


def sinfinder(j):
    return lambda x: np.sin(x*j)

def cosfinder(j):
    return lambda x: np.cos(x*j)

def diccreator():
    '''
    Creates a dictionary of basis functions. This is called over and over again. Although there is probably a better way to do this I will not do it any other way.
    '''
    i = 0
    diccreate = {}

    # Add polynomials
    while i < 2:
        diccreate ["polynom" + str(i)] = np.polynomial.polynomial.Polynomial.basis(deg = i)
        i = i + 1

    j = 1

    # Add sinousides.
    while j < 100:
        
        diccreate["sin (x/" + str(j) + ")"] = sinfinder(1/j)
        
        diccreate["sin (x*" + str(j) + ")"] = sinfinder(j)

        diccreate["sin (x/" + str(j) + ")"] = cosfinder(1/j)
        
        diccreate["sin (x*" + str(j) + ")"] = cosfinder(j)
        
        j = j + 1
        
    return diccreate

def normalized (a):
    '''
    normalizes a vector for Orthoanalg thing. Basically returns unit vector.
    '''
    normfactor = max(a) - min(a)
    for i in a:
        normfactor += i**2

    normfactor = normfactor ** (1/2)
    newarray = (a) / normfactor 
    
    return newarray , normfactor

def dotprod (funclist, x_full, y_fullnorm):
    '''
    returns a list of dotproducts and a new y-list for the next iteration
    '''
    
    dic = diccreator()
    listofdotprod = []

    # For all remaining functions, take the dot product of their prediction with the current yvalues and store in a list
    for func in funclist:
        pred = []
        for x in x_full:
            pred.append(dic[func](x))

        pred = np.array(pred)
        pred, norm = normalized(pred)
        listofdotprod.append([func, np.dot(pred.T, y_fullnorm)])

    # Take max value in this list and preform y - ag to find new yvalue list.
    newy = []
    func = max(listofdotprod, key = lambda x : x[1])
    for x in x_full:
        z = func[1]*dic[func[0]](x)
        newy.append(z[0])
    newy =np.array(newy)

    newy = y_fullnorm - newy
    return listofdotprod, newy

def greedyregression(x_train, x_valid, y_train, y_valid, dic):
    '''
    returns a list of functions and a constants
    '''
       
    x_full = np.vstack([x_train, x_valid])
    y_full = np.vstack([y_train, y_valid])

    y_fullnorm, normalizefactor = normalized(y_full)

    funcer =  []
    for i in dic.keys():
        funcer.append(i)
        
    listofdotprod, newy =dotprod (funcer, x_full, y_fullnorm)
    neyy, normfact = normalized(newy)

    # Initialize everything
    
    k = 1
    N = len(x_full)
    funclist = []
    function = max(listofdotprod, key = lambda x : x[1])
    funclist.append(function)
    error1 = MDL(x_full, y_fullnorm, funclist, k)
    funcer.remove(function[0])
    error2 = error1
    i = 0
    numberpos = 0
    while numberpos < 3:
        # The interesting  condition in this while loop stems from an observation that the MDL error would
        # sometimes fluctuate between increasing and decreasing. Basically the algorithm counts the number
        # of times the MDL increases and once it is 3, it terminates.
        error1 = error2
        listofdotprod, newy =dotprod (funcer, x_full, neyy)
        neyy, normfact = normalized(newy)
        function = max(listofdotprod, key = lambda x : x[1])
        funclist.append(function)
        error2 = MDL(x_full, y_fullnorm, funclist, k)
        funcer.remove(function[0])
        if error2-error1 > 0:
            numberpos += 1
        i = i + 1

    return funclist, normalizefactor



    

