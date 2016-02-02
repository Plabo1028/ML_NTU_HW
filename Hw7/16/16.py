import sys
import numpy as np
import math
from random import *
import pylab as pl

class TreeNode:
    def __init__(self, i, v):
        self.dim = i
        self.val = v
        self.sign = 0
        self.left = None
        self.right = None

def readData(path):
    X = []
    y = []
    with open(path) as f:
        for line in f:
            # print line
            # raw_input('pause')
            tmp_line = line.strip().split(' ')
            tmp_x = []

            for i in tmp_line[:-1]: tmp_x.append(float(i))

            X.append(tmp_x)
            y.append(float(tmp_line[-1]))
    return np.array(X),np.array(y)

def step1_split(X, y):

    def learn_decisionStump(X,y):
        ''' calculate median of interval '''
        theta_interval = []
        for i in range(X.shape[0]-1):
            if i == 0:
                theta_interval.append((X[i]+X[i+1])/2)
            elif i == X.shape[0]-1:
                theta_interval.append(float("inf"))
            else:
                theta_interval.append((X[i]+X[i+1])/2)

        Error = float("inf")

        for theta in theta_interval:
            ly = y[np.where(X<theta)]
            ry = y[np.where(X>=theta)]
            err = ly.shape[0]*GiniIndex(ly) + ry.shape[0]*GiniIndex(ry)
            if Error > err:
                Error = err
                target_theta = theta
        return Error, target_theta

    sorted_index = []
    for i in range(X.shape[1]): sorted_index.append(np.argsort(X[:,i]))

    Error = float("inf")
    dim = -1
    val = 0 # val == theta which split data into 1 -1

    for i in range(X.shape[1]):
        Xi = X[sorted_index[i], i]
        yi = y[sorted_index[i]]
        ''' min cost and best theta of dim i '''
        err, theta = learn_decisionStump(Xi, yi)
        ''' argmin error and best theta of data X'''
        if Error > err:
            Error = err
            dim = i #which dim we split
            val = theta
    ''' step2_binary tree '''
    lX = X[np.where(X[:,dim] < val)]
    ly = y[np.where(X[:,dim] < val)]
    rX = X[np.where(X[:,dim] >= val)]
    ry = y[np.where(X[:,dim] >= val)]
    return lX, ly, rX, ry, dim, val

def GiniIndex(y):
    N = int(y.shape[0])
    if N==0: return 0
    n1 = sum(y==1)
    n2 = sum(y==-1)
    '''impurity == 0'''
    if n1==0: return 0
    if n2==0: return 0
    return 1.0 - math.pow(1.0*n1/N,2) - math.pow(1.0*n2/N,2)
    # return float(1.0 - math.pow(float(n1/N),2) - math.pow(float(n2/N),2))

def CRT(X,y):
    if X.shape[0]==0: return None # none case
    if GiniIndex(y)==0:
        node = TreeNode(-1, -1)
        node.sign = 1 if y[0]==1 else -1
        return node
    ''' step2_binary tree '''
    lX, ly, rX, ry, dim, val = step1_split(X,y)
    node = TreeNode( dim, val)
    ''' step3_recurrence '''
    node.left = CRT(lX, ly)
    node.right = CRT(rX, ry)
    return node

def predict(root, X):
    if root.val==-1 : return root.sign
    if X[root.dim] < root.val:
        return predict(root.left, X)
    else:
        return predict(root.right, X)

def Bagging(X, y):
    sampleX = np.zeros(X.shape)
    sampley = np.zeros(y.shape)
    flips = [randint(0,X.shape[0]-1) for i in range(X.shape[0])]
    sampleX = X[flips]
    sampley = y[flips]
    return sampleX, sampley

def calculate_Error(model, X, y):
    error_count = 0
    for i in range(X.shape[0]):
        error_count = error_count + (1 if predict(model, X[i])!=y[i] else 0)
    return 1.0*error_count/X.shape[0]

if __name__ == '__main__':
    ''' train '''
    X,y = readData("../hw7_train.dat")
    T = 30000
    trees = []
    error_rate = 0
    error_histogram = {}
    for i in range(T):
        print i
        Xi,yi = Bagging(X,y)
        model = CRT(Xi,yi)
        trees.append(model)

        tmp_error = calculate_Error(model, X, y)
        error_rate += tmp_error
        if float(tmp_error) not in error_histogram:
            error_histogram[float(tmp_error)] = 1
        else:
            error_histogram[float(tmp_error)] = error_histogram[float(tmp_error)] + 1

    print error_rate = error_rate/T

    pl.bar(error_histogram.keys(),error_histogram.values(), width=0.001)
    pl.xlim([min(error_histogram.keys()), max(error_histogram.keys())])
    pl.ylim(0, (max(error_histogram.values())))
    pl.xlabel('Ein(gt)')
    pl.ylabel('Frequency')
    pl.title('16')
    pl.savefig('16')
