import sys
import numpy as np
import math
from random import *

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

if __name__ == '__main__':
    ''' train '''
    X,y = readData("../hw7_train.dat")
    root = CRT(X,y)
    ''' test '''
    error = 0
    for i in range(X.shape[0]):
        error += (1 if predict(root, X[i]) != y[i] else 0)
    print float(error)/X.shape[0]




