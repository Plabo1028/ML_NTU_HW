import sys
import numpy as np
import math
from random import *
import pylab as pl

def readData(path):
    X = []
    y = []
    with open(path) as f:
        for line in f:
            one = line.strip('\n').split(' ')
            tmp_x = []
            for i in one[:-1]:
                tmp_x.append(float(i))
            X.append(tmp_x)
            y.append(float(one[-1]))
    return np.array(X),np.array(y),len(X)

def cal_error(data_size,X, y, r, err_type):
    err = 0
    if err_type == 'ein':
        test_X = X
        test_y = y
        size = data_size
    elif err_type == 'eout':
        test_X,test_y,test_data_size = readData('../hw8_test.dat')
        size = test_data_size

    for i in range(size-1):
        m = KNN(r, X, y, test_X[i])
        if test_y[i] != m:
            err += 1
    return err/float(size)

def KNN(r, X, y, test_x):
    ''' caculate distance between X[i] and x_all '''
    distance = np.sum((X-test_x)**2, axis=1)
    # print distance*float(-r)
    similarity = np.exp(distance*float(-r))
    # print (similarity)
    ''' vote X[i] '''
    m = np.sum(similarity*y)
    # print similarity*y
    # raw_input(m)
    return -1 if m<0 else 1

if __name__ == '__main__':
    X,y,data_size = readData('../hw8_train.dat')

    error_histogram = {}

    for r in [0.001, 0.1, 1, 10, 100]:
        print 'R{0} Error:{1}\n'.format(r,cal_error(data_size,X,y,r,'eout'))


