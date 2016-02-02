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

def cal_error(data_size,X, y, k, err_type):
    err = 0
    if err_type == 'ein':
        test_X = X
        test_y = y
        size = data_size
    elif err_type == 'eout':
        test_X,test_y,test_data_size = readData('../hw8_test.dat')
        size = test_data_size

    for i in range(size-1):
        m = KNN( k, X, y, test_X[i])
        if test_y[i] != m:
            err += 1
    return err/float(size)

def KNN(k, X, y, test_x):
    ''' caculate distance between X[i] and x_all '''
    distance = np.sum((X-test_x)**2, axis=1)
    order = np.argsort(distance)
    m = 0
    ''' vote X[i] '''
    for i in range(k):
        m += y[order[i]]
    return -1 if m<0 else 1

if __name__ == '__main__':
    X,y,data_size = readData('../hw8_train.dat')

    error_histogram = {}
    for k in [1,3,5,7,9]:
        error_histogram[k]=cal_error(data_size,X,y,k,'eout')

    command = ''
    for k in sorted(error_histogram.iterkeys()):
        command += 'K='+str(k)+'=>Error:'+str(error_histogram[k])+'\n'

    pl.plot(error_histogram.keys(),error_histogram.values(),'b*')
    pl.xlim([min(error_histogram.keys()), max(error_histogram.keys())])
    pl.ylim(0, (max(error_histogram.values())))
    pl.xlabel('K')
    pl.ylabel('Eout_k')
    pl.title('13&14')
    pl.text(5,0.2,command,color='blue')
    pl.savefig('13&14')
