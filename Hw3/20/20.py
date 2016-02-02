import sys
import numpy as np
import math
from random import *

def read_input_data(path):
    x = []
    y = []
    with open(path) as f:
        for line in f:
            data = line.strip().split(' ')
            tmp_x = data[:-1]

            x.append(tmp_x)
            y.append(float(data[-1]))

    return np.float32(x),np.float32(y)

def predict(X, y, w):
    probability = 1/(1+np.exp((-1)*np.dot(X, w)))
    predicts = np.where( probability >=0.5,1.0,-1.0)
    Eout = sum(predicts!=y)
    return Eout / float(predicts.shape[0])

def calculate_gradient(X,y,w):
    s = np.dot(X,w)*y
    # s = np.dot(w, X.transpose())*y
    theta = 1.0/(1+np.exp(s))
    # print theta.reshape(-1,1)
    # raw_input('pause')
    gradient_all = (-1)*theta.reshape(-1,1)*y.reshape(-1,1)*X
    gradient_average = np.sum(gradient_all, axis=0)
    return gradient_average / gradient_all.shape[0]


if __name__ == '__main__':
    # read train data
    X,y = read_input_data("hw3_train.dat")
    print 'X dimension before add 1 column: ',X.shape
    # add '1' column
    X = np.concatenate((np.ones(X.shape[0]).reshape(-1,1),X),axis=1)
    print 'y dimension: ',y.shape
    print 'X dimension: ',X.shape
    T = 2000
    learning_rate = 0.001
    w = np.zeros(X.shape[1], dtype = float)
    print 'w dimension: ',w.shape

    for i in range(0,T):
        tmpi = i % 1000
        gradient = calculate_gradient(X[tmpi], y[tmpi], w)
        w = w - learning_rate*gradient
        # print w
        # raw_input('pause')

    '''predict'''
    test_X, test_y = read_input_data("hw3_test.dat")
    test_X = np.concatenate((np.ones(test_X.shape[0]).reshape(-1,1),test_X),axis=1)
    Eout = predict(test_X, test_y, w)
    print 'W: ',w
    print 'Eout: ',Eout
