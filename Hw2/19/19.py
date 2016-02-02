import sys
import numpy as np
import math
from random import *

def read_input_data(path):
    x = []
    y = []
    for line in open(path).readlines():
        items = line.strip().split(' ') # read one line into items
        tmp_x = []
        for i in range(0,len(items)-1): tmp_x.append(float(items[i]))
        x.append(tmp_x)
        y.append(float(items[-1]))
    # return array
    return np.array(x),np.array(y)

def Ein(x,y):
    Ein = x.shape[0]
    final_theta = 0.0
    for a in range(len(x)+1):
        if a == 0:
            theta = (x[0]-1)/2
        elif a == 100:
            theta = (1+x[a-1])/2
        else:
            theta = (x[a]+x[a-1])/2
        y_positive = np.where(x>theta,1,-1 )
        y_negative = np.where(x<theta,1,-1)
        error_positive = sum(y_positive!=y)
        error_negative = sum(y_negative!=y)
        if error_positive < error_negative:
            if error_positive < Ein:
                Ein = error_positive
                sign = 1
                final_theta = theta
        else:
            if error_negative < Ein:
                Ein = error_negative
                sign = -1
                final_theta = theta
    if final_theta==float("inf"):
        final_theta = 1.0
    if final_theta==float("-inf"):
        final_theta = -1.0
    return Ein, final_theta, sign

if __name__ == '__main__':
    x,y = read_input_data("hw2_train.dat")
    total_Ein = x.shape[0]
    theta = 0
    sign = 1
    index = 0
    # implementation (a) step
    for i in range(0,x.shape[1]): # i is x dimension
        input_x = x[:,i] # arrary size is data_size*1
        input_data = np.transpose(np.array([input_x,y])) # add y into data
        input_data = input_data[np.argsort(input_data[:,0])] # sort
        curr_Ein, curr_theta, curr_sign = Ein(input_data[:,0],input_data[:,1])
        if total_Ein>curr_Ein:
            total_Ein = curr_Ein
            theta = curr_theta
            sign = curr_sign
            index = i
    print '================================================================='
    print 'Ein: ',((float(total_Ein)/float(x.shape[0])))
    print 'Best theta: ',theta
    print 'Best sign: ',sign
    print 'Best dimension: ',index
    # test process
    test_x,test_y = read_input_data("hw2_test.dat")
    test_x = test_x[:,index]
    predict_y = np.array([])
    if sign==1:
        predict_y = np.where(test_x>theta,1.0,-1.0)
    else:
        predict_y = np.where(test_x<theta,1.0,-1.0)
    Eout = sum(predict_y!=test_y)
    print '================================================================='
    print 'Eout: ',(Eout*1.0)/test_x.shape[0]
