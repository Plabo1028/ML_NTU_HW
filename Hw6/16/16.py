import numpy as np
import sys, math
import pylab as pl
from random import *

def calculate_Ein(x,y,u):
    ''' calculate median of interval '''
    theta_interval = []
    for i in range(x.shape[0]-1):
        # print i
        if i == 0:
            theta_interval.append(float('-inf'))
            theta_interval.append((x[i]+x[i+1])/2)
        elif i == x.shape[0]-1:
            theta_interval.append(float("inf"))
        else:
            theta_interval.append((x[i]+x[i+1])/2)
    # print np.array(theta_interval)
    # theta_interval = np.array( [float("-inf")]+[ (x[i]+x[i+1])/2 for i in range(0, x.shape[0]-1) ]+[float("inf")] )
    # print 'theta_interval\n',theta_interval
    # print 'u'
    # raw_input('pause')

    Sum_U = sum(u)
    sign = 1
    target_theta = 0.0

    for theta in theta_interval:
        y_positive = np.where(x > theta,1,-1)
        y_negative = np.where(x < theta,1,-1)
        '''calculate error like orange square in ppt'''
        error_positive = sum((y_positive != y)*u)
        error_negative = sum((y_negative != y)*u)

        if error_positive < error_negative:
            # positive ray because low error_positvie
            if Sum_U > error_positive:
                Sum_U = error_positive
                sign = 1
                target_theta = theta
        else:
            # negative ray because low error_negative
            if Sum_U > error_negative:
                Sum_U = error_negative
                sign = -1
                target_theta = theta

    ''' -inf and inf two case'''
    if target_theta == float("inf"):
        target_theta = 1.0
    if target_theta == float("-inf"):
        target_theta = -1.0

    '''calculate scalingFactor like purple diamond in ppt'''
    scalingFactor = 0.0
    errorRate = 0

    if sign == 1:
        # positive ray
        # print 'sign=1'
        error = float(sum((np.where(x > target_theta,1,-1) != y)*u))
        errorRate = error/float(sum(u))
        # print 'error',error
        # print 'errorRate',errorRate
        # raw_input('pause in sign=1')
        try:
            scalingFactor = math.sqrt( (1-errorRate)/errorRate )
        except:
            scalingFactor = 0.5
        # update weight error*scalingFactor + correct/scalingFactor
        # using np.where to distinguish true of false
        # Ein = sum(np.where(X[:,index_t] > theta_t,1,-1)!=y)/float(X.shape[0])
        u_next = (np.where(x > target_theta,1,-1) != y )*u*scalingFactor + (np.where(x > target_theta,1,-1) == y)*u/scalingFactor
    else:
        # negative ray
        # print 'sign=-1'
        error = float(sum((np.where(x < target_theta,1,-1) != y)*u))
        errorRate = error/float(sum(u))
        # print 'error',error
        # print 'errorRate',errorRate
        # raw_input('pause in sign=-1')
        try:
            scalingFactor = math.sqrt( (1-errorRate)/errorRate )
        except:
            scalingFactor = 0.5
        # update weight error*scalingFactor + correct/scalingFactor
        # using np.where to distinguish true of false
        # Ein = sum(np.where(X[:,index_t] < theta_t,1,-1)!=y)/float(X.shape[0])
        u_next = (np.where(x < target_theta,1,-1) != y )*u*scalingFactor + (np.where(x < target_theta,1,-1) == y)*u/scalingFactor
    alpha = math.log(scalingFactor,math.e)
    # print errorRate
    '''
    errorRate
    u_next : update the weight
    alpha : ln(scalingFactor)
    target_theta : distinguish positive or negative
    sign : -1 iff negative ray ,1 iff positive ray
    '''
    return errorRate, u_next, alpha, target_theta, sign

def readData(path):
    X = []
    y = []
    with open(path) as f:
        for line in f:
            items = line.strip().split(' ')
            tmp_X = []
            for i in range(0,len(items)-1):
                tmp_X.append(float(items[i]))
            X.append(tmp_X)
            y.append(float(items[-1]))
            # raw_input(line)
    return np.array(X),np.array(y)

if __name__ == '__main__':
    T = 300
    '''initial'''
    X,y = readData('../train.dat')
    u = np.ones(X.shape[0])/X.shape[0]
    u_next = u
    sorted_index = []
    for i in range(0, X.shape[1]): sorted_index.append(np.argsort(X[:,i]))

    # alpha == weight
    alpha = np.ones(T)
    theta = np.ones(T)
    sign = np.ones(T)
    index = np.zeros(T)
    Ein = np.zeros(T)
    U = np.zeros(T)
    errorRate = np.zeros(T)

    mini_error = 1
    for t in range(0,T):
        # best parameter in iteration t
        alpha_t = 1
        theta_t = 1
        sign_t = 1
        index_t = 1
        Eu = float("inf")

        for i in range(0,X.shape[1]):
            '''i means x dim or y dim '''
            xi = X[sorted_index[i],i]
            yi = y[sorted_index[i]]
            errorRate_this_time, u_this_time, alpha_this_time, theata_this_time, sing_this_time = calculate_Ein(xi, yi, u[sorted_index[i]])

            if Eu > errorRate_this_time :
                Eu = errorRate_this_time
                if mini_error > errorRate_this_time:
                    mini_error = errorRate_this_time

                index_t = i
                u_next = u_this_time
                alpha_t = alpha_this_time
                # Ein_t = Ein_this_time
                theta_t = theata_this_time
                sign_t = sing_this_time
                errorRate_t = errorRate_this_time
        index[t] = index_t
        U[t] = sum(u)
        u[sorted_index[index_t]] = u_next
        alpha[t] = alpha_t
        theta[t] = theta_t
        sign[t] = sign_t
        errorRate[t] = errorRate_t
        if sign_t == 1:
            Ein[t] = sum(np.where(X[:,index_t] > theta_t,1,-1)!=y)/float(X.shape[0])
        else:
            Ein[t] = sum(np.where(X[:,index_t] < theta_t,1,-1)!=y)/float(X.shape[0])

    min_errorRate = float('inf')
    for t in range(T):
        tmp = errorRate[t]
        print ('ER_%d:%.5f\t' % ((t+1),tmp)),
        # if t % 10 == 9:
        #     print ''
        if min_errorRate > tmp:
            min_errorRate = tmp
    print 'Q16=>min_errorRate:',min_errorRate

