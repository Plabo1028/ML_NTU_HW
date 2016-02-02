import sys
import numpy as np
import math
import random
import pylab as pl

def readData(path):
    X = []
    with open(path) as f:
        for line in f:
            one = line.strip('\n').split(' ')
            tmp_x = []
            for i in one:
                tmp_x.append(float(i))
            X.append(tmp_x)
    return np.array(X),len(X)

def cal_error(X,y,centers):
    error_total = 0
    for i in range(X.shape[0]):
        error_total += np.sum((X[i]-centers[y[i]])**2,axis=0)
    return error_total/X.shape[0]

def Kmeans(X, k, round):
    centers_index = random.sample(xrange(0,X.shape[0]-1), k)
    centers = X[centers_index]
    stop_error = float("inf")
    for r in range(round):
        y = update_category(X, k, centers)
        centers = update_centers(X, y, k)
        if r%10 == 0:
            tmp_error = cal_error(X,y,centers)
            if stop_error > tmp_error:
                stop_error = tmp_error
            elif stop_error == tmp_error:
                # print 'break in round ',r
                break

    return centers,y

def update_category(X, k, centers):
    y = []
    for i in range(X.shape[0]):
        category = -1
        distance = float("inf")
        for j in range(k):
            tmp_d = np.sum((X[i] - centers[j])**2,axis=0)
            if tmp_d < distance:
                distance = tmp_d
                category = j
        y.append(category)
    return np.array(y)

def update_centers(X, y, k):
    centers = []
    for j in range(k):
        center = np.sum(X[np.where(y==j)],axis=0)/float(X[np.where(y==j)].shape[0])
        centers.append(center)
    return np.array(centers)

if __name__ == '__main__':
    X,data_size = readData('../hw8_nolabel_train.dat')
    times = 500
    round = 500

    error_histogram = {}
    for k in [2,4,6,8,10]:
        print 'k',k
        err = 0
        for t in range(times):
            if t%50==0:
                print 't',t
            centers,y = Kmeans(X,k,round)
            err += cal_error(X,y,centers)
        error_histogram[k] = err/float(times)

    for k in [2,4,6,8,10]:
        print 'K{0} Error:{1}\n'.format(k,error_histogram[k])

