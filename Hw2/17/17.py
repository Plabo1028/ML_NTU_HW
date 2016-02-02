import numpy as np
import pylab as pl
import math
import sys
from random import *

def generate_input_data(time_seed,data_size):
    np.random.seed(time_seed)
    raw_X = np.sort(np.random.uniform(-1,1,data_size))
    noised_y = np.sign(raw_X)*np.where(np.random.random(raw_X.shape[0])<0.2,-1,1)
    return raw_X, noised_y

def Ein(x,y):
    Ein = x.shape[0]
    final_theta = 0.0
    for a in range(len(x)+1):
        if a == 0:
            theta = (x[0]-1)/2
        elif a == 20:
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

ans17 = {}
ans18 = {}
if __name__ == '__main__':
    Loop = 5000
    data_size = 20
    total_Eout = 0
    total_Ein = 0
    # repeat experiment Loop times
    for i in range(0,Loop):
        # generate
        x,y = generate_input_data(i,data_size)
        # Ein of this loop, best_theta, best_sign
        curr_Ein, theta, sign = Ein(x,y)
        total_Ein = total_Ein + curr_Ein
        # store the Ein of two number after point
        store_Ein= format((curr_Ein/float(data_size)),'.2f')
        # Eout of this loop
        curr_Eout = 0.5+0.3*sign*(abs(theta)-1)
        total_Eout = total_Eout + curr_Eout
        store_Eout= format((curr_Eout),'.2f')
        if float(store_Ein) not in ans17:
            ans17[float(store_Ein)] = 1
        else:
            ans17[float(store_Ein)] = ans17[float(store_Ein)] + 1
        if float(store_Eout) not in ans18:
            ans18[float(store_Eout)] = 1
        else:
            ans18[float(store_Eout)] = ans18[float(store_Eout)] + 1
    print '================================================================='
    print 'Average Ein: ',(float(total_Ein) / float(Loop*data_size))
    print '================================================================='
    print 'Average Eout: ',(float(total_Eout) / float(Loop))

    #plot figure
    pl.bar(ans17.keys(),ans17.values(), width=0.001)
    pl.xlim([min(ans17.keys()), max(ans17.keys())])
    pl.ylim(0, (max(ans17.values())))
    pl.xlabel('Ein')
    pl.ylabel('Frequency')
    pl.title('17')
    pl.savefig('17')

