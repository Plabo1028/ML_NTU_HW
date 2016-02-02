import numpy as np
from numpy import dtype
import random
import pylab as pl

def generate_data():
    x1 = np.random.uniform(-1,1,1000)
    x2 = np.random.uniform(-1,1,1000)
    X = np.zeros((1000,3), dtype = float)
    f_result = np.zeros((1000,1), dtype = float)

    for item in range(1000):
        X[item][0] = 1.0
        X[item][1] = x1[item]
        X[item][2] = x2[item]
        f_result[item] = np.sign(x1[item]*x1[item] + x2[item]*x2[item] - 0.6)
    return X,f_result

sum_all = 0
all_error = {}
for times in range(1000):
    '''generate_data'''
    X,f_result = generate_data()
    f_noise_result = np.zeros((1000,1), dtype = float)
    np.copyto(f_noise_result,f_result)
    for j in range(1000):
        probability = random.uniform(0,10)
        if probability >= 9 :
            f_noise_result[j] = f_noise_result[j] * (-1)

    '''linear regression'''
    W_lin =  np.linalg.pinv(X) # pseudo-inverse x^
    W_lin = np.asmatrix(np.dot(W_lin, f_noise_result)) #  pseudo-inverse * y
    '''predict'''
    ans = np.dot(X, W_lin)
    test_result = []
    for i in range(1000):
        test_result.append(np.sign(ans[i][0] * f_noise_result[i][0]) != 1)
    error = np.average(np.array(test_result))
    print 'Number {0} , error {1}'.format(times,error)
    if error not in all_error:
        all_error[error] = 1
    else:
        all_error[error] = all_error[error] +1
    sum_all += error
print "average error ",sum_all/1000

pl.bar(all_error.keys(),all_error.values(), width=0.001)
pl.xlim([min(all_error.keys()), max(all_error.keys())])
pl.ylim(0, (max(all_error.values())))
pl.xlabel('Ein')
pl.ylabel('Frequency')
pl.title('13')
pl.savefig('13')
