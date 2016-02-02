import numpy as np
from numpy import dtype
import random
import pylab as pl

def generate_data():
    x1 = np.random.uniform(-1,1,1000)
    x2 = np.random.uniform(-1,1,1000)
    X = np.zeros((1000,6), dtype = float)
    f_result = np.zeros((1000,1), dtype = float)

    for item in range(1000):
        # print x1[item]
        # print x2[item]
        X[item][0] = 1.0
        X[item][1] = x1[item]
        X[item][2] = x2[item]
        X[item][3] = x1[item] * x2[item]
        X[item][4] = x1[item] * x1[item]
        X[item][5] = x2[item] * x2[item]
        f_result[item] = np.sign(x1[item]*x1[item] + x2[item]*x2[item] - 0.6)
        # print X[item]
        # raw_input('pause')
    return X,f_result
time = 1000
W_3 = {}
W_lin_total = np.zeros((6,1), dtype = float)
for times in range(time):
    '''generate_data'''
    X,f_result = generate_data()
    f_noise_result = np.zeros((1000,1), dtype = float)
    np.copyto(f_noise_result,f_result)
    for j in range(1000):
        probability = random.uniform(0,10)
        if probability >= 9 :
            f_noise_result[j] = f_result[j] * (-1)
    '''linear regression'''
    W_lin =  np.linalg.pinv(X) # pseudo-inverse x^
    # W_lin = np.asmatrix(np.dot(W_lin, f_noise_result)) #  pseudo-inverse * y
    W_lin = np.dot(W_lin, f_noise_result)
    W_lin_total += W_lin
    # print W_lin
    # print format(W_lin.item(3),'.3f')
    tmp = format(W_lin.item(3),'.3f')
    print '\nNumber ',times
    print 'W_lin ',W_lin

    if float(tmp) not in W_3:
        W_3[float(tmp)] = 1
    else:
        W_3[float(tmp)] = W_3[float(tmp)] + 1
    # print W_lin
    # print W_lin_total
    # raw_input('pause')
print '\naverage W:\n',W_lin_total/float(time)
print 'average W_3:',W_lin_total[3]/float(time)


# '''predict'''
# W_lin_total = W_lin_total/1000
# # print W_lin_total.shape
# all_error = {}
# for times in range(1000):
#     X,f_result = generate_data()
#     f_noise_result = np.zeros((1000,1), dtype = float)

#     np.copyto(f_noise_result,f_result)
#     for j in range(1000):
#         probability = random.uniform(0,10)
#         # print j
#         # print probability
#         if probability >= 9 :
#             f_noise_result[j] = f_result[j] * (-1)
#             # print f_result[j]
#             # print f_noise_result[j]
#             # raw_input('pause')

#     ans = np.dot(X, W_lin_total)
#     test_result = []
#     for i in range(1000):
#         test_result.append(np.sign(ans[i][0] * f_noise_result[i][0]) != 1)
#     error = np.average(np.array(test_result))
#     print 'Number {0} , error {1}'.format(times,error)
#     if error not in all_error:
#         all_error[error] = 1
#     else:
#         all_error[error] = all_error[error] +1
#     # raw_input('pause')

pl.bar(W_3.keys(),W_3.values(), width=0.001)
pl.xlim([min(W_3.keys()), max(W_3.keys())])
pl.ylim(0, (max(W_3.values())))
pl.xlabel('W_3')
pl.ylabel('Frequency')
pl.title('14')
pl.savefig('14')

# pl.bar(all_error.keys(),all_error.values(), width=0.001)
# pl.xlim([min(all_error.keys()), max(all_error.keys())])
# pl.ylim(0, (max(all_error.values())))
# pl.xlabel('error')
# pl.ylabel('Frequency')
# pl.title('15')
# pl.savefig('15')

