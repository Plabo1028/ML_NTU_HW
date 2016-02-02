import numpy as np
from scipy import linalg
'''train data'''
X_train = []
Y_train = []
with open("hw4_train.dat") as f:
    for line in f:
        '''X_train'''
        x = line.split()[:-1]
        tmp = [1.0]
        for a in x:
            # print a
            tmp.append(float(a))
        X_train.append(tmp)

        '''Y_train'''
        Y_train.append(float(line.split()[-1]))
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print 'X_train.shape: ',X_train.shape
print 'Y_train.shape: ',Y_train.shape

'''test data'''
X_test = []
Y_test = []
with open("hw4_test.dat") as f:
    for line in f:
        '''X_test'''
        x = line.split()[:-1]
        tmp = [1.0]
        for a in x:
            # print a
            tmp.append(float(a))
        X_test.append(tmp)

        '''Y_test'''
        Y_test.append(float(line.split()[-1]))
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print 'X_test.shape: ',X_test.shape
print 'Y_test.shape: ',Y_test.shape

# raw_input('pause')
lam_ans13 = 11.26

# Z_inv = (np.dot(X_train.transpose(), X_train) + lam_ans13 * np.eye(X_train.shape[1])).I
# W = np.dot(Z_inv,  np.dot(X_train.transpose(), Y_train))

Z_inv = linalg.inv(np.dot(X_train.transpose(),X_train)+lam_ans13*np.eye(X_train.shape[1]))
W = np.dot(np.dot(Z_inv,X_train.transpose()),Y_train)
print W.shape

def calculate_Error(W, X, y):
    scores = np.dot(W, X.transpose())
    predicts = np.where(scores>=0,1.0,-1.0)
    Eout = sum(predicts!=y)
    return (Eout*1.0) / predicts.shape[0]

Ein = calculate_Error(W, X_train, Y_train)
Eout = calculate_Error(W, X_test, Y_test)

print 'Ein: ',Ein
print 'Eout: ',Eout

