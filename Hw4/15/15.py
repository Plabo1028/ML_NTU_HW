import numpy as np
import pylab as pl

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

# X_val = X_train[121:,:,:]
# Y_val = Y_train[121:]

# X_train = X_train[:120,:,:]
# Y_train = Y_train[:120]

print 'X_train.shape: ',X_train.shape
print 'Y_train.shape: ',Y_train.shape
# print 'X_val.shape: ',X_val.shape
# print 'Y_val.shape: ',Y_val.shape
# raw_input('pause')
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

def calculate_Error(W, X, y):
    scores = np.dot(W, X.transpose())
    predicts = np.where(scores>=0,1.0,-1.0)
    E = sum(predicts!=y)
    return (E*1.0) / float(predicts.shape[0])

def W_rigde_regression(X,y,lam):
    Z_v = np.linalg.inv(np.dot(X.transpose(),X)+lam*np.eye(X.shape[1]))
    return  np.dot(np.dot(Z_v,X.transpose()),y)

lam_set = [ i for i in range(2,-11,-1) ]

min_Ein = 1
min_Eout = 1
target_lambda = 2

dic_15 = {}
for lam in lam_set:
    W = W_rigde_regression(X_train,Y_train,pow(10, lam))
    Ein = calculate_Error(W, X_train, Y_train)
    Eout = calculate_Error(W, X_test, Y_test)
    dic_15[lam] = Eout
    print 'lambda: ',lam
    print 'Ein: ',Ein
    print 'Eout: ',Eout
    print '==============='
    if Eout < min_Eout:
        target_lambda = lam
        # min_Ein = Ein
        min_Eout = Eout

print 'best'
print 'ans lambda: ',target_lambda
# print 'Ein: ',min_Ein
W = W_rigde_regression(X_train,Y_train,pow(10, target_lambda))
Eout = calculate_Error(W, X_test, Y_test)
print 'min Eout: ',Eout


pl.bar(dic_15.keys(),dic_15.values())
pl.xlim([min(dic_15.keys()), max(dic_15.keys())])
pl.ylim(0, (max(dic_15.values())))
pl.xlabel('lambda')
pl.ylabel('Eout')
pl.title('15')
pl.savefig('15')
