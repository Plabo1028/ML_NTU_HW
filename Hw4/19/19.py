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

def calculate_Error(W, X, y):
    scores = np.dot(W, X.transpose())
    predicts = np.where(scores>=0,1.0,-1.0)
    E = sum(predicts!=y)
    return (E*1.0) / float(predicts.shape[0])

def W_rigde_regression(X,y,lam):
    Z_v = np.linalg.inv(np.dot(X.transpose(),X)+lam*np.eye(X.shape[1]))
    return  np.dot(np.dot(Z_v,X.transpose()),y)

lam_set = [ i for i in range(2,-11,-1) ]

min_Ecv = 1
target_lambda = 2

X_train_set = []
Y_train_set = []
for i in range(5):
    X_train_set.append(X_train[i*40:(i+1)*40])
    Y_train_set.append(Y_train[i*40:(i+1)*40])
    # print len(X_train_set)
    # print len(Y_train_set)
# raw_input('pause')

dic_19 = {}
for lam in lam_set:
    total_Ecv = 0
    for i in range(5):
        train_x = []
        train_y = []
        for j in range(5):
            if j!=i :
                train_x.extend( X_train[j*40:(j+1)*40] )
                train_y.extend( Y_train[j*40:(j+1)*40] )
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        W = W_rigde_regression(train_x, train_y, pow(10, lam))
        Ecv = calculate_Error(W, X_train_set[i], Y_train_set[i])
        total_Ecv += Ecv

    this_lam_Ecv = total_Ecv/5.0
    dic_19[lam] = this_lam_Ecv
    # print 'lam: ',lam
    # print 'Ecv: ',this_lam_Ecv
    # print '======================'
    if this_lam_Ecv < min_Ecv:
        target_lambda = lam
        min_Ecv = this_lam_Ecv
print 'best'
print 'ans lambda: ',target_lambda
print 'min Eval: ',min_Ecv


pl.bar(dic_19.keys(),dic_19.values())
pl.xlim([min(dic_19.keys()), max(dic_19.keys())])
pl.ylim(0, (max(dic_19.values())))
pl.xlabel('lambda')
pl.ylabel('Ecv')
pl.title('19')
pl.savefig('19')
