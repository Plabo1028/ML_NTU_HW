import numpy as np

'''train data'''
X_train = []
Y_train = []
with open("hw4_train.dat") as f:
    for line in f:
        '''X_train'''
        x = line.split()[:-1]
        tmp = ['1.0']
        for a in x:
            # print a
            tmp.append(a)
        X_train.append(tmp)

        '''Y_train'''
        Y_train.append(line.split()[-1])
X_train = np.float32(X_train)
Y_train = np.float32(Y_train)
print 'X_train.shape: ',X_train.shape
print 'Y_train.shape: ',Y_train.shape

'''test data'''
X_test = []
Y_test = []
with open("hw4_test.dat") as f:
    for line in f:
        '''X_test'''
        x = line.split()[:-1]
        tmp = ['1.0']
        for a in x:
            # print a
            tmp.append(a)
        X_test.append(tmp)

        '''Y_test'''
        Y_test.append(line.split()[-1])
X_test = np.float32(X_test)
Y_test = np.float32(Y_test)
print 'X_test.shape: ',X_test.shape
print 'Y_test.shape: ',Y_test.shape

raw_input('pause')
lam_ans13 = 11.26

Z_inv = np.linalg.inv(np.dot(X_train.transpose(),X_train)+lam_ans13*np.eye(X_train.shape[1]))
W = np.dot(np.dot(Z_inv,X_train.transpose()),Y_train)

scores = np.dot(W, X_test.transpose())
predicts = np.where(scores>=0,1.0,-1.0)
Eout = sum(predicts!=Y_test)
print (Eout*1.0) / predicts.shape[0]

# def calculate_E(w, x, y):
#     scores = np.dot(w, x.transpose())
#     predicts = np.where(scores>=0,1.0,-1.0)
#     Eout = sum(predicts!=y)
#     return (Eout*1.0) / predicts.shape[0]


# lam = 10.0
# train_limit =200
# def calculate_W_rigde_regression(x,y,LAMBDA):
#     Z_v = np.linalg.inv(np.dot(x.transpose(),x)+LAMBDA*np.eye(x.shape[1]))
#     return  np.dot(np.dot(Z_v,x.transpose()),y)
# for item_lam in xrange(2,-11,-1):
#     lam = np.power(10.0,item_lam)
#     W = np.dot(np.asmatrix(np.dot(X_train[:train_limit].T, X_train[:train_limit]) + lam * Iden).I,  np.dot(X_train[:train_limit].T, Y_train[:train_limit]))

#     Ein = 0
#     Eval = 0
#     Eout = 0
#     for item in xrange(0,train_limit):
#         Erro = (np.sign(X_train[item]*W * Y_train[item]) !=1)[0] * 1
#         Ein += Erro[0][0]

#     for item in xrange(train_limit,200):
#         Erro = (np.sign(X_train[item]*W * Y_train[item]) !=1)[0] * 1
#         Eval += Erro[0][0]

#     for item in xrange(1000):
#         Erro = (np.sign(X_test[item]*W * Y_test[item]) !=1)[0] * 1
#         Eout += Erro
#     print "item_lam: ", item_lam,
#     print "Ein: ", Ein*1.0/train_limit,
#     #print "Eval", Eval*1.0/(200-train_limit),
#     print "Eout", Eout*1.0/1000
