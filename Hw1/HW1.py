import random
import numpy as np
import pylab as pl

def Draw_Update_His(dic,name):
    pl.bar(dic.keys(),dic.values(), width=1)
    pl.xlim([min(dic.keys()), max(dic.keys())])
    pl.ylim(0, (max(dic.values())))
    pl.xlabel('Update number')
    pl.ylabel('Frequency')
    pl.title(name)
    pl.savefig(name)

def Draw_Error_His(dic,name):
    pl.bar(dic.keys(),dic.values(), width=0.001)
    pl.xlim([min(dic.keys()), max(dic.keys())])
    pl.ylim(0, (max(dic.values())))
    pl.xlabel('Error rate')
    pl.ylabel('Frequency')
    pl.title(name)
    pl.savefig(name)

def checkSin(X_W):
    temp = []
    for a in (np.sign(X_W)):
        if a == 0:
            temp.append(float(-1))
        else:
            temp.append(a)
    return np.array(temp)

def Pla_NaiveCycle(W_Matrix, X_Matrix, Y_Matrix):
    count = 0
    i = 0
    result = checkSin(np.dot(X_Matrix, W_Matrix))
    while(not np.array_equal(result, Y_Matrix)):
        if(np.not_equal(result, Y_Matrix)[i]):
            count = count + 1
            W_Matrix = W_Matrix + (X_Matrix[i] * Y_Matrix[i])
            result = checkSin(np.dot(X_Matrix, W_Matrix))
        i = i + 1
        if(i >= Y_Matrix.size):
            i = 0
    print "ansQ15 count:{0} index:{1}".format(count,i-1)

def Pla_PreRandom(W_Matrix, X_Matrix, Y_Matrix):
    count = 0
    random_list = random.sample(xrange(400), 400)
    result = checkSin(np.dot(X_Matrix, W_Matrix))
    while(not np.array_equal(result, Y_Matrix)):
        for i in random_list:
            if(np.not_equal(result, Y_Matrix)[i]):
                count = count + 1
                W_Matrix = W_Matrix + (X_Matrix[i] * Y_Matrix[i])
                result = checkSin(np.dot(X_Matrix, W_Matrix))
    return count

def Pla_PreRandom_LearningRate(W_Matrix, X_Matrix, Y_Matrix, LearningRate):
    count = 0
    random_list = random.sample(xrange(400), 400)
    result = checkSin(np.dot(X_Matrix, W_Matrix))
    while(not np.array_equal(result, Y_Matrix)):
        for i in random_list:
            if(np.not_equal(result, Y_Matrix)[i]):
                count = count + 1
                W_Matrix = W_Matrix + \
                    (LearningRate * (X_Matrix[i] * Y_Matrix[i]))
                result = checkSin(np.dot(X_Matrix, W_Matrix))
    return count

def Test(W_Matrix, X_Matrix, Y_Matrix):
    err = float(0)
    result = checkSin(np.dot(X_Matrix, W_Matrix))
    for a in np.not_equal(result, Y_Matrix):
        if a:
            err = err + 1
    return err / float(Y_Matrix.size)

def Pocket(W_Matrix, X_Matrix, Y_Matrix, Num_Update):
    count = 0
    W_Matrix_better = W_Matrix
    while(count < Num_Update):
        i = random.randint(0, 399)
        result = checkSin(np.dot(X_Matrix, W_Matrix))
        if(np.not_equal(result, Y_Matrix)[i]):
            count = count + 1
            W_Matrix = W_Matrix + (X_Matrix[i] * Y_Matrix[i])
            if(Test(W_Matrix, X_Matrix, Y_Matrix) <= Test(W_Matrix_better, X_Matrix, Y_Matrix)):
                W_Matrix_better = W_Matrix
    return W_Matrix_better

def Pla_limitNumber(W_Matrix, X_Matrix, Y_Matrix, Num_Update):
    count = 0
    while(count < Num_Update):
        i = random.randint(0, 399)
        result = checkSin(np.dot(X_Matrix, W_Matrix))
        if(np.not_equal(result, Y_Matrix)[i]):
            count = count + 1
            W_Matrix = W_Matrix + (X_Matrix[i] * Y_Matrix[i])
    return W_Matrix
# PLA Q15~17

Pla_X = []
Pla_Y = []
with open("PlaTrain.dat") as f:
    # data processing
    for line in f:
        Data = line.split()
        Pla_X.append(Data[0:4])
        Pla_Y.append(float(Data[4]))
    Pla_X = [[1] + x for x in Pla_X]
    Pla_X_Matrix = np.array(Pla_X, dtype=float)  # add X_0 = 1  size(400,5)
    Pla_Y_Matrix = np.array(Pla_Y, dtype=float)  # sieze(400,1)
f.close

# Q15
Pla_W_Matrix = np.array([0, 0, 0, 0, 0])
Pla_NaiveCycle(Pla_W_Matrix, Pla_X_Matrix, Pla_Y_Matrix)

# Q16
times = 2000
ansQ16 = 0
dicQ16 = {}
for i in range(times):
    Pla_W_Matrix = np.array([0, 0, 0, 0, 0])
    count = Pla_PreRandom(Pla_W_Matrix, Pla_X_Matrix, Pla_Y_Matrix)
    if count not in dicQ16:
        dicQ16[count] = 1
    else:
        dicQ16[count] = dicQ16[count] + 1
    ansQ16 = ansQ16 + count
print "ansQ16:{0}".format(ansQ16 / times)
Draw_Update_His(dicQ16,"Q16")

# Q17
times = 2000
eta = 0.5
ansQ17 = 0
dicQ17 = {}
for i in range(times):
    Pla_W_Matrix = np.array([0, 0, 0, 0, 0])
    count = Pla_PreRandom_LearningRate(Pla_W_Matrix, Pla_X_Matrix, Pla_Y_Matrix, eta)
    if count not in dicQ17:
        dicQ17[count] = 1
    else:
        dicQ17[count] = dicQ17[count] + 1
    ansQ17 = ansQ17 + count
print "ansQ17:{0}".format(ansQ17 / times)
Draw_Update_His(dicQ17,"Q17")

# Pocket Q18~Q20
Pocket_X = []
Pocket_Y = []
with open("PocketTrain.dat") as f:
    # data processing
    for line in f:
        Data = line.split()
        Pocket_X.append(Data[0:4])
        Pocket_Y.append(float(Data[4]))
    Pocket_X = [[1] + x for x in Pocket_X]
    Pocket_X_Matrix = np.array(
        Pocket_X, dtype=float)  # add X_0 = 1  size(400,5)
    Pocket_Y_Matrix = np.array(Pocket_Y, dtype=float)  # sieze(400,1)
f.close

Test_X = []
Y_head = []
with open("PocketTest.dat") as f:
    # data processing
    for line in f:
        Data = line.split()
        Test_X.append(Data[0:4])
        Y_head.append(float(Data[4]))
    Test_X = [[1] + x for x in Test_X]
    Test_X_Matrix = np.array(Test_X, dtype=float)  # add X_0 = 1  size(400,5)
    Y_head_Matrix = np.array(Y_head, dtype=float)  # sieze(400,1)
f.close

# Q18
times = 2000
ansQ18 = 0
Num_Update = 50
dicQ18 = {}
for i in range(times):
    Pocket_W_Matrix = np.array([0, 0, 0, 0, 0])
    Pocket_W_Matrix = Pocket(
        Pocket_W_Matrix, Pocket_X_Matrix, Pocket_Y_Matrix, Num_Update)
    count = Test(Pocket_W_Matrix, Test_X_Matrix, Y_head_Matrix)
    if count not in dicQ18:
        dicQ18[count] = 1
    else:
        dicQ18[count] = dicQ18[count] + 1
    ansQ18 = ansQ18 + count
print "ansQ18:{0}".format(ansQ18 / times)
Draw_Error_His(dicQ18,"Q18")

# Q19
times = 2000
ansQ19 = 0
Num_Update = 50
dicQ19 = {}
for i in range(times):
    Pocket_W_Matrix = np.array([0, 0, 0, 0, 0])
    Pocket_W_Matrix = Pla_limitNumber(
        Pocket_W_Matrix, Pocket_X_Matrix, Pocket_Y_Matrix, Num_Update)
    count = Test(Pocket_W_Matrix, Test_X_Matrix, Y_head_Matrix)
    if count not in dicQ19:
        dicQ19[count] = 1
    else:
        dicQ19[count] = dicQ19[count] + 1
    ansQ19 = ansQ19 + count
print "ansQ19:{0}".format(ansQ19 / times)
Draw_Error_His(dicQ19,"Q19")

# Q20
times = 2000
ansQ20 = 0
Num_Update = 100
dicQ20 = {}
for i in range(times):
    Pocket_W_Matrix = np.array([0, 0, 0, 0, 0])
    Pocket_W_Matrix = Pocket(
        Pocket_W_Matrix, Pocket_X_Matrix, Pocket_Y_Matrix, Num_Update)
    count = Test(Pocket_W_Matrix, Test_X_Matrix, Y_head_Matrix)
    if count not in dicQ20:
        dicQ20[count] = 1
    else:
        dicQ20[count] = dicQ20[count] + 1
    ansQ20 = ansQ20 + count
print "ansQ20:{0}".format(ansQ20 / times)
Draw_Error_His(dicQ20,"Q20")
