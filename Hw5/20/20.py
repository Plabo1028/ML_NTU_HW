import sys
sys.path.append('/Users/Plabo/libsvm')
from svmutil import *
import numpy as np
import math
from random import *

def readData(path, target_class):
    x = []
    y = []
    x_size = 2
    f = open(path)
    # record each x column's max value for scaling
    min_max = []
    for i in range(x_size):
        min_max.append([float("inf"),float("-inf")])
    # raw_input(min_max)
    for line in f.readlines():
        datas = line.strip().split(' ')
        items = []
        for field in datas:
            if field!='':
                items.append(field)
        # print items
        tmp_x = []
        for i in range(1,len(items)):
            val = float(items[i])
            if val < min_max[i-1][0]:
                min_max[i-1][0] = val
            elif val > min_max[i-1][1]:
                min_max[i-1][1] = val
            tmp_x.append(val)
        y.append(1 if float(items[0])==target_class else -1)
        x.append(tmp_x)
    f.close()
    return x,y,x_size

if __name__ == '__main__':
    ## Q20
    fw = open('ans20','w')
    T = 100
    test_size = 1000
    ans20 = {}
    x,y,x_size = readData('../features.train',0)
    for i in range(T):
        test_indexs = np.random.random_integers(0,len(x)-1,test_size)
        train_x = []
        train_y = []
        for i in range(len(x)):
            if not (i in test_indexs):
                train_x.append(x[i])
                train_y.append(y[i])
        test_x = np.array(x)[test_indexs].tolist()
        test_y = np.array(y)[test_indexs].tolist()
        problem = svm_problem(train_y,train_x)

        min_Eval = float("inf")
        min_gamma = -1

        for g in range(0,5,+1):
            print g
            '''parameter'''
            g_par = '-g {0}'.format(math.pow(10,g))
            parameter = g_par +' -t 2 -c 0.1'
            params = svm_parameter(parameter)

            '''model'''
            model = svm_train(problem, params)

            '''test'''
            p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
            Eval = 1-p_acc[0]/100.0
            fw.write("gamma:"+str(g)+"\t Eval:"+str(Eval)+'\n')

            if min_Eval > Eval:
                min_Eval = Eval
                min_gamma = g
        if ans20.has_key(min_gamma):
            ans20[min_gamma] += 1
        else:
            ans20[min_gamma] = 1
    for k,v in ans20.items():
        fw.writelines("gamma:"+str(k)+" times:"+str(v)+'\n')
