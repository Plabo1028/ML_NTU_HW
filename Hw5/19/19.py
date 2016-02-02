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
    ## Q19
    fw = open('ans19','w')
    x,y,x_size = readData('../features.train',0)
    test_x,test_y,test_x_size = readData('../features.test',0)

    for g in range(0,5,+1):
        print g
        '''problem'''
        problem = svm_problem(y,x)

        '''parameter'''
        g_par = '-g {0}'.format(math.pow(10,g))
        parameter = g_par +' -t 2 -c 0.1'
        params = svm_parameter(parameter)

        '''model'''
        model = svm_train(problem, params)
        model_name = 'model'+str(g)
        svm_save_model(model_name,model)

        '''test'''
        test_model = svm_load_model(model_name)
        p_label, p_acc, p_val = svm_predict(test_y, test_x, test_model)
        fw.write("Eout:"+str(1-p_acc[0]/100.0)+'\n')

