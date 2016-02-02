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
    ## Q18
    fw = open('ans18','w')
    x,y,x_size = readData('../features.train',0)

    for c in range(-3,2,+1):
        # print c
        '''problem'''
        problem = svm_problem(y,x)

        '''parameter'''
        print 'c',c
        c_par = '-c {0}'.format(math.pow(10,c))
        parameter = c_par +' -t 2 -g 100'
        params = svm_parameter(parameter)

        '''model'''
        model = svm_train(problem, params)
        model_name = 'model'+str(c)
        svm_save_model(model_name,model)

        f = open(model_name)
        W = [0 for i in range(x_size)]
        if_reach_SV = False
        for line in f.readlines():
            # print line
            if line.strip()=="SV":
                if_reach_SV = True
                continue
            if if_reach_SV:

                items = line.strip().split(' ')
                alpha_yn = float(items[0])
                for i in range(x_size):
                    '''W = alpha*y*z'''
                    W[i] = W[i] + alpha_yn*float(items[i+1].split(':')[1].strip())
                '''store free support vector'''
                OneSample = True
                if abs(alpha_yn) != math.pow(10,c):
                    x_test = []
                    if (float(items[0])) >= 0:
                        y_test = [1]
                    else:
                        y_test = [-1]
                    if OneSample:
                        OneSample = False
                        for i in range(x_size):
                            x_test.append(float(items[i+1].split(':')[1].strip()))
        '''caculate Distance'''
        test_model = svm_load_model(model_name)
        p_label, p_acc, p_val = svm_predict(y_test, [x_test], test_model)
        W = np.array(W)
        p_val_dis = p_val[0][0]
        print p_val_dis
        fw.write('log10_C:{0},Distance:{1}\n'.format(c,abs(float(p_val_dis))/np.dot(W,np.transpose(W) )))

