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
    # Q16
    fw = open('ans16','w')
    x,y,x_size = readData('../features.train',8)
    for c in range(-6,3,+2):
        '''problem'''
        problem = svm_problem(y,x)

        '''parameter'''
        print 'c',c
        c_par = '-c {0}'.format(math.pow(10,c))
        parameter = c_par +' -t 1 -g 1 -r 1 -d 2 -e 0.1 -h 0 -q'
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
                # raw_input('')
                for i in range(x_size):
                    '''W = alpha*y*z'''
                    W[i] = W[i] + alpha_yn*float(items[i+1].split(':')[1].strip())
        test_model = svm_load_model(model_name)
        p_label, p_acc, p_val = svm_predict(y, x, test_model)
        fw.write('log10_C:{0},Ein:{1}\n'.format(c,str(1-p_acc[0]/100.0)))
