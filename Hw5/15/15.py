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
    # Q15
    fw = open('ans15','w')
    x,y,x_size = readData('../features.train',0)
    # print x
    print y
    raw_input('pause')
    for c in range(-6,3,+2):
        '''problem'''
        problem = svm_problem(y,x)

        '''parameter'''
        print 'c',c
        c_par = '-c {0}'.format(math.pow(10,c))
        parameter = c_par +' -t 0 '
        params = svm_parameter(parameter)

        '''model'''
        model = svm_train(problem, params)
        model_name = 'model'+str(c)
        svm_save_model(model_name,model)

        f = open(model_name)
        W = [0 for i in range(x_size)]
        sum_of_alpha = 0
        if_reach_SV = False
        for line in f.readlines():
            # print line
            if line.strip()=="SV":
                if_reach_SV = True
                continue
            if if_reach_SV:
                items = line.strip().split(' ')
                alpha_yn = float(items[0])
                # raw_input(items)
                sum_of_alpha = sum_of_alpha + abs(alpha_yn)
                for i in range(x_size):
                    '''W = alpha*y*z'''
                    W[i] = W[i] + alpha_yn*float(items[i+1].split(':')[1].strip())
        W = np.array(W)
        fw.write('log10_C:{0},|W|:{1}\n'.format(c,np.dot(W,np.transpose(W))))


