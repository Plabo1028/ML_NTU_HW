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
    # for min_max in x_max_min:
    #    print "min:"+str(min_max[0])+";max:"+str(min_max[1])
    # scaling x & y
    # for non_scaled in x:
    #     for i in range(len(non_scaled)):
    #         non_scaled[i] = (non_scaled[i]-x_max_min[i][0]) / (x_max_min[i][1]-x_max_min[i][0])
    # return x,y,x_size


max_sum_of_alhpha = float("-inf")
    for target_class in range(0,10,2):
        x,y,x_size = readData('../features.train',target_class)
        # raw_input('after readdata')
        problem = svm_problem(y,x)
        # set SVM parameter
        params = svm_parameter('-c 0.01 -t 1 -g 1 -r 1')
        model = svm_train(problem, params)
        svm_save_model('model',model)
        # get W
        f = open('model')
        W = [0 for i in range(x_size)];
        sum_of_alpha = 0
        if_reach_SV = False
        for line in f.readlines():
            print line
            if line.strip()=="SV":
                if_reach_SV = True
                continue
            if if_reach_SV:
                items = line.strip().split(' ')
                print 'line',line
                print 'items',items
                alphan_yn = float(items[0])
                print 'alphan_yn',alphan_yn
                # raw_input('')
                sum_of_alpha = sum_of_alpha + abs(alphan_yn)
                for i in range(x_size):
                    W[i] = W[i] + alphan_yn*float(items[i+1].split(':')[1].strip())
        # w=np.array(W)
        # fw.writelines(str(np.dot(w,w.T))+'\t')
        fw.writelines(str(sum_of_alpha)+'\n')
        max_sum_of_alhpha = sum_of_alpha if sum_of_alpha>max_sum_of_alhpha else max_sum_of_alhpha
        f.close()
        #test_x,test_y,test_x_size = read_input_data('test.dat',target_class)
        #if x_size!= test_x_size: sys.exit(-1)
        p_label, p_acc, p_val = svm_predict(y, x, model)
        fw.writelines("class:"+str(target_class)+";Ein:"+str(1-p_acc[0]/100.0)+'\n')
    fw.writelines(str(max_sum_of_alhpha)+'\n')
    fw.close()
