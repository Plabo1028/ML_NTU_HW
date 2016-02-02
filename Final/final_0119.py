import scipy.io as sio
import numpy as np
import sys
from sklearn import preprocessing
sys.path.append('/home/plabo/libsvm')
from svmutil import *
import random
import math
import timeit

def load_data_fromMat(path):
    Data = sio.loadmat(path)
    try:
        X = Data['Feature']
        y = Data['Label']
        X = np.float64(X)
        y = np.float64(y)
        return X,y
    except:
        X = Data['Feature']
        X = np.float64(X)
        return X

def ch_Xarray2list(data):
    data_list = []
    for line in data:
        one_data = []
        for i in line:
            one_data.append(i)
        data_list.append(one_data)
    return data_list

def ch_yarray2list(data):
    data_list = []
    for line in data:
        for i in line:
            data_list.append(int(line))
    return data_list

def minmax(data):
    minmax = preprocessing.MinMaxScaler()
    Data_minmax = minmax.fit_transform(np.array(data))
    return Data_minmax

def scale(data):
    # print data[:3]
    scale = preprocessing.StandardScaler()
    Data_scale = scale.fit_transform(np.array(data))
    # raw_input(Data_scale[:3])
    return Data_scale

def maxabs(data):
    maxabs = preprocessing.MaxAbsScaler()
    Data_maxabs = maxabs.fit_transform(np.array(data))
    return Data_maxabs

def shuffle_val(X,y,ratio):
    data = []
    data_size = X.shape[0]
    feature_size = X.shape[1]
    train_data_size = int(data_size * ratio)

    for i in range(data_size):
        tmp_X = X[i]
        one_line = np.concatenate((tmp_X,y[i]))
        data.append(one_line)

    random.shuffle(data)

    split_index = [0,int(data_size*ratio),data_size]

    X = np.zeros((data_size,feature_size))
    y = np.zeros((data_size,1))
    for i in range(data_size):
        X[i] = data[i][:feature_size]
        y[i] = data[i][feature_size]

    X_train = np.array(X[split_index[0]:split_index[1]])
    X_val = np.array(X[split_index[1]:split_index[2]])
    y_train = np.array(y[split_index[0]:split_index[1]])
    y_val = np.array(y[split_index[1]:split_index[2]])

    return X_train,y_train,X_val,y_val

def predict(model_path,X_test_minmax_list):

    def output(path,p_label,model_path):
        w = open(str('submit'+str(model_path)+'.csv'),'w')
        with open(path) as f:
            index = 0
            for line in f:
                id,label=line.split(',')
                # print id
                w.write(id)
                w.write(',')

                # if float(p_label[index]) >= 0.53:
                #     tmp = 1
                # elif float(p_label[index]) < 0:
                #     tmp = 0
                # else:
                #     tmp = float(p_label[index])/0.53
                # w.write(str(tmp))

                w.write(str(p_label[index]))
                w.write('\n')
                index += 1

    m = svm_load_model(model_path)
    p_label, p_acc, p_val = svm_predict([0]*len(X_test_minmax_list),X_test_minmax_list,m)
    # print p_label[:10]
    # print p_acc
    output('Data/sampleSubmission.csv',p_label,model_path)

def data_preparation(X,y,ratio,shuffle_type):
    '''Shuffle => cut into train,validation => change2list'''
    if shuffle_type == 'random':
        X_train,y_train,X_val,y_val = shuffle_val(X,y,ratio)
    elif shuffle_type == 'dependLabel':
        X_train,y_train,X_val,y_val = shuffle_val_dependLabel(X,y,ratio)
    X_train_list = ch_Xarray2list(X_train)
    y_train_list = ch_yarray2list(y_train)
    X_val_list = ch_Xarray2list(X_val)
    y_val_list = ch_yarray2list(y_val)
    return X_train_list,y_train_list,X_val_list,y_val_list

def cal_normal_different(Data_path = 'Data/ML_Train17.mat',iter_time=100):

    minmax_err = 0
    err = 0
    scale_err = 0
    maxabs_err = 0

    for i in range(iter_time):

        print '\n==== Times:{0}  Generate new data  ============================================'.format(i)
        X,y = load_data_fromMat(Data_path)
        index = np.random.random_integers(0,X.shape[0]-1,10000)
        X=X[index]
        y=y[index]
        print 'X.shape:',X.shape
        print 'y.shape:',y.shape
        X_minmax = minmax(X)
        X_scale = scale(X)
        X_maxabs = maxabs(X)
        print 'Shuffle => cut into train,validation => change2list....................'
        X_train_list,y_train_list,X_val_list,y_val_list = data_preparation(X,y,0.9)
        scale_X_train_list,scale_y_train_list,scale_X_val_list,scale_y_val_list = data_preparation(X_scale,y,0.9)
        minmax_X_train_list,minmax_y_train_list,minmax_X_val_list,minmax_y_val_list = data_preparation(X_minmax,y,0.9)
        maxabs_X_train_list,maxabs_y_train_list,maxabs_X_val_list,maxabs_y_val_list = data_preparation(X_maxabs,y,0.9)

        '''parameter'''
        parameter = '-g 100.0 -t 2 -h 0 -m 10000 -c 1.0'
        params = svm_parameter(parameter)
        ''' train '''
        print '=============  no normalized  ================================================'
        problem = svm_problem(y_train_list,X_train_list)
        model = svm_train(problem, params)
        p_label, p_acc, p_val = svm_predict(y_val_list, X_val_list, model)
        err += (100.0-p_acc[0])

        print '=============  minmax normalized  ============================================'
        minmax_problem = svm_problem(minmax_y_train_list,minmax_X_train_list)
        minmax_model = svm_train(minmax_problem, params)
        minmax_p_label, minmax_p_acc, minmax_p_val = svm_predict(minmax_y_val_list, minmax_X_val_list, minmax_model)
        minmax_err += (100.0-minmax_p_acc[0])

        print '=============  scale normalized  ============================================='
        scale_problem = svm_problem(scale_y_train_list,scale_X_train_list)
        scale_model = svm_train(scale_problem, params)
        scale_p_label, scale_p_acc, scale_p_val = svm_predict(scale_y_val_list, scale_X_val_list, scale_model)
        scale_err += (100.0-scale_p_acc[0])

        print '=============  maxabs normalized  ============================================'
        maxabs_problem = svm_problem(maxabs_y_train_list,maxabs_X_train_list)
        maxabs_model = svm_train(maxabs_problem, params)
        maxabs_p_label, maxabs_p_acc, maxabs_p_val = svm_predict(maxabs_y_val_list, maxabs_X_val_list, maxabs_model)
        maxabs_err += (100.0-maxabs_p_acc[0])

    return err/float(iter_time),minmax_err/float(iter_time),scale_err/float(iter_time),maxabs_err/float(iter_time),parameter

def shuffle_val_dependLabel(X,y,ratio):
    data = []
    data_size = X.shape[0]
    feature_size = X.shape[1]
    label0 = []
    label1 = []

    for i in range(data_size):
        tmp_X = X[i]
        one_line = np.concatenate((tmp_X,y[i]))
        data.append(one_line)

        if y[i] == 0:
            label0.append(i)
        elif y[i] == 1:
            label1.append(i)

    random.shuffle(label1)
    random.shuffle(label0)

    train_index = label1[:int(len(label1)*ratio)]
    val_index = label1[int(len(label1)*ratio):]
    train_index += label0[:int(len(label0)*ratio)]
    val_index += label0[int(len(label0)*ratio):]

    X_train = np.zeros((len(train_index),feature_size))
    X_val = np.zeros((len(val_index),feature_size))
    y_train = np.zeros((len(train_index),1))
    y_val = np.zeros((len(val_index),1))
    print len(train_index)
    print len(val_index)
    print X_train.shape
    print X_val.shape
    print y_train.shape
    print y_val.shape
    # raw_input('pause')
    count = 0
    for i in train_index:
        # print i
        try:
            X_train[count] = data[i][:feature_size]
            y_train[count] = data[i][feature_size]
        except:
            print i
            print data[i]
            print X_train[count]
            raw_input('pause')
        count += 1
    count = 0
    for j in val_index:
        X_val[count] = data[j][:feature_size]
        y_val[count] = data[j][feature_size]
        count += 1

    return X_train,y_train,X_val,y_val

def cal_shuffle_different(Data_path = 'Data/ML_Train17.mat',iter_time=100):
    random_err = 0
    de_rerr = 0

    for i in range(iter_time):

        print '\n==== Times:{0}  Generate new data  ============================================'.format(i)
        X,y = load_data_fromMat(Data_path)
        index = np.random.random_integers(0,X.shape[0]-1,10000)
        X=X[index]
        y=y[index]

        print 'X.shape:',X.shape
        print 'y.shape:',y.shape
        print 'Shuffle => cut into train,validation => change2list....................'
        X_train_list,y_train_list,X_val_list,y_val_list = data_preparation(X,y,0.9,'random')
        de_X_train_list,de_y_train_list,de_X_val_list,de_y_val_list = data_preparation(X,y,0.9,'dependLabel')

        '''parameter'''
        parameter = '-g 100.0 -t 2 -h 0 -m 10000 -c 1.0'
        params = svm_parameter(parameter)
        ''' train '''
        print '=============  random shuffle  ================================================'
        problem = svm_problem(y_train_list,X_train_list)
        model = svm_train(problem, params)
        p_label, p_acc, p_val = svm_predict(y_val_list, X_val_list, model)
        random_err += (100.0-p_acc[0])

        print '=============  shuffle depend Label  =========================================='
        de_problem = svm_problem(de_y_train_list,de_X_train_list)
        de_model = svm_train(de_problem, params)
        de_p_label, de_p_acc, de_p_val = svm_predict(de_y_val_list, de_X_val_list, de_model)
        de_rerr += (100.0-de_p_acc[0])

    return random_err/float(iter_time),de_rerr/float(iter_time),parameter


def train():
    # print 'Load data.........................................'
    # X,y = load_data_fromMat('Data/train_29feature.mat')

    # index = np.random.random_integers(0,X.shape[0],5000)
    # X=X[index]
    # y=y[index]
    # print 'X.shape:',X.shape
    # print 'y.shape:',y.shape

    # print 'Minmax......................................'
    # X_minmax = minmax(X)
    # y_minmax = minmax(y)

    # print 'Shuffle and cut into train,validation....................'
    # X_train,y_train,X_val,y_val= shuffle_val(X_minmax,y_minmax,0.9)
    # print 'X_train.shape:',X_train.shape
    # print 'y_train.shape:',y_train.shape
    # print 'X_val.shape:',X_val.shape
    # print 'y_val.shape:',y_val.shape


    # print 'Data prepocess for svm....................................'
    # X_train_list = ch_Xarray2list(X_train)
    # y_train_list = ch_yarray2list(y_train)
    # X_val_list = ch_Xarray2list(X_val)
    # y_val_list = ch_yarray2list(y_val)

    # print 'SVM.................................................'
    # acc = 0
    # best_g = 0
    # best_c = 0
    # best_d = 0
    # degree = 3
    # error = float('inf')
    # for g in range(-5,5,+1):

    #     for c in range(-5,5,+1):

    #         for d in range(3,7,+1):

    #             parameter = ''
    #             g_par = '-g {0}'.format(math.pow(degree,g))
    #             parameter = g_par +' -t 2 -h 0 -m 12000'
    #             c_par = ' -c {0}'.format(math.pow(degree,c))
    #             parameter = parameter + c_par
    #             d_par = ' -d {0}'.format(str(d))
    #             parameter = parameter + d_par
    #             print '==============================================================='
    #             params = svm_parameter(parameter)

    #             ''' train '''
    #             problem = svm_problem(y_train_list,X_train_list)
    #             model = svm_train(problem, params)

    #             '''test'''
    #             p_label, p_acc, p_val = svm_predict(y_val_list, X_val_list, model)
    #             print 'Acc:{0}% Error:{1} when parameter {2}'.format(p_acc[0],p_acc[1],parameter)

    #             if p_acc[0] > acc:
    #                 best_d = d
    #                 best_g = g
    #                 best_c = c
    #                 acc = p_acc[0]
    #                 model_name = 'model/model'+str(p_acc[0])
    #                 svm_save_model(model_name,model)
    #             # if error > p_acc[1]:
    #             #     error = p_acc[1]
    #             #     best_g = g
    #             #     best_c = c
    #             #     best_d = d
    #             #     model_name = 'model/model'+str(p_acc[1])
    #             #     svm_save_model(model_name,model)
    # print 'Best Acc:{0}% Error:{1} when -g {2} -c {3} -d {4}'.format(acc,error,math.pow(degree,best_g),math.pow(degree,best_c),best_d)


    X,y = load_data_fromMat('Data/train_29feature.mat')
    # X = X[:100]
    # y = y[:100]
    X_minmax = minmax(X)
    X_train_list = ch_Xarray2list(X_minmax)
    y_train_list = ch_yarray2list(y)
    # parameter = ''
    # g_par = '-g {0}'.format(math.pow(degree,best_g))
    # parameter = g_par +' -t 3 -h 0 -m 12000'
    # c_par = ' -c {0}'.format(math.pow(degree,best_c))
    # parameter = parameter + c_par
    # d_par = ' -d {0}'.format(str(d))
    # parameter = parameter + d_par
    parameter = '-g 0.037 -c 27.0 -t 3 -m 12000'
    params = svm_parameter(parameter)
    problem = svm_problem(y_train_list,X_train_list)
    model = svm_train(problem, params)
    svm_save_model('model/best_model_with'+str(parameter),model)

if __name__ == '__main__':
    start_time = timeit.default_timer()
    # w = open('Shuffle_Record.txt','w')
    ''' calculate normalized between minmax maxabs scale'''
    # err, minmax_err, scale_err, maxabs_err, parameter = cal_normal_different(iter_time=1000)
    # w.write('Err: '+str(err)+'%\n')
    # w.write('Minmax_Err: '+str(minmax_err)+'%\n')
    # w.write('Scale_Err: '+str(scale_err)+'%\n')
    # w.write('Maxabs_Err: '+str(maxabs_err)+'%\n')
    # w.write('Parameter: '+str(parameter)+'\n')
    '''train'''
    # train()
    '''predict'''
    X_test = load_data_fromMat('Data/test_29feature.mat')
    X_test_minmax = minmax(X_test)
    X_test_minmax_list = ch_Xarray2list(X_test_minmax)
    predict('best_model_with-g 0.037 -c 27.0 -t 3 -m 12000',X_test_minmax_list)
    ''' calculate different shuffle algorithm'''
    # random_err, de_rerr, parameter = cal_shuffle_different(iter_time=1000)
    # w.write('Random_Err: '+str(random_err)+'%\n')
    # w.write('Depend_label_Err: '+str(de_rerr)+'%\n')
    # w.write('Parameter: '+str(parameter)+'\n')

    end_time = timeit.default_timer()
    running_time = (end_time-start_time)
    print 'Running_time: '+str(running_time)+' second\n'
    # w.write('\nRunning_time: '+str(running_time)+' second\n')

