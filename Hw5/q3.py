import sys
sys.path.append('/Users/Plabo/libsvm')
from svmutil import *
import numpy as np
import math
from random import *

y=[-1,-1,-1,+1,+1,+1,+1]
x=[[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]]

problem = svm_problem(y,x)

params = svm_parameter('-t 1 -g 1 -r 1 -d 2')

'''model'''
model = svm_train(problem, params)
svm_save_model('model3',model)

f = open('model3')
if_reach_SV = False

W = [0,0]

for line in f.readlines():
    # print line
    if line.strip()=="SV":
        if_reach_SV = True
        continue
    if if_reach_SV:
        items = line.strip().split(' ')
        alpha_yn = float(items[0])
        # raw_input(items)
        '''W = alpha*y*z'''
        index = items[1].split(':')[0].strip()
        W[int(index)-1] = W[int(index)-1] + alpha_yn*float(items[1].split(':')[1].strip())
print W
