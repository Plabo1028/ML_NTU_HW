import numpy as np
from cvxopt import matrix
from cvxopt import solvers

y=[-1,-1,-1,+1,+1,+1,+1]
x=[[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]]
X = np.array(x)
y = np.array(y)
print 'X.shape:{0}\nX:{1}\n'.format(X.shape,X) #(7,2)
print 'Y.shape:{0}\nY:{1}\n'.format(y.shape,y) #(7,)
def kernel(X1,X2):
    inner = (1+np.dot(X1,X2))
    return np.dot(inner,inner)


if __name__ == '__main__':
    K = np.zeros([7,7])

    for i in range(0,7):
        for j in range(0,7):
            K[i,j]=kernel(X[i],X[j])
    print 'K',K

    y_tmp = np.matrix([-1,-1,-1,+1,+1,+1,+1])

    P = matrix(np.outer(y_tmp,y_tmp) * K)
    q = matrix(np.ones(7) * -1)
    A = matrix(y_tmp,tc='d')
    b = matrix(0.0)
    G = matrix(np.diag(np.ones(7) * -1))
    h = matrix(np.zeros(7))
    solution = solvers.qp(P, q, G, h, A, b)
    print solution['x']



