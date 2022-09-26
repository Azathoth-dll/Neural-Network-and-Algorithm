# 20003068

import numpy as np
def w_improve(W0,X,activation,learning_rate,D):
    #D=np.matrix(D)
    W0=np.matrix(W0)
    X=np.matrix(X)
    W1=W0
    while(1):
        i=0
        while(i<2):
            net1=W1*(X[i].T)
            output=D[i]-activation(net1)
            W1=W1+learning_rate*output*X[i]
            print('net%d='%(i+1),net1)
            if(output==0):
                break
            else:
                i=i+1
        if(output==0):
            break

def sgn(net):
    if(net>0):
        return 1
    elif net==0:
        return 0
    else:
        return -1

W0=[0,1,0]
X=[[2,1,-1],[0,-1,-1]]
D=[-1,1]
activation=sgn
learning_rate=1
w_improve(W0,X,activation,learning_rate,D)
