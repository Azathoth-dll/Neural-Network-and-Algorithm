# 学号:20003068

import numpy as npy
def W_improve(X,W0,learning_rate,Activation):
    length_X=len(X)
    if(length_X>1):
        X=npy.matrix(X)
        W0=npy.matrix(W0)
        Wi=W0
        i=0
        while(i<length_X):
            Xi=X[i]
            Xi=Xi.T
            neti=Wi*Xi
            neti=neti.tolist()
            Wi=Wi+learning_rate*Activation(neti[0][0])*(Xi.T)
            print('net%d='%(i+1),neti)
            print('W%d='%(i+1),Wi)
            i=i+1
    else:
        X=npy.matrix(X)
        X=X.T
        W0=npy.matrix(W0)
        W0=W0
        net1=W0*X
        net1=net1.tolist()
        W1=W0+learning_rate*Activation(net1[0][0])*(X.T)
        print('net1=',net1)
        print('W1=',W1)
    return

def sgn(net):
    if(net>0):
        return 1
    elif net==0:
        return 0
    else:
        return -1
    
def tanh(net):
    e=2.71828
    result=(1-e**(-net))/(1+e**(-net))
    return result

learning_rate=1
Activation=tanh
W0=[1,-1,0,0.5]
X=[[1,-2,1.5,0],[1,-0.5,-2,-1.5],[0,1,-1,1.5],[1,-1,0,0.5]]
W_improve(X,W0,learning_rate,Activation)
