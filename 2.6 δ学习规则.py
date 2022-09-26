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
            #print(net1)
            net1=net1.tolist()
            net1=net1[0][0]
            output=D[i]-activation(net1,0)
            W1=W1+learning_rate*output*(activation(net1,1))*X[i]
            #print('net%d='%(i+1),net1)
            print('W%d='%(i+1),W1)
            if(output==0):
                break
            else:
                i=i+1
        if(output==0):
            break

def Sigmoid(net,n):
    e=2.71828
    if(n==0):
        result=1/(1+e**(-net))
        return result
    elif(n==1):
        result=1/(2*(1-Sigmoid(net,0)))
        return result
    
W0=[1,0,1]
X=[[2,0,-1],[1,-2,-1]]
D=[-1,1]
activation=Sigmoid
learning_rate=0.25
w_improve(W0,X,activation,learning_rate,D)