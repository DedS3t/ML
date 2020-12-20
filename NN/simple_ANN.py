

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
dig=load_digits()
onehot_target=pd.get_dummies(dig.target)
x_train, x_val, y_train, y_val=train_test_split(dig.data,onehot_target,test_size=0.1)

def sigmoid(s):
    return 1/(1+np.exp(-s))
def softmax(s):
    exps=np.exp(s-np.max(s,axis=1,keepdims=True))
    return exps/np.sum(exps,axis=1,keepdims=True)
def sigmoid_derv(x):
    return x*(1-x)
def cross_entropy(pred,real):
    n_samples=real.shape[0]
    res=pred-real
    return res/n_samples
def error(pred,real):
    n_samples=real.shape[0]
    logp=-np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss=np.sum(logp)/n_samples
    return loss
class NN:
    def __init__(self, x, y):
        self.x=x
        self.y=y
        neurons=128 
        self.lr=0.5
        ip_dim=x.shape[1]
        op_dim=y.shape[1]
        self.w1=np.random.randn(ip_dim, neurons)
        self.b1=np.zeros((1, neurons))         
        self.w2=np.random.randn(neurons, neurons)
        self.b2=np.zeros((1, neurons))
        self.w3=np.random.randn(neurons, op_dim)
        self.b3=np.zeros((1, op_dim))

    def feedforward(self):
        # feed forward process is simple - sigmoid((inputs*weights)+bias)
        self.a1=sigmoid(np.dot(self.x,self.w1)+self.b1)
        self.a2=sigmoid(np.dot(self.a1,self.w2)+self.b2)
        self.a3=sigmoid(np.dot(self.a2,self.w3)+self.b3)
    def backprop(self):
        loss=error(self.a3,self.y)
        print("Error :",loss)
        a3_delta=cross_entropy(self.a3,self.y) # w3
        z2_delta=np.dot(a3_delta,self.w3.T)
        a2_delta=z2_delta*sigmoid_derv(self.a2) # w2
        z1_delta=np.dot(a2_delta,self.w2.T)
        a1_delta=z1_delta*sigmoid_derv(self.a1) # w1
        self.w3-=self.lr*np.dot(self.a2.T, a3_delta)
        self.b3-=self.lr*np.sum(a3_delta, axis=0, keepdims=True)
        self.w2-=self.lr*np.dot(self.a1.T, a2_delta)
        self.b2-=self.lr*np.sum(a2_delta, axis=0)
        self.w1-=self.lr*np.dot(self.x.T, a1_delta)
        self.b1-=self.lr*np.sum(a1_delta, axis=0)
    def predict(self,data):
        self.x=data
        self.feedforward()
        return self.a3.argmax()

model=NN(x_train/16.0,np.array(y_train))
epochs=1500
for x in range(epochs):
    model.feedforward()
    model.backprop()
def get_acc(x,y):
    acc=0
    for xx,yy in zip(x,y):
        s=model.predict(xx)
        if s==np.argmax(yy):
            acc+=1
    return acc/len(x)*100
print("Training accuracy : ", get_acc(x_train/16, np.array(y_train)))
print("Test accuracy : ", get_acc(x_val/16, np.array(y_val)))
'''
z = xw + b                
a = sig(z) or softmax(z)   
c = -(y*log(a3))
From the output all we have to do is find the error and how much does each weight influence the output. 
 -find the derivative of cost function w.r.t w3.
'''