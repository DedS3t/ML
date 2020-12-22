# A support vector machine implementation
# Finds a dicision boundry which creates a sepration beetwen 2 classes.
# How it works finds optimal values of w*(weights/normal) and b* (intercept). The optimul values are found by minimizing cost function.
# The svm is defined with f(x)=sign(w*x+b*)

import numpy as np
import pandas as pd 
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts 
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils import shuffle


# uses hingle loss function J(w)=1/2 ||w||^2 + C (1/N∑max(0,1-yi*(w*xi+b)))
def compute_cost(W,X,Y):
    # hinge loss
    N=X.shape[0]
    distances=1-Y*(np.dot(X,W))
    distances[distances<0]=0 # equivalent to max(0,distance)
    hinge_loss=reg_strength*(np.sum(distances)/N)
    cost=1/2*np.dot(W,W)+hinge_loss
    return cost


# Gradient of cost function is ∇w J(w)=1/N Σ {w if max(0,1-yi*(w*xi))=0 and   w-Cyixi otherwise }

def calculate_cost_gradient(W,X_batch,Y_batch):
    if type(Y_batch) == np.float64  or type(Y_batch)==np.int64:
        Y_batch=np.array([Y_batch])
        X_batch=np.array([X_batch])
    
    distance=1-(Y_batch*np.dot(X_batch,W))
    dw=np.zeros(len(W))
    for ind,d in enumerate(distance): 
        # if gradient is negative, this means its past the minimum. So we keep the weights the same
        if max(0,d)==0:
            di=W
        else:
            di=W-(reg_strength*Y_batch[ind]*X_batch[ind])
        dw+=di 
    dw=dw/len(Y_batch)
    return dw

# gradient decent algorithm explained
# 1. Find the gradient of cost function
# 2. move opposite of the gradient by rate i.e w’ = w’ — ∝(∇J(w’))
# repeat steps until convergence i.e we found w' where J(w) is smallest



def sgd(features,outputs):
    max_epochs=5000
    nth=0
    weights=np.zeros(features.shape[1])
    prev_cost=float("inf")
    cost_threshold=0.01
    # stochastic gradient descent
    for epoch in range(1,max_epochs):
        X,Y=shuffle(features,outputs)
        for ind,x in enumerate(X):
            ascent=calculate_cost_gradient(weights,x,Y[ind])
            weights=weights-(learning_rate*ascent)
        if epoch==2**nth or epoch==max_epochs-1:
            cost=compute_cost(weights,features,outputs)
            print("Epoch is:{} and Cost is: {}".format(epoch,cost))

            if abs(prev_cost-cost)<cost_threshold*prev_cost:
                return weights
            prev_cost=cost
            nth+=1
    return weights



def init():
    data=pd.read_csv('../data/data.csv')
    # transformas values into numeric
    diagnosis_map={'M':1,'B':-1}
    data['diagnosis']=data['diagnosis'].map(diagnosis_map)
    data.drop(data.columns[[-1,0]],axis=1,inplace=True)

    # normalization: converting a range of values into a certain interval ([-1,1] or [0,1]). Improves speed of learning
    Y=data.loc[:,'diagnosis']
    X=data.iloc[:,1:] 
    X_normalized=MinMaxScaler().fit_transform(X.values)
    X=pd.DataFrame(X_normalized)
    # split data
    X.insert(loc=len(X.columns),column='intercept',value=1) # add intercept column
    X_train,X_test,y_train,y_test=tts(X,Y,test_size=0.2)
    W=sgd(X_train.to_numpy(),y_train.to_numpy())
    y_test_predicted=np.array([])
    for i in range(X_test.shape[0]):
        yp=np.sign(np.dot(W,X_test.to_numpy()[i]))
        y_test_predicted=np.append(y_test_predicted,yp)
    print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
    print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
    print("precision on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))

reg_strength=10000
learning_rate=0.000001
init()
