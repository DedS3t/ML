# linear regression implemeneted with gradient descent


'''
equation is in the form y=mx+c
we have to use gradient descent to find m and c
First we take the cost function which I decide to use MSE. 
E=1/n∑(y_i-ŷ_i)^2
which becomes
E=1/n∑(y_i-(mx_i+c))^2

Then we need to calculate the partial derivative of the cost function wrt m and c

D_m = -2/n∑x_i(y_i-ŷ_i)
D_c = -2/n ∑(y_i-ŷ_i)

Now after each epoch to update the values we would use to formulas:
where L is the learning rate
m=m-L*D_m
c=c-L*D_c
'''


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self,lr=0.0001):
        self.lr=lr 
        self.m=0
        self.c=0
    def costFunction(self,y,y_hat):
        return (1/len(y) * sum((y-y_hat)**2))

    def fit(self,X,y,epochs=1000):
        for i in range(epochs):
            y_hat=self.m*X+self.c
            self.m=self.m-self.lr*(-2/len(X) * sum(X*(y-y_hat)))
            self.c=self.c-self.lr*(-2/len(X) * sum(y-y_hat))
            if i%10==0:
                print(f'The error is {self.costFunction(y,self.m*X+self.c)}')
        return self.m,self.c


data=pd.read_csv('../data/data2.csv')
X=data.iloc[:,0]
y=data.iloc[:,1]
regression=LinearRegression()
m,c=regression.fit(X,y)
y_hat=m*X+c
plt.scatter(X,y)
plt.plot([min(X),max(X)],[min(y_hat),max(y_hat)],color='red')
plt.show()