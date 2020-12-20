# My implementation of the Least Squares method for lin reg in python 
# It is one of the simplest implementations but it does achieve good accuracy on simple data.
# HOW IT WORKS
# We are trying to find the coef b1 and b0  ŷ=b1x+b0
# b1 = Σ(x-x̅)(y-ȳ)/Σ(x-x̅)2
# b0 = y̅ - (x̅*b1) 


import matplotlib.pyplot as plt 
import numpy as np

class linearRegression:
    def __init__(self,Xval,Yval):
        self.X=Xval
        self.y=Yval
        self.b1=-1
        self.b0=-1
    def __getSquared(self,arr,mean):
        temp=0
        for i in arr:
            temp+=(i-mean)**2
        return temp
    def __mean(self,arr):
        return sum(arr)/len(arr)
    def __multiply(self,arr,arr2,xMean,yMean):
        temp=0
        for i in range(0,len(arr)):
            temp+=((arr[i]-xMean)*(arr2[i]-yMean))
        return temp
    def calculate(self):
        xMean=self.__mean(self.X)
        yMean=self.__mean(self.y)
        b1=(self.__multiply(self.X,self.y,xMean,yMean)/self.__getSquared(self.X,xMean))
        b0=yMean-(xMean*b1)
        self.b1=b1
        self.b0=b0
        print("y=",self.b1,"x+",b0,"")
        return (self.b0,self.b1)
    def predict(self,X):
        if self.b1==-1 and self.b0==-1:
            self.calculate()
        return ((self.b1*X)+self.b0)


if __name__=="__main__":
    X=[4,5,6,7]
    y=[41.04,43.04,45.25,47.27] 
    regression=linearRegression(X,y)
    b0,b1=regression.calculate()
    x_max=np.max(X)+100
    x_min=np.min(X)-100
    x=np.linspace(x_min,x_max,100)
    y1=b0+b1*x
    plt.plot(x,y1,color='#00ff00',label="Lin reg")
    plt.scatter(X,y,color="#ff0000",label="data")
    plt.show()
