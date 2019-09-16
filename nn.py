#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:21:09 2019

@author: sid
"""
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1,keepdims=1)

def feedforward(X,parameters):
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    Z = sigmoid(np.dot(X, W1 )+b1)
    Y = softmax(np.dot(Z,W2)+b2)
    cache=(Z,Y)
    return cache
def cost(cache, y_train):
    cost=0
    Z,Y= cache
    m,n=y_train.shape
    for row in range(m):
         cost =cost -np.dot(y_train[row], np.transpose(np.log(Y[row]))) -np.dot((1-y_train[row]),np.transpose(np.log(1-Y[row])))
    print("cost is :",cost/m)
    return 0
def backprop(X,y,parameters,cache):
    (Z,Y)=cache
    W2=parameters["W2"]

    Y_grad=Y-y
    W2_grad=np.dot(Z.T, Y_grad)
    Z_grad=np.dot(Y_grad,W2.T )
    W1_grad=np.dot(X.T,Z_grad)
    b1_grad=np.sum(Z_grad, axis=0, keepdims=1)
    b2_grad=np.sum(Y_grad, axis=0, keepdims=1)
    gradients={"W2_grad":W2_grad, "b2_grad":b2_grad, "W1_grad":W1_grad, "b1_grad":b1_grad }
    return gradients
def neuralnet(X_train, y_train, alpha, parameters, epochs):
    for i in range(epochs):
        print(i)
        cache=feedforward(X_train,parameters)
        cost(cache,y_train)
        gradients=backprop(X_train,y_train,parameters,cache) 
        parameters["W1"]-=alpha*gradients["W1_grad"]
        parameters["W2"]-=alpha*gradients["W2_grad"]
        parameters["b1"]-=alpha*gradients["b1_grad"]
        parameters["b2"]-=alpha*gradients["b2_grad"]
    return cache
    
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
y_train = y_train.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_train= encoder.fit_transform(y_train)
W1=np.random.randn(4,10)
W2=np.random.randn(10,3)
b1=np.random.randn(1,10)*0.01
b2=np.random.randn(1,3)*0.01
parameters={}
parameters["W1"]=W1
parameters["W2"]=W2
parameters["b1"]=b1
parameters["b2"]=b2
alpha=0.001
epochs=1000
X_train=(X_train-X_train.mean())/X_train.std()
cache=neuralnet(X_train,y_train, alpha,parameters,epochs)
