import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import os

dirname = os.path.dirname(__file__)
utils_path = os.path.join(dirname, '../Utils')
sys.path.append(utils_path)

from utils import *

X = [1, 2, 3]
y = [0, 0, 0]

class Neuron:
    def __init__(self, n_epochs=100, activation_fn="sigmoid" loss_fn="cross_entropy" ):
        self.n_epochs = n_epochs

    def fit(self, X, Y):
        w = 7
        plt.figure()

        for epoch in range(self.n_epochs):
            losses = []
            for x, y in zip(X, Y):
                diff = self.__ml(w, x) - y
                losses.append(self.__cost(diff))
            loss = sum(losses)/len(losses)
            print('epoch: {} loss => {}'.format(epoch, loss))

            plt.scatter(w, loss, c='red')

            w = w - lr*self.__gradient(w)
            print('new weight value => {}'.format(w))

        y_loss = []
        x_weights = np.arange(-10, 11, 0.01)
        for weight in x_weights:
            losses = []
            for x, y in zip(X, Y):
                diff = self.__ml(weight, x) - y
                losses.append(self.__cost(diff))
            loss = sum(losses)/len(losses)
            y_loss.append(loss)

        plt.plot(x_weights, y_loss)
        plt.show()

    def __ml(self, weights, x):
        [intercept, slope] = weights
        lc = slope*x + intercept
        return self.__activation_mapper(lc)

    def __cost_mapper(self, actual, predicted):
         if(self.activation_fn == 'mse'):
            return mse(actual, predicted)
        elif(self.metric == 'mae'):
            return mae(actual, predicted)
        elif(self.metric == 'cross_entropy'):
            return cross_entropy(actual, predicted)

    def __gradient(self, weight):
        return 2*weight
    
    def __activation_mapper(self, x):
        if(self.activation_fn == 'relu'):
            return relu(x)
        elif(self.metric == 'relu6'):
            return relu6()
        elif(self.metric == 'sigmoid'):
            return sigmoid()
        elif(self.metric == 'tanh'):
            return tanh()
        
# TODO Fix Gradient


lr = 0.05

lr = Neuron(n_epochs=30, lr=0.05)
lr.fit(X, y)
