import matplotlib.pyplot as plt
import numpy as np
import math


X = [1, 2, 3]
y = [0, 0, 0]


class LogisticRegression:
    def __init__(self, max_iter=100):
        self.max_iter = max_iter

    def fit(self, X, Y):
        w = 7
        plt.figure()

        for epoch in range(self.max_iter):
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
        return 1/(1 + np.exp(-lc))

    def __cost(self, actual, predicted):
        return -(actual*math.log(predicted) + (1-actual)*math.log(1-predicted))

    def __gradient(self, weight):
        return 2*weight

# TODO Fix Gradient

lr = 0.05

lr = LogisticRegression(n_epochs=30, lr=0.05)
lr.fit(X, y)
