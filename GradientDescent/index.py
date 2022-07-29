import matplotlib.pyplot as plt
import numpy as np

X = [1, 2, 3]
y = [0, 0, 0]

# x*w = y
'''
1 * w = 0
2 * w = 0
3 * w = 0

'''


class GradientDescent:
    def __init__(self, n_epochs, lr):
        self.n_epochs = n_epochs
        self.lr = lr

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

            w = w - self.lr*self.__gradient(w)
            print('new weight value => {}'.format(w))

        # For plotting
        y_loss = []
        x_weights = np.arange(-10, 11, 0.01)
        for weight in x_weights:
            losses = []
            for x, y in zip(X, Y):
                diff = self.__ml(weight, x) - y
                losses.append(self.__cost(diff))
            loss = sum(losses)/len(losses)
            y_loss.append(loss)
        plt.xlabel('Weights')
        plt.ylabel('Loss')
        plt.title('Weights vs Loss')
        plt.plot(x_weights, y_loss)
        plt.show()

    def __ml(self, w, l):
        return l*w

    def __cost(self, diff):
        return diff**2

    def __gradient(self, weight):
        return 2*weight


gd = GradientDescent(n_epochs=100, lr=0.05)
gd.fit(X, y)
