import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as skL

X = [[2], [5], [10], [15]]
y = [[25000], [46000], [100000], [180000]]

''''
X => 6*1
y => 6*1
f = 1*6    6*1   =>     1*1
s = 1*6    6*1   =>     1*1
'''
test_x = 4


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        y = np.array(y)
        X = np.array(X)
        X = np.append(X, np.ones(X.shape), axis=1)

        first_part = np.matmul(X.transpose(), X)
        first_part_inv = np.linalg.inv(first_part)

        second_part = np.matmul(X.transpose(), y)
        coefs = np.matmul(first_part_inv, second_part)
        self.coefs = coefs.flatten()
        print(self.coefs)

    def predict(self, test_x):
        test_x = np.array([test_x, 1])
        return np.sum(test_x*self.coefs)


model = LinearRegression()
model.fit(X, y)
print(model.predict(12))

model = skL(fit_intercept=False)
model.fit(X, y)
print(model.coef_)
print(model.intercept_)
print(model.predict([[12]]))
