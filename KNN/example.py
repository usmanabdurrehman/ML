import matplotlib.pyplot as plt
import numpy as np

# This file shows the plots using matplotlib


def simple_regression_plot():
    X = [[1], [2], [3], [4], [5]]
    y = [[25000], [40000], [60000], [90000], [150000]]

    plt.figure()
    plt.scatter(X, y, c='blue')
    plt.scatter([3.5], [47000], c="black")
    plt.xlabel('Yrs of experience')
    plt.ylabel('Salary')
    plt.title('Train data with test sample')

    plt.show(block=False)


def simple_classification_plot():
    X = [[1, 20000], [2, 25000], [10, 40000]]
    y = [[0], [0], [1]]

    y = np.array(y).ravel()
    X = np.array(X)

    test_y = [7, 35000]

    colors = ['red', 'green', 'blue']
    plt.figure()
    for i in range(len(colors)):
        plt.scatter(X[:, 0][y == i], X[:, 1][y == i], c=colors[i])
    plt.scatter(test_y[0], test_y[1], c='black')
    plt.legend(['Junior', 'Senior', 'Test Sample'], loc=0)
    plt.title('Train data with test sample')
    plt.xlabel('Yrs of experience')
    plt.ylabel('Salary')
    plt.show(block=False)


def k_more_than_1():
    X = [[1, 10000], [2, 25000], [3, 35000]]
    y = [[0], [1], [2]]

    y = np.array(y).ravel()
    X = np.array(X)

    test_y = [1.5, 20000]

    colors = ['red', 'green', 'blue']
    plt.figure()
    for i in range(len(colors)):
        plt.scatter(X[:, 0][y == i], X[:, 1][y == i], c=colors[i])
    plt.scatter(test_y[0], test_y[1], c='black')
    plt.legend(['Junior', 'Associate', 'Senior', 'Test Sample'], loc=0)
    plt.title('Train data with test sample')
    plt.xlabel('Yrs of experience')
    plt.ylabel('Salary')
    plt.show(block=False)


# simple_classification_plot()
# k_more_than_1()
simple_regression_plot()

plt.show()
