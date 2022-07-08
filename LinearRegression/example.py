import matplotlib.pyplot as plt
import numpy as np

# This file shows the plots using matplotlib


def find_y(coefs, x):
    [b1, b0] = coefs
    return x*b1 + b0


def simple_regression_plot():
    X = [[2], [5], [10], [15]]
    y = [[25000], [46000], [100000], [180000]]

    plt.figure()
    plt.scatter(X, y, c='blue')
    plt.xlabel('Yrs of experience')
    plt.ylabel('Salary')
    plt.title('Plotting X vs y')

    plt.show(block=False)


def simple_regression_plot_with_best_fit_line():
    coefs = [11959.18367347, -7923.46938776]

    X = [[2], [5], [10], [15]]
    y = [[25000], [46000], [100000], [180000]]

    plt.figure()
    plt.scatter(X, y, c='blue')
    plt.plot([1, 17], [find_y(coefs, 1), find_y(coefs, 17)], c="green")
    plt.plot([3, 20], [3000, 170000], c="red")
    plt.plot([5, 12], [2500, 200000], c="red")
    plt.xlabel('Yrs of experience')
    plt.ylabel('Salary')
    plt.title('Trying to find best fit line')

    plt.show(block=False)


def simple_regression_plot_with_error_difference():
    coefs = [10000, -5000]

    X = np.array([[2], [5], [10], [15]]).ravel()
    y = np.array([[25000], [46000], [100000], [180000]]).ravel()

    plt.figure()
    plt.scatter(X, y, c='blue')
    plt.plot([1, 17], [find_y(coefs, 1), find_y(coefs, 17)], c="red")
    errors = []
    for _X, _y in zip(X, y):
        errors.append((_y-find_y(coefs, _X))**2)
        # print(_y-find_y(coefs, _X))
        plt.plot([_X, _X], [_y, find_y(coefs, _X)], c='blue')
    print(np.sum(errors))
    plt.xlabel('Yrs of experience')
    plt.ylabel('Salary')
    plt.title('Random Line with LSS')

    plt.show(block=False)


def simple_regression_plot_with_best_fit_error_difference():
    coefs = [11959.18367347, -7923.46938776]

    X = np.array([[2], [5], [10], [15]]).ravel()
    y = np.array([[25000], [46000], [100000], [180000]]).ravel()

    plt.figure()
    plt.scatter(X, y, c='blue')
    plt.plot([1, 17], [find_y(coefs, 1), find_y(coefs, 17)], c="green")
    errors = []
    for _X, _y in zip(X, y):
        errors.append((_y-find_y(coefs, _X))**2)
        # print(_y-find_y(coefs, _X))
        plt.plot([_X, _X], [_y, find_y(coefs, _X)], c='blue')
    print(np.sum(errors))
    plt.xlabel('Yrs of experience')
    plt.ylabel('Salary')
    plt.title('Best Fit line with LSS')

    plt.show(block=False)


# simple_regression_plot()
# simple_regression_plot_with_best_fit_line()
simple_regression_plot_with_error_difference()
simple_regression_plot_with_best_fit_error_difference()

plt.show()
