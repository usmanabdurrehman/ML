import matplotlib.pyplot as plt
import numpy as np

from utils import euclidean, manhattan, minkowski

X = [[1], [1.7], [2], [3], [5], [6.8]]
y = [[0], [0.9], [1.3], [3], [5.1], [7]]

test_x = 4


class KNNRegressor:
    def __init__(self, n_neighbors=5, metric='euclidean', p=2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.y = np.array(y).ravel()
        self.X = np.array(X)

    def __plot_2D_data_and_test_sample(self, test_x, target):
        plt.figure()
        plt.scatter(self.X, self.y, c='blue')
        plt.scatter(test_x, target, c="black")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Train data with test sample k={}'.format(self.n_neighbors))

        plt.show(block=False)

    def __calculate_distances(self, test_x):
        if(self.metric == 'euclidean'):
            return [euclidean(test_x, sample) for sample in self.X]
        elif(self.metric == 'manhattan'):
            return [manhattan(test_x, sample) for sample in self.X]
        elif(self.metric == 'minkowski'):
            return [minkowski(test_x, sample, self.p) for sample in self.X]

    def predict(self, test_x):
        try:
            some_var = self.X
        except AttributeError:
            raise Exception(
                "This KNeighborsClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        else:
            distance_list = self.__calculate_distances(test_x)
            indicesOfNeighbors = np.array(distance_list).argsort()[
                :self.n_neighbors]
            valuesOfNeighbors = self.y[indicesOfNeighbors]
            target = np.sum(valuesOfNeighbors)/len(valuesOfNeighbors)
            self.__plot_2D_data_and_test_sample(test_x, target)
            return target


knn = KNNRegressor(n_neighbors=2)
knn.fit(X, y)
print(knn.predict(test_x))

plt.show()
