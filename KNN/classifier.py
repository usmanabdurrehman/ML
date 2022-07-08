import matplotlib.pyplot as plt
import numpy as np
import sys
import os

dirname = os.path.dirname(__file__)
utils_path = os.path.join(dirname, '../Utils')
sys.path.append(utils_path)

from utils import euclidean, manhattan, minkowski

X = [[3.9, 39000], [4.6, 46000], [2.8, 28000], [3, 30000], [4.4, 44000]]
y = [[0], [1], [0], [0], [1]]

test_x = [4, 40000]

colors = ['red', 'green']


class KNNClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean', p=2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.y = np.array(y).ravel()
        self.X = np.array(X)

    def __plot_2D_data_and_test_sample(self, class_of_closest_sample, max_distance):
        plt.figure()
        plt.subplot(1, 2, 1)
        for i in range(len(np.unique(self.y))):
            plt.scatter(self.X[:, 0][self.y == i],
                        self.X[:, 1][self.y == i], c=colors[i])
        plt.scatter(test_x[0], test_x[1],
                    c='black',  marker='*')
        draw_circle = plt.Circle(
            (test_x[0], test_x[1]), max_distance, color='b', fill=False)
        plt.gca().add_artist(draw_circle)
        plt.legend(['Red', 'Green', 'Test Sample'], loc=0)
        plt.title('Train data with test sample k={}'.format(
            self.n_neighbors))
        plt.xlim([2, 6])
        plt.ylim([20000, 60000])

        plt.subplot(1, 2, 2)
        for i in range(len(np.unique(self.y))):
            plt.scatter(self.X[:, 0][self.y == i], self.X[:, 1]
                        [self.y == i], c=colors[i])
        plt.legend(['Red', 'Green'], loc=0)
        plt.title('Train data with test sample classified k={}'.format(
            self.n_neighbors))
        plt.scatter(test_x[0], test_x[1],
                    c=colors[class_of_closest_sample], marker='*')
        plt.xlim([2, 6])
        plt.ylim([20000, 60000])
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
            indices_of_neighbors = np.array(distance_list).argsort()[
                :self.n_neighbors]
            classes_of_closest_sample = self.y[indices_of_neighbors]
            _, counts = np.unique(
                classes_of_closest_sample, return_counts=True)
            are_classes_distributed_equally = np.max(counts) == np.min(counts)

            if(are_classes_distributed_equally):
                smallest_distance = min(distance_list)
                index_of_smallest_distance = distance_list.index(
                    smallest_distance)
                class_of_closest_sample = self.y[index_of_smallest_distance]
            else:
                class_of_closest_sample = max(
                    classes_of_closest_sample, key=list(classes_of_closest_sample).count)

            max_distance = max(sorted(distance_list)[:self.n_neighbors])
            self.__plot_2D_data_and_test_sample(
                class_of_closest_sample, max_distance)
            return class_of_closest_sample


for k in range(1, 6):
    knn = KNNClassifier(n_neighbors=k)
    knn.fit(X, y)
    knn.predict(test_x)

plt.show()
