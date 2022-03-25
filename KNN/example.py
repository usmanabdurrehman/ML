import matplotlib.pyplot as plt
import numpy as np

X = [[1, 20000], [10, 40000], [2, 25000]]
y = [[0], [1], [0]]

y = np.array(y).ravel()
X = np.array(X)

test_y = [7, 35000]

colors = ['red', 'green']
plt.figure()
for i in range(len(colors)):
    plt.scatter(X[:, 0][y == i], X[:, 1][y == i], c=colors[i])
plt.scatter(test_y[0], test_y[1], c='black')
plt.legend(['Junior', 'Senior', 'Test Sample'], loc=0)
plt.title('Train data with test sample')
plt.xlabel('Yrs of experience')
plt.ylabel('Salary')
plt.show()

