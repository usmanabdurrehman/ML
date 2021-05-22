'''
Plotting into
'''

import matplotlib.pyplot as plt

'''
x1	x2		y
1	1		0
1	2		0
2	2		0
3	4		1
3	3.5		1
4	4.1		1
'''
X = [
	[1,1],
	[1,2],
	[2,2],
	[3,4],
	[3,3.5],
	[4,4.1]
	]
y = [[0],[0],[0],[1],[1],[1]]

plt.scatter(X[0][0],X[0][1],c='r')
plt.scatter(X[1][0],X[1][1],c='r')
plt.scatter(X[2][0],X[2][1],c='r')
plt.scatter(X[3][0],X[3][1],c='g')
plt.scatter(X[4][0],X[4][1],c='g')
plt.scatter(X[5][0],X[5][1],c='g')
plt.show()
