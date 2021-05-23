'''
Intro + Plotting
'''

'''
#######################################################
###############   Classifcation Plot   ################
#######################################################
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
Sufi
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

x = pd.DataFrame({'x1':[1,1,2,3,4,4],'x2':[1,2,2,4,3.5,4.1],'y':[0,0,0,1,1,1]})
x.to_csv('classification_data.csv',index=False)

plt.figure()
plt.scatter(X[0][0],X[0][1],c='r')
plt.scatter(X[1][0],X[1][1],c='r')
plt.scatter(X[2][0],X[2][1],c='r')
plt.scatter(X[3][0],X[3][1],c='g')
plt.scatter(X[4][0],X[4][1],c='g')
plt.scatter(X[5][0],X[5][1],c='g')
# Can display the legend using this command
# plt.legend(['Red','Red','Red','Green','Green','Green'],loc=0)
plt.title('Manual Way for Classification Plotting')

'''
Instead of doing this manually we can also make use of numpy
for assigning a color to a specific class easily
'''

colors = ['red','green']
y = np.array(y).ravel()
X = np.array(X)
plt.figure()
for i in range(len(colors)):
	plt.scatter(X[:,0][y==i],X[:,1][y==i],c=colors[i])
plt.legend(['Class1','Class2'],loc=0)
plt.title('Optimal Way for Classification Plotting')


'''
#######################################################
###############   Regression Plot  ####################
#######################################################
'''

X = [
	[1],
	[1.3],
	[2],
	[3],
	[3],
	[4]
	]
y = [[0],[1],[2],[2.5],[3],[2]]

x = pd.DataFrame({'Experience':[1,1.3,2,3,3,4],'Salary':[0,1,2,2.5,3,2]})
x.to_csv('regression_data.csv',index=False)

plt.figure()
plt.plot(X,y)
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Regression Plot')
plt.legend(['Salary'],loc=0)

plt.show()
