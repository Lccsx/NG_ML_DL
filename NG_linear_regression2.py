import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  linear_model

path = r'C:\Users\东邪\PycharmProjects\Coursera-ML-AndrewNg-Notes-master\code\ex1-linear regression\ex1data2.txt'
data = pd.read_csv(path, names=['Size', 'Bedrooms', 'Price'])
print(data.head(5))

data = (data - data.mean())/data.std()  #归一化
print(data.head())

data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]#X是所有行去掉最后一列
y = data.iloc[:, cols-1:cols]

X = np.matrix(X.values) #X是字典？
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0, 0]))


'''
def computecost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1]) #有疑问，ravel
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computecost(X, y, theta)
    return  theta, cost

alpha = 0.01
iters = 1000

[g, cost] = gradientDescent(X, y, theta, alpha, iters)
computecost(X, y, g)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Irterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training EPOPch')
plt.show()
'''