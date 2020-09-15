import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import  linear_model

data = pd.read_csv(r'C:\Users\东邪\PycharmProjects\Coursera-ML-AndrewNg-Notes-master\code\ex1-linear regression\ex1data1.txt', header=None, names=['Population', 'Profit'])
print(data.head(8))
print(data.describe())
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()
data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:, 0:cols-1]#X是所有行去掉最后一列
y = data.iloc[:, cols-1:cols]#y是所有行，最后一列
#你开始了
model = linear_model.LinearRegression()
model.fit(X, y)
x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
'''
def computecost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

data.insert(0, 'Ones', 1)
print(data)

cols = data.shape[1]
X = data.iloc[:, 0:cols-1]#X是所有行去掉最后一列
y = data.iloc[:, cols-1:cols]

X = np.matrix(X.values) #X是字典？
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

print(X.shape)
print(y.shape)
print(theta.shape)

print(computecost(X, y, theta))#计算初始代价函数

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

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()


fig ,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
'''