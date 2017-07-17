import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour


data = pd.read_csv('D:/data/Maths_dataset_Siraj/1_OrderDataset.csv', header = None, names=['X', 'Y'])
#print(data)
#print(data.head())

data.plot(kind='scatter', x='X', y='Y', figsize=(12,8))
show()

def computeCost(X, y, m, b):
    inner = np.power(((X * m + b) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, m, b):
    for i in range(iterations):
        cost = computeCost(X, y, m, b)
        print('cost is',cost)
        m = m - (learning_rate/len(X)) * np.sum((X * m) - y)
    return m, cost


# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
m = np.matrix(np.array([0]))
b = np.matrix(np.array([0]))
print(X.shape, m.shape, y.shape)
iterations = 50
learning_rate = 0.01

wt, cost = gradientDescent(X, y, m, b)
print('final cost:- ', cost, 'final weight:-', wt)

print(computeCost(X, y, wt, b))

x = np.linspace(data.X.min(), data.X.max(), 100)
f = (wt[0, 0] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.X, data.Y, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Predicted Profit vs. Population Size')
show()