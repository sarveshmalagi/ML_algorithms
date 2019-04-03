import numpy as np
from matplotlib import pyplot as plt

m=100
X = 2 * np.random.rand(m,1)
Y = 4 + 3*X + np.random.rand(m,1)

theta = np.random.randn(2,1)

n_iterations = 50
lr = 0.1

X_c = np.c_[np.ones((m,1)), X]

for i in range(n_iterations):
    gradients = (2/m)*X_c.T.dot(X_c.dot(theta)-Y)
    theta = theta - lr * gradients

print(theta)

X_test = np.array([[0],[2]])
X_test_c = np.c_[np.ones((2,1)), X_test]
Y_test = X_test_c.dot(theta)

plt.plot(X_test, Y_test, 'b-')
plt.plot(X,Y, 'ro')
plt.show()