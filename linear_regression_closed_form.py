import numpy as np
from matplotlib import pyplot as plt
import random

n = 20


X = 2 * np.random.rand(100,1)
Y = 4 + 3*X + np.random.rand(100,1)


def calculate_mse(m, c, x, y):

    err = 0
    for x_i,y_i in zip(x,y):
        err = err + (y_i - (m*x_i + c))**2

    return err/float(n)

X_c = np.c_[np.ones((100,1)), X]
X_transpose = np.transpose(X_c)

theta = np.linalg.inv(X_c.T.dot(X_c)).dot(X_c.T).dot(Y)
print(theta)

X_test = np.array([[0],[2]])
X_test_c = np.c_[np.ones((2,1)), X_test]
Y_test = X_test_c.dot(theta)

plt.plot(X_test, Y_test, 'b-')
plt.plot(X,Y, 'ro')
plt.show()