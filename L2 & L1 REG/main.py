# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# def predict_poly(x, koeff):
#     res = 0
#     xx = [x ** (len(koeff) - n - 1) for n in range(len(koeff))]
#
#     for i, k in enumerate(koeff):
#         res += k * xx[i]
#
#     return res
#
#
# x = np.arange(0, 10.1, 0.1)
#
# y = 1/(1 + 10 * x**2)
# x_train, y_train = x[::2], y[::2]
#
# N = len(x)
#
# z_train = np.polyfit(x_train, y_train, 5)
# print(z_train)

def l2_reg():
    x = np.arange(0, 10.1, 0.1)
    y = np.array([a ** 3 - 10 * a ** 2 + 3 * a + 500 for a in x])
    x_train, y_train = x[::2], y[::2]
    N = 13
    L = 20
    X = np.array([[a ** n for n in range(N)] for a in x])
    IL = np.array([[L if i == j else 0 for j in range(N)]for i in range(N)])
    IL[0][0] = 0
    X_train = X[::2]
    Y = y_train

    A = np.linalg.inv(X_train.T @ X_train + IL)
    w = Y @ X_train @ A
    print(w)
    yy = [np.dot(w, x) for x in X]
    plt.plot(x, yy)
    plt.plot(x, y)
    plt.grid(True)
    plt.show()

# l2_reg()


def l1_reg():

    def loss(w, x, y):
        M = np.dot(w, x) * y
        return 2 / (1 + np.exp(M))

    def df(w, x ,y):
        L1 = 1.0
        M = np.dot(w, x) * y
        return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y + L1 * w

    x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
    x_train = [x + [10 * x[0], 10 * x[1], 5 * (x[0] + x[1])] for x in x_train]
    x_train = np.array(x_train)
    y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

    fn = len(x_train[0])
    n_train = len(x_train)
    w = np.zeros(fn)
    nt = 0.00001
    lm = 0.01
    N = 5000

    Q = np.mean([loss(x, w, y) for x, y in zip(x_train, y_train)])
    Q_plot = [Q]

    for i in range(N):
        k = np.random.randint(0, n_train - 1)
        ek = loss(w, x_train[k], y_train[k])
        w = w - nt * df(w, x_train[k], y_train[k])
        Q = lm * ek + (1 - lm) * Q
        Q_plot.append(Q)

    Q = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])
    print(w)
    print(Q)
    plt.plot(Q_plot)
    plt.grid(True)
    plt.show()

l1_reg()
