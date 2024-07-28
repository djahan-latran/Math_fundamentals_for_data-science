import pandas as pd
import numpy as np

data = pd.read_csv("https://bit.ly/2KF29Bd", header=0)

X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

n = data.shape[0]

m = 0.0
b = 0.0

sample_size = 1  # size of sample
L = 0.0001  # learning rate
epochs = 100000  # amount of iterations

# stochastic gradient descent
for i in range(epochs):
    idx = np.random.choice(n, sample_size, replace=False)
    x_sample = X[idx]
    y_sample = Y[idx]

    Y_pred = m * x_sample + b

    # derivative of loss function d/dm (derivative of the function that calculates the sum of squares)
    D_m = (-2 / sample_size) * sum(x_sample * (y_sample - Y_pred))

    # derivative of loss function d/db (derivative of the function that calculates the sum of squares)
    D_b = (-2 / sample_size) * sum(y_sample - Y_pred)

    m = m - L * D_m  # update m
    b = b - L * D_b  # update b

    if i % 100 == 0:
        print(i, m, b)

print(f"y = {m}x + {b}")