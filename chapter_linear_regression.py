import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Basic linear regression with scikit-learn

# import data
df = pd.read_csv("https://bit.ly/3goOAnt", delimiter=",")
#print(df)

# extract x-values
X = df.values[:, :-1]

# extract y-values
Y = df.values[:, -1]

# find best-fit line
fit = LinearRegression().fit(X, Y)

# get m and b ( slope and y-intercept)
m = fit.coef_.flatten()
b = fit.intercept_.flatten()

# plot in diagram
plt.plot(X, Y, "o")
plt.plot(X, m*X+b)
#plt.show()


# calculate sum of squares

points = pd.read_csv("https://bit.ly/3goOAnt").itertuples()

m = 1.93939
b = 4.73333

sum_of_squares = 0

for p in points:
    actual_y = p.y
    predict_y = m * p.x + b
    residual = (actual_y - predict_y) ** 2
    sum_of_squares += residual
    
#print(sum_of_squares)

