import pandas as pd
from math import sqrt

"""calculating the standard error of the estimate"""

# load data and format them into a list of pandas tuples
points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())

# how many pairs there are
n = len(points)

# regression line that was calculated beforehand
m = 1.939
b = 4.733

# calculating the standard error of the estimate (Se)
S_e = sqrt((sum((p.y - (m * p.x + b))**2 for p in points)) / (n - 2))

print(S_e)