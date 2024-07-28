import pandas as pd

# import data
df = pd.read_csv("https://bit.ly/2KF29Bd")

# calculate the correlation coefficient of the x and y data
correlations = df.corr(method="pearson")

# the closer the cc is to 1, the better the data correlates positively
print(correlations)
