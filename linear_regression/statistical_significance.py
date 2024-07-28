from scipy.stats import t
from math import sqrt
import pandas as pd

"""Proving that the correlation is not random"""

# load data
df = pd.read_csv("https://bit.ly/2KF29Bd")

# correlation coefficient
corr_coeff = df.corr(method="pearson")
# extract corr_coeff value from df
r = round(corr_coeff.iloc[0, 1], 6)

# set hypothesis
# H0: p = 0 (no correlation)
# H1: p != 0 (correlation)

# sample size
n = 10

# lower and upper critical value of t-distribution for 95% confidence with degrees of freedom n-1 = 9
lower_cv = t(n-1).ppf(.025)
upper_cv = t(n-1).ppf(0.975)

# calculate the test value
test_value = r / sqrt((1 - r**2) / (n - 2))

print(f"Test value = {test_value}")
print(f"Critical range: {lower_cv}, {upper_cv}")

if test_value > upper_cv or test_value < lower_cv:
    print("Correlation proven. Reject H0")
else:
    print("Correlation not proven. Failed to reject H0")

# calculate the p-value
if test_value > 0:
    p_value = 1.0 - t(n-1).cdf(test_value)
else:
    p_value = t(n-1).cdf(test_value)

p_value = p_value * 2
print(f"P-value: {p_value}")