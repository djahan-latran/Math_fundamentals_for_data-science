"""Übung 1: Führen Sie eine einfache lineare Regression aus, um die Werte m und b zu bestimmen,
die den Verlust minimieren (Summe der Quadrate). Datensatz: https://bit.ly/3C8JzrM"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import matplotlib.pyplot as plt
from scipy.stats import t
from math import sqrt
import numpy as np

df = pd.read_csv("https://bit.ly/3C8JzrM")

X = df.values[:, :-1]
Y = df.values[:, -1]

model = LinearRegression()
fit = model.fit(X, Y)

m = fit.coef_.flatten()
b = fit.intercept_.flatten()
print(f"slope:{m}, intercept: {b}")

plt.plot(X,Y,"o")
plt.plot(X, m*X+b)
plt.show()


"""Übung 2: Berechnen Sie den Korrelationskoeffizienten 
und die statistische Signifikanz dieser Daten (bei 95% Vertrauen). Ist die Korrelation sinnvoll?"""

corr_coeff = df.corr(method="pearson")

# extract the value of r
r = corr_coeff.iloc[1, 0]
print(f"r = {r}")

# sample size -> n-1 degrees of freedom
n = df.shape[0]

# lower and upper critical values (range of no rejection)
lower_cv = t(n-1).ppf(0.025)
upper_cv = t(n-1).ppf(0.975)
print(f"Lower cv: {lower_cv}, upper cv: {upper_cv}")

# calc the test value
test_value = r / sqrt((1 - r**2) / (n - 2))
print(f"Test value: {test_value}")

if test_value > upper_cv or test_value < lower_cv:
    print(f"Correlation proven: Reject H0")
else:
    print(f"Correlation not proven: Failed to reject H0")

p_value = 1.0 - t(n-1).cdf(test_value)
print(f"P-value: {p_value}")


"""Übung 3: Wenn ich eine Vorhersage für x=50 treffe, wie groß ist dann das 95%ige Vorhersageintervall
für den vorhergesagten Wert von y?"""

# get points
points = list(pd.read_csv("https://bit.ly/3C8JzrM").itertuples())

# prediction interval for x=50
x_mean = np.mean(X)
x_0 = 50

t_value = t(n-2).ppf(.975)


# calc the standard error
standard_error = sqrt(sum((p.y - (m * p.x + b))**2 for p in points) / (n-2))
# calc the margin of error
margin_of_error = t_value * standard_error * sqrt(1 + (1/n) + (n * (x_0 - x_mean)**2) /
                                                  (n * sum(p.x **2 for p in points) - sum(p.x for p in points)**2))
# predicated value at x=50
predicted_y = int(m * x_0 + b)

# calc upper and lower value
lower_boundary = predicted_y - margin_of_error
upper_boundary = predicted_y + margin_of_error

print(f"prediction interval: {lower_boundary} - {upper_boundary}")


"""Übung 4: Beginnen Sie Ihre Regression von vorn und führen Sie eine Aufteilung in Trainings- und Testdaten durch.
Experimentieren Sie mit Kreuzvalidierung. Liefert die lineare Regression gute und konsistente Ergebnisse
für die Testdaten? Warum bzw. warum nicht?"""

df = pd.read_csv("https://bit.ly/3C8JzrM")

X = df.values[:, :-1]
Y = df.values[:, -1]

kfold = KFold(n_splits=3, random_state=7, shuffle=True)

model = LinearRegression()

results = cross_val_score(model, X, Y, cv=kfold)

print(f"mean: {results.mean()}, std: {results.std()}")

