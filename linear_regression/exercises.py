"""Übung 1: Führen Sie eine einfache lineare Regression aus, um die Werte m und b zu bestimmen,
die den Verlust minimieren (Summe der Quadrate). Datensatz: https://bit.ly/3C8JzrM"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
#plt.show()

"""Übung 2: Berechnen Sie den Korrelationskoeffizienten 
und die statistische Signifikanz dieser Daten (bei 95% Vertrauen). Ist die Korrelation sinnvoll?"""

corr_coeff = df.corr(method="pearson")
print(corr_coeff)