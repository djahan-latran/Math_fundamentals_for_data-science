import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# loading the data
df = pd.read_csv("https://bit.ly/3cIH97A", delimiter=",")

# get variable values
X = df.values[:, :-1]
Y = df.values[:, -1]

# split train- and testdata
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

# create linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# calculate the coefficient of determination
result = model.score(X_test, Y_test)

print(f"The coefficient of determination is: {result}")

# cross validation with 3 splits
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
results = cross_val_score(model, X, Y, cv=kfold)

print(f"The coefficients of determination of the cross validation are: {results}")
print(f"MSE: mean = {results.mean()}, stde = {results.std()}")

