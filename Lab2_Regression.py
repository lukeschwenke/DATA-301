
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

piazza_data = pd.read_csv("./labData.csv")
piazza_data

X = piazza_data[["contributions"]].values
y = piazza_data[["Grade"]].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1693)
X_train
y_train

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

plt.scatter(X_train, y_train)
plt.show()

plt.scatter(X_train, y_train, color="black")
plt.plot(X_train, regression.predict(X_train), color="red")
plt.show()

regression.predict(X_train)

plt.scatter(X_train, y_train, color="black")
plt.plot(X_train, regression.predict(X_train), color ="red")
plt.title("Piazza Contributions and Grades (Training Data)")
plt.xlabel("Piazza Contributions")
plt.ylabel("Grade")
plt.show()

y_predictions = regression.predict(X_test)
y_predictions
[y_test, y_predictions]

plt.scatter(X_train, y_train, color="black")
plt.scatter(X_test, y_test, color="blue")
plt.plot(X_train, regression.predict(X_train), color ="red")
plt.title("Piazza Contributions and Grades")
plt.xlabel("Piazza Contributions")
plt.ylabel("Grade")
plt.show() #How well does red line represent blue points

piazza_data 

#Multiple Linear Regression
X = piazza_data[["contributions", "days online", "views", "questions", "answers"]].values
y = piazza_data[["Grade"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1693)

# Feature Scaling (only do for X variables)
from sklearn.preprocessing import StandardScaler

scale_X = StandardScaler()
X_train_scaled = scale_X.fit_transform(X_train)
X_test_scaled = scale_X.transform(X_test)

multiple_regression = LinearRegression()
multiple_regression.fit(X_train_scaled, y_train)

y_predictions = multiple_regression.predict(X_test_scaled)
[y_test, y_predictions]

from sklearn.preprocessing import PolynomialFeatures
X = piazza_data[["contributions", "days online", "views", "questions", "answers"]].values
y = piazza_data[["Grade"]].values

poly_data = PolynomialFeatures(degree=2) #Creating a second order poly --> Keeps data as originally were and then adds them all squared
X_poly = poly_data.fit_transform(X) 

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.25, random_state=1693)
poly_reg = LinearRegression()
poly_reg.fit(X_train, y_train)

y_predictions = poly_reg.predict(X_test)
y_predictions

X = piazza_data[["contributions"]].values
y = piazza_data[["Grade"]].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1693)

scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

poly_data = PolynomialFeatures(degree=2)
poly_reg = LinearRegression()
poly_reg.fit(poly_data.fit_transform(X_train), y_train)

plt.scatter(X_test, y_test, color="black", label="Truth")
plt.scatter(X_test, lin_reg.predict(X_test), color = "green", label = "Linear")
plt.scatter(X_test, poly_reg.predict(poly_data.fit_transform(X_test)), color="blue", label="Poly")
plt.xlabel("Piazza Contributions")
plt.ylabel("Grade")
plt.legend()
plt.show()

