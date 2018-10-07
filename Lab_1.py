
#Importing Data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("L1Data.csv")
data

X = data[["Class", "Age", "Funds"]].values #.values just takes the content, not Class Age Functions 0 1 2 etc.
X

Y = data[["Sale"]].values
X[1][0]
Y

from sklearn.preprocessing import Imputer 
imputing_configuration = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputed_values = imputing_configuration.fit(X[:,[1,2]])
X[:,[1,2]] = imputed_values.transform(X[:,[1,2]])
X

from sklearn.preprocessing import LabelEncoder 
discreteCoder_X = LabelEncoder()
X[:,0] = discreteCoder_X.fit_transform(X[:,0])
X

from sklearn.preprocessing import OneHotEncoder 
discreteCoder_X_dummies = OneHotEncoder(categorical_features = [0]) # Only apply to first column
X = discreteCoder_X_dummies.fit_transform(X).toarray()
X

discreteCoder_Y = LabelEncoder()
Y = discreteCoder_Y.fit_transform(Y)
Y

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 1693)
X_train
X_test
Y_train
Y_test

from sklearn.preprocessing import StandardScaler

scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)

X_test = scale_X.transform(X_test)
X_train
X_test

