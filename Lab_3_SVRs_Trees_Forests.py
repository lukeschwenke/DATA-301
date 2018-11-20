import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

students_math = pd.read_csv("./studentmat.csv")
students_port = pd.read_csv("./studentpor.csv")
# Predict WALC (1 most 5 least amount of alc consumed)

students_port.shape
all_student_rows = [students_math, students_port]
all_students = pd.concat(all_student_rows, ignore_index = True)
all_students

X = all_students[["age", "address", "traveltime", "failures", "higher", "internet",
                  "romantic", "famrel", "freetime", "goout", "absences"]].values
from sklearn.preprocessing import LabelEncoder
discreteCoder_X = LabelEncoder()

X[:,1] = discreteCoder_X.fit_transform(X[:,1]) #Fits data to 0 and 1 then transform 
X[:,4] = discreteCoder_X.fit_transform(X[:,4])
X[:,5] = discreteCoder_X.fit_transform(X[:,5])
X[:,6] = discreteCoder_X.fit_transform(X[:,6])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

y = all_students[["Walc"]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1693)

scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

from sklearn.svm import SVR 
svr_regression = SVR(kernel = "linear", epsilon = 1.0)
svr_regression.fit(X_train, y_train)

new_studentA = [[18, 1, 3, 3, 0, 0, 1, 2, 5, 2, 5]]
new_studentA

new_student_scaledA = scale_X.transform(new_studentA)
studentA_prediction = svr_regression.predict(new_student_scaledA)
studentA_prediction

print("First new student (A):" + str(studentA_prediction))

new_studentB = [[18, 0, 3, 3, 0, 0, 1, 2, 1, 1, 5]]
new_student_scaledB = scale_X.transform(new_studentB)
studentB_prediction = svr_regression.predict(new_student_scaledB)
print("First new student (B):" + str(studentB_prediction))

from sklearn import tree

DT_regression = tree.DecisionTreeRegressor(random_state = 1693, max_depth = 3)
DT_regression.fit(X_train, y_train)

tree.export_graphviz(DT_regression, out_file="tree.dot", feature_names = ["age", "address", "traveltime", "failures", "higher", "internet",
                  "romantic", "famrel", "freetime", "goout", "absences"])

studentA_prediction_RT = DT_regression.predict(new_student_scaledA)
print("First new student:" + str(studentA_prediction_RT))

studentB_prediction_RT = DT_regression.predict(new_student_scaledB)
print("First new student:" + str(studentB_prediction_RT))

from sklearn.ensemble import RandomForestRegressor
RF_regression = RandomForestRegressor(n_estimators = 100, random_state = 1693)
RF_regression.fit(X_train, y_train)

studentA_prediction_RF = RF_regression.predict(new_student_scaledA)
print("First new student:" + str(studentA_prediction_RF))

studentB_prediction_RF = RF_regression.predict(new_student_scaledB)
print("First new student:" + str(studentB_prediction_RF))

from sklearn.metrics import mean_absolute_error 
rf_MAD = mean_absolute_error(y_test, RF_regression.predict(X_test))
rf_MAD

RT_MAD = mean_absolute_error(y_test, DT_regression.predict(X_test))
SVR_MAD = mean_absolute_error(y_test, svr_regression.predict(X_test))

print("Random Forest:" + str(rf_MAD))
print("Decision Tree:" + str(RT_MAD))
print("Supper Vector Regression:" + str(SVR_MAD))

#Problem Set Question
new_studentFIND = [[20, 1, 3, 1, 0, 1, 1, 2, 3, 2, 5]]
new_student_scaledFIND = scale_X.transform(new_studentFIND)
studentFIND_prediction = svr_regression.predict(new_student_scaledFIND)
print("First new student FIND:" + str(studentFIND_prediction))

studentFIND_prediction_RF = RF_regression.predict(new_student_scaledFIND)
print("First new student:" + str(studentFIND_prediction_RF))

