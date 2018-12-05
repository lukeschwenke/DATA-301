import pandas as pd
dta = pd.read_csv("dailyinmatesincustody.csv")

dta = dta.dropna(subset =  ["GENDER"])
dta.head()

dta["AGE"].isnull().sum()

X = dta[["AGE", "GENDER"]]
y = dta[["INFRACTION"]]

X = pd.get_dummies(X)
X = X.drop(columns = ["GENDER_M"])

y = pd.get_dummies(y)
y = y.drop(columns=["INFRACTION_N"])

X = X.values
y = y.values

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
dt_classifier = DecisionTreeClassifier(random_state=1693, max_depth=2, min_samples_leaf=2)
dt_classifier.fit(X,y)

k_fold = cross_val_score(estimator = dt_classifier, X=X, y=y, cv=10, scoring="accuracy")

k_fold.mean()

params = [{'max_depth':[10, 15, 20], 'min_samples_leaf':[2,4,6,8,10,12,14,16,18,20]}]

gSearch = GridSearchCV(estimator = dt_classifier,
                      param_grid = params,
                      scoring = "accuracy",
                      cv = 5)

gSearch_results = gSearch.fit(X,y)
gSearch_results.best_params_
gSearch_results.best_score_

