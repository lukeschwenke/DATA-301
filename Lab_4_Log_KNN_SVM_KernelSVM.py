import pandas as pd
data = pd.read_csv("./data.csv")

data.shape
list(data)

X = data.iloc[:,2:4]
y = data.iloc[:,1]

X = X.values
y = y.values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1693)
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression(random_state = 1693)
logistic_classifier.fit(X_train, y_train)
y_pred = logistic_classifier.predict(X_test)
y_pred

# PREDICTION FOR LOG 
log_x = [[15.78, 17.89]]
scaled_log_x = scale_X.transform(log_x)
log_x_prediction = logistic_classifier.predict_proba(scaled_log_x)
log_x_prediction

y_test

y_pred_probabilities = logistic_classifier.predict_proba(X_test)
y_pred_probabilities

from sklearn.metrics import confusion_matrix
confMat = confusion_matrix(y_test, y_pred)
confMat

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

print_cm(confMat, ["Benign", "Malignant"])

#75 cases we predicted benign, and it actually was benign
#52 cases we predicted malignant, and it actually was malignant
#6 cases we predicted malignant, and it was actually benign (False Positive)
#10 cases we predicted benign, and it was actually malignant (False Negative)

# KNN
from sklearn.neighbors import KNeighborsClassifier

nn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p=2)
nn_classifier.fit(X_train, y_train)

y_pred_NN = nn_classifier.predict(X_test)
y_pred_NN

# PREDICTION FOR KNN
knn_x = [[17.18, 8.65]]
scaled_knn_x = scale_X.transform(knn_x)
knn_x_prediction = nn_classifier.predict(scaled_knn_x)
knn_x_prediction

y_test

confMat_NN = confusion_matrix(y_test, y_pred_NN)
print_cm(confMat_NN, ["Benign", "Malignant"])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder

def viz_cm(model, labels):

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    pred = model.predict(np.array([X1.ravel(), X2.ravel()]).T)

    discreteCoder = LabelEncoder()
    pred = discreteCoder.fit_transform(pred)

    plt.contourf(X1, X2, pred.reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Classification')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    plt.show()

viz_cm(model = nn_classifier, labels = ["Radius Mean", "Texture Mean"])

#SVM 
from sklearn.svm import SVC
svc_classifier = SVC(kernel="linear", random_state=1693)
svc_classifier.fit(X_train, y_train)

y_pred_SVC = svc_classifier.predict(X_test)

confMat_SVC = confusion_matrix(y_test, y_pred_SVC)
print_cm(confMat_SVC, ["Benign", "Malignant"])

# PREDICTION FOR SVM
svm_x = [[15.78, 17.89]]
scaled_svm_x = scale_X.transform(svm_x)
svm_x_prediction = svc_classifier.predict(scaled_svm_x)
svm_x_prediction

viz_cm(model = svc_classifier, labels = ["Radius Mean", "Texture Mean"])
#Simpler model (SVC) is doing better than KNN since we have 16 false neg/pos in this one vs. 19 in the other.

from sklearn.svm import SVC
kernelSVC_classifier = SVC(kernel="rbf", random_state=1693)
kernelSVC_classifier.fit(X_train, y_train)

y_pred_SVC_kernel = kernelSVC_classifier.predict(X_test)

confMat_SVC_kernel = confusion_matrix(y_test, y_pred_SVC_kernel)
print_cm(confMat_SVC_kernel, ["Benign", "Malignant"])

viz_cm(model = svc_classifier, labels = ["Radius Mean", "Texture Mean"])

