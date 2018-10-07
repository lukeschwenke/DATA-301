
# coding: utf-8

# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./data.csv")


# In[15]:


X = data.iloc[:,2:4]
y = data.iloc[:, 1]

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1693)
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)


# In[16]:


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


# In[17]:


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


# In[18]:


from sklearn.naive_bayes import GaussianNB
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)


# In[19]:


y_pred = bayes_classifier.predict(X_test)
y_pred


# In[20]:


y_test


# In[21]:


from sklearn.metrics import confusion_matrix
confMat = confusion_matrix(y_test, y_pred)
print_cm(confMat, ["Benign", "Malignant"])


# In[22]:


viz_cm(bayes_classifier, ["Radius Mean", "Texture Mean"])


# In[24]:


from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state = 1693, max_depth = 3)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)


# In[25]:


dt_confMat = confusion_matrix(y_test, dt_pred)
print_cm(dt_confMat, ["Benign", "Malignant"])


# In[26]:


viz_cm(dt_classifier, ["Radius Mean", "Texture Mean"])


# In[28]:


from sklearn import tree
tree.export_graphviz(dt_classifier, out_file = "tree_dt.dot", feature_names= ["Radius Mean", "Texture Mean"])


# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


rt_classifier = RandomForestClassifier(n_estimators = 1000, random_state=1693, max_depth=3)
rt_classifier.fit(X_train, y_train)


# In[31]:


rt_pred = rt_classifier.predict(X_test)


# In[32]:


rt_confMat = confusion_matrix(y_test, rt_pred)
print_cm(rt_confMat, ["Benign", "Malignant"])


# In[33]:


viz_cm(rt_classifier, ["Radius Mean", "Texture Mean"])


# In[34]:


#Test Cases (Look at False Positives - We predicted Malignant but it was actually Benign)

# Random Forest
rt_classifier = RandomForestClassifier(n_estimators = 1000, random_state=1693, max_depth=5)
rt_classifier.fit(X_train, y_train)
rt_pred = rt_classifier.predict(X_test)
rt_confMat = confusion_matrix(y_test, rt_pred)
print_cm(rt_confMat, ["Benign", "Malignant"])
#Answer: 7 False Positives


# In[35]:


# Decision / Classification Tree
dt_classifier = DecisionTreeClassifier(random_state = 1693, max_depth = 5)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
dt_confMat = confusion_matrix(y_test, dt_pred)
print_cm(dt_confMat, ["Benign", "Malignant"])
#Answer: 6 False Positives


# In[36]:


# Bayes
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)
y_pred = bayes_classifier.predict(X_test)
confMat = confusion_matrix(y_test, y_pred)
print_cm(confMat, ["Benign", "Malignant"])
#Answer: 9 False Positives

