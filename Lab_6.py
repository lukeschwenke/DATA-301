
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import zipimport
importer = zipimport.zipimporter("nltk.zip")
importer.load_module("nltk")

import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import download

download("stopwords")


# In[12]:


wine_data = pd.read_csv("./winemagdata130kv2.csv", quoting=2)
wine_data = wine_data[["description", "points"]]
wine_data = wine_data.sample(1000, random_state=1693).reset_index(drop=True)
print(wine_data.shape)


# In[36]:


# Use Regex to remove and replace unnecessary characters
wine_descriptions = re.sub("[^a-zA-Z0-9 ]", "", wine_data["description"][0])
wine_descriptions = wine_descriptions.lower()
wine_descriptions = wine_descriptions.split()

# Remove english stopwords
stopwords.words("english")
wine_descriptions = [word for word in wine_descriptions if not word in set(stopwords.words("english"))]

# Bring words to their roots
stemmer = PorterStemmer()
wine_descriptions = [stemmer.stem(word) for word in wine_descriptions]

# Join everything together with a space in between
wine_descriptions = " ".join(wine_descriptions)

wine_descriptions


# In[60]:


get_ipython().run_cell_magic('time', '', '#Unique to Jupyter - Time whatever is in this snippet of code\nwine_data = pd.read_csv("./winemagdata130kv2.csv", quoting=2)\nwine_data = wine_data[["description", "points"]]\nwine_data = wine_data.sample(1000, random_state=1693).reset_index(drop=True)\n\ncorpus = []\n\nfor i in range(0, len(wine_data)):\n    wine_descriptions = re.sub("[^a-zA-Z0-9 ]", "", wine_data["description"][i])\n    wine_descriptions = wine_descriptions.lower()\n    wine_descriptions = wine_descriptions.split()\n    stopwords.words("english")\n    wine_descriptions = [word for word in wine_descriptions if not word in set(stopwords.words("english"))]\n    stemmer = PorterStemmer()\n    wine_descriptions = [stemmer.stem(word) for word in wine_descriptions]\n    wine_descriptions = " ".join(wine_descriptions)\n    corpus.append(wine_descriptions)\n    \n\n# Create Sparse Matrix\nfrom sklearn.feature_extraction.text import CountVectorizer\ncountVec = CountVectorizer()\nX_raw = countVec.fit_transform(corpus) #Goes through every sentence, extracts the words (fit), and puts the corresponding counts (transform)\nX = X_raw.toarray()\n\n# Set above 90 points (good wine) = 1, else 0\ny = wine_data["points"]\ny = y.where(y>90, other=0).where(y <= 90, other=1).values\n\n# Use logistic classifier\nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1693)\n\nfrom sklearn.linear_model import LogisticRegression\nclassifier = LogisticRegression(random_state=1693)\nclassifier.fit(X_train,y_train)\n\n# Predict\ny_pred = classifier.predict(X_test)\ny_pred\n\n    ')


# In[43]:


print(wine_data["description"][1])
print(corpus[1])


# In[50]:


X
print(pd.DataFrame(X_raw.A, columns=countVec.get_feature_names()).transpose())


# In[56]:


n, bins, patches = plt.hist(wine_data["points"].values, 10, normed=1, facecolor="blue", alpha=0.7)
plt.show()


# In[61]:


print(y_pred)


# In[64]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confMat = confusion_matrix(y_test, y_pred)
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


# In[65]:


print_cm(confMat, ["Bad Wine", "Good Wine"])
# 12 wines we said were bad but were in fact good
# 41 wines we said were good but were in fact bad


# In[68]:


# Predict w/ Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(random_state=1693)
classifierDT.fit(X_train,y_train)

# Predict
y_pred = classifierDT.predict(X_test)
y_pred

confMat = confusion_matrix(y_test, y_pred)
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
        
print_cm(confMat, ["Bad Wine", "Good Wine"])


# In[71]:


# Predict w/ Naive Bayes

from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train,y_train)

# Predict
y_pred = classifierNB.predict(X_test)
y_pred

confMat = confusion_matrix(y_test, y_pred)
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
        
print_cm(confMat, ["Bad Wine", "Good Wine"])

