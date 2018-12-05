
# coding: utf-8

# In[5]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import KernelPCA
import numpy as np

# Confusion Matrix
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

# Find Correlation Function
def find_correlation(data, threshold=0.5, remove_negative=False): #originally thresh = 0.9
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove.
    Parameters
    -----------
    data : pandas DataFrame
        DataFrame
    threshold : float
        correlation threshold, will remove one of pairs of features with a
        correlation greater than this value.
    remove_negative: Boolean
        If true then features which are highly negatively correlated will
        also be returned for removal.
    Returns
    --------
    select_flat : list
        listof column names to be removed
    """
    corr_mat = data.corr()
    if remove_negative:
        corr_mat = np.abs(corr_mat)
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][corr_mat[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat

# Import Data        
ks_data = pd.read_csv("ksprojects201801.csv")

X = ks_data.drop(["ID","name","pledged","backers", "usd pledged", "usd_pledged_real", "state", "category"], axis=1)
y = ks_data["state"]

y[y != "successful"] = "Failed"
y = pd.get_dummies(y)
y = y.drop("Failed", axis = 1)

start_date = pd.to_datetime("2000-1-1")
X["deadline"] = (pd.to_datetime(X["deadline"]) - start_date).dt.days
X["launched"] = (pd.to_datetime(X["launched"]) - start_date).dt.days
X["duration"] = X["deadline"] - X["launched"]
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=1693)


# In[6]:


len(find_correlation(X, threshold = 0.5, remove_negative = True))


# In[ ]:


# N_components = 5
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=1693)
pca = PCA(n_components = 5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(pca.explained_variance_ratio_)

plt.plot(pca.explained_variance_ratio_)
plt.xticks([0,1,2,3,4])
plt.ylabel("PCA Variance Explained")
plt.show()

# N_components = 2
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=1693)
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

plt.plot(pca.explained_variance_ratio_)
plt.xticks([0,1])
plt.ylabel("PCA Variance Explained")
plt.show()

# Bayes without PCA
print("\nUsing All Data")
from sklearn.naive_bayes import GaussianNB
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)
y_pred = bayes_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confMat = confusion_matrix(y_test, y_pred)
print_cm(confMat, ["Successful", "Failed"])
print(accuracy_score(y_test, y_pred))

# Bayes with PCA 
print("\nPCA with Two Components")
from sklearn.naive_bayes import GaussianNB
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train_pca, y_train)
y_pred_pca = bayes_classifier.predict(X_test_pca)
from sklearn.metrics import confusion_matrix
confMat = confusion_matrix(y_test, y_pred)
print_cm(confMat, ["Successful", "Failed"])
print(accuracy_score(y_test, y_pred_pca))


# In[9]:


# Kernel PCA - Allows for non-linearities in multi-dimensional space where we are trying to reduct down the dimensions

ks_data = pd.read_csv("ksprojects201801.csv")
ks_data = ks_data.sample(2500, random_state = 1693)

X = ks_data.drop(["ID","name","pledged","backers", "usd pledged", "usd_pledged_real", "state", "category"], axis=1)
y = ks_data["state"]

y[y != "successful"] = "Failed"
y = pd.get_dummies(y)
y = y.drop("Failed", axis = 1)

start_date = pd.to_datetime("2000-1-1")
X["deadline"] = (pd.to_datetime(X["deadline"]) - start_date).dt.days
X["launched"] = (pd.to_datetime(X["launched"]) - start_date).dt.days
X["duration"] = X["deadline"] - X["launched"]
X = pd.get_dummies(X)

#print(find_correlation(X, threshold = 0.5, remove_negative = True))
X = X.drop(find_correlation(X, threshold = 0.7, remove_negative = True), axis=1) #Remove these variables due to collinearity

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=1693)

kernel_pca = KernelPCA(kernel = "rbf")
x_train_kPCA = kernel_pca.fit_transform(X_train)
x_test_kPCA = kernel_pca.transform(X_test)


# In[3]:


# Find Correlation and Bayes
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)
y_pred_FC = bayes_classifier.predict(X_test)
confMat = confusion_matrix(y_test, y_pred_FC)
print_cm(confMat, ["Successful", "Failed"])


# In[4]:


explained_variance = np.var(x_train_kPCA, axis = 0)
kernelPCA_explained_variance_ratio = explained_variance / np.sum(explained_variance)
print(kernelPCA_explained_variance_ratio)


# In[5]:


plt.plot(kernelPCA_explained_variance_ratio)
plt.ylabel("Kernel PCA Variance Explained")
plt.show()


# In[6]:


kernel_pca = KernelPCA(kernel = "rbf", n_components = 250) # Very sensitive to n_components / dimensions
X_train_kPCA = kernel_pca.fit_transform(X_train)
X_test_kPCA  = kernel_pca.transform(X_test)


# In[7]:


bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train_kPCA, y_train)
y_pred_kPCA = bayes_classifier.predict(X_test_kPCA)
confMat = confusion_matrix(y_test, y_pred_kPCA)
print_cm(confMat, ["Successful", "Failed"])
print(accuracy_score(y_test, y_pred_kPCA))

