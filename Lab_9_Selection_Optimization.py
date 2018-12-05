
# coding: utf-8

# In[22]:


import pandas as pd
dta = pd.read_csv("dailyinmatesincustody.csv")


# In[23]:


dta.head()


# In[24]:


dta = dta.dropna(subset =  ["GENDER"])


# In[25]:


dta.head()


# In[26]:


dta["AGE"].isnull().sum()


# In[27]:


X = dta[["AGE", "GENDER"]]


# In[28]:


y = dta[["INFRACTION"]]


# In[29]:


X = pd.get_dummies(X)


# In[30]:


X = X.drop(columns = ["GENDER_M"])


# In[31]:


y = pd.get_dummies(y)


# In[32]:


y = y.drop(columns=["INFRACTION_N"])


# In[33]:


X = X.values
y = y.values


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
dt_classifier = DecisionTreeClassifier(random_state=1693, max_depth=2, min_samples_leaf=2)
dt_classifier.fit(X,y)


# In[47]:


k_fold = cross_val_score(estimator = dt_classifier, X=X, y=y, cv=10, scoring="accuracy")


# In[48]:


k_fold.mean()


# In[49]:


params = [{'max_depth':[10, 15, 20], 'min_samples_leaf':[2,4,6,8,10,12,14,16,18,20]}]

gSearch = GridSearchCV(estimator = dt_classifier,
                      param_grid = params,
                      scoring = "accuracy",
                      cv = 5)


# In[50]:


gSearch_results = gSearch.fit(X,y)


# In[51]:


gSearch_results.best_params_


# In[40]:


gSearch_results.best_score_

