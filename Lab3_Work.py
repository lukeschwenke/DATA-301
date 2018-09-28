
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

students_math = pd.read_csv("./studentmat.csv")
students_port = pd.read_csv("./studentpor.csv")
# Predict WALC (1 most 5 least amount of alc consumed)


# In[3]:


students_port.shape


# In[4]:


all_student_rows = [students_math, students_port]


# In[5]:


all_students = pd.concat(all_student_rows, ignore_index = True)


# In[6]:


all_students


# In[12]:


X = all_students[["age", "address", "traveltime", "failures", "higher", "internet",
                  "romantic", "famrel", "freetime", "goout", "absences"]].values
from sklearn.preprocessing import LabelEncoder
discreteCoder_X = LabelEncoder()


# In[14]:


X[:,1] = discreteCoder_X.fit_transform(X[:,1]) #Fits data to 0 and 1 then transform 
X[:,4] = discreteCoder_X.fit_transform(X[:,4])
X[:,5] = discreteCoder_X.fit_transform(X[:,5])
X[:,6] = discreteCoder_X.fit_transform(X[:,6])


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

y = all_students[["Walc"]].values


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1693)


# In[18]:


scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)


# In[19]:


from sklearn.svm import SVR 


# In[20]:


svr_regression = SVR(kernel = "linear", epsilon = 1.0)
svr_regression.fit(X_train, y_train)


# In[21]:


new_studentA = [[18, 1, 3, 3, 0, 0, 1, 2, 5, 2, 5]]


# In[22]:


new_studentA


# In[23]:


new_student_scaledA = scale_X.transform(new_studentA)
studentA_prediction = svr_regression.predict(new_student_scaledA)


# In[24]:


studentA_prediction


# In[25]:


print("First new student (A):" + str(studentA_prediction))


# In[29]:


new_studentB = [[18, 0, 3, 3, 0, 0, 1, 2, 1, 1, 5]]
new_student_scaledB = scale_X.transform(new_studentB)
studentB_prediction = svr_regression.predict(new_student_scaledB)
print("First new student (B):" + str(studentB_prediction))


# In[30]:


from sklearn import tree


# In[31]:


DT_regression = tree.DecisionTreeRegressor(random_state = 1693, max_depth = 3)
DT_regression.fit(X_train, y_train)


# In[32]:


tree.export_graphviz(DT_regression, out_file="tree.dot", feature_names = ["age", "address", "traveltime", "failures", "higher", "internet",
                  "romantic", "famrel", "freetime", "goout", "absences"])


# In[35]:


studentA_prediction_RT = DT_regression.predict(new_student_scaledA)
print("First new student:" + str(studentA_prediction_RT))


# In[38]:


studentB_prediction_RT = DT_regression.predict(new_student_scaledB)
print("First new student:" + str(studentB_prediction_RT))


# In[39]:


from sklearn.ensemble import RandomForestRegressor
RF_regression = RandomForestRegressor(n_estimators = 100, random_state = 1693)
RF_regression.fit(X_train, y_train)


# In[40]:


studentA_prediction_RF = RF_regression.predict(new_student_scaledA)
print("First new student:" + str(studentA_prediction_RF))


# In[41]:


studentB_prediction_RF = RF_regression.predict(new_student_scaledB)
print("First new student:" + str(studentB_prediction_RF))


# In[45]:


from sklearn.metrics import mean_absolute_error 
rf_MAD = mean_absolute_error(y_test, RF_regression.predict(X_test))


# In[48]:


rf_MAD


# In[50]:


RT_MAD = mean_absolute_error(y_test, DT_regression.predict(X_test))
SVR_MAD = mean_absolute_error(y_test, svr_regression.predict(X_test))


# In[52]:


print("Random Forest:" + str(rf_MAD))
print("Decision Tree:" + str(RT_MAD))
print("Supper Vector Regression:" + str(SVR_MAD))


# In[53]:


#Problem Set Question
new_studentFIND = [[20, 1, 3, 1, 0, 1, 1, 2, 3, 2, 5]]
new_student_scaledFIND = scale_X.transform(new_studentFIND)
studentFIND_prediction = svr_regression.predict(new_student_scaledFIND)
print("First new student FIND:" + str(studentFIND_prediction))


# In[54]:


studentFIND_prediction_RF = RF_regression.predict(new_student_scaledFIND)
print("First new student:" + str(studentFIND_prediction_RF))

