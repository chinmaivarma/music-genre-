#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

df=pd.read_csv('features_30_sec.csv')
df.head()


# In[12]:


X = df.iloc[: ,1:59].values
Y = df['label'].values


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0,test_size=0.3, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[90]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)


# In[ ]:


y_pred=knn.predict(x_test)


# In[91]:


print("Training set score: {:.3f}".format(knn.score(x_train, y_train)))
print("Test set score: {:.3f}".format(knn.score(x_test, y_test)))


# In[ ]:




