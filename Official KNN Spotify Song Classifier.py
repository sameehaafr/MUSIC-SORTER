#!/usr/bin/env python
# coding: utf-8

# STEP 1: Access Spotify Credentials

# STEP 2: Download playlists music attribute data into csv

# STEP 3: Import needed libraries

# In[1]:


from sklearn.neighbors import NearestNeighbors
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns


# STEP 4: Read in csv file to each playlist and graph each attribute to spot differences. I am reading in my happy and euphonious playlists.

# In[3]:


happy = pd.read_csv("happy_all.csv")
euphon = pd.read_csv("euphonious_all.csv")


# below is an example of one of the attributes I graphed, you can do this for all attributes

# In[5]:


plt.plot(happy.energy, color='blue')
plt.plot(euphon.energy, color='red')
plt.xlabel("song #")
plt.ylabel("energy intensity")
plt.title("energy intensity across all songs in both playlists")


# STEP 5: Once you choose the attributes with the largest differences between the 2 playlists, combine both playlists datasets keeping wanted attributes then read in the new csv file

# In[7]:


happy_eupho = pd.read_csv("happy_eupho_new.csv")
happy_eupho.head()


# check for null values

# In[8]:


happy_eupho.isnull().values.any()


# check for integer datatypes (numpy arrays only work with integers)

# In[9]:


happy_eupho.dtypes


# In[10]:


happy_eupho['danceability']=happy_eupho['danceability'].astype(int)


# STEP 6: Normalize/Standardize the dataset

# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(happy_eupho.drop('target', axis=1))


# In[12]:


scaled_features = scaler.transform(happy_eupho.drop('target',axis=1))
scaled_features


# below is the pandas dataframe with the standardized values

# In[13]:


happy_eupho_feat = pd.DataFrame(scaled_features, columns = happy_eupho.columns[:-1])
happy_eupho_feat.head()


# STEP 7: Split dataset into training and testing sets

# In[15]:


from sklearn.model_selection import train_test_split
X = happy_eupho_feat
y = happy_eupho['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30, shuffle=True)


# STEP 8: Train model

# find k value

# In[16]:


import math
math.sqrt(len(y_test))


# In[17]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean')
knn.fit(X_train, y_train)


# STEP 9: Make predictions

# In[19]:


prediction = knn.predict(X_test)
prediction


# STEP 10: Evaluate Predictions using the Classification Report

# In[21]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, prediction))


# STEP 11: Evaluate alternative K-values for better predictions
# 

# In[22]:


error_rate = []
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    prediction_i = knn.predict(X_test)
    error_rate.append(np.mean(prediction_i != y_test))


# Plot error rate

# In[24]:


plt.plot(error_rate, color='blue')
plt.title('error rate vs. k values')
plt.xlabel('k values')
plt.ylabel('error rate')


# STEP 11: Adjust K value according to the graph (lowest error rate)

# In[26]:


knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(X_train, y_train)
prediction= knn.predict(X_test)
prediction


# In[27]:


print(classification_report(y_test, prediction))

