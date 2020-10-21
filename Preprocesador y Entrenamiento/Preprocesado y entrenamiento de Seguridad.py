#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[ ]:





# In[4]:


data=pd.read_csv("Segu.csv")


# In[5]:



data.head()


# In[6]:


data.info()


# In[7]:


data_used=data


# In[12]:


data_used


# In[9]:


data_used=pd.get_dummies(data_used,columns=["protocol_type"])


# In[10]:


data_used=pd.get_dummies(data_used,columns=["service"])


# In[11]:


data_used=pd.get_dummies(data_used,columns=["flag"])


# In[13]:


data_used=pd.get_dummies(data_used,columns=["attack"])


# In[15]:


data_used.info()


# In[16]:


data_used.dropna(axis=0)


# In[17]:


data_used.info()


# In[2]:


data_used.to_csv("SeguConTitulosPre.csv")


# In[3]:


from keras.models import Sequential
from keras.layers import Dense
import csv
import numpy
import json
import os


# In[4]:


dataset = numpy.loadtxt("SeguSinTitulosPre2.csv", delimiter=",")
X = dataset[:,1:]
Y = dataset[:,0]


# In[25]:


Y


# In[5]:


model = Sequential()
model.add(Dense(30, input_dim=134, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)


# In[6]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serializar los pesos a HDF5
model.save_weights("CTF.h5")
print("Modelo Guardado!")



