#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')
import datetime as dt
import numpy as np
import pandas as pd
import pywt
from pandas import read_csv, DataFrame
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model


# In[16]:


df = read_csv("C:\\Users\\Nitin Rana\\Documents\\GitHub\\beacon\\data\\input\\Nick_WT.txt")
print (df.shape)
# SCALE EACH FEATURE INTO [0, 1] RANGE
Y = df.iloc[:,1]
X = df.iloc[:,1:]
sX = minmax_scale(X, axis = 0)
ncol = sX.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(sX, Y, train_size = 0.5, random_state = seed(2017))
X_train.shape


# In[22]:


### AN EXAMPLE OF DEEP AUTOENCODER WITH MULTIPLE LAYERS
# InputLayer (None, 10)
#      Dense (None, 20)
#      Dense (None, 10)
#      Dense (None, 5)
#      Dense (None, 3)
#      Dense (None, 5)
#      Dense (None, 10)
#      Dense (None, 20)
#      Dense (None, 10)
 
input_dim = Input(shape = (ncol, ))
# DEFINE THE DIMENSION OF ENCODER ASSUMED 3
encoding_dim = 3

# DEFINE THE ENCODER LAYERS
encoded1 = Dense(10, activation = 'relu')(input_dim) 
encoded2 = Dense(8, activation = 'relu')(encoded1)
encoded3 = Dense(5, activation = 'relu')(encoded2)
encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)

### encoded4 = Dense(encoding_dim, activation = 'relu', activity_regularizer=regularizers.l1(10e-5))(encoded3)
# DEFINE THE DECODER LAYERS
decoded1 = Dense(5, activation = 'relu')(encoded4)
decoded2 = Dense(8, activation = 'relu')(decoded1)
decoded3 = Dense(10, activation = 'relu')(decoded2)
decoded4 = Dense(ncol, activation = 'sigmoid')(decoded3)
# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(input = input_dim, output = decoded4)
# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(X_train, X_train, nb_epoch = 500, batch_size = 256, shuffle = True, validation_data = (X_test, X_test))
# THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(input = input_dim, output = encoded4)
encoded_input = Input(shape = (encoding_dim, ))
encoded_out = encoder.predict(X_test)
encoded_out[0:]


# In[ ]:




