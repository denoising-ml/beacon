import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import regularizers
from keras.optimizers import SGD
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from numpy.random import seed
from keras.callbacks import EarlyStopping


#### df_merged_data is the final output after wavelet has been run
df_merged_data = pd.read_csv("wavelet_file_output_with_TA.csv")
df_merged_data.drop(["Unnamed: 0"], inplace=True, axis=1)
# wv_data.head()

#Split Data
X_train, X_test = train_test_split(df_merged_data, train_size = 0.75, random_state = seed(2017))

wv_train = pd.DataFrame(X_train, index=None)
wv_train.to_csv("preprocessing/wv_train.csv")

wv_test = pd.DataFrame(X_test)
wv_test.to_csv("preprocessing/wv_test.csv")

def sae():
    ### AN EXAMPLE OF DEEP AUTOENCODER WITH MULTIPLE LAYERS
    # InputLayer (None, 18)
    #      Dense (None, 10)
    #      Dense (None, 5)
    #      Dense (None, 10)
    #      Dense (None, 18)

    ncol = X_train.shape[1]
    print(ncol)
    input_dim = Input(shape = (ncol, ))
    # DEFINE THE DIMENSION OF ENCODER ASSUMED 3
    encoding_dim = 5
    # DEFINE THE ENCODER LAYERS
    encoded = Dense(ncol, activation = 'relu', activity_regularizer=regularizers.l1(10e-5))(input_dim)
    encoded = Dense(10, activation = 'relu', activity_regularizer=regularizers.l1(10e-5))(encoded)
    encoded = Dense(encoding_dim, activation = 'relu', activity_regularizer=regularizers.l1(10e-5))(encoded)
    # DEFINE THE DECODER LAYERS
    decoded = Dense(10, activation = 'relu', activity_regularizer=regularizers.l1(10e-5))(encoded)
    decoded = Dense(ncol, activation = 'sigmoid', activity_regularizer=regularizers.l1(10e-5))(decoded) #sigmoid  tanh
    # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
    autoencoder = Model(input = input_dim, output = decoded)
    # CONFIGURE AND TRAIN THE AUTOENCODER
    sgd = SGD(lr=0.01, decay=10e-5, momentum=0.9, nesterov=True) #0.001, 0.003, 0.01, 0.03, 0.1, 0.3
    autoencoder.compile(optimizer = sgd, loss = 'mean_squared_error', metrics=['accuracy'])
    autoencoder.summary()
    #Stop processing when accuracy stop increasing
    early_stop = EarlyStopping(monitor='acc', patience=1, verbose=1)

    history = autoencoder.fit(X_train, X_train, nb_epoch = 100, batch_size = 60, shuffle = False,
                            validation_data = (X_test, X_test), callbacks=[early_stop])
    # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
    encoder = Model(input = input_dim, output = encoded)
    encoded_input = Input(shape = (encoding_dim, ))
    encoded_out = encoder.predict(X_test)
    encoded_out[0:]

    encoded_train = pd.DataFrame(encoder.predict(X_train))
    encoded_train = encoded_train.add_prefix('feature_')
    print(encoded_train.head())
    encoded_train.to_csv("features/autoencoded_train_data.csv")

    encoded_test = pd.DataFrame(encoder.predict(X_test))
    encoded_test = encoded_test.add_prefix('feature_')
    encoded_test.head()
    encoded_test.to_csv("features/autoencoded_test_data.csv")

#https://stackoverflow.com/questions/19256930/python-how-to-normalize-time-series-data
