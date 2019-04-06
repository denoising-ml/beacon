from numpy.random import choice
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Reference: http://philipperemy.github.io/keras-stateful-lstm/
'''


sequence_len = 20
np.random.seed(20190604)

def create_data(N):
    # create an input matrix of zeros except in the first column where exactly half of the values are 1.
    # each row is a sequence/time-steps of 20 input. Each input has 1 feature only.
    # the first column is also stored as label
    # our model must understand that if the first value of the sequence is 1, then the result is 1, 0 otherwise

    x_train = np.zeros((N, sequence_len))
    y_train = np.zeros((N, 1))
    one_indexes = choice(a=N, size=int(N/2), replace=False)
    x_train[one_indexes, 0] = 1
    y_train[one_indexes, 0] = 1

    input_shape = (N, sequence_len, 1)
    x_train = np.reshape(x_train, input_shape)
    y_train = np.reshape(y_train, (N, 1))

    return x_train, y_train

def learn_stateless(trainX, trainY, testX, testY):

    print('Build STATE-LESS model...')

    # https://stackoverflow.com/questions/38714959/understanding-keras-lstms
    # dimension of LSTM output
    output_unit = 10

    model = Sequential()

    # input shape define the length of input sequence and the number of features per input
    # for example if use 10 sequence of prices from 3 stocks to predict an outcome, then input shape is (10, 3)
    model.add(LSTM(output_unit, input_shape=(sequence_len, 1), stateful=False, return_sequences=False))

    # we will map output units from LSTM to just one binary outcome
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, batch_size=50, nb_epoch=5, validation_data=(testX, testY), shuffle=False)

    return model

def study():

    trainX, trainY = create_data(1000)
    testX, testY = create_data(200)

    model = learn_stateless(trainX, trainY, testX, testY)

    validationX = np.zeros((1, sequence_len, 1))
    validationX[0, 0, 0] = 1
    predictY = model.predict(validationX)
    print("Prediction should be close to 1 and getting {}".format(predictY))

    validationX[0, 0, 0] = 0
    predictY = model.predict(validationX)
    print("Prediction should be close to 0 and getting {}".format(predictY))

if __name__ == "__main__":
    study()