import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

from tensorflow import set_random_seed

set_random_seed(20190410)
from numpy.random import seed

seed(20190410)


def model(x_train, y_train):
    """
    Build a many-to-one LSTM model
    Args


    Returns


    """
    assert y_train.ndim == 1
    assert x_train.ndim == 3

    x_shape = x_train.shape
    input_shape = (x_shape[1], x_shape[2])

    _model = Sequential()
    _model.add(LSTM(10, input_shape=input_shape, return_sequences=False, activation='relu'))
    _model.add(Dense(1))
    _model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse', 'mae', 'mape'])
    _model.fit(x_train, y_train, epochs=1000, batch_size=50)

    return _model


def shape_data(data, label, time_step, sliding_step):
    if data.ndim == 1:
        num_features = 1
    else:
        num_features = data.shape[1]

    # slide over the time series (N, features) to produce a matrix of (M, time_step, features)
    # where each row is a time sequence of features
    # output matrix will have number of rows (M) = (length - time_step)/sliding_step + 1

    output_length = int((len(data) - time_step) / sliding_step + 1)

    output_data = np.zeros((output_length, time_step, num_features))
    output_label = np.zeros(output_length)

    for i in range(0, output_length):
        begin_index = i * sliding_step
        stop_index = begin_index + time_step
        sequence_data = data[begin_index:stop_index, :]
        output_data[i] = sequence_data
        output_label[i] = label[stop_index - 1]

    return output_data, output_label


def performance(y_test, y_predict):
    assert len(y_test) == len(y_predict)

    actual_direction = np.sign(y_test[1:] - y_test[0:-1])
    predict_direction = np.sign(y_predict[1:] - y_test[0:-1])
    correct_direction = actual_direction * predict_direction
    correct_count = (correct_direction > 0).sum()
    n = len(actual_direction)
    accuracy = correct_count / n

    print("total prediction = {}".format(n))
    print("correct count = {} ".format(correct_count))
    print("accuracy = {} ".format(accuracy))

    return accuracy


if __name__ == "__main__":
    df = pd.read_csv('../data/input/HSI_figshare.csv')
    df = df.drop(['Date', 'Time'], axis=1)

    """
    Prepare input and label data
    label is the closing price in next(future) time step
    convert to numpy array
    """
    df_close = df.loc[:, "Closing Price"]
    y = df_close.iloc[1:].values

    df_input = df
    df_input = df_input.iloc[:-1]
    x = df_input.values

    # split data into training and test set
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(x, y, train_size=0.8, shuffle=False)

    # shape data
    time_step = 2
    sliding_step = 1
    x_train, y_train = shape_data(x_train_raw, y_train_raw, time_step, sliding_step)
    x_test, y_test = shape_data(x_test_raw, y_test_raw, time_step, sliding_step)

    print("Data sizes")
    print("x_train = {}".format(x_train.shape))
    print("y_train".format(y_train.shape))
    print("x_test = {}".format(x_test.shape))
    print("y_test = {}".format(y_test.shape))

    # build model
    model = model(x_train, y_train)
    history = model.history

    # pyplot.plot(history.history['mean_absolute_percentage_error'])
    # pyplot.show()

    prediction = model.predict(x_test)
    print(prediction.shape)

    prediction = prediction.reshape(-1)
    print(prediction.shape)

    pyplot.plot(y_test, label="Actual")
    pyplot.plot(prediction, label="Prediction")
    pyplot.legend()
    pyplot.show()

    performance(y_test, prediction)

