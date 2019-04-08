import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


def model_m2m(time_step, num_features):
    """
    Build a many-to-many LSTM model
    Args


    Returns


    """
    _model = Sequential()

    _model.add(LSTM(5, input_shape=(time_step, num_features), return_sequences=True, activation='relu'))
    _model.add(Dense(1))
    _model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse', 'mae', 'mape'])

    return _model


def shape_data_m2m(data, label, time_step, num_features):

    # drop some rows so we can reshape

    # split into training and test sets
    _x_train, _x_test, _y_train, _y_test = train_test_split(data, label, train_size=0.8, shuffle=False)

    _x_train = _x_train.reshape(-1, time_step, num_features)
    _x_test = _x_test.reshape(-1, time_step, num_features)

    _y_train = _y_train.reshape(-1, time_step, 1)
    _y_test = _y_test.reshape(-1, time_step, 1)

    return _x_train, _x_test, _y_train, _y_test


if __name__ == "__main__":
    df = pd.read_csv('../data/input/HSI_figshare.csv')
    df = df.drop(['Date', 'Time'], axis=1)

    """
    Prepare input and label data
    label is the closing price in next(future) time step
    """
    df_close = df.loc[:, "Closing Price"]
    df_close = df_close.iloc[1:]
    df_input = df
    df_input = df_input.iloc[:-1]

    # as numpy array
    x = df_input.values
    y = df_close.values

    time_step = 1
    num_features = 19

    x_train, x_test, y_train, y_test = shape_data_m2m(x, y, time_step, num_features)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    model = model_m2m(time_step, num_features)
    history = model.fit(x_train, y_train, epochs=500, batch_size=100)
    pyplot.plot(history.history['mean_absolute_percentage_error'])
    pyplot.show()

    y_predict = model.predict(x_test)

