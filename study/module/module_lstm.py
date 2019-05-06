from typing import Dict

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import set_random_seed

set_random_seed(20190410)
from numpy.random import seed

seed(20190410)


def lstm_model(x_train, y_train, epochs):
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
    _model.add(LSTM(units=10,
                    input_shape=input_shape,
                    activation='relu'))
    _model.add((Dense(1)))
    _model.compile(loss="mean_squared_error",
                   optimizer="adam",
                   metrics=['mse', 'mae', 'mape'])

    # fit model
    _model.fit(x_train,
               y_train,
               epochs=epochs,
               batch_size=50)

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
    accuracy = correct_count / len(actual_direction)

    print("total prediction = {}".format(len(actual_direction)))
    print("correctness = {} ".format(correct_count))
    print("accuracy = {}".format(accuracy))

    return accuracy


def fit_predict(
        config: Dict,
        train_in_file: str,
        train_expected_file: str,
        train_predicted_file: str,
        test_in_file: str,
        test_expected_file: str,
        test_predicted_file: str):
    print('------------------ LSTM Start -------------------')
    df_in_train = pd.read_csv(train_in_file, index_col=0).values
    df_expected_train = pd.read_csv(train_expected_file, index_col=0).values
    df_in_test = pd.read_csv(test_in_file, index_col=0).values
    df_expected_test = pd.read_csv(test_expected_file, index_col=0).values

    print('In train: {} {}'.format(train_in_file, df_in_train.shape))
    print('Expected train: {} {}'.format(train_expected_file, df_expected_train.shape))
    print('In test: {} {}'.format(test_in_file, df_in_test.shape))
    print('Expected test: {} {}'.format(test_expected_file, df_expected_test.shape))
    print('Config: {}'.format(config))

    # shape data
    time_step = config.get('time_step', 3)
    sliding_step = config.get('sliding_step', 1)
    in_train, expected_train = shape_data(df_in_train, df_expected_train, time_step, sliding_step)
    in_test, expected_test = shape_data(df_in_test, df_expected_test, time_step, sliding_step)

    print("Data sizes")
    print("x_train = {}".format(in_train.shape))
    print("y_train = {}".format(expected_train.shape))
    print("x_test = {}".format(in_test.shape))
    print("y_test = {}".format(expected_test.shape))

    # build model
    epochs = config.get('epochs', 800)
    model = lstm_model(x_train=in_train, y_train=expected_train, epochs=epochs)
    history = model.history

    # pyplot.plot(history.history['mean_absolute_percentage_error'])
    # pyplot.show()

    predicted_train = model.predict(in_train)
    predicted_train = predicted_train.reshape(-1)

    predicted_test = model.predict(in_test)
    predicted_test = predicted_test.reshape(-1)
    performance(expected_test, predicted_test)

    pd.DataFrame(predicted_train).to_csv(train_predicted_file)
    pd.DataFrame(predicted_test).to_csv(test_predicted_file)

    print('------------------ LSTM End -------------------')


if __name__ == "__main__":
    df = pd.read_csv('../../data/input/HSI_figshare.csv')
    df = df.drop(['date', 'time'], axis=1)

    """
    Prepare input and label data
    label is the closing price in next(future) time step
    convert to numpy array
    """
    df_close = df.loc[:, "close"]
    y = df_close.iloc[1:]

    df_input = df
    x = df_input.iloc[:-1]

    # split data into training and test set
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(x, y, train_size=0.8, shuffle=False)

    in_train_file = '../output/encoder/tmp_in_train_data.csv'
    expected_train_file = '../output/encoder/tmp_expected_train_data.csv'
    predicted_train_file = '../output/encoder/tmp_predicted_train_data.csv'
    in_test_file = '../output/encoder/tmp_in_test_data.csv'
    expected_test_file = '../output/encoder/tmp_expected_test_data.csv'
    predicted_test_file = '../output/encoder/tmp_predicted_test_data.csv'

    pd.DataFrame(x_train_raw).to_csv(in_train_file)
    pd.DataFrame(y_train_raw).to_csv(expected_train_file)
    pd.DataFrame(x_test_raw).to_csv(in_test_file)
    pd.DataFrame(y_test_raw).to_csv(expected_test_file)

    config = {}

    fit_predict(config=config,
                train_in_file=in_train_file,
                train_expected_file=expected_train_file,
                train_predicted_file=predicted_train_file,
                test_in_file=in_test_file,
                test_expected_file=expected_test_file,
                test_predicted_file=predicted_test_file)

    df_in_test_data = pd.read_csv(in_test_file)
    df_out_test_data = pd.read_csv(predicted_test_file)

    df_display = df_in_test_data.iloc[:, [1]].copy()
    df_display.columns = ['in']
    df_display['out'] = df_out_test_data.iloc[:, 1].copy()

    df_display.plot()
    plt.show()
