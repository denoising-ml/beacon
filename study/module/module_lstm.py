from typing import Dict
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.optimizers as optimizers
import json
import matplotlib.pyplot as plt
import os
import keras


def lstm_model(x_train,
               y_train,
               cell_neurons,
               epochs,
               batch_size,
               layers,
               kernel_initializer='uniform',
               tensorboard_dir=None):

    """
    All except the last LSTMs return their full output sequences (return_sequences=True),
    but the last one only returns the last time step in its output sequence, thus dropping the temporal dimension
    (i.e. converting the input sequence into a single vector).

    -------------------------------------------------------------
    LSTM-1         Input:     (none, time_step, features)
                   Output:    (none, time_step, cell_neurons)
    -------------------------------------------------------------
    LSTM-2..(N-1)  Input:     (none, time_step, cell_neurons)
                   Output:    (none, time_step, cell_neurons)
    -------------------------------------------------------------
    LSTM-N         Input:     (none, time_step, cell_neurons)
                   Output:    (none, cell_neurons)
    -------------------------------------------------------------
    Dense          Input:     (none, cell_neurons)
                   Output:    (none, 1)

    If only 1 layer:
    -------------------------------------------------------------
    LSTM-1         Input:     (none, time_step, features)
                   Output:    (none, cell_neurons)
    -------------------------------------------------------------
    Dense          Input:     (none, cell_neurons)
                   Output:    (none, 1)
    """
    assert y_train.ndim == 1
    assert x_train.ndim == 3

    x_shape = x_train.shape
    input_shape = (x_shape[1], x_shape[2])

    _model = Sequential()

    if layers == 1:
        _model.add(LSTM(units=cell_neurons,
                        input_shape=input_shape,
                        activation='relu',
                        kernel_initializer=kernel_initializer))

    else:
        _model.add(LSTM(units=cell_neurons,
                        input_shape=input_shape,
                        return_sequences=True,
                        activation='relu',
                        kernel_initializer=kernel_initializer))

        layers -= 1

        if layers > 1:
            _model.add(LSTM(units=cell_neurons, return_sequences=True))
            layers -=1

        _model.add(LSTM(units=cell_neurons, activation='relu'))

    # dense output layer
    _model.add((Dense(1)))

    _model.compile(loss="mean_squared_error",
                   optimizer=optimizers.Adadelta(lr=0.2),
                   metrics=['mse', 'mae', 'mape'])

    # callbacks
    callbacks = []
    if tensorboard_dir is not None:
        tbCallBack = keras.callbacks.TensorBoard(log_dir=tensorboard_dir,
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True)
        callbacks.append(tbCallBack)

    # fit model
    _model.fit(x_train,
               y_train,
               epochs=epochs,
               batch_size=batch_size,
               callbacks=callbacks)

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
        test_predicted_file: str,
        tensorboard_dir: str = None):

    print('------------------ LSTM Start -------------------')
    df_in_train = pd.read_csv(train_in_file, index_col=0).values
    df_expected_train = pd.read_csv(train_expected_file, index_col=0).values
    df_in_test = pd.read_csv(test_in_file, index_col=0).values
    df_expected_test = pd.read_csv(test_expected_file, index_col=0).values

    print('[Train Set] Input: {} {}'.format(train_in_file, df_in_train.shape))
    print('[Train Set] Label: {} {}'.format(train_expected_file, df_expected_train.shape))
    print('[Test Set] Input: {} {}'.format(test_in_file, df_in_test.shape))
    print('[Test Set] Label: {} {}'.format(test_expected_file, df_expected_test.shape))
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
    cell_neurons = config.get('cell_neurons', 5)
    epochs = config.get('epochs', 800)
    batch_size = config.get('batch_size', 60)
    layers = config.get('layers', 1)

    model = lstm_model(x_train=in_train,
                       y_train=expected_train,
                       cell_neurons=cell_neurons,
                       epochs=epochs,
                       batch_size=batch_size,
                       layers=layers,
                       tensorboard_dir=tensorboard_dir)

    history = model.history

    # pyplot.plot(history.history['mean_absolute_percentage_error'])
    # pyplot.show()

    predicted_train = model.predict(in_train)
    predicted_train = predicted_train.reshape(-1)

    predicted_test = model.predict(in_test)
    predicted_test = predicted_test.reshape(-1)
    performance(expected_test, predicted_test)

    pd.DataFrame(predicted_train, columns=['predict']).to_csv(train_predicted_file)
    pd.DataFrame(predicted_test, columns=['predict']).to_csv(test_predicted_file)

    # save model
    if tensorboard_dir is not None:
        model.save(tensorboard_dir + '/model_lstm.h5')

    print('------------------ LSTM End -------------------')
    return model


def plot(label_file, predict_file):
    df_label = pd.read_csv(label_file)
    df_predict = pd.read_csv(predict_file)

    df_display = df_label.iloc[:, [1]].copy()
    df_display.columns = ['actual']
    df_display['predict'] = df_predict.iloc[:, 1].copy()

    df_display.plot()
    plt.show()


if __name__ == "__main__":
    # Change directory
    directory = 'C:/temp/beacon/study_20190629_162057/run_0/'

    # Training and test data
    in_train_file = directory + 'run_0_train_lstm_input.csv'
    in_test_file = directory + 'run_0_test_lstm_input.csv'
    expected_train_file = directory + 'run_0_train_lstm_label.csv'
    expected_test_file = directory + 'run_0_test_lstm_label.csv'

    # put all output files in a sub folder
    file_prefix = directory + 'analyze_lstm/'

    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)

    predict_train_file = file_prefix + 'train_lstm_predict.csv'
    predict_test_file = file_prefix + 'test_lstm_predict.csv'

    config = {
        'epochs': 1000,
        'cell_neurons': 8,
        'time_step': 4,
        'layers': 1,
        'batch_size': 60
    }

    with open(file_prefix + 'config.json', 'w') as outfile:
        json.dump(config, outfile, indent=4)

    fit_predict(config=config,
                train_in_file=in_train_file,
                train_expected_file=expected_train_file,
                train_predicted_file=predict_train_file,
                test_in_file=in_test_file,
                test_expected_file=expected_test_file,
                test_predicted_file=predict_test_file)

    plot(expected_train_file, predict_train_file)
    plot(expected_test_file, predict_test_file)




