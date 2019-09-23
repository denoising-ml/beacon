from typing import Dict
import pandas as pd
import keras
from keras.layers import Input, Dense
from keras.models import Model
import keras.optimizers as optimizers
import sklearn.preprocessing as preprocessing
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import os
import numpy as np
import json


def sae(train_x,
        validate_x,
        hidden_dimensions,
        epochs,
        hidden_activation='relu',
        kernel_initializer='random_uniform',
        bias_initializer='zeros',
        loss_metric='mean_squared_error',
        batch_size=10,
        learning_rate=0.05,
        tensorboard_dir=None):

    assert train_x.ndim == 2
    assert validate_x.ndim == 2
    assert validate_x.shape[1] == train_x.shape[1]

    n_features = train_x.shape[1]
    input_dim = Input(shape=(n_features,))

    # encoder layers
    last_encoded_layer = input_dim
    for hidden_dim in hidden_dimensions:
        last_encoded_layer = Dense(hidden_dim,
                                   activation=hidden_activation,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer)(last_encoded_layer)

    # hidden decoder layers, skip the inner most dimension
    last_decoded_layer = last_encoded_layer
    for hidden_dim in reversed(hidden_dimensions[:-1]):
        last_decoded_layer = Dense(hidden_dim,
                                   activation=hidden_activation,
                                   kernel_initializer=keras.initializers.Constant(value=0.1),
                                   bias_initializer='zeros')(last_decoded_layer)

    # output layer
    last_decoded_layer = Dense(n_features, activation='linear')(last_decoded_layer)

    # autoencoder model
    autoencoder = Model(input=input_dim, output=last_decoded_layer)
    autoencoder.compile(optimizer=optimizers.Adadelta(lr=learning_rate), loss=loss_metric, metrics=['mse', 'mae', 'mape'])
    autoencoder.summary()

    # encoder model
    encoder = Model(input=input_dim, output=last_encoded_layer)

    # callbacks
    callbacks = []
    if tensorboard_dir is not None:
        tbCallBack = keras.callbacks.TensorBoard(log_dir=tensorboard_dir,
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True)
        callbacks.append(tbCallBack)

    # configure and fit the model
    autoencoder.fit(train_x,
                    train_x,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False,
                    validation_data=(validate_x, validate_x),
                    callbacks=callbacks)

    return autoencoder, encoder


def fit_predict(
        config: Dict,
        train_in_file: str,                     # training - input features
        train_in_scaled_file: str,              # training - scaled input features (input into model)
        train_encoder_file: str,                # training - encoded features
        train_decoder_scaled_file: str,         # training - decoded features (scaled)
        train_decoder_file: str,                # training - decoded features (in original scale)
        test_in_file: str,                      # test     - input features
        test_in_scaled_file: str,               # test     - scaled input features (input into model)
        test_encoder_file: str,                 # test     - encoded features
        test_decoder_scaled_file: str,          # test     - decoded features (scaled)
        test_decoder_file: str,                 # test     - decoded features (in original scale)
        loss_plot_file: str = None,
        tensorboard_dir: str = None,
        chart_dir: str = None,
        model_store_dir: str = None):

    print('------------------ SAE Start --------------------')
    df_in_train = pd.read_csv(train_in_file, index_col=0)
    df_in_test = pd.read_csv(test_in_file, index_col=0)

    column_names = df_in_train.columns

    print('In train: {} {}'.format(train_in_file, df_in_train.shape))
    print('In test: {} {}'.format(test_in_file, df_in_test.shape))
    print('Config: {}'.format(config))

    apply_scaler = config.get('scaler', True)

    # normalize data if required
    if apply_scaler:
        # fit scaler using training data
        scaler = preprocessing.StandardScaler()
        scaler.fit(df_in_train)

        # transform data
        _in_train_scaled = scaler.transform(df_in_train)
        _in_test_scaled = scaler.transform(df_in_test)

        scaler_train = scaler
        scaler_test = scaler

        # create data frame from numpy array
        df_in_train_scaled = pd.DataFrame(data=_in_train_scaled, columns=column_names)
        df_in_test_scaled = pd.DataFrame(data=_in_test_scaled, columns=column_names)
    else:
        df_in_train_scaled = df_in_train
        df_in_test_scaled = df_in_test

    if train_in_scaled_file is not None:
        df_in_train_scaled.to_csv(train_in_scaled_file)
    if test_in_scaled_file is not None:
        df_in_test_scaled.to_csv(test_in_scaled_file)

    hidden_dim = config.get('hidden_dim', [16, 8])
    epochs = config.get('epochs', 1000)
    loss_metric = config.get('loss_metric', 'mean_squared_error')
    batch_size = config.get('batch_size', 10)
    learning_rate = config.get('learning_rate', 0.2)
    hidden_activation = config.get('hidden_activation', 'relu')

    autoencoder, encoder = sae(train_x=df_in_train_scaled,
                               validate_x=df_in_test_scaled,
                               hidden_dimensions=hidden_dim,
                               epochs=epochs,
                               hidden_activation=hidden_activation,
                               loss_metric=loss_metric,
                               batch_size=batch_size,
                               learning_rate=learning_rate,
                               tensorboard_dir=tensorboard_dir)

    # get decoder output
    out_decoder_train_scaled = autoencoder.predict(df_in_train_scaled)
    out_decoder_test_scaled = autoencoder.predict(df_in_test_scaled)

    # create data frame for decoder output
    df_out_decoder_train_scaled = pd.DataFrame(data=out_decoder_train_scaled, columns=column_names)
    df_out_decoder_test_scaled = pd.DataFrame(data=out_decoder_test_scaled, columns=column_names)

    if train_decoder_scaled_file is not None:
        df_out_decoder_train_scaled.to_csv(train_decoder_scaled_file)
    if test_decoder_scaled_file is not None:
        df_out_decoder_test_scaled.to_csv(test_decoder_scaled_file)

    # de-normalize decoder output if required
    if apply_scaler:
        out_train = scaler_train.inverse_transform(out_decoder_train_scaled)
        out_test = scaler_test.inverse_transform(out_decoder_test_scaled)
    else:
        out_train = out_decoder_train_scaled
        out_test = out_decoder_test_scaled

    # create data frame and save decoder output
    df_out_train = pd.DataFrame(data=out_train, columns=column_names)
    df_out_train.to_csv(train_decoder_file)

    df_out_test = pd.DataFrame(data=out_test, columns=column_names, index=df_in_test.index)
    df_out_test.to_csv(test_decoder_file)

    # get encoder output
    out_encoder_train = encoder.predict(df_in_train_scaled)
    out_encoder_test = encoder.predict(df_in_test_scaled)

    # save encoder output
    pd.DataFrame(out_encoder_train).to_csv(train_encoder_file)
    pd.DataFrame(out_encoder_test).to_csv(test_encoder_file)

    # save loss
    if loss_plot_file is not None:
        plot_loss(autoencoder, loss_plot_file)

    # plot comparison charts
    if chart_dir is not None:
        # create directory if not already exists
        if not os.path.exists(chart_dir):
            os.makedirs(chart_dir)

        plot_comparison(chart_dir, 'train_original', df_in_train, df_out_train)
        plot_comparison(chart_dir, 'train_scaled', df_in_train_scaled, df_out_decoder_train_scaled)
        plot_comparison(chart_dir, 'test_scaled', df_in_test_scaled, df_out_decoder_test_scaled)
        plot_comparison(chart_dir, 'test_original', df_in_test, df_out_test)

    # save model
    if tensorboard_dir is not None:
        autoencoder.save(tensorboard_dir + '/model_autoencoder.h5')
        encoder.save(tensorboard_dir + '/model_encoder.h5')

    # save to store
    if model_store_dir is not None:
        autoencoder.save(model_store_dir + '/model_autoencoder.h5')
        encoder.save(model_store_dir + '/model_encoder.h5')

    print('------------------ SAE End ----------------------')
    return autoencoder, encoder


def plot_loss(model, loss_plot_file):
    # training and validation loss
    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']

    fig, axs = plt.subplots(4)

    axs[0].plot(range(len(loss)), loss, 'bo', label='Training')
    axs[0].plot(range(len(val_loss)), val_loss, 'r+', label='Validation')
    axs[0].set_title('loss')
    axs[0].legend()

    loss = model.history.history['mean_absolute_percentage_error']
    val_loss = model.history.history['val_mean_absolute_percentage_error']

    axs[1].plot(range(len(loss)), loss, 'bo', label='Training')
    axs[1].plot(range(len(val_loss)), val_loss, 'r+', label='Validation')
    axs[1].set_title('mean absolute percentage error')
    axs[1].legend()

    loss = model.history.history['mean_squared_error']
    val_loss = model.history.history['val_mean_squared_error']
    axs[2].plot(range(len(loss)), loss, 'bo', label='Training')
    axs[2].plot(range(len(val_loss)), val_loss, 'r+', label='Validation')
    axs[2].set_title('mean squared error')
    axs[2].legend()

    loss = model.history.history['mean_absolute_error']
    val_loss = model.history.history['val_mean_absolute_error']
    axs[3].plot(range(len(loss)), loss, 'bo', label='Training')
    axs[3].plot(range(len(val_loss)), val_loss, 'r+', label='Validation')
    axs[3].set_title('mean absolute error')
    axs[3].legend()

    plt.savefig(loss_plot_file)


def plot_comparison(_directory, _key, df_in, df_out):
    column_names = df_in.columns

    total_loss = [0, 0]

    df_losses = pd.DataFrame(columns=['feature', 'mape', 'mse'])

    # skip first index column
    for column in range(len(column_names)):
        column_name = column_names[column]
        column_name_in = 'in_' + column_name
        column_name_out = 'out_' + column_name

        # create two columns of in and out data
        df_display = df_in.iloc[:, [column]].copy()
        df_display.columns = [column_name_in]
        out_data = df_out.iloc[:, column].copy()
        df_display[column_name_out] = out_data

        if df_display.isnull().values.any():
            print("Nan detected in extracting column " + column)

        # plot
        plot_file = _directory + _key + '_' + column_name + '.png'
        mape, mse = plot_inout(df_display, column_name, column_name_in, column_name_out, plot_file)
        total_loss = np.vstack((total_loss, [mape, mse]))

        # append to data frame
        df_losses = df_losses.append({'feature': column_names[column], 'mape': mape, 'mse': mse}, ignore_index=True)

    # save losses to csv
    df_losses.to_csv(_directory + _key + '_losses.csv')

    print('Total loss on data {}: {}'.format(_key, np.sum(total_loss, axis=0)))


def plot_inout(_df, name, in_name, out_name, plot_file):

    in_column = _df[in_name]
    out_column = _df[out_name]

    if in_column.isnull().values.any():
        print("Nan detected in input column when producing " + plot_file)
    elif out_column.isnull().values.any():
        print("Nan detected in output column when producing " + plot_file)

    # compute losses between in and out scaled features
    mape = skmet.mean_absolute_error(in_column, out_column)
    mse = skmet.mean_squared_error(in_column, out_column)

    print('----- Column: {} | MAPE: {:.4f} | MSE: {:.4f}'.format(name, mape, mse))

    # save plots
    ax = plt.gca()
    _df.plot(ax=ax)
    plt.title('SAE | Column: {} | MAPE: {:.4f} | MSE: {:.4f}'.format(name, mape, mse))
    plt.savefig(plot_file)
    plt.clf()

    return [mape, mse]


if __name__ == "__main__":
    # change this folder and input files
    directory = 'C:/temp/beacon/study_20190922_142437/repeat_1/run_0/'
    in_train_file = directory + 'run_0_train_dwt_denoised.csv'
    in_test_file = directory + 'run_0_test_dwt_denoised.csv'

    # put all output files in a sub folder
    file_prefix = directory + 'analyze_sae/'
    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)

    # test - input features after scaling
    in_scaled_test_file = file_prefix + 'test_dwt_denoised_scaled.csv'

    # test - decoder features output (scaled)
    predicted_scaled_test_file = file_prefix + 'test_sae_decoder_scaled.csv'

    # test - decoder features output (in original scale)
    predicted_test_file = file_prefix + 'test_sae_decoder.csv'

    dimensions = {
        'model_1': [15, 10]
    }

    for key, value in dimensions.items():

        config = {'hidden_dim': value,
                  'epochs': 1000,
                  'loss_metric': 'mean_absolute_error',
                  'batch_size': 17,
                  'learning_rate': 0.07
                  }

        with open(file_prefix + 'config.json', 'w') as outfile:
            json.dump(config, outfile, indent=4)

        autoencoder, encoder = fit_predict(config=config,
                                           train_in_file=in_train_file,
                                           train_in_scaled_file=file_prefix + "train_dwt_denoised_scaled.csv",
                                           train_encoder_file=file_prefix + 'train_sae_encoder.csv',
                                           train_decoder_scaled_file=file_prefix + 'train_sae_decoder_scaled.csv',
                                           train_decoder_file=file_prefix + 'train_sae_decoder.csv',
                                           test_in_file=in_test_file,
                                           test_in_scaled_file=in_scaled_test_file,
                                           test_encoder_file=file_prefix + 'test_sae_encoder.csv',
                                           test_decoder_scaled_file=predicted_scaled_test_file,
                                           test_decoder_file=predicted_test_file,
                                           loss_plot_file=file_prefix + 'plot_sae_loss.png',
                                           chart_dir=file_prefix + 'charts/',
                                           tensorboard_dir=file_prefix + 'tensorboard/')
