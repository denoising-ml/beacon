from typing import Dict
import pandas as pd
import keras
from keras.layers import Input, Dense
from keras.models import Model
import sklearn.preprocessing as preprocessing
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import os
import numpy as np


def sae(train_x,
        validate_x,
        hidden_dimensions,
        epochs,
        optimizer="adadelta",
        loss="mean_squared_error",
        #kernel_initializer=keras.initializers.Constant(value=0.2),
        kernel_initializer='random_uniform',
        bias_initializer='zeros',
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
                                   activation='relu',
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer)(last_encoded_layer)

    # hidden decoder layers, skip the inner most dimension
    last_decoded_layer = last_encoded_layer
    for hidden_dim in reversed(hidden_dimensions[:-1]):
        last_decoded_layer = Dense(hidden_dim,
                                   activation='relu',
                                   kernel_initializer=keras.initializers.Constant(value=0.1),
                                   bias_initializer='zeros')(last_decoded_layer)

    # output layer
    last_decoded_layer = Dense(n_features, activation='linear')(last_decoded_layer)

    # autoencoder model
    autoencoder = Model(input=input_dim, output=last_decoded_layer)
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae', 'mape'])
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
                    batch_size=50,
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
        chart_dir: str = None
):

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
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(df_in_train)

        # transform data
        in_train_scaled = scaler.transform(df_in_train)
        in_test_scaled = scaler.transform(df_in_test)

        scaler_train = scaler
        scaler_test = scaler

        # create data frame from numpy array
        df_in_train_scaled = pd.DataFrame(data=in_train_scaled, columns=column_names)
        df_in_test_scaled = pd.DataFrame(data=in_test_scaled, columns=column_names)
    else:
        df_in_train_scaled = df_in_train
        df_in_test_scaled = df_in_test

    if train_in_scaled_file is not None:
        df_in_train_scaled.to_csv(train_in_scaled_file)
    if test_in_scaled_file is not None:
        df_in_test_scaled.to_csv(test_in_scaled_file)

    hidden_dim = config.get('hidden_dim', [16, 8])
    epochs = config.get('epochs', 1000)

    autoencoder, encoder = sae(train_x=in_train_scaled,
                               validate_x=in_test_scaled,
                               epochs=epochs,
                               hidden_dimensions=hidden_dim,
                               tensorboard_dir=tensorboard_dir)

    # get decoder output
    out_decoder_train_scaled = autoencoder.predict(in_train_scaled)
    out_decoder_test_scaled = autoencoder.predict(in_test_scaled)

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

    df_out_test = pd.DataFrame(data=out_test, columns=column_names)
    df_out_test.to_csv(test_decoder_file)

    # get encoder output
    out_encoder_train = encoder.predict(in_train_scaled)
    out_encoder_test = encoder.predict(in_test_scaled)

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

    print('------------------ SAE End ----------------------')
    return autoencoder, encoder


def plot_loss(model, loss_plot_file):
    # training loss
    loss = model.history.history['loss']

    # validation loss
    val_loss = model.history.history['val_loss']

    figure = plt.gcf()
    figure.clf()

    plt.plot(range(len(loss)), loss, 'bo', label='Training loss')
    plt.plot(range(len(val_loss)), val_loss, 'r+', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(loss_plot_file)
    figure.clf()
    # plt.show()


def plot_comparison(directory, key, df_in, df_out):
    column_names = df_in.columns

    total_loss = [0, 0]

    # skip first index column
    for column in range(len(column_names)):
        column_name = column_names[column]
        column_name_in = 'in_' + column_name
        column_name_out = 'out_' + column_name

        # create two columns of in and out data
        df_display = df_in.iloc[:, [column]].copy()
        df_display.columns = [column_name_in]
        df_display[column_name_out] = df_out.iloc[:, column].copy()

        # plot
        plot_file = directory + key + '_' + column_name + '.png'
        losses = plot_inout(df_display, column_name, column_name_in, column_name_out, plot_file)
        total_loss = np.vstack((total_loss, losses))

    print('Total loss on data {}: {}'.format(key, np.sum(total_loss, axis=0)))


def plot_inout(_df, name, in_name, out_name, plot_file):

    in_column = _df[in_name]
    out_column = _df[out_name]

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


def study_sae():
    # change this folder and input files
    directory = 'C:/temp/beacon/study_20190623_201725/run_0/'
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
        '10x5': [10, 10]
    }

    for key, value in dimensions.items():

        config = {'hidden_dim': value,
                  'epochs': 1000
                  }

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
                                           chart_dir=file_prefix + 'charts/')

        # save model
        autoencoder.save(file_prefix + 'model_autoencoder.h5')
        encoder.save(file_prefix + 'model_encoder.h5')

        # Plot predicted vs actual using test data
        df_in_test_data = pd.read_csv(in_test_file)
        df_in_scaled_test_data = pd.read_csv(in_scaled_test_file)
        df_out_scaled_test_data = pd.read_csv(predicted_scaled_test_file)
        df_out_test_data = pd.read_csv(predicted_test_file)

        plot_comparison(file_prefix, key + '_test_original', df_in_test_data, df_out_test_data)
        plot_comparison(file_prefix, key + '_test_scaled', df_in_scaled_test_data, df_out_scaled_test_data)


if __name__ == "__main__":
    study_sae()
