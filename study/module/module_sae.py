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
        tensorboard_dir=None):

    assert train_x.ndim == 2
    assert validate_x.ndim == 2
    assert validate_x.shape[1] == train_x.shape[1]

    n_features = train_x.shape[1]
    input_dim = Input(shape=(n_features,))

    # encoder layers
    last_encoded_layer = input_dim
    for hidden_dim in hidden_dimensions:
        last_encoded_layer = Dense(hidden_dim, activation='relu')(last_encoded_layer)

    # hidden decoder layers, skip the inner most dimension
    last_decoded_layer = last_encoded_layer
    for hidden_dim in reversed(hidden_dimensions[:-1]):
        last_decoded_layer = Dense(hidden_dim, activation='relu')(last_decoded_layer)

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
):

    print('------------------ SAE Start --------------------')
    df_in_train = pd.read_csv(train_in_file, index_col=0)
    df_in_test = pd.read_csv(test_in_file, index_col=0)

    print('In train: {} {}'.format(train_in_file, df_in_train.shape))
    print('In test: {} {}'.format(test_in_file, df_in_test.shape))
    print('Config: {}'.format(config))

    apply_scaler = config.get('scaler', True)

    # normalize data if required
    if apply_scaler:
        # fit scaler using training data
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(df_in_train)

        df_in_train_scaled = scaler.transform(df_in_train)
        df_in_test_scaled = scaler.transform(df_in_test)

        scaler_train = scaler
        scaler_test = scaler

    else:
        df_in_train_scaled = df_in_train
        df_in_test_scaled = df_in_test

    if train_in_scaled_file is not None:
        pd.DataFrame(df_in_train_scaled).to_csv(train_in_scaled_file)
    if test_in_scaled_file is not None:
        pd.DataFrame(df_in_test_scaled).to_csv(test_in_scaled_file)

    hidden_dim = config.get('hidden_dim', [16, 8])
    epochs = config.get('epochs', 1000)

    autoencoder, encoder = sae(train_x=df_in_train_scaled,
                               validate_x=df_in_test_scaled,
                               epochs=epochs,
                               hidden_dimensions=hidden_dim,
                               tensorboard_dir=tensorboard_dir)

    # get decoder output
    out_decoder_train_scaled = autoencoder.predict(df_in_train_scaled)
    out_decoder_test_scaled = autoencoder.predict(df_in_test_scaled)

    if train_decoder_scaled_file is not None:
        pd.DataFrame(out_decoder_train_scaled).to_csv(train_decoder_scaled_file)
    if test_decoder_scaled_file is not None:
        pd.DataFrame(out_decoder_test_scaled).to_csv(test_decoder_scaled_file)

    # de-normalize decoder output if required
    if apply_scaler:
        out_train = scaler_train.inverse_transform(out_decoder_train_scaled)
        out_test = scaler_test.inverse_transform(out_decoder_test_scaled)
    else:
        out_train = out_decoder_train_scaled
        out_test = out_decoder_test_scaled

    # save decoder output
    pd.DataFrame(out_train).to_csv(train_decoder_file)
    pd.DataFrame(out_test).to_csv(test_decoder_file)

    # get encoder output
    out_encoder_train = encoder.predict(df_in_train_scaled)
    out_encoder_test = encoder.predict(df_in_test_scaled)

    # save encoder output
    pd.DataFrame(out_encoder_train).to_csv(train_encoder_file)
    pd.DataFrame(out_encoder_test).to_csv(test_encoder_file)

    # save loss
    if loss_plot_file is not None:
        plot_loss(autoencoder, loss_plot_file)

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


def study_sae():
    # change this folder and input files
    directory = 'C:/temp/beacon/study_20190619_131015/run_0/'
    in_train_file = directory + 'run_0_train_dwt_denoised.csv'
    in_test_file = directory + 'run_0_test_dwt_denoised.csv'

    # put all output files in a sub folder
    file_prefix = directory + 'sae/'
    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)

    # test - input features after scaling
    in_scaled_test_file = file_prefix + 'test_dwt_denoised_scaled.csv'

    # test - decoder features output (scaled)
    predicted_scaled_test_file = file_prefix + 'test_sae_decoder_scaled.csv'

    # test - decoder features output (in original scale)
    predicted_test_file = file_prefix + 'test_sae_decoder.csv'

    dimensions = {
        '10x5': [16, 16, 16, 16, 16, 16, 16, 16]
    }

    for key, value in dimensions.items():

        config = {'hidden_dim': value,
                  'epochs': 1000
                  }

        fit_predict(
            config=config,
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
        )

        # Plot predicted vs actual close

        df_in_test_data = pd.read_csv(in_test_file)
        df_in_scaled_test_data = pd.read_csv(in_scaled_test_file)
        df_out_scaled_test_data = pd.read_csv(predicted_scaled_test_file)
        df_out_test_data = pd.read_csv(predicted_test_file)

        column_names = df_in_test_data.columns

        losses_scaled = [0, 0]
        losses_orig = [0, 0]

        # skip first index column
        for column in range(1, len(column_names)):
            column_name = column_names[column]

            # create two columns of in and out scaled features
            df_display = df_in_scaled_test_data.iloc[:, [column]].copy()
            df_display.columns = ['in_scaled']
            df_display['out_scaled'] = df_out_scaled_test_data.iloc[:, column].copy()

            plot_file = file_prefix + key + '_scaled_' + column_name + '.png'
            losses = analyse_losses(df_display, column_name + '_scaled', 'in_scaled', 'out_scaled', plot_file)
            losses_scaled = np.vstack((losses_scaled, losses))

            # create two columns of in and out features in original scale
            df_display = df_in_test_data.iloc[:, [column]].copy()
            df_display.columns = ['in_original']

            df_display['out_original'] = df_out_test_data.iloc[:, column].copy()

            plot_file = file_prefix + key + '_original_' + column_name + '.png'
            losses = analyse_losses(df_display, column_name + '_original', 'in_original', 'out_original', plot_file)
            losses_orig = np.vstack((losses_orig, losses))

    print('Losses on scaled data: {}'.format(np.sum(losses_scaled, axis=0)))
    print('Losses on original data: {}'.format(np.sum(losses_orig, axis=0)))


def analyse_losses(_df, name, in_name, out_name, plot_file):
    # compute losses between in and out scaled features
    mape = skmet.mean_absolute_error(_df[in_name], _df[out_name])
    mse = skmet.mean_squared_error(_df[in_name], _df[out_name])

    print('----- Column: {} / MAPE: {:.4f} / MSE: {:.4f}'.format(name, mape, mse))

    # save plots
    ax = plt.gca()
    _df.plot(ax=ax)
    plt.title('Column: {} / SAE Losses/ MAPE: {:.4f} / MSE: {:.4f}'.format(name, mape, mse))
    plt.savefig(plot_file)
    plt.clf()

    return [mape, mse]


if __name__ == "__main__":
    study_sae()
