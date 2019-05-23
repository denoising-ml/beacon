from typing import Dict

import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as skmet
import matplotlib.pyplot as plt


def sae(train_x,
        validate_x,
        hidden_dimensions,
        epochs,
        optimizer="adadelta",
        loss="mean_squared_error"):
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
    last_decoded_layer = Dense(n_features, activation='sigmoid')(last_decoded_layer)

    # autoencoder model
    autoencoder = Model(input=input_dim, output=last_decoded_layer)
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae', 'mape'])
    autoencoder.summary()

    # encoder model
    encoder = Model(input=input_dim, output=last_encoded_layer)

    # configure and fit the model
    autoencoder.fit(train_x,
                    train_x,
                    epochs=epochs,
                    batch_size=50,
                    shuffle=False,
                    validation_data=(validate_x, validate_x))

    return autoencoder, encoder


def fit_predict(
        config: Dict,
        train_in_file: str,
        train_encoder_file: str,
        train_decoder_file: str,
        test_in_file: str,
        test_encoder_file: str,
        test_decoder_file: str,
        loss_plot_file: str = None,
        train_in_scaled_file: str = None,
        train_decoder_scaled_file: str = None,
        test_in_scaled_file: str = None,
        test_decoder_scaled_file: str = None
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
        scaler = MinMaxScaler()
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
                               hidden_dimensions=hidden_dim)

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
    file_prefix = 'c:/Temp/beacon/study_sae/run_1_'
    in_test_file = file_prefix + 'test_dwt_denoised.csv'
    in_scaled_test_file = file_prefix + 'test_dwt_denoised_scaled.csv'
    predicted_scaled_test_file = file_prefix + 'test_sae_decoder_scaled.csv'
    predicted_test_file = file_prefix + 'test_sae_decoder.csv'

    dimensions = {
        '10x5': [10, 10, 10, 10, 10]
    }

    for key, value in dimensions.items():

        config = {'hidden_dim': value,
                  'epochs': 500
                  }

        fit_predict(
            config=config,
            train_in_file=file_prefix + 'train_dwt_denoised.csv',
            train_encoder_file=file_prefix + 'train_sae_encoder.csv',
            train_decoder_file=file_prefix + 'train_sae_decoder.csv',
            test_in_file=in_test_file,
            test_encoder_file=file_prefix + 'test_sae_encoder.csv',
            test_decoder_file=predicted_test_file,
            loss_plot_file=file_prefix + 'plot_sae_loss.png',
            test_in_scaled_file=in_scaled_test_file,
            test_decoder_scaled_file=predicted_scaled_test_file
        )

        # Plot predicted vs actual close

        df_in_test_data = pd.read_csv(in_test_file)
        df_in_scaled_test_data = pd.read_csv(in_scaled_test_file)
        df_out_scaled_test_data = pd.read_csv(predicted_scaled_test_file)
        df_out_test_data = pd.read_csv(predicted_test_file)

        column_names = df_in_test_data.columns.tolist()
        # df_display = df_in_test_data.iloc[:, [1]].copy()
        # df_display.columns = ['in']
        # df_display['out'] = df_out_test_data.iloc[:, 1].copy()

        all_mape = []
        all_mse = []

        for column in range(1, 19):
            plot_file = file_prefix + key + '_' + str(column) + '.png'

            df_display = df_in_scaled_test_data.iloc[:, [column]].copy()
            df_display.columns = ['in_scaled']
            df_display['out_scaled'] = df_out_scaled_test_data.iloc[:, column].copy()

            # mape = skmet.mean_absolute_error(df_in_scaled_test_data, df_out_scaled_test_data)
            # mse = skmet.mean_squared_error(df_in_scaled_test_data, df_out_scaled_test_data)
            mape = skmet.mean_absolute_error(df_display['in_scaled'], df_display['out_scaled'])
            mse = skmet.mean_squared_error(df_display['in_scaled'], df_display['out_scaled'])

            all_mape.append(mape)
            all_mse.append(mse)

            column_name = column_names[column]
            print('----- Column: {} / MAPE: {:.4f} / MSE: {:.4f}'.format(column_name, mape, mse))

            ax = plt.gca()
            df_display.plot(ax=ax)
            # plt.show()
            plt.title('Column: {} / SAE: {} / MAPE: {:.4f} / MSE: {:.4f}'.format(column_name, key, mape, mse))
            plt.savefig(plot_file)
            plt.clf()


if __name__ == "__main__":
    study_sae()