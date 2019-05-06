from typing import Dict

import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
        train_predicted_file: str,
        test_in_file: str,
        test_predicted_file: str,
        loss_plot_file: str = None
):
    print('------------------ SAE Start --------------------')
    df_in_train = pd.read_csv(train_in_file, index_col=0)
    df_in_test = pd.read_csv(test_in_file, index_col=0)

    print('In train: {} {}'.format(train_in_file, df_in_train.shape))
    print('In test: {} {}'.format(test_in_file, df_in_test.shape))
    print('Config: {}'.format(config))

    apply_scaler = config.get('scaler', True)

    if apply_scaler:
        scaler = MinMaxScaler()

        scaler.fit(df_in_train)
        df_in_train_scaled = scaler.transform(df_in_train)
        df_in_test_scaled = scaler.transform(df_in_test)
    else:
        df_in_train_scaled = df_in_train
        df_in_test_scaled = df_in_test

    hidden_dim = config.get('hidden_dim', [16, 8])
    epochs = config.get('epochs', 1000)

    autoencoder, encoder = sae(train_x=df_in_train_scaled,
                               validate_x=df_in_test_scaled,
                               epochs=epochs,
                               hidden_dimensions=hidden_dim)

    out_train_scaled = autoencoder.predict(df_in_train_scaled)
    out_test_scaled = autoencoder.predict(df_in_test_scaled)

    if apply_scaler:
        out_train = scaler.inverse_transform(out_train_scaled)
        out_test = scaler.inverse_transform(out_test_scaled)
    else:
        out_train = out_train_scaled
        out_test = out_test_scaled

    pd.DataFrame(out_train).to_csv(train_predicted_file)
    pd.DataFrame(out_test).to_csv(test_predicted_file)

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


if __name__ == "__main__":
    df_data = pd.read_csv('../../data/input/HSI_figshare.csv')
    df_data = df_data.drop(['date', 'time'], axis=1)

    in_train_file = '../output/encoder/tmp_in_train_data.csv'
    predicted_train_file = '../output/encoder/tmp_predicted_train_data.csv'
    in_test_file = '../output/encoder/tmp_in_test_data.csv'
    predicted_test_file = '../output/encoder/tmp_predicted_test_data.csv'
    loss_plot_file = '../output/encoder/tmp_sae_loss_plot.png'

    # Split Data
    x_train, x_test = train_test_split(df_data, train_size=0.75, shuffle=False)
    pd.DataFrame(x_train).to_csv(in_train_file)
    pd.DataFrame(x_test).to_csv(in_test_file)

    config = {'hidden_dim': [16, 8],
              'epochs': 10
              }

    fit_predict(config=config,
                train_in_file=in_train_file,
                train_predicted_file=predicted_train_file,
                test_in_file=in_test_file,
                test_predicted_file=predicted_test_file,
                loss_plot_file=loss_plot_file)

    df_in_test_data = pd.read_csv(in_test_file)
    df_out_test_data = pd.read_csv(predicted_test_file)

    df_display = df_in_test_data.iloc[:, [1]].copy()
    df_display.columns = ['in']
    df_display['out'] = df_out_test_data.iloc[:, 1].copy()
