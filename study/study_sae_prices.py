import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def sae(train_x,
        validate_x,
        hidden_dimensions,
        optimizer="adadelta",
        loss="mean_squared_error",
        epochs=1000,
        last_activation="sigmoid"):
    assert train_x.ndim == 2
    assert validate_x.ndim == 2
    assert validate_x.shape[1] == train_x.shape[1]

    n_features = train_x.shape[1]
    print(n_features)
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
    last_decoded_layer = Dense(n_features, activation=last_activation)(last_decoded_layer)

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

    # output encoded and decoded output from train data
    encoded_train = pd.DataFrame(encoder.predict(train_x))
    encoded_train = encoded_train.add_prefix('feature_')
    print(encoded_train.head())
    encoded_train.to_csv("output/encoder/encoded_train_data.csv")

    decoded_train = pd.DataFrame(autoencoder.predict(train_x))
    decoded_train = decoded_train.add_prefix('decoded_')
    decoded_train.to_csv("output/encoder/decoded_train_data.csv")

    # output encoded and decoded output from validation data
    encoded_test = pd.DataFrame(encoder.predict(validate_x))
    encoded_test = encoded_test.add_prefix('feature_')
    encoded_test.head()
    encoded_test.to_csv("output/encoder/encoded_validate_data.csv")

    decoded_test = pd.DataFrame(autoencoder.predict(validate_x))
    decoded_test = decoded_test.add_prefix('decoded_')
    decoded_test.to_csv("output/encoder/decoded_validate_data.csv")

    return autoencoder, encoder


def plot_loss(model):
    # training loss
    loss = model.history.history['loss']

    # validation loss
    val_loss = model.history.history['val_loss']

    plt.figure()
    plt.plot(range(len(loss)), loss, 'bo', label='Training loss')
    plt.plot(range(len(val_loss)), val_loss, 'r+', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df_data = pd.read_csv('../data/input/HSI_figshare.csv')
    df_data = df_data.drop(['Date', 'Time'], axis=1)
    # df_data = df_data.loc[:, ['Closing Price', 'Open Price', 'High price', 'Low Price']]

    # scale the data
    scaler = MinMaxScaler()

    scaler.fit(df_data)
    df_data_scaled = scaler.transform(df_data)

    # Split Data
    X_train, X_test = train_test_split(df_data, train_size=0.75, shuffle=False)
    X_train_scaled, X_test_scaled = train_test_split(df_data_scaled, train_size=0.75, shuffle=False)

    wv_train = pd.DataFrame(X_train_scaled, index=None)
    wv_train.to_csv("output/encoder/scaled_train_data.csv")

    wv_test = pd.DataFrame(X_test_scaled)
    wv_test.to_csv("output/encoder/scaled_test_data.csv")

    df_data.iloc[len(X_train_scaled):, :].to_csv("output/encoder/orig_test_data.csv")

    structures = [
        [8],
        [16],
        [16, 8],
        [16, 12, 8]
    ]

    df_X_test_scaled = pd.DataFrame(data=X_test_scaled)
    df_display = df_X_test_scaled.iloc[:, [0]]
    df_display.columns = ['X_test_close']

    for index, structure in enumerate(structures):
        autoencoder, encoder = sae(X_train_scaled, X_test_scaled, structure)

        Y_test_scaled = autoencoder.predict(X_test_scaled)
        Y_test = scaler.inverse_transform(Y_test_scaled)

        df_Y_test_scaled = pd.DataFrame(data=Y_test_scaled)
        df_display['Y_test_close ' + str(structure)] = df_Y_test_scaled.iloc[:, 0]

    df_display.plot()
    plt.show()

    pd.DataFrame(autoencoder.predict(Y_test)).to_csv("output/encoder/predict_test_data.csv")

    # serialize model to JSON
    model_json = autoencoder.to_json()
    with open("output/encoder/price_model.json", "w") as json_file:
        json_file.write(model_json)

        # serialize weights to HDF5
        autoencoder.save_weights("output/encoder/price_model.h5")

    print("Saved model to disk")