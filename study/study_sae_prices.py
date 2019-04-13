import pandas as pd
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split


df_data = pd.read_csv('../data/input/HSI_figshare.csv')
df_data = df_data.drop(['Date', 'Time'], axis=1)
df_data = df_data.loc[:, ['Closing Price', 'Open Price', 'High price', 'Low Price']]

# Split Data
X_train, X_test = train_test_split(df_data, train_size=0.75, shuffle=False)

wv_train = pd.DataFrame(X_train, index=None)
wv_train.to_csv("output/encoder/wv_train.csv")

wv_test = pd.DataFrame(X_test)
wv_test.to_csv("output/encoder/wv_test.csv")


def sae(hidden_dimensions):
    ncol = X_train.shape[1]
    print(ncol)
    input_dim = Input(shape=(ncol,))

    # encoder layers
    # first layer is input layer
    encoded = Dense(ncol, activation='relu')(input_dim)

    # hidden encoder layers
    last_encoded_layer = encoded
    for hidden_dim in hidden_dimensions:
        last_encoded_layer = Dense(hidden_dim, activation='relu')(last_encoded_layer)

    # hidden decoder layers, skip the inner most dimension
    last_decoded_layer = last_encoded_layer
    for hidden_dim in reversed(hidden_dimensions[:-1]):
        last_decoded_layer = Dense(10, activation='relu')(last_decoded_layer)

    # output layer
    last_decoded_layer = Dense(ncol, activation='linear')(last_decoded_layer)

    # autoencoder model
    autoencoder = Model(input=input_dim, output=last_decoded_layer)

    # encoder model
    encoder = Model(input=input_dim, output=last_encoded_layer)

    # configure and fit the model
    sgd = SGD(lr=0.01, decay=10e-5, momentum=0.9, nesterov=True)  # 0.001, 0.003, 0.01, 0.03, 0.1, 0.3
    autoencoder.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mse', 'mae', 'mape'])
    autoencoder.summary()
    history = autoencoder.fit(X_train,
                              X_train,
                              nb_epoch=100,
                              batch_size=60,
                              shuffle=False,
                              validation_data=(X_test, X_test))

    # output encoded and decoded output from train data
    encoded_train = pd.DataFrame(encoder.predict(X_train))
    encoded_train = encoded_train.add_prefix('feature_')
    print(encoded_train.head())
    encoded_train.to_csv("output/encoder/encoded_train_data.csv")

    decoded_train = pd.DataFrame(autoencoder.predict(X_train))
    decoded_train = decoded_train.add_prefix('decoded_')
    decoded_train.to_csv("output/encoder/decoded_train_data.csv")

    # output encoded and decoded output from test data
    encoded_test = pd.DataFrame(encoder.predict(X_test))
    encoded_test = encoded_test.add_prefix('feature_')
    encoded_test.head()
    encoded_test.to_csv("output/encoder/encoded_test_data.csv")

    decoded_test = pd.DataFrame(autoencoder.predict(X_test))
    decoded_test = decoded_test.add_prefix('decoded_')
    decoded_test.to_csv("output/encoder/decoded_test_data.csv")


if __name__ == "__main__":
    sae([2])
