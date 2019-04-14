# mlp for regression with mse loss function
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot


if __name__ == "__main__":
    # generate regression dataset
    X, orig_y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
    y = orig_y.reshape(len(orig_y), 1)

    # standardize dataset
    X = StandardScaler().fit_transform(X)

    y_scaler = StandardScaler()
    y_scaler.fit(y)
    y = y_scaler.transform(y)[:, 0]

    # split into train and test
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    # define model
    model = Sequential()
    model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=opt)

    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)

    # evaluate the model
    train_mse = model.evaluate(trainX, trainy, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

    # plot loss during training
    pyplot.title('Loss / Mean Squared Error')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # plot prediction vs actual
    predict_y = model.predict(testX)
    predict_yy = y_scaler.inverse_transform(predict_y)
    pyplot.plot(predict_yy, label="prediction")
    pyplot.plot(orig_y[n_train:], label="actual")
    pyplot.legend()
    pyplot.show()


