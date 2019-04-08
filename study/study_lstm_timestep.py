from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed

n_neurons = 5
np.random.seed(20190604)


def create_data(length):
    # the prediction problem is to output the same number
    x = array([i / float(length) for i in range(length)])
    return x, x


def one_to_one(x, y):
    # input_shape has the structure of (time_step, features)
    # output shape refers to y (ignore the batch size)
    # in one-to-one case, input_shape = (1, 1), output_shape = (1)
    # we pass in each item one at a time and provide a single number as output/label
    # effectively single time step
    n = len(x)
    x = x.reshape(n, 1, 1)
    y = y.reshape(n, 1)

    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x, y, epochs=1000, batch_size=n)
    return model


def many_to_one(x, y):
    # input_shape = (5, 1), output_shape = (1, 5)
    # we pass in a sequence of items (5 time steps), and provide a single vector of labels as output
    # so from a sequence of 5 inputs, we want to predict a single vector of 5 output at the end of sequence
    n = len(x)
    x = x.reshape(1, n, 1)
    y = y.reshape(1, n)

    model = Sequential()

    # by setting return_sequences=False we only produce an output at the last time step
    model.add(LSTM(n_neurons, input_shape=(n, 1), return_sequences=False))
    model.add(Dense(n))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x, y, epochs=1000, batch_size=1)
    return model


def many_to_many(x, y):
    # input_shape = (5, 1), output_shape = (5, 1)
    # we pass in a sequence of items, and predict an output at each time step
    n = len(x)
    x = x.reshape(1, n, 1)
    y = y.reshape(1, n, 1)

    model = Sequential()

    # by setting return_sequences=True, an output will be produced at each time step
    # that's why the model expect Y to be of shape (n, 1) so there is one label per time step
    model.add(LSTM(n_neurons, input_shape=(n, 1), return_sequences=True))

    # without time distributed layer, the dense layer will have one set of weights/bias per time step
    # so we do not take much advantage of the full capability for sequencing learning
    # by having the time distributed layer before dense layer, there is just one set of weights/bias shared by
    # all time steps
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x, y, epochs=1000, batch_size=1)
    return model


def study():
    x, y = create_data(5)

    predict_data = [0.2, 0.3, 0.6, 0.7, 0.9]

    model = one_to_one(x, y)
    o2o_predict = model.predict(np.reshape(predict_data, (-1, 1, 1)))

    model = many_to_one(x, y)
    m2o_predict = model.predict(np.reshape(predict_data, (1, 5, 1)), batch_size=1)

    model = many_to_many(x, y)
    m2m_predict = model.predict(np.reshape(predict_data, (1, 5, 1)), batch_size=1)

    print("One-to-one prediction")
    print(o2o_predict)

    print("Many-to-one prediction")
    print(m2o_predict)

    print("Many-to-many prediction")
    print(m2m_predict)


if __name__ == "__main__":
    study()