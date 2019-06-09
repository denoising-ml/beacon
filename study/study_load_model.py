from keras.models import load_model

if __name__ == "__main__":
    model = load_model('C:/temp/beacon/study_20190609_211802/run_1/run_1_model_encoder.h5')

    model.summary()

    weights = model.layers[1].get_weights()

    print('Weights:')
    print(weights)


