from keras.models import load_model

if __name__ == "__main__":
    model = load_model('C:/temp/beacon/study_20190623_172123/run_0/analyze_sae/model_encoder.h5')
    #model = load_model('C:/temp/beacon/study_20190623_172123/run_0/run_0_model_encoder.h5')

    print(model.summary())

    for layer in model.layers:
        weights = layer.get_weights()  # list of numpy arrays
        print(weights)
