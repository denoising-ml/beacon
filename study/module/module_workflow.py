import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import study.module.module_sae as sae_layer
import study.module.module_lstm as lstm_layer
import study.module.module_pre_backtrader as pre_backtrader_layer
import study.module.module_backtrader as backtrader_layer
import study.module.module_dwt as dwt_layer
import json
import os


class StudyFilenames:
    def get_train_filename(self, desc: str):
        return self.get_filename('train', desc)

    def get_test_filename(self, desc: str):
        return self.get_filename('test', desc)

    def get_model_filename(self, desc: str):
        return self.get_filename('model', desc, 'h5')

    def get_filename(self, data_type: str, desc: str, file_type: str = 'csv'):
        return '{}/run_{}_{}_{}.{}'.format(
            self.directory, self.run_number, data_type, desc, file_type)

    def get_directory(self, folder):
        return '{}/{}/'.format(self.directory, folder)

    def __init__(self, run_number, study_number):
        self.run_number = run_number
        self.study_number = study_number

        # Create dir is not exist
        self.root = 'c:/temp/beacon/study_{}'.format(study_number)

        self.directory = self.root + '/run_{}'.format(run_number)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.config = self.get_filename('train', 'config', 'json')

        self.train_dates = self.get_train_filename('dates')
        self.test_dates = self.get_test_filename('dates')

        self.train_input = self.get_train_filename('input')
        self.test_input = self.get_test_filename('input')

        # DWT denoised output (scaled)
        self.train_dwt_denoised = self.get_train_filename('dwt_denoised')
        self.test_dwt_denoised = self.get_test_filename('dwt_denoised')

        # DWT original denoised output
        self.train_dwt_denoised_orig = self.get_train_filename('dwt_denoised_orig')
        self.test_dwt_denoised_orig = self.get_test_filename('dwt_denoised_orig')

        # SAE encoder scaled input features
        self.train_in_scaled_file = self.get_train_filename('sae_scaled_input')
        self.test_in_scaled_file = self.get_test_filename('sae_scaled_input')

        # SAE encoder output
        self.train_sae_encoder = self.get_train_filename('sae_encoder')
        self.test_sae_encoder = self.get_test_filename('sae_encoder')

        # SAE decoder scaled output
        self.train_decoder_scaled_file = self.get_train_filename('sae_decoder_scaled')
        self.test_decoder_scaled_file = self.get_test_filename('sae_decoder_scaled')

        # SAE decoder output
        self.train_sae_decoder = self.get_train_filename('sae_decoder')
        self.test_sae_decoder = self.get_test_filename('sae_decoder')

        self.sae_loss_plot = self.get_filename('plot', 'sae_loss', 'png')

        # LSTM inputs
        self.train_lstm_input = self.get_train_filename('lstm_input')
        self.test_lstm_input = self.get_test_filename('lstm_input')

        # LSTM label
        self.train_lstm_label = self.get_train_filename('lstm_label')
        self.test_lstm_label = self.get_test_filename('lstm_label')

        # LSTM prediction
        self.train_lstm_predict = self.get_train_filename('lstm_predict')
        self.test_lstm_predict = self.get_test_filename('lstm_predict')

        self.backtrader_mktdata = self.get_filename('backtrader', 'mktdata')

        self.plot_backtrader = self.get_filename('plot', 'backtrader', 'pdf')
        self.plot_pyfolio = self.get_filename('plot', 'pyfolio', 'pdf')
        self.plot_accuracy = self.get_filename('plot', 'accuracy', 'pdf')

        # model files
        self.model_autoencoder = self.get_model_filename('autoencoder')
        self.model_encoder = self.get_model_filename('encoder')
        self.model_lstm = self.get_model_filename('lstm')

        # TensorBoard logs
        self.tensorboard_sae = self.get_directory('tensorboard_sae')
        self.tensorboard_lstm = self.get_directory('tensorboard_lstm')

        # SAE comparison charts directory
        self.chart_sae = self.get_directory('chart_sae')


def generate_config(
        epochs=1000,
        sae_hidden_dim=None,
        sae_scaler=True,
        sae_loss_metric='mean_squared_error',
        sae_learning_rate=0.2,
        sae_batch_size=10,
        sae_hidden_activation='relu',
        lstm_cell_neurons=10,
        lstm_time_step=3,
        lstm_layers=1,
        lstm_batch_size=60):

    if sae_hidden_dim is None:
        sae_hidden_dim = [16, 8]

    config = {
        'dwt_layer': {
            'dwt_levels': 4,
            'dwt_mode': 'hard'
        },
        'sae_layer': {
            'hidden_dim': sae_hidden_dim,
            'epochs': epochs,
            'hidden_activation': sae_hidden_activation,
            'scaler': sae_scaler,
            'loss_metric': sae_loss_metric,
            'learning_rate': sae_learning_rate,
            'batch_size': sae_batch_size
        },
        'lstm_layer': {
            'epochs': epochs,
            'cell_neurons': lstm_cell_neurons,
            'time_step': lstm_time_step,
            'layers': lstm_layers,
            'batch_size': lstm_batch_size
        },
        'pre_backtrader_layer': {
        },
        'backtrader_layer': {
        }
    }

    return config


def study_hsi(config, run_number, study_number):
    _filenames = StudyFilenames(run_number, study_number)

    # Save config
    with open(_filenames.config, 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # Load raw data
    df_raw_data = pd.read_csv('../data/input/HSI_figshare.csv')

    # make a copy of date column
    df_dates = df_raw_data['date'].copy()

    # remove date column from input data
    df_raw_data = df_raw_data.drop(['date', 'time'], axis=1)

    # we split using an explicit index to ensure the data sets contain even rows
    split_index = 1526

    # Split train and test data
    x_train = df_raw_data.iloc[0:split_index, :]
    x_test = df_raw_data.iloc[split_index:, :]

    # Save input files
    pd.DataFrame(x_train).to_csv(_filenames.train_input)
    pd.DataFrame(x_test).to_csv(_filenames.test_input)

    # Split train and test dates
    dates_train = df_dates[0:split_index]
    dates_test = df_dates[split_index:]

    # Save dates index file
    pd.DataFrame(dates_train).to_csv(_filenames.train_dates)
    pd.DataFrame(dates_test).to_csv(_filenames.test_dates)

    # Study
    start(config, _filenames)


def start(_config, filenames):

    # Add engineered features
    add_engineered_features(filenames)

    # Remove some features
    remove_features(filenames)

    # DWT layer
    run_dwt(_config, filenames)

    # SAE layer
    run_sae(_config, filenames)

    # Prepare LSTM input
    prepare_lstm_input(filenames)

    # LSTM layer
    run_lstm(_config, filenames)

    # trading performance
    run_backtrader(_config, filenames)


def add_engineered_features(filenames):
    pass


def remove_features(_filenames):
    # load data
    df_in_train = pd.read_csv(_filenames.train_input, index_col=0)
    df_in_test = pd.read_csv(_filenames.test_input, index_col=0)

    columns_to_remove = ['Volume', 'ATR', 'HIBOR']

    for column_name in columns_to_remove:
        df_in_train = df_in_train.drop(column_name, 1)
        df_in_test = df_in_test.drop(column_name, 1)

    # save data
    df_in_train.to_csv(_filenames.train_input)
    df_in_test.to_csv(_filenames.test_input)


def run_dwt(_config, _filenames):
    dwt_config = _config['dwt_layer']

    dwt_layer.denoise(config=dwt_config,
                      train_in_file=_filenames.train_input,
                      train_denoise_file=_filenames.train_dwt_denoised,
                      test_in_file=_filenames.test_input,
                      test_denoise_file=_filenames.test_dwt_denoised,
                      denoise_columns=['close', 'open', 'high', 'low'])

    '''
    # Prepare LSTM label data using denoised close
    denoised_train = pd.read_csv(_filenames.train_dwt_denoised)
    denoised_test = pd.read_csv(_filenames.test_dwt_denoised)

    # make a copy
    denoised_train.to_csv(_filenames.train_dwt_denoised_orig, index=False)
    denoised_test.to_csv(_filenames.test_dwt_denoised_orig)

    x_train, y_train = prepare_label(denoised_train)
    x_test, y_test = prepare_label(denoised_test)

    x_train.to_csv(_filenames.train_dwt_denoised, index=False)
    pd.DataFrame(y_train).to_csv(_filenames.train_lstm_label)

    x_test.to_csv(_filenames.test_dwt_denoised, index=False)
    pd.DataFrame(y_test).to_csv(_filenames.test_lstm_label)
    '''


def run_sae(_config, _filenames):
    sae_config = _config['sae_layer']

    autoencoder, encoder = sae_layer.fit_predict(config=sae_config,
                                                 train_in_file=_filenames.train_dwt_denoised,
                                                 train_in_scaled_file=_filenames.train_in_scaled_file,
                                                 train_encoder_file=_filenames.train_sae_encoder,
                                                 train_decoder_scaled_file=_filenames.train_decoder_scaled_file,
                                                 train_decoder_file=_filenames.train_sae_decoder,
                                                 test_in_file=_filenames.test_dwt_denoised,
                                                 test_in_scaled_file=_filenames.test_in_scaled_file,
                                                 test_encoder_file=_filenames.test_sae_encoder,
                                                 test_decoder_scaled_file=_filenames.test_decoder_scaled_file,
                                                 test_decoder_file=_filenames.test_sae_decoder,
                                                 loss_plot_file=_filenames.sae_loss_plot,
                                                 tensorboard_dir=_filenames.tensorboard_sae,
                                                 chart_dir=_filenames.chart_sae)

    # save model
    autoencoder.save(_filenames.model_autoencoder)
    encoder.save(_filenames.model_encoder)


def prepare_lstm_input(_filenames):
    df_train_features = pd.read_csv(_filenames.train_sae_encoder, index_col=0)
    df_test_features = pd.read_csv(_filenames.test_sae_encoder, index_col=0)

    df_train_data = pd.read_csv(_filenames.train_input)
    df_test_data = pd.read_csv(_filenames.test_input)

    assert_same_rows(df_train_features, df_train_data)
    assert_same_rows(df_test_features, df_test_data)

    # Today's features are used to predict tomorrow's close.
    # Inputs          | Label
    # =============================
    # data[0]...     | close[1]
    # data[1]...     | close[2]
    # ...
    # data[T-1]...   | close[T]

    # remove last line from feature
    df_train_features = df_train_features.iloc[:-1]
    df_test_features = df_test_features.iloc[:-1]

    # The label, close[1] .. close[T]
    # the double square brackets [["close"]] are required to create a new data frame, ["close"] will create a series
    df_train_label = df_train_data[["close"]].copy()
    df_train_label = df_train_label.iloc[1:]

    df_test_label = df_test_data[["close"]].copy()
    df_test_label = df_test_label.iloc[1:]

    assert_same_rows(df_train_features, df_train_label)
    assert_same_rows(df_test_features, df_test_label)

    # save
    df_train_features.to_csv(_filenames.train_lstm_input)
    df_test_features.to_csv(_filenames.test_lstm_input)

    df_train_label.to_csv(_filenames.train_lstm_label)
    df_test_label.to_csv(_filenames.test_lstm_label)


def assert_same_rows(df1, df2):
    assert df1.shape[0] == df2.shape[0]


def run_lstm(_config, _filenames):
    lstm_config = _config['lstm_layer']

    model = lstm_layer.fit_predict(config=lstm_config,
                                   train_in_file=_filenames.train_lstm_input,
                                   train_expected_file=_filenames.train_lstm_label,
                                   train_predicted_file=_filenames.train_lstm_predict,
                                   test_in_file=_filenames.test_lstm_input,
                                   test_expected_file=_filenames.test_lstm_label,
                                   test_predicted_file=_filenames.test_lstm_predict)

    # save model
    model.save(_filenames.model_lstm)


def run_backtrader(_config, _filenames):
    matplotlib.use('Agg')

    pre_backtrader_config = _config['pre_backtrader_layer']

    pre_backtrader_layer.create_trading_file(config=pre_backtrader_config,
                                             test_dates_file=_filenames.test_dates,
                                             test_input_file=_filenames.test_input,
                                             test_predicted_file=_filenames.test_lstm_predict,
                                             backtrader_mkt_data_file=_filenames.backtrader_mktdata)

    backtrader_config = _config['backtrader_layer']

    backtrader_layer.run_backtrader(config=backtrader_config,
                                    backtrader_mkt_data_file=_filenames.backtrader_mktdata,
                                    backtrader_plot_file=_filenames.plot_backtrader,
                                    pyfolio_plot_file=_filenames.plot_pyfolio)

    # Plot graphs for manual visual verification
    df_in_test_data = pd.read_csv(_filenames.test_input)
    df_expected_test_data = pd.read_csv(_filenames.test_lstm_label)
    df_predicted_test_data = pd.read_csv(_filenames.test_lstm_predict)

    df_display = df_in_test_data.iloc[:, [1]].copy()
    df_display.columns = ['in']
    df_display['expected'] = df_expected_test_data.iloc[:, 1].copy()
    df_display['predicted'] = df_predicted_test_data.iloc[:, 1].copy()

    df_display.plot()
    # plt.show()

    plt.gcf().savefig(_filenames.plot_accuracy)
