from sklearn.model_selection import train_test_split
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

    def get_filename(self, data_type: str, desc: str, file_type: str = 'csv'):
        return '{}/run_{}_{}_{}.{}'.format(
            self.directory, self.run_number, data_type, desc, file_type)

    def __init__(self, run_number, study_number):
        self.run_number = run_number
        self.study_number = study_number

        # Create dir is not exist
        self.directory = 'c:/temp/beacon/study_{}/run_{}'.format(study_number, run_number)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.config = self.get_filename('train', 'config', 'json')

        self.train_dates = self.get_train_filename('dates')
        self.test_dates = self.get_test_filename('dates')

        self.train_input = self.get_train_filename('input')
        self.test_input = self.get_test_filename('input')

        # DWT denoised output
        self.train_dwt_denoised = self.get_train_filename('dwt_denoised')
        self.test_dwt_denoised = self.get_test_filename('dwt_denoised')

        # SAE encoder output
        self.train_sae_encoder = self.get_train_filename('sae_encoder')
        self.test_sae_encoder = self.get_test_filename('sae_encoder')

        # SAE decoder output
        self.train_sae_decoder = self.get_train_filename('sae_decoder')
        self.test_sae_decoder = self.get_test_filename('sae_decoder')

        self.sae_loss_plot = self.get_filename('plot', 'sae_loss', 'png')

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


def generate_config(
        epochs=1000,
        sae_hidden_dim=None
):
    if sae_hidden_dim is None:
        sae_hidden_dim = [16, 8]

    config = {
        'dwt_layer': {
            'dwt_levels': 4,
            'dwt_mode': 'hard'
        },
        'sae_layer': {
            'hidden_dim': sae_hidden_dim,
            'epochs': epochs
        },
        'lstm_layer': {
            'epochs': epochs
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
        json.dump(config, outfile)

    # Load raw data
    df_raw_data = pd.read_csv('../data/input/HSI_figshare.csv')

    # make a copy of date column
    df_dates = df_raw_data['date'].copy()

    # remove date column from input data
    df_raw_data = df_raw_data.drop(['date', 'time'], axis=1)

    # Today's input data is used to predict tomorrow's close.
    # Inputs          | Label
    # =============================
    # data[0]...     | close[1]
    # data[1]...     | close[2]
    # ...
    # data[T-1]...   | close[T]
    #
    # The inputs, data[0] .. data[T-1]
    df_dates = df_dates.iloc[:-1]
    df_raw_input = df_raw_data.iloc[:-1]

    # The label, close[1] .. close[T]
    label_close = df_raw_data.loc[:, "close"]
    label_close = label_close.iloc[1:]

    # Split train and test data
    x_train, x_test, y_train, y_test = train_test_split(df_raw_input, label_close, train_size=0.75, shuffle=False)
    dates_train, dates_test = train_test_split(df_dates, train_size=0.75, shuffle=False)

    # Prepare dates index file
    pd.DataFrame(dates_train).to_csv(_filenames.train_dates)
    pd.DataFrame(dates_test).to_csv(_filenames.test_dates)

    # Prepare input files
    pd.DataFrame(x_train).to_csv(_filenames.train_input)
    pd.DataFrame(x_test).to_csv(_filenames.test_input)

    # study
    start(config, _filenames)


def start(_config, filenames):

    # DWT layer
    run_dwt(_config, filenames)

    # SAE layer
    run_sae(_config, filenames)

    # LSTM layer
    run_lstm(_config, filenames)

    # trading performance
    run_backtrader(_config, filenames)


def run_dwt(_config, _filenames):
    dwt_config = _config['dwt_layer']

    dwt_layer.denoise(config=dwt_config,
                      train_in_file=_filenames.train_input,
                      train_denoise_file=_filenames.train_dwt_denoised,
                      test_in_file=_filenames.test_input,
                      test_denoise_file=_filenames.test_dwt_denoised,
                      denoise_columns=['close', 'open', 'high', 'low'])

    # Prepare LSTM label data using denoised close
    denoised_train = pd.read_csv(_filenames.train_dwt_denoised)
    denoised_test = pd.read_csv(_filenames.test_dwt_denoised)
    lstm_train_label = denoised_train['close']
    lstm_test_label = denoised_test['close']
    pd.DataFrame(lstm_train_label).to_csv(_filenames.train_lstm_label)
    pd.DataFrame(lstm_test_label).to_csv(_filenames.test_lstm_label)


def run_sae(_config, _filenames):
    sae_config = _config['sae_layer']

    sae_layer.fit_predict(config=sae_config,
                          train_in_file=_filenames.train_dwt_denoised,
                          train_encoder_file=_filenames.train_sae_encoder,
                          train_decoder_file=_filenames.train_sae_decoder,
                          test_in_file=_filenames.test_dwt_denoised,
                          test_encoder_file=_filenames.test_sae_encoder,
                          test_decoder_file=_filenames.test_sae_decoder,
                          loss_plot_file=_filenames.sae_loss_plot)


def run_lstm(_config, _filenames):
    lstm_config = _config['lstm_layer']

    lstm_layer.fit_predict(config=lstm_config,
                           train_in_file=_filenames.train_sae_encoder,
                           train_expected_file=_filenames.train_lstm_label,
                           train_predicted_file=_filenames.train_lstm_predict,
                           test_in_file=_filenames.test_sae_encoder,
                           test_expected_file=_filenames.test_lstm_label,
                           test_predicted_file=_filenames.test_lstm_predict)


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