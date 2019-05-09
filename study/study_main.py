import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import study.module.module_sae as sae_layer
import study.module.module_lstm as lstm_layer
import study.module.module_pre_backtrader as pre_backtrader_layer
import study.module.module_backtrader as backtrader_layer
import study.module.module_dwt as dwt_layer


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
        self.directory = 'c:/temp/beacon/study_{}'.format(study_number)
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


def run_study(
        config,
        run_number,
        study_number
):
    matplotlib.use('Agg')

    filenames = StudyFilenames(run_number, study_number)

    # Save config
    with open(filenames.config, 'w') as outfile:
        json.dump(config, outfile)

    # Load raw data
    df_raw_data = pd.read_csv('../data/input/HSI_figshare.csv')
    df_dates = df_raw_data['date'].copy()
    df_raw_data = df_raw_data.drop(['date', 'time'], axis=1)

    # Today's input data is used to predict tomorrow's close.
    # Prepare the data.
    raw_close = df_raw_data.loc[:, "close"]

    raw_close = raw_close.iloc[1:]
    df_dates = df_dates.iloc[:-1]
    df_raw_input = df_raw_data.iloc[:-1]

    # Split train and test data
    x_train, x_test, y_train, y_test = train_test_split(df_raw_input, raw_close, train_size=0.75, shuffle=False)
    dates_train, dates_test = train_test_split(df_dates, train_size=0.75, shuffle=False)

    # Prepare dates index file
    pd.DataFrame(dates_train).to_csv(filenames.train_dates)
    pd.DataFrame(dates_test).to_csv(filenames.test_dates)

    # Prepare input files
    pd.DataFrame(x_train).to_csv(filenames.train_input)
    pd.DataFrame(x_test).to_csv(filenames.test_input)

    # DWT layer
    dwt_config = config['dwt_layer']

    dwt_layer.denoise(config=dwt_config,
                      train_in_file=filenames.train_input,
                      train_denoise_file=filenames.train_dwt_denoised,
                      test_in_file=filenames.test_input,
                      test_denoise_file=filenames.test_dwt_denoised,
                      denoise_columns=['close', 'open', 'high', 'low'])

    # SAE layer
    sae_config = config['sae_layer']

    sae_layer.fit_predict(config=sae_config,
                          train_in_file=filenames.train_dwt_denoised,
                          train_encoder_file=filenames.train_sae_encoder,
                          train_decoder_file=filenames.train_sae_decoder,
                          test_in_file=filenames.test_dwt_denoised,
                          test_encoder_file=filenames.test_sae_encoder,
                          test_decoder_file=filenames.test_sae_decoder,
                          loss_plot_file=filenames.sae_loss_plot)

    # Prepare LSTM expected data
    pd.DataFrame(y_train).to_csv(filenames.train_lstm_label)
    pd.DataFrame(y_test).to_csv(filenames.test_lstm_label)

    # LSTM layer
    lstm_config = config['lstm_layer']

    lstm_layer.fit_predict(config=lstm_config,
                           train_in_file=filenames.train_sae_encoder,
                           train_expected_file=filenames.train_lstm_label,
                           train_predicted_file=filenames.train_lstm_predict,
                           test_in_file=filenames.test_sae_encoder,
                           test_expected_file=filenames.test_lstm_label,
                           test_predicted_file=filenames.test_lstm_predict)

    pre_backtrader_config = config['pre_backtrader_layer']

    pre_backtrader_layer.create_trading_file(config=pre_backtrader_config,
                                             test_dates_file=filenames.test_dates,
                                             test_input_file=filenames.test_input,
                                             test_predicted_file=filenames.test_lstm_predict,
                                             backtrader_mkt_data_file=filenames.backtrader_mktdata)

    backtrader_config = config['backtrader_layer']

    backtrader_layer.run_backtrader(config=backtrader_config,
                                    backtrader_mkt_data_file=filenames.backtrader_mktdata,
                                    backtrader_plot_file=filenames.plot_backtrader,
                                    pyfolio_plot_file=filenames.plot_pyfolio)

    # Plot graphs for manual visual verification
    df_in_test_data = pd.read_csv(filenames.test_input)
    df_expected_test_data = pd.read_csv(filenames.test_lstm_label)
    df_predicted_test_data = pd.read_csv(filenames.test_lstm_predict)

    df_display = df_in_test_data.iloc[:, [1]].copy()
    df_display.columns = ['in']
    df_display['expected'] = df_expected_test_data.iloc[:, 1].copy()
    df_display['predicted'] = df_predicted_test_data.iloc[:, 1].copy()

    df_display.plot()
    # plt.show()

    plt.gcf().savefig(filenames.plot_accuracy)


if __name__ == "__main__":

    study_number = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_number = 0
    epochs = 500

    hidden_layer_combo = [
        [16, 8, 8]
    ]

    for hidden_layer in hidden_layer_combo:
        config = generate_config(
            epochs=epochs,
            sae_hidden_dim=hidden_layer
        )
        run_number += 1
        run_study(config=config, run_number=run_number, study_number=study_number)
