from typing import Dict
import pandas as pd
import numpy as np


def create_trading_file(
        config: Dict,
        test_dates_file: str,
        test_input_file: str,
        test_predicted_file: str,
        backtrader_mkt_data_file: str
):
    print('------------------ Pre-Backtrader Start -------------------')

    df_dates_test = pd.read_csv(test_dates_file, index_col=1, parse_dates=True)
    df_input_test = pd.read_csv(test_input_file, index_col=0)
    df_input_test.reset_index(inplace=True)
    df_predicted_test = pd.read_csv(test_predicted_file, index_col=0)

    print('Dates test: {} {}'.format(test_dates_file, df_dates_test.shape))
    print('Input test: {} {}'.format(test_input_file, df_input_test.shape))
    print('Predicted test: {} {}'.format(test_predicted_file, df_predicted_test.shape))
    print('Config: {}'.format(config))

    # LSTM only starts predicting from the 3rd tick. 3 being the number of nodes.
    lstm_lag = df_input_test.shape[0] - df_predicted_test.shape[0]
    df_dates_test = df_dates_test.iloc[lstm_lag:]
    df_input_test = df_input_test.iloc[lstm_lag:]

    df_backtrader = df_input_test.loc[:, ['open', 'high', 'low', 'close']]
    df_backtrader.columns = ['open', 'high', 'low', 'close']
    df_backtrader['volume'] = 0
    df_backtrader['openinterest'] = 0

    df_backtrader.index = df_dates_test.index
    df_backtrader.index.names = ['datetime']

    df_trading = df_predicted_test.copy()
    df_trading.columns = ['predicted']
    df_trading.index = df_dates_test.index
    df_trading['close'] = df_backtrader['close'].copy()
    df_trading['signal'] = np.where(
        df_trading['predicted'].sub(df_trading['close']) > 0, 1, -1
    )
    df_trading['signal'] = np.where(
        df_trading['predicted'].sub(df_trading['close']) == 0, 0, df_trading['signal']
    )

    df_backtrader['signal'] = df_trading['signal'].copy()
    df_backtrader.to_csv(backtrader_mkt_data_file)

    print('------------------ Pre-Backtrader End -------------------')