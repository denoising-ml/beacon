from typing import Dict, List

import pywt
import pandas as pd
from statsmodels import robust
import math
import numpy as np

'''
Boundary problem and data leakage: a caveat for wavelet based forecasting:
https://www.jcer.or.jp/jcer_download_log.php?post_id=38552&file_post_id=38554
'''

def denoise(
        config: Dict,
        train_in_file: str,
        train_denoise_file: str,
        test_in_file: str,
        test_denoise_file: str,
        denoise_columns: List = None,
        test_cheat: bool = False):

    assert denoise_columns is not None

    print('------------------ DWT Start -------------------')
    df_in_train = pd.read_csv(train_in_file, index_col=0)
    df_in_test = pd.read_csv(test_in_file, index_col=0)

    print('In train: {} {}'.format(train_in_file, df_in_train.shape))
    print('In test: {} {}'.format(test_in_file, df_in_test.shape))
    print('Config: {}'.format(config))

    # If skip DWT
    if not config.get('apply_dwt', True):
        train_index = range(0, df_in_train.shape[0])
        df_in_train['new_index'] = train_index
        df_in_train.set_index(keys='new_index', inplace=True)
        df_in_train.to_csv(train_denoise_file)

        test_index = range(0, df_in_test.shape[0])
        df_in_test['new_index'] = test_index
        df_in_test.set_index(keys='new_index', inplace=True)
        df_in_test.to_csv(test_denoise_file)
        return

    # sanity checks
    assert (df_in_train.shape[0] % 2 == 0), "Number of training rows must be even"
    assert (df_in_test.shape[0] % 2 == 0), "Number of testing rows must be even"

    levels = config.get('dwt_levels', 4)
    mode = config.get('dwt_mode', 'hard')

    df_denoise_train = denoise_dataframe(df_in_train, denoise_columns, levels, mode)

    # denoise test set with moving window to prevent future leak
    if test_cheat:
        df_denoise_test = denoise_dataframe(df_in_test, denoise_columns, levels, mode)
    else:
        df_in_combine = pd.concat([df_in_train, df_in_test])
        window_size = len(df_in_train)
        df_denoise_test = denoise_dataframe_with_moving_window(df_in_combine,
                                                               denoise_columns,
                                                               window_size,
                                                               levels,
                                                               mode)

        # take only the last N rows of data, which belongs to test set
        df_denoise_test = df_denoise_test.tail(len(df_in_test))

    print('Denoise train: {} {}'.format(train_denoise_file, df_denoise_train.shape))
    print('Denoise test: {} {}'.format(test_denoise_file, df_denoise_test.shape))
    df_denoise_train.to_csv(train_denoise_file)
    df_denoise_test.to_csv(test_denoise_file)


def compute_threshold(coeffs):
    """
    Compute threshold using VisuShrink approach which employs a single universal threshold to all
    wavelet detail coefficients, as proposed by Donoho and Johnstone

    Args:
        coeffs: array of detail wavelet coefficients
        L: length of time series data

    Returns:
        A threshold (double)

    """
    sigma = robust.mad(coeffs) * 1.4826
    return sigma * math.sqrt(2 * math.log(len(coeffs)))


def dwt_denoise(data: pd.Series, levels: int, mode: str):
    """
    De-noise a time series using discrete wavelet transform

    Args:
        data: Panda series
        label: A string for data label

    Returns:
        An array of de-noise series

    """
    # print('-------------')
    # print(data)
    wavelet_coeffs = pywt.wavedec(data, 'haar', level=levels)
    # np.savetxt("output/dwt/orig_dwt_coeffs_{}.txt".format(label), wavelet_coeffs, fmt='%s')

    # compute threshold using last level of coefficients which mainly consists of noise
    threshold = compute_threshold(wavelet_coeffs[-1])
    threshold_coeffs = [None] * len(wavelet_coeffs)
    for (i, coeffs) in enumerate(wavelet_coeffs):
        # soft threshold on wavelet coefficients
        threshold_coeffs[i] = pywt.threshold(coeffs, threshold, mode=mode)

    # reconstruct data using thresholded coefficients
    return pywt.waverec(threshold_coeffs, 'haar')


def dwt_denoise_moving_window(_data_array, _window_size, _levels, _mode):
    _num_data = len(_data_array)
    _output = np.zeros(_num_data)
    _output[0:_window_size] = dwt_denoise(_data_array[0:_window_size], _levels, _mode)

    for i in range(_window_size, _num_data):
        from_index = i - _window_size + 1
        to_index = i + 1
        tmp_series = dwt_denoise(_data_array[from_index:to_index], _levels, _mode)
        _output[i] = tmp_series[-1]

    return _output


def denoise_dataframe(_df, _work_columns, _levels, _mode):
    # create an empty data frame
    _df_target = pd.DataFrame()

    for column in _df.columns:
        if column in _work_columns:
            series = pd.Series(dwt_denoise(data=_df[column], levels=_levels, mode=_mode))
            _df_target = pd.concat([_df_target, series.rename(column)], axis=1)
        else:
            _df_target[column] = pd.Series(_df[column].values)

    return _df_target


def denoise_dataframe_with_moving_window(_df, _work_columns, _window_size, _levels, _mode):
    # create an empty data frame
    _df_target = pd.DataFrame(index=_df.index)

    for column in _df.columns:
        if column in _work_columns:
            _data = dwt_denoise_moving_window(_df[column], _window_size, _levels, _mode)
            series = pd.Series(_data, index=_df.index)
            _df_target = pd.concat([_df_target, series.rename(column)], axis=1)
        else:
            _df_target[column] = _df[column].copy()

    return _df_target


if __name__ == "__main__":
    config = {}
    file_dir = '../output/study_20190424_174701/'

    denoise(config=config,
            train_in_file='../../data/input/HSI_figshare.csv',
            train_denoise_file=file_dir + 'train_denoise.csv',
            test_in_file='../../data/input/HSI_figshare.csv',
            test_denoise_file=file_dir + 'test_denoise.csv',
            denoise_columns=['close', 'open', 'high', 'low'])