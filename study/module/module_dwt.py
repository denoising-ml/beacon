from typing import Dict, List

import pywt
import pandas as pd
from statsmodels import robust
import math


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


def dwt_denoise(data: pd.Series, label: str, levels: int, mode: str):
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

    # np.savetxt("output/dwt/thresholded_dwt_coeffs_{}.txt".format(label), threshold_coeffs, fmt='%s')
    # reconstruct data using thresholded coefficients
    # print(pywt.waverec(threshold_coeffs, 'haar'))
    # print(len(pywt.waverec(threshold_coeffs, 'haar')))
    return pywt.waverec(threshold_coeffs, 'haar')


def denoise(
        config: Dict,
        train_in_file: str,
        train_denoise_file: str,
        test_in_file: str,
        test_denoise_file: str,
        denoise_columns: List = None
):
    print('------------------ DWT Start -------------------')
    df_in_train = pd.read_csv(train_in_file, index_col=0)
    df_in_test = pd.read_csv(test_in_file, index_col=0)

    print('In train: {} {}'.format(train_in_file, df_in_train.shape))
    print('In test: {} {}'.format(test_in_file, df_in_test.shape))
    print('Config: {}'.format(config))

    levels = config.get('dwt_levels', 4)
    mode = config.get('dwt_mode', 'hard')

    df_denoise_train = pd.DataFrame()
    for column in df_in_train.columns:
        if denoise_columns is not None and column in denoise_columns:

            denoised = pd.Series(dwt_denoise(data=df_in_train[column],
                                             label=column,
                                             levels=levels,
                                             mode=mode))

            df_denoise_train[column] = denoised
        else:
            df_denoise_train[column] = df_in_train[column].copy()

    df_denoise_test = pd.DataFrame()
    for column in df_in_test.columns:
        if denoise_columns is not None and column in denoise_columns:
            denoised = pd.Series(dwt_denoise(data=df_in_test[column],
                                             label=column,
                                             levels=levels,
                                             mode=mode))
            if denoised.size > df_in_test[column].size:
                denoised = denoised[:-1]

            df_denoise_test[column] = denoised
        else:
            df_denoise_test[column] = pd.Series(df_in_test[column].values)

    print('Denoise train: {} {}'.format(train_denoise_file, df_denoise_train.shape))
    print('Denoise test: {} {}'.format(test_denoise_file, df_denoise_test.shape))
    df_denoise_train.to_csv(train_denoise_file)
    df_denoise_test.to_csv(test_denoise_file)


if __name__ == "__main__":
    config = {}
    file_dir = '../output/study_20190424_174701/'

    denoise(config=config,
            train_in_file='../../data/input/HSI_figshare.csv',
            train_denoise_file=file_dir + 'train_denoise.csv',
            test_in_file='../../data/input/HSI_figshare.csv',
            test_denoise_file=file_dir + 'test_denoise.csv',
            denoise_columns=['close', 'open', 'high', 'low'])
