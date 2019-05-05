import pywt
import pandas as pd
import numpy as np
from statsmodels import robust
import math
from study_noise import visualise, summarise


def trim_coefficients(wavelet_coeffs):
    """
    The time series can be denoised by removing some coefficients from the projections in details.
    Compute threshold using last level of coefficients

    :param
        wavelet_coeffs: array of DWT detailed coefficients
    :return:
        trimmed coefficients
    """
    threshold = compute_threshold(wavelet_coeffs[-1])
    threshold_coeffs = [None] * len(wavelet_coeffs)
    for (i, coeffs) in enumerate(wavelet_coeffs):
        # soft threshold on wavelet coefficients
        threshold_coeffs[i] = pywt.threshold(coeffs, threshold, mode='hard')

    return threshold_coeffs


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


def dwt_denoise(data, label, level=4, rounds=1):
    """
    De-noise a time series using discrete wavelet transform

    Args:
        data: Panda series
        level: A number of decomposition level
        label: A string for data label
        round: A number indicating rounds of denoising

    Returns:
        An array of de-noise series

    """

    # The components of the decomposition are divided into the approximation (a) and details (d) at different levels
    # Approximation represents the major feature of the signal and the details describe the detailed changes and noise.
    for i in range(rounds):
        wavelet_coeffs = pywt.wavedec(data, 'haar', level=level)
        np.savetxt("output/dwt/orig_dwt_coeffs_{}_{}.txt".format(label, i), wavelet_coeffs, fmt='%s')

        # trim coefficients
        threshold_coeffs = trim_coefficients(wavelet_coeffs)

        # save
        np.savetxt("output/dwt/thresholded_dwt_coeffs_{}_{}.txt".format(label, i), threshold_coeffs, fmt='%s')

        # reconstruct data using thresholded coefficients
        denoised_data = pywt.waverec(threshold_coeffs, 'haar')
        np.savetxt("output/dwt/denoise_{}_{}.txt".format(label, i), denoised_data, fmt='%s')

        data = pd.Series(denoised_data, index=data.index)

    return data


def study_JPM():
    date_parser = lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce')
    _df = pd.read_csv('JPM.csv', parse_dates=['date'], date_parser=date_parser, index_col=0)
    return _df


def study_HSI():
    date_parser = lambda x: pd.to_datetime(x, format='%Y%m%d', errors='coerce')
    _df = pd.read_csv('../data/input/HSI_figshare.csv', parse_dates=['date'], date_parser=date_parser, index_col=0)
    return _df


if __name__ == "__main__":
    df = study_HSI()

    df['open_denoise'] = pd.Series(dwt_denoise(df["open"], level=2, label="open", rounds=2), index=df.index)
    df['high_denoise'] = pd.Series(dwt_denoise(df["high"], level=2, label="high", rounds=2), index=df.index)
    df['low_denoise'] = pd.Series(dwt_denoise(df["low"], level=2, label="low", rounds=2), index=df.index)
    df['close_denoise'] = pd.Series(dwt_denoise(df["close"], level=2, label="close", rounds=2), index=df.index)

    df.to_csv('output/dwt/dwt_output.csv', sep=',')

    '''
    Two types of noise: white noise (constant power spanning all frequencies) and 
    colored noise (different power at different bands).
    One way to check quality of denoise is to test if the errors are serially correlated or not.
    A serially correlated series can be modeled as AR or MA process.
    If the errors still contain serial correlation, then some information is missed in the denoise series. 
    '''
    print("Close price errors analysis")
    print("===========================")
    close_errors = df["close"] - df['close_denoise']
    summarise(close_errors, lags=20)
    visualise(close_errors)

    print("Close price analysis")
    print("====================")
    summarise(df["close"])

    print("Denoised close price analysis")
    print("=============================")
    summarise(df["close_denoise"])





