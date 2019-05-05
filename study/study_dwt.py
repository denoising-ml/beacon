import pywt
import pandas as pd
import numpy as np
from statsmodels import robust
import math
from study_noise import visualise, summarise

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


def dwt_denoise(data, label):
    """
    De-noise a time series using discrete wavelet transform

    Args:
        data: Panda series
        label: A string for data label

    Returns:
        An array of de-noise series

    """

    # The components of the decomposition are divided into the approximation (a) and details (d) at different levels
    # Approximation represents the major feature of the signal and the details describe the detailed changes and noise.
    wavelet_coeffs = pywt.wavedec(data, 'haar', level=4)
    np.savetxt("output/dwt/orig_dwt_coeffs_{}.txt".format(label), wavelet_coeffs, fmt='%s')

    # The time series can be denoised by removing some coefficients from the projections in details.
    # Compute threshold using last level of coefficients
    threshold = compute_threshold(wavelet_coeffs[-1])
    threshold_coeffs = [None] * len(wavelet_coeffs)
    for (i, coeffs) in enumerate(wavelet_coeffs):
        # soft threshold on wavelet coefficients
        threshold_coeffs[i] = pywt.threshold(coeffs, threshold, mode='hard')

    np.savetxt("output/dwt/thresholded_dwt_coeffs_{}.txt".format(label), threshold_coeffs, fmt='%s')

    # reconstruct data using thresholded coefficients
    denoised_data = pywt.waverec(threshold_coeffs, 'haar')
    np.savetxt("output/dwt/denoise_{}.txt".format(label), denoised_data, fmt='%s')

    return denoised_data


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

    df['open_denoise'] = pd.Series(dwt_denoise(df["open"], "open"), index=df.index)
    df['high_denoise'] = pd.Series(dwt_denoise(df["high"], "high"), index=df.index)
    df['low_denoise'] = pd.Series(dwt_denoise(df["low"], "low"), index=df.index)
    df['close_denoise'] = pd.Series(dwt_denoise(df["close"], "close"), index=df.index)

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





