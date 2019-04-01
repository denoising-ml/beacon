import pywt
import pandas as pd
import numpy as np
from statsmodels import robust
import math


def compute_threshold(coeffs, L):
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
    return sigma * math.sqrt(2 * math.log(L))


def dwt_denoise(data, label):
    """
    De-noise a time series using discrete wavelet transform

    Args:
        data: Panda series
        label: A string for data label

    Returns:
        An array of de-noise series

    """

    wavelet_coeffs = pywt.wavedec(data, 'haar', level=2)
    np.savetxt("orig_dwt_coeffs_{}.txt".format(label), wavelet_coeffs, fmt='%s')

    # compute threshold using last level of coefficients which mainly consists of noise
    threshold = compute_threshold(wavelet_coeffs[-1], len(data))
    threshold_coeffs = [None] * len(wavelet_coeffs)
    for (i, coeffs) in enumerate(wavelet_coeffs):
        # soft threshold on wavelet coefficients
        threshold_coeffs[i] = pywt.threshold(coeffs, threshold, mode='soft')

    np.savetxt("thresholded_dwt_coeffs_{}.txt".format(label), threshold_coeffs, fmt='%s')

    # reconstruct data using thresholded coefficients
    return pywt.waverec(threshold_coeffs, 'haar')


if __name__ == "__main__":
    date_parser = lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce')
    df = pd.read_csv('JPM.csv', parse_dates=['date'], date_parser=date_parser, index_col=0)

    df['open_denoise'] = pd.Series(dwt_denoise(df["open"], "open"), index=df.index)
    df['high_denoise'] = pd.Series(dwt_denoise(df["high"], "high"), index=df.index)
    df['low_denoise'] = pd.Series(dwt_denoise(df["low"], "low"), index=df.index)
    df['close_denoise'] = pd.Series(dwt_denoise(df["close"], "close"), index=df.index)

    df.to_csv('dwt_output.csv', sep=',')



