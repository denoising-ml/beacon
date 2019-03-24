import pywt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    date_parser = lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce')
    df = pd.read_csv('JPM.csv', parse_dates=['date'], date_parser=date_parser, index_col=0)

    data = df["close"]

    wavelet_coeffs = pywt.wavedec(data, 'haar', level=2)
    np.savetxt("orig_coeffs.txt", wavelet_coeffs, fmt='%s')

    print(np.shape(wavelet_coeffs))

    # soft threshold on wavelet coefficients
    # https://dsp.stackexchange.com/questions/15823/feature-extraction-reduction-using-dwt
    threshold_coeffs = [None] * len(wavelet_coeffs)
    for (i, coeffs) in enumerate(wavelet_coeffs):
        threshold = np.std(coeffs)/2
        threshold_coeffs[i] = pywt.threshold(coeffs, threshold, mode='soft')

    np.savetxt("filtered_coeffs.txt", threshold_coeffs, fmt='%s')

    # reconstruct data using original coefficients
    data_rec1 = pywt.waverec(wavelet_coeffs, 'haar')
    df['data_rec1'] = pd.Series(data_rec1, index=df.index)

    # reconstruct data using threshold coefficients
    data_rec2 = pywt.waverec(threshold_coeffs, 'haar')
    df['data_rec2'] = pd.Series(data_rec2, index=df.index)

    df.to_csv('dwt_output.csv', sep=',')



