import pywt
import pandas as pd

if __name__ == "__main__":
    date_parser = lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce')
    df = pd.read_csv('JPM.csv', parse_dates=['date'], date_parser=date_parser, index_col=0)

    print(df.head)

    wavelet = pywt.Wavelet('haar')
    print(wavelet)

    phi, psi, x = wavelet.wavefun(level=5)

