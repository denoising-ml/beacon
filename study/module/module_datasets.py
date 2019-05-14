import pandas as pd
import os

def load_JPM():

    _filename = os.path.join(os.path.dirname(__file__), '../JPM.csv')
    date_parser = lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce')
    _df = pd.read_csv(_filename, parse_dates=['date'], date_parser=date_parser, index_col=0)
    return _df


def load_HSI():

    _filename = os.path.join(os.path.dirname(__file__), '../../data/input/HSI_figshare.csv')
    date_parser = lambda x: pd.to_datetime(x, format='%Y%m%d', errors='coerce')
    _df = pd.read_csv(_filename, parse_dates=['date'], date_parser=date_parser, index_col=0)
    return _df


if __name__ == "__main__":
    df = load_HSI()
    print(df.loc['2008-8-1':'2008-8-31'])

