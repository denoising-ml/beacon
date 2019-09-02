"""
We study the difference between applying DWT to an entire data window vs a rolling window

"""
import study.module.module_datasets as datasets
from study.module.module_dwt import dwt_denoise
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def denoise(_data_array, _window_size):
    _num_data = len(_data_array)
    _output = np.zeros(_num_data)
    _output[0:window_size] = dwt_denoise(_data_array[0:_window_size], 4, 'hard')

    for i in range(_window_size, _num_data):
        from_index = i - window_size + 1
        to_index = i + 1
        tmp_series = dwt_denoise(_data_array[from_index:to_index], 4, 'hard')
        _output[i] = tmp_series[-1]

    return _output


if __name__ == "__main__":
    df = datasets.load_HSI()

    data = df["close"]
    num_data = len(data)

    # one pass
    s1 = pd.Series(dwt_denoise(data, 4, 'hard'), index=data.index)

    # moving window
    data_array = data.values
    window_size = 1500
    s2 = denoise(data_array, window_size)
    s2 = denoise(s2, window_size)

    # concatenate
    output = pd.concat([data, s1.rename('one_pass')], axis=1)
    output = pd.concat([output, pd.Series(s2, name="moving_window", index=data.index)], axis=1)

    sns.set_style("darkgrid")
    output.plot()
    plt.show()
