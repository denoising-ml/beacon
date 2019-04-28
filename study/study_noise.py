from random import gauss
from random import seed
from pandas import Series
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from pandas.plotting import lag_plot
from matplotlib import pyplot
import nolds
from scipy import stats

def visualise(_series):
    fig, axes = pyplot.subplots(nrows=2, ncols=3)

    axes[0, 0].title.set_text('Data')
    _series.plot(ax=axes[0, 0])

    axes[0, 1].title.set_text('Historgram')
    _series.hist(ax=axes[0, 1])

    axes[0, 2].title.set_text('Lag plot')
    lag_plot(_series, ax=axes[0, 2])

    axes[1, 0].title.set_text('Autocorrelation Function (ACF)')
    autocorrelation_plot(_series, ax=axes[1, 0])


    axes[1, 1].title.set_text('Partial correlation function (PACF)')
    plot_pacf(_series, lags=50, ax=axes[1, 1])

    pyplot.show()


def calc_acf(y, lags=50):
    lag_acf = acf(y, nlags=lags)
    return lag_acf


def calc_pacf(y, lags=50):
    lag_pacf = pacf(y, nlags=30, method='ols')
    return lag_pacf


def calc_aproxEntropy(y, N=2, r=0.2):
    """
    Calculate approximate entropy (ApEn)
    :param y: data
    :param N: size of sequence vector to compare
    :param r: distance threshold for two template vectors to be considered equal (default: 0.2 * std(data))
    :return: a float value of entropy (smaller number means more predictable i.e. more regular)
    """
    entropy = nolds.sampen(y, emb_dim=N, tolerance=r)
    return entropy


def summarise(_series, lags=50):

    print("Data statistics")
    print(stats.describe(_series))

    print("Autocorrelation coefficients")
    print(calc_acf(_series, lags=lags))

    print("Partial-autocorrelation coefficients")
    print(calc_pacf(_series, lags=lags))

    print("Sample approximate entropy = {}".format(calc_aproxEntropy(_series)))


if __name__ == "__main__":

    # seed random number generator
    seed(1)

    # create white noise series
    series = [gauss(0.0, 1.0) for i in range(1000)]

    series = Series(series)

    summarise(series)

    visualise(series)



