import backtrader as bt
import pandas as pd
from typing import Dict
from backtrader.feeds import GenericCSVData
from study.module.backtrader_report import BacktraderReport
from study.module.pyfolio_analyzer import PyfolioAnalyzer
from study.module.pyfolio_report import PyfolioReport
import os


class GenericCSVWithSignal(GenericCSVData):
    # Add a more lines to the inherited ones from the base class
    lines = ('signal',)

    # openinterest in GenericCSVData has index 6 ... add 1
    # add the parameter to the parameters inherited from the base class
    params = (('signal', 7),)


class SignalStrategy(bt.Strategy):
    def __init__(self):
        pass

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                print('Buy Executed. Price: {:.2f}, Cost: {:.2f}, Comm {:.2f}'.format(
                    order.executed.price,
                    order.executed.value,
                    order.executed.comm))

            else:  # Sell
                print('Sell Executed. Price: {:.2f}, Cost: {:.2f}, Comm {:.2f}'.format(
                    order.executed.price,
                    order.executed.value,
                    order.executed.comm))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('Order Canceled/Margin/Rejected')

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        print('Operating PnL: Gross: {:.2f}. Net: {:.2f}'.format(trade.pnl, trade.pnlcomm))

    def next(self):
        # print('Close: {:.2f}. Signal: {:.0f}'.format(self.data.close[0], self.data.signal[0]))

        if self.data.signal[0] > 0:
            if self.position.size <= 0:
                print('Buy')
                self.order_target_percent(target=1.0)
        elif self.data.signal[0] < 0:
            if self.position.size >= 0:
                print('Sell')
                self.order_target_percent(target=-1.0)
        else:
            if self.position.size != 0:
                print('Neutralize')
                self.order_target_percent(target=0.0)


def run_backtrader(
        config: Dict,
        backtrader_mkt_data_file: str,
        backtrader_plot_file: str = None,
        pyfolio_plot_file: str = None,
        pyfolio_dir: str = None):

    print('------------------ Backtrader Start -------------------')

    date_parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce')
    df_mktdata = pd.read_csv(backtrader_mkt_data_file, index_col=0, parse_dates=['datetime'], date_parser=date_parser)

    print('Mkt Data: {} {}'.format(backtrader_mkt_data_file, df_mktdata.shape))

    if config is not None:
        print('Config: {}'.format(config))

    # Run backtrader
    cerebro = bt.Cerebro()
    cerebro.adddata(GenericCSVWithSignal(dataname=backtrader_mkt_data_file, dtformat=('%Y-%m-%d'), timeframe=bt.TimeFrame.Days))

    # disable margin/cash before accepting an order into the system
    cerebro.broker.set_checksubmit(False)
    # Set our desired cash start
    cerebro.broker.setcash(1000000.0)
    # Set the commission
    cerebro.broker.setcommission(commission=0.0)
    # Print out the starting conditions
    print('Starting Portfolio Value: {:.2f}'.format(cerebro.broker.getvalue()))

    # add Pyfolio analyzer
    cerebro.addanalyzer(PyfolioAnalyzer, _name='pyfolio')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharperatio')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')

    cerebro.addstrategy(SignalStrategy)

    strats = cerebro.run()
    strategy = strats[0]

    if pyfolio_dir is not None:
        # Pyfolio output in csv format
        if not os.path.exists(pyfolio_dir):
            os.makedirs(pyfolio_dir)

        pyfoliozer = strategy.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        returns.to_csv(pyfolio_dir + '/returns.csv')
        positions.to_csv(pyfolio_dir + '/positions.csv')
        transactions.to_csv(pyfolio_dir + '/transactions.csv')
        gross_lev.to_csv(pyfolio_dir + '/gross_leverage.csv')

    # Print out the final result
    print('Final Portfolio Value: {:.2f}'.format(cerebro.broker.getvalue()))

    if backtrader_plot_file is not None:
        # Backtrader report
        BacktraderReport().run_report(strategy=strategy, outfilename=backtrader_plot_file)

    if pyfolio_plot_file is not None:
        # Pyfolio report
        PyfolioReport().createPyfolioReport(strategy.analyzers.getbyname('pyfolio'), pyfolio_plot_file)

    # cerebro.plot()

    print('------------------ Backtrader End -------------------')


if __name__ == "__main__":
    config = {
    }

    directory = 'C:/temp/beacon/study_20190901_172240/run_0/'

    run_backtrader(config=config,
                   backtrader_mkt_data_file=directory + 'run_0_backtrader_mktdata.csv',
                   backtrader_plot_file=directory + 'backtrader_report.pdf',
                   pyfolio_plot_file=directory + 'pyfolio_report.pdf',
                   pyfolio_dir=directory + 'pyfolio')
