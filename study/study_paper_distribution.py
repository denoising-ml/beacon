import study.module.module_workflow as workflow
from datetime import datetime
from dateutil.relativedelta import relativedelta
import study.module.module_utils as utils
import study.module.module_backtrader as backtrader
import pandas as pd
import matplotlib.pyplot as plt
from study_paper import run


def gen_paper_config():
    return gen_config(datetime(2008, 7, 1), 2*12, 3, 3, 4*5)


def gen_one_config():
    return gen_config(start_date=datetime(2013, 7, 1),
                      training_months=2*12,
                      validation_months=3,
                      test_months=3,
                      runs=1,
                      repeats=100)


def gen_config(start_date, training_months, validation_months, test_months, runs, repeats: int = 1):
    _config = {
        'start_date': start_date,
        'training_months': training_months,
        'validation_months': validation_months,
        'test_months': test_months,
        'runs': runs,
        'repeats': repeats
    }
    return utils.DotDict(_config)


def run_study():

    # df = datasets.load_HSI()

    # config = gen_paper_config()
    config = gen_one_config()

    study_number = datetime.now().strftime('%Y%m%d_%H%M%S')

    trading_files = []
    pyfolio_dirs = []

    start_date = config.start_date

    # go through each run period
    for i in range(config.runs):
        for j in range(config.repeats):
            print('run_{}_{}'.format(i, j))

            # generate file names config
            file_names = workflow.StudyFilenames(run_number=i, study_number=study_number, repeat_number=j)

            run(file_names, start_date, config.training_months, config.validation_months, config.test_months)

            # collect file name of backtrader market data file
            pyfolio_dirs.append(file_names.pyfolio_dir)

        # only use the last repetition to build the overall backtest
        # this can be changed, no specific reason last run is chosen
        trading_files.append(file_names.backtrader_mktdata)

        # move to next training start date
        start_date = start_date + relativedelta(months=+3)

    run_backtrader_for_all_runs(root_dir=file_names.root, trading_files=trading_files)

    gen_stats_for_all_runs(root_dir=file_names.root, pyfolio_dirs=pyfolio_dirs)


def gen_stats_for_all_runs(root_dir: str, pyfolio_dirs: list):
    print('------------------ Overall Stats Start -------------------')

    returns = []
    for pyfolio_dir in pyfolio_dirs:
        df_returns = pd.read_csv(pyfolio_dir + '/returns.csv', index_col=0)
        returns.append(df_returns.sum()[0])

    plt.gcf().clf()
    plt.hist(returns)
    plt.gcf().savefig(root_dir + '/overall_returns.png')

    print('------------------ Overall Stats End -------------------')


def run_backtrader_for_all_runs(root_dir: str, trading_files: list):

    # concatenate all market data files into one
    i = 0
    master_file = root_dir + '/master_trades.csv'
    with open(master_file, 'w') as outfile:
        for fname in trading_files:
            with open(fname) as infile:
                # only take header from first file, skip header for the rest
                if i > 0:
                    next(infile)
                i = i + 1

                for line in infile:
                    outfile.write(line)

    # calculate overall performance
    backtrader.run_backtrader(None,
                              master_file,
                              root_dir + '/master_backtrader.pdf',
                              root_dir + '/master_pyfolio.pdf')


if __name__ == "__main__":

    run_study()

    # Test stats generation
    # study_dir = 'C:/Temp/beacon/study_20190903_100727'
    # pyfolio_dirs = [study_dir + '/run_0/pyfolio', study_dir + '/run_1/pyfolio']
    # gen_stats_for_all_runs(root_dir=study_dir, pyfolio_dirs=pyfolio_dirs)