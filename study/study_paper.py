import study.module.module_workflow as workflow
import study.module.module_datasets as datasets
from datetime import datetime
from dateutil.relativedelta import relativedelta
import study.module.module_datasets as datasets
import study.module.module_backtrader as backtrader
import pandas as pd


def run(_filenames, _start_date, training_years, validation_months, test_months):

    training_start_date = _start_date
    validation_start_date = training_start_date + relativedelta(years=training_years)
    test_start_date = validation_start_date + relativedelta(months=validation_months)

    training_end_date = validation_start_date - relativedelta(days=1)
    validation_end_date = test_start_date - relativedelta(days=1)
    test_end_date = test_start_date + relativedelta(months=+test_months) - relativedelta(days=1)

    config = workflow.generate_config(
        epochs=500,
        sae_hidden_dim=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        lstm_cell_neurons=8,
        lstm_time_step=4,
        lstm_batch_size=60
    )

    df = datasets.load_HSI()
    data_train = df[training_start_date: training_end_date]
    data_validation = df[validation_start_date: validation_end_date]
    data_test = df[test_start_date: test_end_date]

    print("Training period: {} - {}".format(training_start_date, training_end_date))

    print("Validation period: {} - {}".format(validation_start_date, validation_end_date))

    print("Test period: {} - {}".format(test_start_date, test_end_date))

    # remove first row if not even
    data_train = make_even_rows(data_train)
    data_validation = make_even_rows(data_validation)
    data_test = make_even_rows(data_test)

    # Save input files
    pd.DataFrame(data_train).to_csv(_filenames.train_input)
    pd.DataFrame(data_test).to_csv(_filenames.test_input)

    # Save dates index files
    dates_train = data_train.index
    dates_test = data_test.index
    pd.DataFrame(dates_train).to_csv(_filenames.train_dates)
    pd.DataFrame(dates_test).to_csv(_filenames.test_dates)

    # Study
    workflow.start(config, _filenames)


def make_even_rows(_df):
    if _df.shape[0] % 2 != 0:
        _df = _df.iloc[1:]
    return _df


if __name__ == "__main__":
    df = datasets.load_HSI()

    start_date = datetime(2008, 7, 1)

    study_number = datetime.now().strftime('%Y%m%d_%H%M%S')

    runs = 1

    trading_files = [None] * runs

    for i in range(runs):
        print('run_{}'.format(i))

        filenames = workflow.StudyFilenames(i, study_number)

        run(filenames, start_date, 2, 3, 3)
        start_date = start_date + relativedelta(months=+3)

        # collect file name of backtrader market data file
        trading_files[i] = filenames.backtrader_mktdata

    # concatenate all market data files into one
    i = 0
    master_file = filenames.root + '/master_trades.csv'
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
                              filenames.root + '/master_backtrader.pdf',
                              filenames.root + '/master_pyfolio.pdf')
