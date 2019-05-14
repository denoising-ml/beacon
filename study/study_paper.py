import study.module.module_workflow as workflow
import study.module.module_datasets as datasets
from datetime import datetime
from dateutil.relativedelta import relativedelta
import study.module.module_datasets as datasets
import pandas as pd


def prepare_label(dataframe):
    # Today's input data is used to predict tomorrow's close.
    # Inputs          | Label
    # =============================
    # data[0]...     | close[1]
    # data[1]...     | close[2]
    # ...
    # data[T-1]...   | close[T]

    # The inputs, data[0] .. data[T-1]
    df_input = dataframe.iloc[:-1]

    # The label, close[1] .. close[T]
    df_label = dataframe.loc[:, "close"]
    df_label = df_label.iloc[1:]

    return df_input, df_label


def run(_filenames, _start_date, training_years, validation_months, test_months):

    training_start_date = _start_date
    validation_start_date = training_start_date + relativedelta(years=training_years)
    test_start_date = validation_start_date + relativedelta(months=validation_months)

    training_end_date = validation_start_date - relativedelta(days=1)
    validation_end_date = test_start_date - relativedelta(days=1)
    test_end_date = test_start_date + relativedelta(months=+test_months) - relativedelta(days=1)

    config = workflow.generate_config(
        epochs=500,
        sae_hidden_dim=[16, 8, 8]
    )

    df = datasets.load_HSI()
    data_train = df[training_start_date: training_end_date]
    data_validation = df[validation_start_date: validation_end_date]
    data_test = df[test_start_date: test_end_date]

    x_train, y_train = prepare_label(data_train)
    x_validate, y_validate = prepare_label(data_validation)
    x_test, y_test = prepare_label(data_test)

    print("Training period: {} - {}".format(training_start_date, training_end_date))
    print("Training x[{}], label[{}]".format(x_train.shape, y_train.shape))

    print("Validation period: {} - {}".format(validation_start_date, validation_end_date))
    print("Validation x[{}], label[{}]".format(x_validate.shape, y_validate.shape))

    print("Test period: {} - {}".format(test_start_date, test_end_date))
    print("Test x[{}], label[{}]".format(x_test.shape, y_test.shape))

    # Prepare dates index file
    #pd.DataFrame(dates_train).to_csv(_filenames.train_dates)
    #pd.DataFrame(dates_test).to_csv(_filenames.test_dates)

    # Prepare input files
    pd.DataFrame(x_train).to_csv(_filenames.train_input)
    pd.DataFrame(x_test).to_csv(_filenames.test_input)


if __name__ == "__main__":
    df = datasets.load_HSI()

    start_date = datetime(2008, 7, 1)

    for i in range(24):
        print('run_{}'.format(i))
        study_number = datetime.now().strftime('%Y%m%d_%H%M%S')

        filenames = workflow.StudyFilenames(i, study_number)

        run(filenames, start_date, 2, 3, 3)
        start_date = start_date + relativedelta(months=+3)






