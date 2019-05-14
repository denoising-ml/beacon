import study.module.module_workflow as workflow
import study.module.module_datasets as datasets
from datetime import datetime
from dateutil.relativedelta import relativedelta


def run(_start_date, training_years, validation_months, test_months):

    training_start_date = _start_date
    validation_start_date = training_start_date + relativedelta(years=training_years)
    test_start_date = validation_start_date + relativedelta(months=validation_months)

    training_end_date = validation_start_date - relativedelta(days=1)
    validation_end_date = test_start_date - relativedelta(days=1)
    test_end_date = test_start_date + relativedelta(months=+test_months) - relativedelta(days=1)

    print("Training period: {} - {}".format(training_start_date, training_end_date))
    print("Validation period: {} - {}".format(validation_start_date, validation_end_date))
    print("Test period: {} - {}".format(test_start_date, test_end_date))


if __name__ == "__main__":
    df = datasets.load_HSI()

    start_date = datetime(2008, 7, 1)

    for i in range(24):
        print('run_{}'.format(i))
        run(start_date, 2, 3, 3)
        start_date = start_date + relativedelta(months=+3)






