from unittest import TestCase
import numpy as np
import study_lstm as study


class TestStudyLSTM(TestCase):
    def test_performance(self):
        # actual direction = [+1, +1, -1, -1]
        y_test = np.array([100, 110, 120, 115, 105])

        # predict direction = [-1, +1, -1, -1]
        y_predict = np.array([90, 95, 120, 115, 105])

        assert study.performance(y_test, y_predict) == 3 / 4

    def test_shape_data(self):
        data = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [11, 12, 13, 14]])

        label = np.array([1, 2, 3, 4])

        # test 1
        res_data, res_label = study.shape_data(data, label, time_step=1, sliding_step=2)
        expect_data = np.array([[[1, 2, 3, 4]],
                                [[9, 10, 11, 12]]])

        expect_label = np.array([1, 3])

        assert np.array_equal(expect_data, res_data)
        assert np.array_equal(expect_label, res_label)

        # test 2
        res_data, res_label = study.shape_data(data, label, time_step=2, sliding_step=1)

        expect_data = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]],
                                [[5, 6, 7, 8], [9, 10, 11, 12]],
                                [[9, 10, 11, 12], [11, 12, 13, 14]]])

        expect_label = np.array([2, 3, 4])

        assert np.array_equal(expect_data, res_data)
        assert np.array_equal(expect_label, res_label)