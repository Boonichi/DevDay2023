import numpy as np
import pandas as pd
import unittest
from devday2023.to_share.Evaluator import rmr_score

eps = 1e-9

class TestRmrScore(unittest.TestCase):
    def test_rmr_score_with_outliers(self):
        # Create actual and predict arrays with outliers
        actual = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 100000000])
        predict = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])

        # Calculate expected rmr score without outliers
        expected_actual = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) + eps
        expected_predict = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
        expected_mx = sum(abs(expected_actual - expected_predict)) / sum(expected_actual)
        expected_rmr_score = expected_mx * 100

        # Calculate actual rmr score with outliers removed
        actual_rmr_score = rmr_score(actual, predict)

        # Assert equality of expected and actual rmr score
        self.assertEqual(expected_rmr_score, actual_rmr_score)

    def test_rmr_score_without_outliers(self):
        # Create actual and predict arrays without outliers
        actual = np.array([3, 4, 5, 6, 7])
        predict = np.array([4, 5, 6, 7, 8])

        # Calculate expected rmr score without outliers
        expected_actual = actual + eps
        expected_predict = predict
        expected_mx = sum(abs(expected_actual - expected_predict)) / sum(expected_actual)
        expected_rmr_score = expected_mx * 100

        # Calculate actual rmr score with outliers removed
        actual_rmr_score = rmr_score(actual, predict)

        # Assert equality of expected and actual rmr score
        self.assertEqual(expected_rmr_score, actual_rmr_score)


if __name__ == '__main__':
    unittest.main()
