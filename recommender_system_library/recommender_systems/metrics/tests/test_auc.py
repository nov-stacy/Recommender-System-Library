import unittest

import numpy as np

from recommender_systems.metrics import roc_auc


class TestMeanSquareError(unittest.TestCase):

    TRUE_INTEREST = [np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 1, 1]),
                     np.array([1, 0, 0]), np.array([1, 0, 1]), np.array([1, 1, 0])]
    PREDICTED_RATINGS = [np.array([0.1, 0.2, 0.3]) for _ in range(len(TRUE_INTEREST))]
    RESULT = np.array([1.0, 0.5, 1.0, 0.0, 0.5, 0.0])
    NOT_TRUE_INTEREST_1 = np.array([0, 1, 2])
    NOT_TRUE_INTEREST_2 = [np.array([0, 1, 2]), np.array([1, 2])]
    NOT_TRUE_INTEREST_3 = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]]
    NOT_TRUE_INTEREST_4 = [np.array(['1', '2', '3']) for _ in range(len(TRUE_INTEREST))]
    NOT_TRUE_INTEREST_5 = [np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])]
    NOT_TRUE_INTEREST_6 = [np.array([[0, 0, 1]]), np.array([[0, 1, 0]]), np.array([[0, 1, 1]]),
                           np.array([[1, 0, 0]]), np.array([1, 0, 1]), np.array([[1, 1, 0]])]
    NOT_TRUE_INTEREST_7 = [np.array([2, 0, 1]), np.array([0, 2, 0]), np.array([0, 1, 2]),
                           np.array([1, 0, 2]), np.array([2, 0, 1]), np.array([1, 2, 0])]
    NOT_PREDICTED_RATINGS_1 = np.array([1, 2])
    NOT_PREDICTED_RATINGS_2 = [np.array([1, 2]), np.array([1, 2, 3])]
    NOT_PREDICTED_RATINGS_3 = [[0.1, 0.2, 0.3] for _ in range(len(TRUE_INTEREST))]
    NOT_PREDICTED_RATINGS_4 = [np.array(['1', '2', '3']) for _ in range(len(TRUE_INTEREST))]
    NOT_PREDICTED_RATINGS_5 = [np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])]
    NOT_PREDICTED_RATINGS_6 = [np.array([[0.1, 0.2, 0.3]]) for _ in range(len(TRUE_INTEREST))]

    def test_correct_behavior(self) -> None:
        for true, predicted, result in zip(self.TRUE_INTEREST, self.PREDICTED_RATINGS, self.RESULT):
            self.assertAlmostEqual(roc_auc([true], [predicted]), result)

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, roc_auc, self.NOT_TRUE_INTEREST_1, self.PREDICTED_RATINGS)
        self.assertRaises(TypeError, roc_auc, self.TRUE_INTEREST, self.NOT_PREDICTED_RATINGS_1)
        self.assertRaises(TypeError, roc_auc, self.NOT_TRUE_INTEREST_3, self.PREDICTED_RATINGS)
        self.assertRaises(TypeError, roc_auc, self.TRUE_INTEREST, self.NOT_PREDICTED_RATINGS_3)
        self.assertRaises(TypeError, roc_auc, self.NOT_TRUE_INTEREST_4, self.PREDICTED_RATINGS)
        self.assertRaises(TypeError, roc_auc, self.TRUE_INTEREST, self.NOT_PREDICTED_RATINGS_4)

    def test_raise_value_error(self) -> None:
        self.assertRaises(ValueError, roc_auc, self.NOT_TRUE_INTEREST_2, self.PREDICTED_RATINGS)
        self.assertRaises(ValueError, roc_auc, self.TRUE_INTEREST, self.NOT_PREDICTED_RATINGS_2)
        self.assertRaises(ValueError, roc_auc, self.NOT_TRUE_INTEREST_5, self.PREDICTED_RATINGS)
        self.assertRaises(ValueError, roc_auc, self.TRUE_INTEREST, self.NOT_PREDICTED_RATINGS_5)
        self.assertRaises(ValueError, roc_auc, self.NOT_TRUE_INTEREST_6, self.PREDICTED_RATINGS)
        self.assertRaises(ValueError, roc_auc, self.TRUE_INTEREST, self.NOT_PREDICTED_RATINGS_6)
        self.assertRaises(ValueError, roc_auc, self.NOT_TRUE_INTEREST_7, self.PREDICTED_RATINGS)