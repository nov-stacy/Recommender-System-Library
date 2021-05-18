import unittest

import numpy as np

from recommender_systems.metrics import mean_absolute_error


class TestMeanSquareError(unittest.TestCase):

    TRUE_RATINGS = [np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2]), np.array([0, 0.1]), np.array([0.1])]
    PREDICTED_RATINGS = [ratings + 0.1 for ratings in TRUE_RATINGS]
    RESULT = np.array([0.1, 0.1, 0.1])
    NOT_TRUE_RATINGS_1 = np.array([0, 1, 2])
    NOT_TRUE_RATINGS_2 = [np.array([0, 1, 2]), np.array([1, 2])]
    NOT_TRUE_RATINGS_3 = [[0, 1, 2], [1, 2], [0, 1], [1]]
    NOT_TRUE_RATINGS_4 = [np.array(['1', '2', '3']) for _ in range(len(TRUE_RATINGS))]
    NOT_TRUE_RATINGS_5 = [np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])]
    NOT_TRUE_RATINGS_6 = [np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.2]]), np.array([[0, 0.1]]), np.array([[0.1]])]
    NOT_PREDICTED_RATINGS_1 = np.array([1, 2])
    NOT_PREDICTED_RATINGS_2 = [np.array([1, 2]), np.array([1, 2, 3])]
    NOT_PREDICTED_RATINGS_3 = [[1, 2], [1, 2, 3], [1, 2, 3], [1]]
    NOT_PREDICTED_RATINGS_4 = [np.array(['1', '2', '3']) for _ in range(len(TRUE_RATINGS))]
    NOT_PREDICTED_RATINGS_5 = [np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])]
    NOT_PREDICTED_RATINGS_6 = [np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.2]]), np.array([[0, 0.1]]), np.array([[0.1]])]

    def test_correct_behavior(self) -> None:
        for true, predicted, result in zip(self.TRUE_RATINGS, self.PREDICTED_RATINGS, self.RESULT):
            self.assertAlmostEqual(mean_absolute_error([true], [predicted]), result)

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, mean_absolute_error, self.NOT_TRUE_RATINGS_1, self.PREDICTED_RATINGS)
        self.assertRaises(TypeError, mean_absolute_error, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_1)
        self.assertRaises(TypeError, mean_absolute_error, self.NOT_TRUE_RATINGS_3, self.PREDICTED_RATINGS)
        self.assertRaises(TypeError, mean_absolute_error, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_3)
        self.assertRaises(TypeError, mean_absolute_error, self.NOT_TRUE_RATINGS_4, self.PREDICTED_RATINGS)
        self.assertRaises(TypeError, mean_absolute_error, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_4)

    def test_raise_value_error(self) -> None:
        self.assertRaises(ValueError, mean_absolute_error, self.NOT_TRUE_RATINGS_2, self.PREDICTED_RATINGS)
        self.assertRaises(ValueError, mean_absolute_error, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_2)
        self.assertRaises(ValueError, mean_absolute_error, self.NOT_TRUE_RATINGS_5, self.PREDICTED_RATINGS)
        self.assertRaises(ValueError, mean_absolute_error, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_5)
        self.assertRaises(ValueError, mean_absolute_error, self.NOT_TRUE_RATINGS_6, self.PREDICTED_RATINGS)
        self.assertRaises(ValueError, mean_absolute_error, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_6)
