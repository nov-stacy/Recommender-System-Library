import unittest

import numpy as np

from recommender_systems.metrics import normalized_discounted_cumulative_gain as ndcg


class TestNormalizedDiscountedcumulativeGain(unittest.TestCase):

    TRUE_RATINGS = [np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.3, 0.2]), np.array([0.2, 0.1, 0.3]),
                    np.array([0.2, 0.3, 0.1]), np.array([0.3, 0.1, 0.2]), np.array([0.3, 0.2, 0.1])]
    PREDICTED_RATINGS = [np.array([0.1, 0.2, 0.3]) for _ in range(len(TRUE_RATINGS))]
    RESULT = np.array([1.0, 0.91567565, 0.9720887, 0.809087002, 0.85784986, 0.77917246])
    NOT_TRUE_RATINGS_1 = np.array([0, 1, 2])
    NOT_TRUE_RATINGS_2 = [np.array([0, 1, 2]), np.array([1, 2])]
    NOT_TRUE_RATINGS_3 = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
    NOT_TRUE_RATINGS_4 = [np.array(['1', '2', '3']) for _ in range(len(TRUE_RATINGS))]
    NOT_TRUE_RATINGS_5 = [np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])]
    NOT_TRUE_RATINGS_6 = [np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.3, 0.2]]), np.array([[0.2, 0.1, 0.3]]),
                          np.array([[0.2, 0.3, 0.1]]), np.array([[0.3, 0.1, 0.2]]), np.array([[0.3, 0.2, 0.1]])]
    NOT_PREDICTED_RATINGS_1 = np.array([1, 2])
    NOT_PREDICTED_RATINGS_2 = [np.array([1, 2]), np.array([1, 2, 3])]
    NOT_PREDICTED_RATINGS_3 = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
    NOT_PREDICTED_RATINGS_4 = [np.array(['1', '2', '3']) for _ in range(len(TRUE_RATINGS))]
    NOT_PREDICTED_RATINGS_5 = [np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])]
    NOT_PREDICTED_RATINGS_6 = [np.array([[0.1, 0.2, 0.3]]) for _ in range(len(TRUE_RATINGS))]

    def test_correct_behavior(self) -> None:
        for true, predicted, result in zip(self.TRUE_RATINGS, self.PREDICTED_RATINGS, self.RESULT):
            self.assertAlmostEqual(ndcg([true], [predicted]), result)

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, ndcg, self.NOT_TRUE_RATINGS_1, self.PREDICTED_RATINGS)
        self.assertRaises(TypeError, ndcg, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_1)
        self.assertRaises(TypeError, ndcg, self.NOT_TRUE_RATINGS_3, self.PREDICTED_RATINGS)
        self.assertRaises(TypeError, ndcg, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_3)
        self.assertRaises(TypeError, ndcg, self.NOT_TRUE_RATINGS_4, self.PREDICTED_RATINGS)
        self.assertRaises(TypeError, ndcg, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_4)

    def test_raise_value_error(self) -> None:
        self.assertRaises(ValueError, ndcg, self.NOT_TRUE_RATINGS_2, self.PREDICTED_RATINGS)
        self.assertRaises(ValueError, ndcg, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_2)
        self.assertRaises(ValueError, ndcg, self.NOT_TRUE_RATINGS_5, self.PREDICTED_RATINGS)
        self.assertRaises(ValueError, ndcg, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_5)
        self.assertRaises(ValueError, ndcg, self.NOT_TRUE_RATINGS_6, self.PREDICTED_RATINGS)
        self.assertRaises(ValueError, ndcg, self.TRUE_RATINGS, self.NOT_PREDICTED_RATINGS_6)
