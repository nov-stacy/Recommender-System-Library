import unittest

import numpy as np

from recommender_system_library.metrics import recall_k


class TestRecallK(unittest.TestCase):

    TRUE_INDICES = [np.array([0, 1, 2]), np.array([1, 2]), np.array([0, 1]), np.array([1]), np.array([0])]
    PREDICTED_INDICES = [np.array([1, 2]), np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1]), np.array([1])]
    RECALL = [2 / 3, 2 / 2, 1 / 2, 1 / 1, 0 / 1]
    TRUE_INDICES_ZERO = [np.array([]), np.array([0])]
    PREDICTED_INDICES_ZERO = [np.array([0]), np.array([])]
    NOT_TRUE_INDICES_1 = np.array([0, 1, 2])
    NOT_TRUE_INDICES_2 = [np.array([0, 1, 2]), np.array([1, 2])]
    NOT_TRUE_INDICES_3 = [[0, 1, 2], [1, 2], [0, 1], [1], [0]]
    NOT_TRUE_INDICES_4 = [np.array([0.5]), np.array([0.5]), np.array([0.5]), np.array([0.5]), np.array([0.5])]
    NOT_TRUE_INDICES_5 = [np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])]
    NOT_TRUE_INDICES_6 = [np.array([[0, 1, 2]]), np.array([[1, 2]]), np.array([[0, 1]]), np.array([[1]]), np.array([[0]])]
    NOT_PREDICTED_INDICES_1 = np.array([1, 2])
    NOT_PREDICTED_INDICES_2 = [np.array([1, 2]), np.array([1, 2, 3])]
    NOT_PREDICTED_INDICES_3 = [[1, 2], [1, 2, 3], [1, 2, 3], [1], [1]]
    NOT_PREDICTED_INDICES_4 = [np.array([0.5]), np.array([0.5]), np.array([0.5]), np.array([0.5]), np.array([0.5])]
    NOT_PREDICTED_INDICES_5 = [np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1])]
    NOT_PREDICTED_INDICES_6 = [np.array([[1, 2]]), np.array([[1, 2, 3]]), np.array([[1, 2, 3]]), np.array([[1]]), np.array([[1]])]

    def test_correct_behavior(self) -> None:
        for true, predicted, result in zip(self.TRUE_INDICES, self.PREDICTED_INDICES, self.RECALL):
            self.assertAlmostEqual(recall_k([true], [predicted]), result)
        self.assertAlmostEqual(recall_k(self.TRUE_INDICES, self.PREDICTED_INDICES), np.mean(self.RECALL))
        self.assertAlmostEqual(recall_k(self.TRUE_INDICES_ZERO, self.PREDICTED_INDICES_ZERO), 0)

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, recall_k, self.NOT_TRUE_INDICES_1, self.PREDICTED_INDICES)
        self.assertRaises(TypeError, recall_k, self.TRUE_INDICES, self.NOT_PREDICTED_INDICES_1)
        self.assertRaises(TypeError, recall_k, self.NOT_TRUE_INDICES_3, self.PREDICTED_INDICES)
        self.assertRaises(TypeError, recall_k, self.TRUE_INDICES, self.NOT_PREDICTED_INDICES_3)
        self.assertRaises(TypeError, recall_k, self.NOT_TRUE_INDICES_4, self.PREDICTED_INDICES)
        self.assertRaises(TypeError, recall_k, self.TRUE_INDICES, self.NOT_PREDICTED_INDICES_4)

    def test_raise_value_error(self) -> None:
        self.assertRaises(ValueError, recall_k, self.NOT_TRUE_INDICES_2, self.PREDICTED_INDICES)
        self.assertRaises(ValueError, recall_k, self.TRUE_INDICES, self.NOT_PREDICTED_INDICES_2)
        self.assertRaises(ValueError, recall_k, self.NOT_TRUE_INDICES_5, self.PREDICTED_INDICES)
        self.assertRaises(ValueError, recall_k, self.TRUE_INDICES, self.NOT_PREDICTED_INDICES_5)
        self.assertRaises(ValueError, recall_k, self.NOT_TRUE_INDICES_6, self.PREDICTED_INDICES)
        self.assertRaises(ValueError, recall_k, self.TRUE_INDICES, self.NOT_PREDICTED_INDICES_6)
