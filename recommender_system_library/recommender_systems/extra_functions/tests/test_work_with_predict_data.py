import unittest

import numpy as np

from recommender_systems.extra_functions.work_with_ratings import calculate_predicted_items


class TestCalculateIssueRankedList(unittest.TestCase):

    TEST_SIZE = 1000
    SIZE = 100
    RATINGS = np.array([1, 2, 3])
    NOT_RATINGS_1 = [1, 2, 3]
    NOT_RATINGS_2 = np.array(['1', '2', '3'])
    MATRIX_RATINGS = np.array([[1, 2, 3], [1, 2, 3]])
    K_ITEMS = 1
    NOT_K_ITEMS_1 = -1, len(RATINGS) + 1
    NOT_K_ITEMS_2 = '1'
    BARRIER_VALUE = 0.5
    NOT_BARRIER_VALUE = '0.5'

    def test_correct_behavior(self) -> None:

        for _ in range(self.TEST_SIZE):

            data = np.random.random(self.SIZE)
            indices, sorted_data = list(zip(*sorted(enumerate(data), key=lambda x: x[1], reverse=True)))
            indices, sorted_data = np.array(indices), np.array(sorted_data)

            for k in range(self.SIZE):
                predict = calculate_predicted_items(data, k_items=k)
                self.assertIsInstance(predict, np.ndarray)
                self.assertTrue(np.array_equal(indices[:k], predict))

            for barrier_value in data:
                predict = calculate_predicted_items(data, barrier_value=barrier_value)
                self.assertIsInstance(predict, np.ndarray)
                self.assertTrue(np.array_equal(indices[sorted_data >= barrier_value], predict))

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, calculate_predicted_items, self.NOT_RATINGS_1, k_items=self.K_ITEMS)
        self.assertRaises(TypeError, calculate_predicted_items, self.NOT_RATINGS_2, k_items=self.K_ITEMS)
        self.assertRaises(TypeError, calculate_predicted_items, self.RATINGS, k_items=self.NOT_K_ITEMS_2)
        self.assertRaises(TypeError, calculate_predicted_items, self.RATINGS, barrier_value=self.NOT_BARRIER_VALUE)

    def test_raise_value_error(self) -> None:
        self.assertRaises(ValueError, calculate_predicted_items, self.RATINGS)
        self.assertRaises(ValueError, calculate_predicted_items, self.RATINGS, k_items=self.NOT_K_ITEMS_1[0])
        self.assertRaises(ValueError, calculate_predicted_items, self.RATINGS, k_items=self.NOT_K_ITEMS_1[1])
        self.assertRaises(ValueError, calculate_predicted_items, self.MATRIX_RATINGS, k_items=self.K_ITEMS)
        self.assertRaises(ValueError, calculate_predicted_items, self.RATINGS, k_items=self.K_ITEMS,
                                                                               barrier_value=self.BARRIER_VALUE)
