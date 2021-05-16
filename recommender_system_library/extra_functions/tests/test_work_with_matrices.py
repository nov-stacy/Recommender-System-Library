import os
import unittest

import numpy as np
from scipy import sparse

from recommender_system_library.extra_functions.work_with_matrices import *


class TestWriteDataToFile(unittest.TestCase):

    MATRIX = sparse.coo_matrix([[1, 2], [3, 4]])
    NOT_MATRIX = [[1, 2], [3, 4]]
    PATH = 'matrix.npz'
    NOT_PATH = 1

    def tearDown(self) -> None:
        if os.path.exists(self.PATH):
            os.remove(self.PATH)

    def test_correct_behavior(self) -> None:
        write_matrix_to_file(self.MATRIX, self.PATH)
        self.assertTrue(os.path.exists(self.PATH))

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, write_matrix_to_file, self.NOT_MATRIX, self.PATH)
        self.assertRaises(TypeError, write_matrix_to_file, self.MATRIX, self.NOT_PATH)


class TestReadMatrixFromFile(unittest.TestCase):

    MATRIX = sparse.coo_matrix([[1, 2], [3, 4]])
    PATH = 'matrix.npz'
    NOT_PATH = 1

    def tearDown(self) -> None:
        if os.path.exists(self.PATH):
            os.remove(self.PATH)

    def test_correct_behavior(self) -> None:
        write_matrix_to_file(self.MATRIX, self.PATH)
        matrix = read_matrix_from_file(self.PATH)
        self.assertIsInstance(matrix, sparse.coo_matrix)
        self.assertTrue(np.array_equal(matrix.toarray(), self.MATRIX.toarray()))

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, read_matrix_from_file, self.NOT_PATH)


class TestGetTrainData(unittest.TestCase):

    DATA = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ROWS = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    COLS = np.array([0, 1, 2, 0, 2, 3, 4, 1, 3, 4])
    MATRIX = sparse.coo_matrix((DATA, (ROWS, COLS)), shape=(3, 5))
    PROPORTIONS = 0.5
    NOT_MATRIX = np.array([[1, 2, 3], [3, 4, 5]])
    NOT_PROPORTION_1 = '1'
    NOT_PROPORTION_2 = -0.5
    NOT_PROPORTION_3 = 1.5

    def test_correct_behavior(self) -> None:
        for proportion in np.arange(0, 1, 0.1):
            matrix = get_train_matrix(self.MATRIX, proportion)
            not_null = np.sum(self.MATRIX.toarray() != 0)
            not_null_1 = np.sum(matrix.toarray() != 0)
            self.assertEqual(int(not_null * proportion), not_null_1)

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, get_train_matrix, self.NOT_MATRIX, self.PROPORTIONS)
        self.assertRaises(TypeError, get_train_matrix, self.MATRIX, self.NOT_PROPORTION_1)

    def test_raise_value_error(self) -> None:
        self.assertRaises(ValueError, get_train_matrix, self.MATRIX, self.NOT_PROPORTION_2)
        self.assertRaises(ValueError, get_train_matrix, self.MATRIX, self.NOT_PROPORTION_3)
