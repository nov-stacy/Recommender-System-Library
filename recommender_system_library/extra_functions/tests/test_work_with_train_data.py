import os
import unittest

import numpy as np
from scipy import sparse

from recommender_system_library.extra_functions.work_with_csv_data import write_matrix_to_npz
from recommender_system_library.extra_functions.work_with_train_data import *


class TestReadMatrixFromFile(unittest.TestCase):

    MATRIX = sparse.coo_matrix([[1, 2], [3, 4]])
    PATH = 'matrix.npz'
    NOT_PATH = 1

    def tearDown(self) -> None:
        if os.path.exists(self.PATH):
            os.remove(self.PATH)

    def test_correct_behavior(self):
        write_matrix_to_npz(self.MATRIX, self.PATH)
        matrix = read_matrix_from_file(self.PATH)
        self.assertIsInstance(matrix, sparse.coo_matrix)
        self.assertTrue(np.array_equal(matrix.toarray(), self.MATRIX.toarray()))

    def test_raise_type_error(self):
        self.assertRaises(TypeError, read_matrix_from_file, self.NOT_PATH)


class TestConstructCooMatrixFromData(unittest.TestCase):

    ROWS = np.array([0, 0, 1])
    COLS = np.array([0, 1, 0])
    DATA = np.array([0.5, 0.5, 0.5])
    DATA_1 = np.array([1, 1, 1])
    SHAPE = (4, 3)
    SHAPE_1 = [4, 3]
    MATRIX = sparse.coo_matrix([[0.5, 0.5, 0], [0.5, 0, 0], [0, 0, 0], [0, 0, 0]])
    MATRIX_1 = sparse.coo_matrix([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    NOT_ROWS_1 = [0, 0, 1]
    NOT_ROWS_2 = np.array([[0, 0], [1, 1]])
    NOT_ROWS_3 = np.array([0, 0]), np.array([0, 0, 1, 1])
    NOT_ROWS_4 = np.array(['0', '0', '1'])
    NOT_COLS_1 = [0, 1, 0]
    NOT_COLS_2 = np.array([[0, 0], [1, 1]])
    NOT_COLS_3 = np.array([0, 1]), np.array([0, 1, 0, 1])
    NOT_COLS_4 = np.array(['0', '1', '0'])
    NOT_DATA_1 = [0.5, 0.5, 0.5]
    NOT_DATA_2 = np.array([[0, 0], [1, 1]])
    NOT_DATA_3 = np.array([0.5, 0.5]), np.array([0.5, 0.5, 0.5, 0.5])
    NOT_DATA_4 = np.array(['0.5', '0.5', '0.5'])
    NOT_SHAPE_1 = 1
    NOT_SHAPE_2 = (4, 3, 2), (4, )
    NOT_SHAPE_3 = ('4', 3), (4, '3')
    NOT_SHAPE_4 = (-4, 3), (3, -4)

    def test_correct_behavior(self):
        matrix_1 = construct_matrix_from_data(self.ROWS, self.COLS, self.DATA, self.SHAPE)
        matrix_2 = construct_matrix_from_data(self.ROWS, self.COLS, self.DATA, self.SHAPE_1)
        matrix_3 = construct_matrix_from_data(self.ROWS, self.COLS, self.DATA_1, self.SHAPE)
        matrix_4 = construct_matrix_from_data(self.ROWS, self.COLS, self.DATA_1, self.SHAPE_1)
        self.assertTrue(np.array_equal(matrix_1.toarray(), self.MATRIX.toarray()))
        self.assertTrue(np.array_equal(matrix_2.toarray(), self.MATRIX.toarray()))
        self.assertTrue(np.array_equal(matrix_3.toarray(), self.MATRIX_1.toarray()))
        self.assertTrue(np.array_equal(matrix_4.toarray(), self.MATRIX_1.toarray()))

    def test_raise_type_error(self):
        self.assertRaises(TypeError, construct_matrix_from_data, self.NOT_ROWS_1, self.COLS, self.DATA, self.SHAPE)
        self.assertRaises(TypeError, construct_matrix_from_data, self.ROWS, self.NOT_COLS_1, self.DATA, self.SHAPE)
        self.assertRaises(TypeError, construct_matrix_from_data, self.ROWS, self.COLS, self.NOT_DATA_1, self.SHAPE)
        self.assertRaises(TypeError, construct_matrix_from_data, self.NOT_ROWS_4, self.COLS, self.DATA, self.SHAPE)
        self.assertRaises(TypeError, construct_matrix_from_data, self.ROWS, self.NOT_COLS_4, self.DATA, self.SHAPE)
        self.assertRaises(TypeError, construct_matrix_from_data, self.ROWS, self.COLS, self.NOT_DATA_4, self.SHAPE)
        self.assertRaises(TypeError, construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_1)
        self.assertRaises(TypeError, construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_3[0])
        self.assertRaises(TypeError, construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_3[1])

    def test_raise_value_error(self):
        self.assertRaises(ValueError, construct_matrix_from_data, self.NOT_ROWS_2, self.COLS, self.DATA, self.SHAPE)
        self.assertRaises(ValueError, construct_matrix_from_data, self.ROWS, self.NOT_COLS_2, self.DATA, self.SHAPE)
        self.assertRaises(ValueError, construct_matrix_from_data, self.ROWS, self.COLS, self.NOT_DATA_2, self.SHAPE)
        self.assertRaises(ValueError, construct_matrix_from_data, self.NOT_ROWS_3[0], self.COLS, self.DATA, self.SHAPE)
        self.assertRaises(ValueError, construct_matrix_from_data, self.ROWS, self.NOT_COLS_3[0], self.DATA, self.SHAPE)
        self.assertRaises(ValueError, construct_matrix_from_data, self.ROWS, self.COLS, self.NOT_DATA_3[0], self.SHAPE)
        self.assertRaises(ValueError, construct_matrix_from_data, self.NOT_ROWS_3[1], self.COLS, self.DATA, self.SHAPE)
        self.assertRaises(ValueError, construct_matrix_from_data, self.ROWS, self.NOT_COLS_3[1], self.DATA, self.SHAPE)
        self.assertRaises(ValueError, construct_matrix_from_data, self.ROWS, self.COLS, self.NOT_DATA_3[1], self.SHAPE)
        self.assertRaises(ValueError, construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_2[0])
        self.assertRaises(ValueError, construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_2[1])
        self.assertRaises(ValueError, construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_4[0])
        self.assertRaises(ValueError, construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_4[1])


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

    def test_correct_behavior(self):
        for proportion in np.arange(0, 1, 0.1):
            matrix = get_train_matrix(self.MATRIX, proportion)
            not_null = np.sum(self.MATRIX.toarray() != 0)
            not_null_1 = np.sum(matrix.toarray() != 0)
            self.assertEqual(int(not_null * proportion), not_null_1)

    def test_raise_type_error(self):
        self.assertRaises(TypeError, get_train_matrix, self.NOT_MATRIX, self.PROPORTIONS)
        self.assertRaises(TypeError, get_train_matrix, self.MATRIX, self.NOT_PROPORTION_1)

    def test_raise_value_error(self):
        self.assertRaises(ValueError, get_train_matrix, self.MATRIX, self.NOT_PROPORTION_2)
        self.assertRaises(ValueError, get_train_matrix, self.MATRIX, self.NOT_PROPORTION_3)
