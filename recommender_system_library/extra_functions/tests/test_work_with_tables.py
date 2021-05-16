import os
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from recommender_system_library.extra_functions.work_with_tables import *
from recommender_system_library.extra_functions import work_with_tables


class TestReadDataFromCSV(unittest.TestCase):

    DATA = pd.DataFrame({'1': [1, 2], '2': [3, 4]})
    PATH = 'test.csv'
    NOT_PATH = 1

    def tearDown(self) -> None:
        if os.path.exists(self.PATH):
            os.remove(self.PATH)

    def test_correct_behavior(self) -> None:
        self.DATA.to_csv(self.PATH, index=None)
        data = read_data_from_csv(self.PATH)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue(data.equals(self.DATA))

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, read_data_from_csv, self.NOT_PATH)


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

    def test_correct_behavior(self) -> None:
        matrix_1 = work_with_tables.__construct_matrix_from_data(self.ROWS, self.COLS, self.DATA, self.SHAPE)
        matrix_2 = work_with_tables.__construct_matrix_from_data(self.ROWS, self.COLS, self.DATA, self.SHAPE_1)
        matrix_3 = work_with_tables.__construct_matrix_from_data(self.ROWS, self.COLS, self.DATA_1, self.SHAPE)
        matrix_4 = work_with_tables.__construct_matrix_from_data(self.ROWS, self.COLS, self.DATA_1, self.SHAPE_1)
        self.assertTrue(np.array_equal(matrix_1.toarray(), self.MATRIX.toarray()))
        self.assertTrue(np.array_equal(matrix_2.toarray(), self.MATRIX.toarray()))
        self.assertTrue(np.array_equal(matrix_3.toarray(), self.MATRIX_1.toarray()))
        self.assertTrue(np.array_equal(matrix_4.toarray(), self.MATRIX_1.toarray()))

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, work_with_tables.__construct_matrix_from_data, self.NOT_ROWS_1, self.COLS, self.DATA, self.SHAPE)
        self.assertRaises(TypeError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.NOT_COLS_1, self.DATA, self.SHAPE)
        self.assertRaises(TypeError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.NOT_DATA_1, self.SHAPE)
        self.assertRaises(TypeError, work_with_tables.__construct_matrix_from_data, self.NOT_ROWS_4, self.COLS, self.DATA, self.SHAPE)
        self.assertRaises(TypeError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.NOT_COLS_4, self.DATA, self.SHAPE)
        self.assertRaises(TypeError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.NOT_DATA_4, self.SHAPE)
        self.assertRaises(TypeError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_1)
        self.assertRaises(TypeError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_3[0])
        self.assertRaises(TypeError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_3[1])

    def test_raise_value_error(self) -> None:
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.NOT_ROWS_2, self.COLS, self.DATA, self.SHAPE)
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.NOT_COLS_2, self.DATA, self.SHAPE)
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.NOT_DATA_2, self.SHAPE)
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.NOT_ROWS_3[0], self.COLS, self.DATA, self.SHAPE)
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.NOT_COLS_3[0], self.DATA, self.SHAPE)
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.NOT_DATA_3[0], self.SHAPE)
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.NOT_ROWS_3[1], self.COLS, self.DATA, self.SHAPE)
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.NOT_COLS_3[1], self.DATA, self.SHAPE)
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.NOT_DATA_3[1], self.SHAPE)
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_2[0])
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_2[1])
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_4[0])
        self.assertRaises(ValueError, work_with_tables.__construct_matrix_from_data, self.ROWS, self.COLS, self.DATA, self.NOT_SHAPE_4[1])


class TestGenerateSparseMatrix(unittest.TestCase):
    
    USER, ITEM, RATINGS = 'user', 'item', 'ratings'
    DATA = pd.DataFrame({USER: [1, 1, 2], ITEM: [1, 2, 1], RATINGS: [0.5, 0.5, 0.5]})
    RESULT_MATRIX = sparse.coo_matrix([[0.5, 0.5], [0.5, 0]])
    PATH = 'test.csv'

    def test_correct_behavior(self) -> None:
        matrix = generate_sparse_matrix(self.DATA, self.USER, self.ITEM, self.RATINGS)
        self.assertTrue(np.allclose(matrix.todense(),  self.RESULT_MATRIX.todense()))

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, generate_sparse_matrix, self.RESULT_MATRIX, self.USER, self.ITEM, self.RATINGS)
        self.assertRaises(TypeError, generate_sparse_matrix, self.DATA, 1, self.ITEM, self.RATINGS)
        self.assertRaises(TypeError, generate_sparse_matrix, self.DATA, self.USER, 1, self.RATINGS)
        self.assertRaises(TypeError, generate_sparse_matrix, self.DATA, self.USER, self.ITEM, 1)

    def test_raise_value_error(self) -> None:
        self.assertRaises(ValueError, generate_sparse_matrix, self.DATA, self.USER + '1', self.ITEM, self.RATINGS)
        self.assertRaises(ValueError, generate_sparse_matrix, self.DATA, self.USER, self.ITEM + '1', self.RATINGS)
        self.assertRaises(ValueError, generate_sparse_matrix, self.DATA, self.USER, self.ITEM, self.RATINGS + '1')
