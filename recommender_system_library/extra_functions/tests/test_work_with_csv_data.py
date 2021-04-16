import os
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from recommender_system_library.extra_functions.work_with_csv_data import *


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


class TestWriteDataToNPZ(unittest.TestCase):

    MATRIX = sparse.coo_matrix([[1, 2], [3, 4]])
    NOT_MATRIX = [[1, 2], [3, 4]]
    PATH = 'matrix.npz'
    NOT_PATH = 1

    def tearDown(self) -> None:
        if os.path.exists(self.PATH):
            os.remove(self.PATH)

    def test_correct_behavior(self) -> None:
        write_matrix_to_npz(self.MATRIX, self.PATH)
        self.assertTrue(os.path.exists(self.PATH))

    def test_raise_type_error(self) -> None:
        self.assertRaises(TypeError, write_matrix_to_npz, self.NOT_MATRIX, self.PATH)
        self.assertRaises(TypeError, write_matrix_to_npz, self.MATRIX, self.NOT_PATH)


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
