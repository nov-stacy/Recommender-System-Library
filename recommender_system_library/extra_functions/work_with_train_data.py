import typing as tp

import scipy.sparse as sparse
import numpy as np


__all__ = [
    'read_matrix_from_file',
    'construct_matrix_from_data',
    'get_train_matrix'
]


def read_matrix_from_file(path_to_file: str) -> sparse.coo_matrix:
    """
    Method to load data from file with sparse view

    Parameters
    ----------
    path_to_file: str
        Path to file with data

    Raises
    ------
    TypeError
        If parameters don't have needed format

    Returns
    -------
    matrix: sparse matrix
    """

    if type(path_to_file) != str:
        raise TypeError('Path should have string format')

    return sparse.load_npz(path_to_file)


def construct_matrix_from_data(rows_indices: np.ndarray, columns_indices: np.ndarray,
                               data: np.ndarray, shape: tp.Tuple[int, int]) -> sparse.coo_matrix:
    """
    Method to generate sparse matrix from data

    Parameters
    ----------
    rows_indices: numpy array
        Line indices where values are
    columns_indices: numpy array
        Column indices where values are
    data: numpy array
        Values
    shape: (int, int)
        Matrix dimension (example: (2, 3))

    Raises
    ------
    TypeError
        If parameters don't have needed format
    ValueError
        If indices and data are not 1D arrays
        If indices and data don't  have same shape
        If indices don't have int values or data don't have int/float values
        If shape don't have two non-negative values

    Returns
    -------
    matrix: sparse matrix
    """

    if type(rows_indices) != np.ndarray or type(columns_indices) != np.ndarray or type(data) != np.ndarray:
        raise TypeError('Indices and data need to have numpy array format')

    if len(rows_indices.shape) != 1 or len(columns_indices.shape) != 1 or len(data.shape) != 1:
        raise ValueError('Indices and data need to be 1D arrays')

    if not(rows_indices.shape == columns_indices.shape == data.shape):
        raise ValueError('Indices and data need to have same shape')

    if rows_indices.dtype != np.int or columns_indices.dtype != np.int or data.dtype not in [np.int, np.float]:
        raise TypeError('Indices need to have int values and data need to have int/float values')

    if type(shape) not in [tuple, list]:
        raise TypeError('Shape need to nave list or tuple format')

    if len(shape) != 2:
        raise ValueError('Shape need to have two values')

    if type(shape[0]) != int or type(shape[1]) != int:
        raise TypeError('Shapes need have int format')

    if shape[0] < 0 or shape[1] < 0:
        raise ValueError('Shape need to have non-negative values')

    return sparse.coo_matrix((data, (rows_indices, columns_indices)), shape=shape)


def get_train_matrix(matrix: sparse.coo_matrix, proportion: float) -> sparse.coo_matrix:
    """
    Method to get train data

    Parameters
    ----------
    matrix: sparse matrix
        2-D matrix, where rows are users, and columns are items and at the intersection
        of a row and a column is the rating that this user has given to this item
    proportion: float
        The relative amount of data in the training dataset

    Raises
    ------
    TypeError
        If parameters don't have needed format
    ValueError
        If proportion is not in [0, 1]

    Returns
    -------
    train data: sparse matrix
    """

    if type(matrix) != sparse.coo_matrix:
        raise TypeError('Matrix should have sparse format')

    if type(proportion) not in [float, np.float64]:
        raise TypeError('Proportion should have float format')

    if proportion < 0 or proportion > 1:
        raise ValueError('Proportion need to be in [0, 1])')

    # get key matrix data: indices and values
    row, col, data = matrix.row, matrix.col, matrix.data

    # shuffle indices and get a part of data
    indices = np.arange(row.shape[0])
    np.random.shuffle(indices)
    indices_train = indices[:int(indices.shape[0] * proportion)]

    # get key matrix data about train samples
    row_train, col_train, data_train = row[indices_train], col[indices_train], data[indices_train]

    return sparse.coo_matrix((data_train, (row_train, col_train)), shape=matrix.shape)
