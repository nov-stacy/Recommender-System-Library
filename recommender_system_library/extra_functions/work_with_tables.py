import typing as tp

import scipy.sparse as sparse
import pandas as pd
import numpy as np


__all__ = [
    'read_data_from_csv',
    'generate_sparse_matrix'
]


def _construct_matrix_from_data(rows_indices: np.ndarray, columns_indices: np.ndarray,
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

    if type(shape[0]) not in [int, np.int64] or type(shape[1]) not in [int, np.int64]:
        raise TypeError('Shapes need have int format')

    if shape[0] < 0 or shape[1] < 0:
        raise ValueError('Shape need to have non-negative values')

    return sparse.coo_matrix((data, (rows_indices, columns_indices)), shape=shape)


def read_data_from_csv(path_to_file: str) -> pd.DataFrame:
    """
    Method to load data from file with table view

    Parameters
    ----------
    path_to_file: str
        Path to file with data

    Raises
    ------
    TypeError
        If parameters don't have string format

    Returns
    -------
    table: pandas DataFrame
    """

    if type(path_to_file) != str:
        raise TypeError('Path should have string format')

    return pd.read_csv(path_to_file)


def generate_sparse_matrix(table: pd.DataFrame, column_user_id: str, column_item_id: str,
                           column_rating: str) -> sparse.coo_matrix:
    """
    Method for generating a two-dimensional matrix user - item from table

    Parameters
    ----------
    table: pandas DataFrame
        Data to be used
    column_user_id: str
        The name of the column where the users ID are stored
    column_item_id: str
        The name of the column where the items ID are stored
    column_rating: str
        The name of the column where the ratings are stored

    Raises
    ------
    ValueError
        If table don't contain the columns
    TypeError
        If parameters don't have needed format

    Returns
    -------
    matrix: sparse matrix
    """

    if type(table) != pd.DataFrame:
        raise TypeError('Table should have pandas table format')

    if type(column_user_id) != str or type(column_item_id) != str or type(column_rating) != str:
        raise TypeError('Columns should have string format')

    if column_user_id not in table or column_item_id not in table or column_rating not in table:
        raise ValueError('Columns should be in table')

    # unique user ID and transformer from them to indices
    user_unique_ids = table[column_user_id].unique()
    dict_from_user_ids_to_indices = dict(zip(user_unique_ids, range(len(user_unique_ids))))

    # unique item ID and transformer from them to indices
    item_unique_ids = table[column_item_id].unique()
    dict_from_item_ids_to_indices = dict(zip(item_unique_ids, range(len(item_unique_ids))))

    # shape of matrix user - item
    matrix_shape = user_unique_ids.shape[0], item_unique_ids.shape[0]

    # indices where the data and the data itself are located
    rows = np.array(table[column_user_id].apply(lambda user_id: dict_from_user_ids_to_indices[user_id]))
    columns = np.array(table[column_item_id].apply(lambda user_id: dict_from_item_ids_to_indices[user_id]))
    data = np.array(table[column_rating])

    return _construct_matrix_from_data(rows, columns, data, matrix_shape)
