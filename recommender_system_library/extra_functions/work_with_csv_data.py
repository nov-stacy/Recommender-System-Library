import scipy.sparse as sparse
import pandas as pd
import numpy as np

from recommender_system_library.extra_functions.work_with_train_data import construct_matrix_from_data


__all__ = [
    'read_data_from_csv',
    'write_matrix_to_npz',
    'generate_sparse_matrix'
]


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


def write_matrix_to_npz(sparse_matrix: sparse.coo_matrix, path_to_file: str) -> None:
    """
    Method to save sparse matrix in file

    Parameters
    ----------
    sparse_matrix: sparse matrix
        Sparse data matrix
    path_to_file: str
        Path to file

    Raises
    ------
    TypeError
        If parameters don't have needed format
    """

    if type(sparse_matrix) != sparse.coo_matrix:
        raise TypeError('Matrix should have sparse format')

    if type(path_to_file) != str:
        raise TypeError('Path should have string format')

    sparse.save_npz(path_to_file, sparse_matrix)


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

    return construct_matrix_from_data(rows, columns, data, matrix_shape)
