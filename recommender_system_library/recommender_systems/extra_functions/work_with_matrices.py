import scipy.sparse as sparse
import numpy as np


__all__ = [
    'write_matrix_to_file',
    'read_matrix_from_file',
    'get_train_matrix'
]


def write_matrix_to_file(sparse_matrix: sparse.coo_matrix, path_to_file: str) -> None:
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
